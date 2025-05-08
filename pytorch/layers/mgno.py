from    typing                  import  Sequence, Union
from    typing_extensions       import  Self

import  torch
from    torch                   import  nn
from    torch_geometric.nn      import  MessagePassing

from    ..layers        import  MLP, GraphKernelLayer
from    ..utils         import  get_activation




##################################################
##################################################


__all__ = [
    # "BETA_MultipoleGraphKernelLayer",
    "MultipoleGraphKernelLayer",
]


##################################################
##################################################


class _InterLevelFlow(MessagePassing):
    """An auxiliiary class to implement the V-cycle algorithm in Multipole Graph Neural Operator
    
    This class aims to implement the upward (without isolevel transform) and the downward pass of a graph in the V-cycle algorithm in Multipole Graph Neural Operator.
    For the forward propagation, the following information is given to the method `forward`:
    
        * The full `node_index`
        * `edge_index` and `edge_attr` of a certain level
    """
    def __init__(
            self,
            node_channels:      int,
            edge_channels:      int,
            kernel_layer:       Sequence[int]   = [512, 1024],
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """The initializer of `_InterLevelFlow`
        ### Arguments
        @ `node_channels` (`int`) and `edge_channels` (`int`):
            * The number of the node and edge features.
        
        @ `kernel_layer` (`Sequence[int]`, default: `[512, 1024]`)
            * The hidden layers of the kernel function.
        
        @ `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__(aggr = 'mean')
        # Save some member variables for representation
        self.__node_channels = node_channels
        # Define the subnetworks
        self.kernel_fcn = MLP(
            [edge_channels] + list(kernel_layer) + [node_channels ** 2],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        return

    
    def forward(
            self,
            x:              torch.Tensor,
            sub_edge_index: torch.LongTensor,
            sub_edge_attr:  torch.Tensor,
        ) -> torch.Tensor:
        """The `forward` method of the class `_InterLevelFlow`
        
        Computes the sum of the input node attributes `x` and the kernel value of `x`.
        
        -----
        ### Arguments
        @ `x` (`torch.Tensor`)
            * The tensor of the node features in the full (merged) graph.
            * Shape: `(num_nodes, node_channels)`
        
        @ `sub_edge_index` (`torch.LongTensor`)
            * The tensor of the edge indices for the subgraph.
            * Shape: `(2, num_edges{level=l})`
        
        @ `sub_edge_attr` (`torch.Tensor`)
            * The tensor of the edge features for the subgraph.
            * Aligned to `edge_index`.
            * Shape: `(num_edges{level=l}, edge_channels)`
        """
        return self.propagate(sub_edge_index, x = x, sub_edge_attr = sub_edge_attr)
    
    
    def message(
            self,
            x_j:            torch.Tensor,
            sub_edge_attr:  torch.Tensor,
        ) -> torch.Tensor:
        kernel = self.kernel_fcn.forward(sub_edge_attr).reshape(-1, self.__node_channels, self.__node_channels)
        return torch.einsum("epq,eq->ep", [kernel, x_j])
    
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + aggr_out




class MultipoleGraphKernelLayer(MessagePassing):
    """## Multipole Graph kernel layer
    ### Kernel integration via message passing with fast multipole method
    -----
    ### Description
    Given a batch of graph-based data of discretized functions, this class computes kernel integration via message passing.
    
    -----
    ### Remark
    1. The input tensor should be of shape `(batch_size, *shape_of_domain, hidden_channels)`.
    2. This class assumes that the input multigraphs are nested.
    """
    def __init__(
            self,
            node_channels:      int,
            edge_channels:      int,
            n_poles:            int                 = 3,
            kernel_layer:       Sequence[int]       = [512, 256, 128],
            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
        ) -> Self:
        """## The initializer of `GraphKernelLayer`
        -----
        ### Arguments
        @ `node_channels` (`int`) and `edge_channels` (`int`):
            * The number of the node and edge features.
        
        @ `kernel_layer` (`Sequence[int]`, default: `[512, 256, 128]`)
            * The sequence of the hidden channels of the kernel functions in each inner/inter level.
            * If `len(kernel_layer) > n_poles`, the redundant terms are ignored; if `len(kernel_layer) < n_poles`, the remaining terms are padded by the last dimension.
        
        @ `activation_name` (`str`, default: "relu") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__(aggr = 'mean')
        # Save some member variables for representation
        self.activation = get_activation(activation_name, activation_kwargs)
        self.__n_poles          = n_poles
        self.__node_channels    = node_channels
        self.__edge_channels    = edge_channels
        
        # Correct `kernel_layer`
        if len(kernel_layer) > n_poles:
            kernel_layer = kernel_layer[:n_poles]
        else:
            kernel_layer += [kernel_layer[-1]] * (n_poles - len(kernel_layer))
        
        # Downward transforms
        self.network_downward   = nn.ModuleList(
            [
                _InterLevelFlow(
                    node_channels       = node_channels,
                    edge_channels       = edge_channels,
                    kernel_layer        = [_ker_dim, _ker_dim],
                    activation_name     = activation_name,
                    activation_kwargs   = activation_kwargs,
                )
                for _ker_dim in kernel_layer[:-1]
            ]
        )
        
        # Isolevel transforms
        self.network_isolevel   = nn.ModuleList(
            [
                GraphKernelLayer(
                    node_channels       = node_channels,
                    edge_channels       = edge_channels,
                    kernel_layer        = [_ker_dim, _ker_dim],
                    activation_name     = activation_name,
                    activation_kwargs   = activation_kwargs,
                )
                for _ker_dim in kernel_layer
            ]
        )
        
        # Upward transforms
        self.network_upward     = nn.ModuleList(
            [
                _InterLevelFlow(
                    node_channels       = node_channels,
                    edge_channels       = edge_channels,
                    kernel_layer        = [_ker_dim, _ker_dim],
                    activation_name     = activation_name,
                    activation_kwargs   = activation_kwargs,
                )
                for _ker_dim in kernel_layer[:-1]
            ]
        )        
        
        return
    
    
    @property
    def n_poles(self) -> int:
        return self.__n_poles

    
    def forward(
                    self,
                    x:          torch.Tensor,
                    edge_index: torch.Tensor,
                    edge_attr:  torch.Tensor,
                    
                    mask_node:              Sequence[Union[torch.LongTensor, Sequence[slice]]],
                    mask_edge:              Sequence[Union[torch.LongTensor, Sequence[slice]]],
                    # levelwise_edge_attr:    Sequence[torch.LongTensor],
        ) -> torch.Tensor:
        """The `forward` method of the class `MultipleGraphKernelLayer`
        
        -----
        ### Arguments
        
        @ `x` (`torch.Tensor`)
            * A batch of node attributes.
            * The node attributes from all subgraphs are merged. To infer to the node attributes of certain subgraphs, use `ranges_node`.
        
        @ `mask_node` and `mask_edge` (`Sequence[Union[torch.LongTensor, Sequence[slice]]]`)
            * Sequences of masks used to refer to information of certain subgraphs.
        
        @ `levelwise_edge_index` (`Sequence[torch.LongTensor]`)
            * A sequence of levelwise edge connectivities.
            
        -----
        ### Remark
        
        As the input has nested vertex sets, the node attributes of the subgraph of each level changes in each downward pass. To use the output of each downward pass, the computed node attributes are saved in the list `storage_x`.
        To this end, this forward method requires
        
            * merged node/edge attributes,
            * ranges to infer the information of each subgraph.
        """
        # Instantiate the storage for the node attributes
        storage_x = [x]
            # Backup the initial tensor (current length is 1)
            # `x` will be modified later, so cloning `x` is redundant
        
        # Downward pass (Note in the downward passes that `x` is not the full graph)
        down: _InterLevelFlow
        for level_up, down in enumerate(self.network_downward, 1):
            _r_edge_up = mask_edge[level_up]
            x = down.forward(x, edge_index[:, _r_edge_up], edge_attr[_r_edge_up])
            x = self.activation.forward(x)
            storage_x.append(x) # Backup each downward pass (final length is `self.__n_poles`)
        
        # One isolevel pass in the coarsest level
        last_iso: GraphKernelLayer = self.network_isolevel[-1]
        _r_edge_iso = mask_edge[-1]
        x = last_iso.forward(x, edge_index[:, _r_edge_iso], edge_attr[_r_edge_iso])
        x = self.activation.forward(x)
        
        # Upward pass
        for cnt in range(1, self.__n_poles):
            idx_up, idx_iso         = -cnt, -(cnt + 1)
            _r_node_iso             = mask_node[idx_iso]
            _r_edge_iso, _r_edge_up = mask_edge[idx_iso], mask_edge[idx_up]
            
            up:     _InterLevelFlow     = self.network_upward[  idx_up ]
            iso:    GraphKernelLayer    = self.network_isolevel[idx_iso]
            
            x_up    = up.forward(x, edge_index[:, _r_edge_up],  edge_attr[_r_edge_up])[_r_node_iso]
            x_iso   = iso.forward(storage_x[idx_iso], edge_index[:, _r_edge_iso], edge_attr[_r_edge_iso])[_r_node_iso]
            x = x.clone()   # DO NOT REMOVE THIS LINE: CLONING IS ESSENTIAL
            x[_r_node_iso] = self.activation.forward(x_up + x_iso)
        
        return x
    
    
    def __repr__(self) -> str:
        return \
            f"MultipoleGraphKernelLayer("\
            f"node_channels={self.__node_channels}, "\
            f"edge_channels={self.__edge_channels}, "\
            f"n_poles={self.__n_poles} (down: {len(self.network_downward)}, iso: {len(self.network_isolevel)}, up: {len(self.network_upward)}))"




##################################################
##################################################

