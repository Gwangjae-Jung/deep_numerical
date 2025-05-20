from    typing                  import  *
from    typing_extensions       import  Self

import  torch
from    torch                   import  nn
from    torch_geometric.data    import  Data

from    ..utils     import  get_activation, warn_redundant_arguments
from    ..layers    import  BaseModule, MLP, GraphKernelLayer
from    ..layers    import  VectorSelfAttention




##################################################
##################################################
__all__ = [
    "GraphNeuralOperator",
    "GNO", "GKN", "GraphKernelNetwork",
    
    "PointTransformerLite",
]


##################################################
##################################################
class GraphNeuralOperator(BaseModule):
    """## Graph Neural Operator (GNO)
    ### Integral operator via graph neural network
    
    -----
    ### Description
    The Graph Neural Operator is an Integral Neural Operator.
    In practice, using full graphs is too massive and requires high computational cost.
    One breakthrough is to use radius graphs, i.e., assuming that kernels decay fast away from the diagonal; another breakthrough is to sample graphs.
    
    Reference: https://openreview.net/pdf?id=fg2ZFmXFO3
    
    -----
    ### Note
    Given `n_layers`, this class instantiates `n_layers` distinct `GraphKernelLayer` objects.
    To reuse a single `GraphKernelLayer` instance for `n_layers` times, use `GraphNeuralOperatorLite`, instead.
    """
    def __init__(
            self,
            
            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            edge_channels:      int,
            
            n_layers:           int,
            lift_layer:         Sequence[int] = [256],
            kernel_layer:       Sequence[int] = [512, 1024],
            project_layer:      Sequence[int] = [256],
            weighted_residual:  bool = True,
            
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """## The initializer of the class `GraphNeuralOperator`
        
        Arguments:
            `in_channels` (`int`): The number of the input channels.
            `hidden_channels` (`int`): The number of the hidden channels.
            `out_channels` (`int`): The number of the output channels.
            `edge_channels` (`int`): The number of the edge channels.
            `n_layers` (`int`): The number of the hidden layers.
            `lift_layer` (`Sequence[int]`, default: `[256]`): The hidden layers in the lift function.
            `kernel_layer` (`Sequence[int]`, default: `[512, 1024]`): The hidden layers in the kernel function.
            `project_layer` (`Sequence[int]`, default: `[256]`): The hidden layers in the projection function.
            `weighted_residual` (`bool`, default: `True`): Whether to use the weighted residual connection.
            `activation_name` (`str`, default: `"relu"`): The activation function which shall be used in each hidden layer.
            `activation_kwargs` (`dict[str, object]`, default: `{}`): The keyword arguments for the activation function.
        """
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"'n_layers' must be greater than 0, but got {n_layers}.")
        # Initialize the member variables
        self.__n_layers = n_layers
        
        # Define the subnetworks
        ## Lift
        self.network_lift   = MLP(
            [in_channels] + lift_layer + [hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        # Hidden layers
        self.network_hidden = nn.ModuleList([])
        for idx in range(n_layers):
            activate = True if idx < n_layers-1 else False
            __gkl_kwargs = {
                'node_channels'     : hidden_channels,
                'edge_channels'     : edge_channels,
                'kernel_layer'      : kernel_layer,
                'weighted_residual' : weighted_residual,
                'activate'          : activate,
                'activation_name'   : activation_name,
                'activation_kwargs' : activation_kwargs,
            }
            self.network_hidden.append(GraphKernelLayer(**__gkl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels] + project_layer + [out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        
        return

    
    def forward(self, G: Data) -> torch.Tensor:
        """The forward propagation in the class `GraphNeuralOperator`
        
        Given a graph, the forward method extracts the node attributes, edge indices and attributes, and compute the kernel integration.
        
        Arguments:
            `G` (`torch_geometric.data.Data`): The input graph, where the attributes og `G` used in the computation are described below.        
                * `G.x` (`torch.Tensor`): A 2-tensor of the node features. (Shape: `(num_nodes, num_node_features)`)
                * `G.edge_index` (`torch.Tensor`): A 2-tensor of the node feature. (Shape: `(num_edges, 2)`)
                * `G.edge_attr` (`torch.Tensor`): A 2-tensor of the edge features. (Shape: `(num_edges, num_edge_features)`) In this class, the edge features are assumed to be a combination of the position and the values of the input function(s).
        """
        x, edge_index, edge_attr = G.x, G.edge_index, G.edge_attr
        x = self.network_lift.forward(x)
        for idx in range(self.__n_layers):
            x = self.network_hidden[idx].forward(x, edge_index, edge_attr)
        x = self.network_projection.forward(x)
        return x




##################################################
##################################################
class PointTransformerLite(BaseModule):
    def __init__(
            self,
            
            dim_domain: int,
            
            in_channels:            int,
            hidden_channels:        int,
            out_channels:           int,
            
            n_layers:               int,
            n_heads:                int = 1,
            lift_layer:             Sequence[int] = [256],
            project_layer:          Sequence[int] = [256],
            
            activation_name:        str = "relu",
            activation_kwargs:      dict[str, object] = {},
            
            **kwargs,
        ) -> Self:
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Initialize the member variables
        self.dim_domain         = dim_domain
        self.n_layers           = n_layers
        self.activation         = get_activation(activation_name, activation_kwargs)
        
        # Define the subnetworks
        ## Lift
        self.network_lift       = MLP(
                                        [in_channels] + lift_layer + [hidden_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        # Hidden layers
        self.vsa    = VectorSelfAttention(
                            channels    = hidden_channels,
                            n_heads     = n_heads,
                            pos_encoder = MLP([dim_domain, 256, 256, hidden_channels]),
                            use_softmax = True,
                            use_linear  = True,
                        )
        ## Projection
        self.network_projection = MLP(
                                        [hidden_channels] + project_layer + [out_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        
        return

    
    def forward(self, G: Data) -> torch.Tensor:
        x, edge_index = G.x, G.edge_index
        pos = x[..., -self.dim_domain:]
        x = self.network_lift.forward(x)
        for _ in range(self.n_layers - 1):
            x = self.vsa.forward(x, edge_index, pos)
            x = self.activation.forward(x)
        x = self.vsa.forward(x, edge_index, pos)
        x = self.network_projection.forward(x)
        return x


##################################################
##################################################
GKN     = GNO       = GraphKernelNetwork        = GraphNeuralOperator


##################################################
##################################################
# End of file