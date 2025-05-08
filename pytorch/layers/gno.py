from    typing                  import  Sequence
from    typing_extensions       import  Self

import  torch
from    torch                   import  nn
from    torch_geometric.nn      import  MessagePassing

from    .general                import  MLP


##################################################
##################################################
__all__ = [
    "GraphKernelLayer",
]


##################################################
##################################################
class GraphKernelLayer(MessagePassing):
    """## Graph kernel layer
    ### Kernel integration via message passing
    -----
    ### Description
    Given a batch of graph-based data of discretized functions, this class computes kernel integration via message passing.
    
    -----
    ### Remark
    1. This class computes as `NNConv` computes.
    2. The input tensor should be of shape `(batch_size, *shape_of_domain, hidden_channels)`.
    """
    def __init__(
            self,
            node_channels:      int,
            edge_channels:      int,
            kernel_layer:       Sequence[int] = [512, 1024],
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """## The initializer of `GraphKernelLayer`
        -----
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
        self.linear     = nn.Linear(node_channels, node_channels)
        self.kernel_fcn = MLP(
                                [edge_channels] + list(kernel_layer) + [node_channels ** 2],
                                activation_name     = activation_name,
                                activation_kwargs   = activation_kwargs
                            )
        return

    
    def forward(
            self,
            x:          torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr:  torch.Tensor,
        ) -> torch.Tensor:
        """The `forward` method of the class `GraphKernelLayer`
        
        -----
        ### Arguments
        @ `x` (`torch.Tensor`)
            * The tensor of the node features.
            * Shape: `(N, node_channels)`
        
        @ `edge_index` (`torch.Tensor`)
            * The tensor of the edge indices.
            * Shape: `(2, num_edges)`
        
        @ `edge_attr` (`torch.Tensor`)
            * The tensor of the edge features.
            * Aligned to `edge_index`.
            * Shape: `(num_edges, edge_channels)`
        """
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)
    
    
    def message(
            self,
            x_j:        torch.Tensor,
            edge_attr:  torch.Tensor,
        ) -> torch.Tensor:
        kernel = self.kernel_fcn.forward(edge_attr).reshape(-1, self.__node_channels, self.__node_channels)
        return torch.einsum("epq,eq->ep", [kernel, x_j])
    
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return aggr_out + self.linear.forward(x)
    
    
    def __repr__(self) -> str:
        return f"GraphKernelLayer(node_dim={self.__node_channels}, kernel_layer={self.kernel_fcn})"


##################################################
##################################################
# End of file