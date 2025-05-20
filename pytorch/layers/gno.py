from    typing                  import  Sequence
from    typing_extensions       import  Self

import  torch
from    torch                   import  nn
from    torch_geometric.nn      import  MessagePassing

from    .general                import  MLP
from    ..utils                 import  get_activation


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
            weighted_residual:  bool = True,
            activate:           bool = True,
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """## The initializer of `GraphKernelLayer`
        
        Arguments:
            `node_channels` (`int`): The number of the node features.
            `edge_channels` (`int`): The number of the edge features.
            `kernel_layer` (`Sequence[int]`, default: `[512, 1024]`): The hidden layers of the kernel function.
            `weighted_residual` (`bool`, default: `True`): Whether to use the weighted residual connection. If `False`, a linear layer is not applied to the node features, and the activation function is not applied to the output.
            `activate` (`bool`, default: `True`): Whether to use the activation function in the kernel function.
            `activation_name` (`str`, default: "tanh"): The name of the activation function.
            `activation_kwargs` (`dict[str, object]`, defaule: `{}`): The keyword arguments for the activation function.
        """
        super().__init__(aggr='mean')
        if weighted_residual == False:
            activate = False
        
        # Save some member variables for representation
        self.__node_channels = node_channels
        
        # Define the subnetworks
        self.linear     = nn.Linear(node_channels, node_channels) if weighted_residual else nn.Identity()
        self.kernel_fcn = MLP(
            [edge_channels, *kernel_layer, node_channels**2],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        self.activation = get_activation(activation_name, **activation_kwargs) if activate else nn.Identity()
        return

    
    def forward(
            self,
            x:          torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr:  torch.Tensor,
        ) -> torch.Tensor:
        """The `forward` method of the class `GraphKernelLayer`
        
        Arguments:
            `x` (`torch.Tensor`): The tensor of the node features.
                * Shape: `(num_nodes, node_channels)`
            `edge_index` (`torch.Tensor`): The tensor of the edge indices.
                * Shape: `(2, num_edges)`
            `edge_attr` (`torch.Tensor`): The tensor of the edge features.
                * Aligned to `edge_index`.
                * Shape: `(num_edges, edge_channels)`
        
        Returns:
            `torch.Tensor`: The tensor of the node features after the kernel integration.
                * Shape: `(num_nodes, node_channels)`
        """
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.activation.forward(x)
        return x
    
    
    def message(
            self,
            x_j:        torch.Tensor,
            edge_attr:  torch.Tensor,
        ) -> torch.Tensor:
        kernel = self.kernel_fcn.forward(edge_attr)
        kernel = kernel.reshape(-1, self.__node_channels, self.__node_channels)
        return torch.einsum("epq,eq->ep", [kernel, x_j])
    
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return aggr_out + self.linear.forward(x)
    
    
    def __repr__(self) -> str:
        return f"GraphKernelLayer(node_dim={self.__node_channels}, kernel_layer={self.kernel_fcn})"


##################################################
##################################################
# End of file