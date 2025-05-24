from    typing                  import  *
from    typing_extensions       import  Self

import  torch

from    ..layers        import  BaseModule, MLP, MultipoleGraphKernelLayer
from    ..utils._dtype  import  MultiLevelGraph


##################################################
##################################################
__all__ = [
    "MultipoleGraphNeuralOperator",
    "MGNO",
]


##################################################
##################################################
class MultipoleGraphNeuralOperator(BaseModule):
    """## Multipole Graph Neural Operator (MGNO)
    ### Integral operator via graph neural network with the V-cycle algorithm
    
    -----
    ### Description
    The Multipole Graph Neural Operator is an Integral Neural Operator, whose kernel is defined on radius graphs.
    Compared with the Graph Neural Operator, the Multipole Graph Neural Operator computes the kernel integration using multi-resolution subgraphs and the V-cycle.
    
    Reference:  https://proceedings.neurips.cc/paper_files/paper/2020/file/4b21cf96d4cf612f239a6c322b10c8fe-Paper.pdf

    -----
    ### Note
    1. This class assumes that the vertex sets of the input multigraphs are nested.
    2. This class instantiates a single hidden layer and use it multiple times.
    """
    def __init__(
            self,
            
            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            edge_channels:      int,
            
            n_layers:           int = 1,
            n_poles:            int = 3,
            lift_layer:         Sequence[int] = [256],
            kernel_layer:       Sequence[int] = [512, 1024],
            project_layer:      Sequence[int] = [256],
            
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """## The initializer of the class `MultipoleGraphNeuralOperator`
        
        Arguments:
            `in_channels` (`int`): The number of the input node features.
            `hidden_channels` (`int`): The number of the hidden node features.
            `out_channels` (`int`): The number of the output node features.
            `edge_channels` (`int`): The number of the edge features.
            
            `n_layers` (`int`, default: `1`): The number of the hidden layers.
            `n_poles` (`int`, default: `3`): The number of poles in the kernel function.
            `lift_layer` (`Sequence[int]`, default: `[256]`): The hidden layers of the lift function.
            `kernel_layer` (`Sequence[int]`, default: `[512, 1024]`): The hidden layers of the kernel function.
            `project_layer` (`Sequence[int]`, default: `[256]`): The hidden layers of the projection function.
            
            `activation_name` (`str`, default: `"relu"`): The name of the activation function. Available: `"relu"`, `"tanh"`, `"sigmoid"`.
            `activation_kwargs` (`dict[str, object]`, defaule: `{}`): The keyword arguments for the activation function.
        """
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"'n_layers' should be a positive integer, but got {n_layers}.")
        
        # Initialize the member variables
        self.__n_layers = n_layers
        
        # Define the subnetworks
        ## Lift
        self.network_lift = MLP(
            [in_channels, *lift_layer, hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        # Hidden layers
        self.network_hidden = MultipoleGraphKernelLayer(
            node_channels       = hidden_channels,
            edge_channels       = edge_channels,
            n_poles             = n_poles,
            kernel_layer        = kernel_layer,
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        ## Projection
        self.network_projection = MLP(
            [hidden_channels, *project_layer, out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        
        return

    
    def forward(self, G: MultiLevelData) -> torch.Tensor:
        """The forward propagation in the class `MultipoleGraphNeuralOperator`
        
        Given a graph, the forward method extracts the node attributes, edge indices and attributes, together with the multigrid information, and compute the kernel integration.
        
        Arguments:
            `G` (`torch_geometric.data.Data`): The input graph. The following attributes are required:
            * `x` (`torch.Tensor`): The node attributes.
            * `edge_index` (`torch.LongTensor`): The edge indices.
            * `edge_attr` (`torch.Tensor`): The edge attributes.
            * `mask_node` (`torch.LongTensor`): The node masking tensor.
            * `mask_edge` (`torch.LongTensor`): The edge masking tensor.

        Returns:
            `torch.Tensor`: The output node attributes.
        """
        x, edge_index, edge_attr    = G.x, G.edge_index, G.edge_attr
        mask_node: torch.LongTensor = getattr(G, self.attr_names('mask_node'))
        mask_edge: torch.LongTensor = getattr(G, self.attr_names('mask_edge'))
        
        x = self.network_lift.forward(x)
        for _ in range(self.n_layers):
            x = self.network_hidden.forward(
                x           = x,
                edge_index  = edge_index,
                edge_attr   = edge_attr,
                mask_node   = mask_node,
                mask_edge   = mask_edge,
            )
        x = self.network_projection.forward(x)
        
        return x
    
    
    @property
    def n_layers(self) -> int:
        return self.__n_layers


##################################################
##################################################
MGNO = MultipoleGraphNeuralOperator


##################################################
##################################################
# End of file