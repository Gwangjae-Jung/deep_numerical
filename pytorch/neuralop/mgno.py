from    typing                  import  *
from    typing_extensions       import  Self

import  torch
from    torch                   import  nn
from    torch_geometric.data    import  Data

from    ..layers                import  BaseModule, MLP, MultipoleGraphKernelLayer
from    ..utils                 import  get_activation, warn_redundant_arguments


##################################################
##################################################
__all__ = [
    # "BETA_MultipoleGraphNeuralOperator",
    # "BETA_MultipoleGraphNeuralOperatorLite",
    "MultipoleGraphNeuralOperator",
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
            
            attr_name_mask_node:            str = "mask_node",
            attr_name_mask_edge:            str = "mask_edge",
            attr_name_levelwise_edge_index: str = "levelwise_edge_index",
            
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `MultipoleGraphNeuralOperator`
        -----
        ### Arguments
        @ `in_channels`, `hidden_channels`, `out_channels` (`int`)
            * The number of the channels in the input, hidden, and output spaces.
            
        @ `edge_channels` (`int`)
            * The number of the edge channels.
        
        @ `n_layers` (`int`)
            * The number of the hidden layers.
        
        @ `lift_layer` (`Sequence[int]`, default: `None`), `kernel_layer` (`Sequence[int]`, default: `(512, 1024)`), and `project_layer` (`Sequence[int]`, default: `None`)
            * The hidden layers in the lift function, kernel function, and the projection function, respectively.
        
        @ `attr_name_*` (`str`)
            * The names of the attributes to be called from the input of the forward method.
            * These shoulb be given in the instantiation, since the corresponding attributes are not provided by `torch_geometric.data.Data`.
        
        @ `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Initialize the member variables
        self.activation         = get_activation(activation_name, activation_kwargs)
        self.__n_layers         = n_layers
        self.__activation_name  = activation_name
        self.__attr_names: dict[str, str] = {
            'mask_node':            attr_name_mask_node,
            'mask_edge':            attr_name_mask_edge,
            'levelwise_edge_index': attr_name_levelwise_edge_index,
        }
        
        # Define the subnetworks
        ## Lift
        self.network_lift       = MLP(
                                        [in_channels] + lift_layer + [hidden_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        # Hidden layers
        if n_layers <= 0:
            self.network_hidden = nn.Identity()
        else:
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
                                        [hidden_channels] + project_layer + [out_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        
        return
    
    
    def attr_names(self, key = str) -> str:
        """
        Available keys:
        * `'mask_node'`
        * `'mask_edge'`
        * `'levelwise_edge_index'`
        """
        return self.__attr_names[key]

    
    def forward(self, G: Data) -> torch.Tensor:
        """The forward propagation in the class `MultipoleGraphNeuralOperator`
        Given a graph, the forward method extracts the node attributes, edge indices and attributes, together with the multigrid information, and compute the kernel integration.
        
        -----
        ### Used attributes
        @ `x` (`torch.Tensor`)
            * The merged node attributes.
        
        @ `edge_index` (`torch.LongTensor`)
            * The merged edge connectivities.

        @ `edge_attr` (`torch.Tensor`)
            * The merged edge attributes.
        
        The following are the attributes which should be defined by the users.
        @ `mask_node_index` abd `mask_edge_index` (`tuple[torch.LongTensor]`)
            * Masking tensors to refer to the node/edge information of the subgraphs.
        """
        x, edge_index, edge_attr    = G.x, G.edge_index, G.edge_attr
        mask_node: torch.LongTensor = getattr(G, self.attr_names('mask_node'))
        mask_edge: torch.LongTensor = getattr(G, self.attr_names('mask_edge'))
        # levelwise_edge_index: torch.LongTensor = getattr(G, self.attr_names('levelwise_edge_index'))
        
        x = self.network_lift.forward(x)
        for _ in range(self.n_layers):
            x = self.network_hidden.forward(
                        x                       = x,
                        edge_index              = edge_index,
                        edge_attr               = edge_attr,
                        mask_node               = mask_node,
                        mask_edge               = mask_edge,
                        # levelwise_edge_index    = levelwise_edge_index,
                    )
            x = self.activation.forward(x)
        x = self.network_projection.forward(x)
        
        return x
    
    
    @property
    def n_layers(self) -> int:
        return self.__n_layers


##################################################
##################################################
# End of file