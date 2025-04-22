from    typing                  import  *
from    typing_extensions       import  Self

import  torch
from    torch                   import  nn
from    torch_geometric.data    import  Data

from    ..utils                 import  get_activation, warn_redundant_arguments
from    ..layers                import  BaseModule, MLP, GraphKernelLayer, VectorSelfAttention




##################################################
##################################################
__all__ = [
    "GraphNeuralOperator",
    "GraphNeuralOperatorLite",
    "GNO", "GKN", "GraphKernelNetwork",
    "GKNLite", "GNOLite", "GraphKernelNetworkLite",
    
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
            
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `GraphNeuralOperator`
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
        
        @ `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Initialize the member variables
        self.n_layers = n_layers
        
        # Define the subnetworks
        ## Lift
        self.network_lift       = MLP(
                                        [in_channels] + lift_layer + [hidden_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        # Hidden layers
        if n_layers > 0:
            __gkl_kwargs = {
                'node_channels'     : hidden_channels,
                'edge_channels'     : edge_channels,
                'kernel_layer'      : kernel_layer,
                'activation_name'   : activation_name,
                'activation_kwargs' : activation_kwargs,
            }
            self.network_hidden = nn.ModuleList([GraphKernelLayer(**__gkl_kwargs)])
            for _ in range(n_layers - 1):
                self.network_hidden.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden.append(GraphKernelLayer(**__gkl_kwargs))
        else:
            self.network_hidden: Callable[[Sequence[torch.Tensor]], torch.Tensor] = \
                                    lambda _input: _input[0]
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
        
        -----
        ### Argument
        @ `G` (`torch_geometric.data.Data`)
            * The input graph.
            
        ### Used attributes
        @ `G.x` (`torch.Tensor`)
            * A 2-tensor of the node features (shape: `(num_nodes, num_node_features)`).
            * Generally, the input node features are the values of the input function(s).
        
        @ `G.edge_index` (`torch.Tensor`)
            * A 2-tensor of the node features (shape: `(num_edges, 2)`).
        
        @ `G.edge_attr` (`torch.Tensor`)
            * A 2-tensor of the edge features (shape: `(num_edges, num_edge_features)`).
            * In this class, the edge features are assumed to be a combination of the position and the values of the input function(s).
        """
        x, edge_index, edge_attr = G.x, G.edge_index, G.edge_attr
        x = self.network_lift.forward(x)
        for cnt in range(self.n_layers - 1):
            x = self.network_hidden[2 * cnt].forward(x, edge_index, edge_attr)
            x = self.network_hidden[2 * cnt + 1].forward(x)
        x = self.network_hidden[-1].forward(x, edge_index, edge_attr)
        x = self.network_projection.forward(x)
        return x
    
    


class GraphNeuralOperatorLite(BaseModule):
    """## Graph Neural Operator Lite (GNO Lite)
    ### Integral operator via graph neural network
    -----
    ### Description
    The Graph Neural Operator is an Integral Neural Operator.
    In practice, using full graphs is too massive and requires high computational cost.
    One breakthrough is to use radius graphs, i.e., assuming that kernels decay fast away from the diagonal; another breakthrough is to sample graphs.
    
    Reference: https://openreview.net/pdf?id=fg2ZFmXFO3

    -----
    ### Note
    Given `n_layers`, this class instantiates a single `GraphKernelLayer` object and reuse it for `n_layers` times.
    To use `n_layers` distinct `GraphKernelLayer` instances, use `GraphNeuralOperator`, instead.
    """
    def __init__(
            self,
            
            in_channels:            int,
            hidden_channels:        int,
            out_channels:           int,
            edge_channels:          int,
            
            n_layers:               int,
            lift_layer:             Sequence[int] = [256],
            kernel_layer:           Sequence[int] = [512, 1024],
            project_layer:          Sequence[int] = [256],
            
            activation_name:        str = "relu",
            activation_kwargs:      dict[str, object] = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `GraphNeuralOperatorLite`
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
        
        @ `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Initialize the member variables
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
        if n_layers > 0:
            self.network_hidden = GraphKernelLayer(
                                        node_channels       = hidden_channels,
                                        edge_channels       = edge_channels,
                                        kernel_layer        = kernel_layer,
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        else:
            self.network_hidden: Callable[[Sequence[torch.Tensor]], torch.Tensor] = \
                                    lambda _input: _input[0]
        ## Projection
        self.network_projection = MLP(
                                        [hidden_channels] + project_layer + [out_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
        
        return

    
    def forward(self, G: Data) -> torch.Tensor:
        """The forward propagation in the class `GraphNeuralOperatorLite`
        Given a graph, the forward method extracts the node attributes, edge indices and attributes, and compute the kernel integration.
        
        -----
        ### Argument
        @ `G` (`torch_geometric.data.Data`)
            * The input graph.
            
        ### Used attributes
        @ `G.x` (`torch.Tensor`)
            * A 2-tensor of the node features (shape: `(num_nodes, num_node_features)`).
            * Generally, the input node features are the values of the input function(s).
        
        @ `G.edge_index` (`torch.Tensor`)
            * A 2-tensor of the node features (shape: `(num_edges, 2)`).
        
        @ `G.edge_attr` (`torch.Tensor`)
            * A 2-tensor of the edge features (shape: `(num_edges, num_edge_features)`).
            * In this class, the edge features are assumed to be a combination of the position and the values of the input function(s).
        """
        x, edge_index, edge_attr = G.x, G.edge_index, G.edge_attr
        x = self.network_lift.forward(x)
        for _ in range(self.n_layers - 1):
            x = self.network_hidden.forward(x, edge_index, edge_attr)
            x = self.activation.forward(x)
        x = self.network_hidden.forward(x, edge_index, edge_attr)
        x = self.network_projection.forward(x)
        return x


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
GKNLite = GNOLite   = GraphKernelNetworkLite    = GraphNeuralOperatorLite


##################################################
##################################################
# End of file