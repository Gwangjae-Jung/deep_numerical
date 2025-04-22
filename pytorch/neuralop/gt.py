from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch   import  nn

from    ..utils             import  get_activation, warn_redundant_arguments
from    ..layers            import  BaseModule, MLP,    \
        GalerkinTypeEncoderBlockSelfAttention       as  GTencoderSA, \
        GalerkinTypeEncoderBlockCrossAttention      as  GTencoderCA



##################################################
##################################################
__all__ = ["GalerkinTransformerSelfAttention", "GalerkinTransformerCrossAttention"]


##################################################
##################################################
class GalerkinTransformerSelfAttention(BaseModule):
    """## Galerkin Transformer with self-attention
    ### Neural Operator based on a softmax-free self-attention
    
    -----
    ### Description
    Galerkin Transformer is a neural operator which adopts the self-attention.
    The self-attention is modified as follows:
        * The softmax is not used.
        * The layer normalization is done for the key and value tensors.
    
    Reference: https://proceedings.neurips.cc/paper/2021/file/d0921d442ee91b896ad95059d13df618-Supplemental.pdf
    """
    def __init__(
            self,
            dim_domain:         int,
            
            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            
            lift_layer:         Sequence[int] = [256],
            n_layers:           int = 4,
            n_heads:            int = 1,
            project_layer:      Sequence[int] = [256],
            
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
            
            **kwargs,
        ) -> Self:
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Save some member variables for representation
        self.__dim_domain       = dim_domain
        self.__in_channels      = in_channels
        self.__hidden_channels  = hidden_channels
        self.__out_channels     = out_channels
        self.__n_layers         = n_layers
        self.__n_heads          = n_heads
        
        # Set the subnetworks
        self.network_lift = MLP(
            [in_channels] + list(lift_layer) + [hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        if n_layers <= 0:
            self.network_hidden = nn.Identity()
        else:
            self.network_hidden = nn.Sequential()
            __enc_kwargs = {
                'dim_domain'            : dim_domain,
                'hidden_channels'       : hidden_channels,
                'n_heads'               : n_heads,
                'mlp_activation_name'   : activation_name,
                'mlp_activation_kwargs' : activation_kwargs,                
            }
            self.network_hidden.append(GTencoderSA(**__enc_kwargs))
            for _ in range(n_layers-1):
                self.network_hidden.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden.append(GTencoderSA(**__enc_kwargs))
        self.network_projection = MLP(
            [hidden_channels] + list(project_layer) + [out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X




class GalerkinTransformerCrossAttention(BaseModule):
    """## Galerkin Transformer with cross-attention
    ### Neural Operator based on a softmax-free cross-attention
    
    -----
    ### Description
    Galerkin Transformer is a neural operator which adopts the cross-attention with the following modifications:
    
    1. The softmax is not used.
    2. The layer normalization is done for the key and value tensors.
    
    Reference: https://proceedings.neurips.cc/paper/2021/file/d0921d442ee91b896ad95059d13df618-Supplemental.pdf
    """
    def __init__(
            self,
            dim_input_domain:   int,
            dim_query_domain:   int,
            
            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            
            lift_layer:         Sequence[int] = [256],
            n_layers:           int = 4,
            n_heads:            int = 1,
            project_layer:      Sequence[int] = [256],
            
            activation_name:    str = "relu",
            activation_kwargs:  dict[str, object] = {},
            
            **kwargs,
        ) -> Self:
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Save some member variables for representation
        self.__dim_input_domain = dim_input_domain
        self.__dim_query_domain = dim_query_domain
        self.__in_channels      = in_channels
        self.__hidden_channels  = hidden_channels
        self.__out_channels     = out_channels
        self.__n_layers         = n_layers
        self.__n_heads          = n_heads
        
        # Set the subnetworks
        self.network_lift = MLP(
                                    [in_channels] + list(lift_layer) + [hidden_channels],
                                    activation_name     = activation_name,
                                    activation_kwargs   = activation_kwargs,
                                )
        if n_layers <= 0:
            self.network_hidden = nn.Identity()
        else:
            self.network_hidden = nn.Sequential()
            __enc_kwargs = {
                'dim_input_domain'      : dim_input_domain,
                'dim_query_domain'      : dim_query_domain,
                'hidden_channels'       : hidden_channels,
                'n_heads'               : n_heads,
                'mlp_activation_name'   : activation_name,
                'mlp_activation_kwargs' : activation_kwargs,                
            }
            self.network_hidden = nn.ModuleList([GTencoderCA(**__enc_kwargs) for _ in range(n_layers)])
            self.activation     = get_activation(activation_name, activation_kwargs)
        self.network_projection = MLP(
                                        [hidden_channels] + list(project_layer) + [out_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs,
                                    )
        
        return
    
    
    def forward(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        ### Arguments
        @ `U` (`torch.Tensor`)
            * `U` is the embedding of the input function.
            * `U` has the shape `(B, *__domain__. C)`.
            * `U` is input to the key map and the value map.
        
        @ `X` (`torch.Tensor`)
            * `X` is the 3-tensor saving the coordinates of the query points.
            * `X` has the shape `(B, size(__domain__), dim(__domain__))`.
            * `X` is input to the query map.
        """
        U = self.network_lift.forward(U)
        for idx in range(self.__n_layers):
            U = self.network_hidden[idx].forward(U, X)
            U = self.activation.forward(U)
        U = self.network_projection.forward(U)
        return U


##################################################
##################################################
# End of file