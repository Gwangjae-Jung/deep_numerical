from    typing      import  Sequence, Dict, Optional
import  torch
from    torch   import  nn
from    deep_numerical.neural   import  BaseModule
from    deep_numerical.neural.layer import  MLP, \
    GalerkinTypeEncoderBlockSelfAttention       as  GTencoderSA, \
    GalerkinTypeEncoderBlockCrossAttention      as  GTencoderCA
from    deep_numerical.neural.utils import  get_activation, positional_encoding


__all__: list[str] = [
    "GalerkinTransformer", "GalerkinTransformerSelfAttention",
    "GalerkinTransformerCrossAttention",
]



class GalerkinTransformerSelfAttention(BaseModule):
    """Galerkin Transformer with self-attention
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
            activation_kwargs:  Dict[str, object] = {},
            
            pos_enc:            bool    = True,
        ) -> None:
        """The initializer of `GalerkinTransformerSelfAttention`
        
        ## Description
        Initializes the Galerkin Transformer model with self-attention encoder blocks.
        
        ## Arguments
        `dim_domain` (`int`): Dimension of the spatial domain.
        `in_channels` (`int`): Number of input function channels.
        `hidden_channels` (`int`): Number of hidden channels in attention and MLP layers.
        `out_channels` (`int`): Number of output channels.
        `lift_layer` (`Sequence[int]`, default: `[256]`): Layer sizes for the lifting MLP.
        `n_layers` (`int`, default: `4`): Number of transformer encoder blocks.
        `n_heads` (`int`, default: `1`): Number of attention heads.
        `project_layer` (`Sequence[int]`, default: `[256]`): Layer sizes for the projection MLP.
        `activation_name` (`str`, default: `"relu"`): Activation function name.
        `activation_kwargs` (`Dict[str, object]`, default: `{}`): Keyword arguments for the activation function.
        `pos_enc` (`bool`, default: `True`): Whether to concatenate positional encodings to the input.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
        
        # Save some member variables for representation
        self.__in_channels      = in_channels
        self.__pos_enc          = pos_enc
        
        # Modify `in_channels` if the (sinusoidal) positional encoding is engaged
        if pos_enc:
            self.__in_channels += 1
        
        # Set the subnetworks
        self.network_lift = MLP(
            [self.__in_channels, *lift_layer, hidden_channels],
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
            [hidden_channels, *project_layer, out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        
        return None
    
    
    def forward(self, X: torch.Tensor, p: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`): The input tensor of shape `(B, s_1, ..., s_d, C_x)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C_x` is the number of the data channels.
            `p` (`Optional[torch.Tensor]`, default: `None`): The parameter tensor of shape `(B, s_1, ..., s_d, C_p)`, where `C_p` is the number of the parameter channels.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C_y)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C_out` is the number of the output channels.
        """
        if p is not None:
            X = torch.cat((X, p), dim=-1)
        if self.__pos_enc:
            pos = positional_encoding(X.shape, 'radial', dtype=X.dtype, device=X.device)
            X = torch.cat((X, pos), dim=-1)
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X



class GalerkinTransformerCrossAttention(BaseModule):
    """Galerkin Transformer with cross-attention
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
            activation_kwargs:  Dict[str, object] = {},
            
            pos_enc:            bool    = True,
            **kwargs,
        ) -> None:
        """The initializer of `GalerkinTransformerCrossAttention`
        
        ## Description
        Initializes the Galerkin Transformer model with cross-attention encoder blocks between input and query domains.
        
        ## Arguments
        `dim_input_domain` (`int`): Dimension of the input function domain.
        `dim_query_domain` (`int`): Dimension of the query domain.
        `in_channels` (`int`): Number of input function channels.
        `hidden_channels` (`int`): Number of hidden channels.
        `out_channels` (`int`): Number of output channels.
        `lift_layer` (`Sequence[int]`, default: `[256]`): Layer sizes for the lifting MLP.
        `n_layers` (`int`, default: `4`): Number of transformer encoder blocks.
        `n_heads` (`int`, default: `1`): Number of attention heads.
        `project_layer` (`Sequence[int]`, default: `[256]`): Layer sizes for the projection MLP.
        `activation_name` (`str`, default: `"relu"`): Activation function name.
        `activation_kwargs` (`Dict[str, object]`, default: `{}`): Keyword arguments for the activation function.
        `pos_enc` (`bool`, default: `True`): Whether to concatenate positional encodings.
        
        ## Returns
        `None`: None.
        """
        
        # Save some member variables for representation
        self.__in_channels      = in_channels
        self.__n_layers         = n_layers
        self.__pos_enc          = pos_enc
        
        # Modify `in_channels` if the (sinusoidal) positional encoding is engaged
        if pos_enc:
            self.__in_channels += 2*dim_input_domain
        
        # Set the subnetworks
        self.network_lift = MLP(
            [in_channels, *lift_layer, hidden_channels],
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
            [hidden_channels, *project_layer, out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        
        return None
    
    
    def forward(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `U` (`torch.Tensor`):
                * `U` is the embedding of the input function.
                * `U` has the shape `(B, *__domain__. C)`.
                * `U` is input to the key map and the value map.
            
            `X` (`torch.Tensor`):
                * `X` is the 3-tensor saving the coordinates of the query points.
                * `X` has the shape `(B, size(__domain__), dim(__domain__))`.
                * `X` is input to the query map.
        """
        if self.__pos_enc:
            pos = positional_encoding(X.shape, 'sinusoidal', dtype=X.dtype, device=X.device)
            X = torch.cat((X, pos), dim=-1)
        U = self.network_lift.forward(U)
        for idx in range(self.__n_layers):
            U = self.network_hidden[idx].forward(U, X)
            U = self.activation.forward(U)
        U = self.network_projection.forward(U)
        return U



GalerkinTransformer = GalerkinTransformerSelfAttention


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()
