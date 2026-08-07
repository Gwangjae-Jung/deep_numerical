from    typing      import  Sequence, Optional, Dict, Tuple
import  torch
from    deep_numerical.utils        import  type_as_real
from    deep_numerical.neural       import  BaseModule
from    deep_numerical.neural.utils import  Activations, positional_encoding
from    deep_numerical.neural.layer import  MLP, FourierLayer


__all__ = ["FourierNeuralOperator", "FNO"]


##################################################
##################################################
class FourierNeuralOperator(BaseModule):
    """## Fourier Neural Operator (FNO)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels.
    By the convolution theorem, the kernel integration can be computed by a convolution under some mild conditions.
    Ignoring the Fourier modes of high frequencies, the Fourier Neural Operator reduces its quadratic computational complexity to quasilinear computational complexity.
    
    Reference: https://openreview.net/pdf?id=c8P9NQVtmnO
    """
    def __init__(
            self,
            n_modes:            Sequence[int],

            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],
            weighted_residual:  bool            = True,
            bias:               bool            = True,
            pos_enc:            bool            = False,
            
            activation_name:    Activations         = "relu",
            activation_kwargs:  Dict[str, object]   = {},
            
            dtype:              Optional[torch.dtype]   = None,
        ) -> None:
        """## The initializer of the class `FourierNeuralOperator`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                The maximum degree of the Fourier modes to be preserved.
                To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
                Note that the length of `n_modes` is the dimension of the domain.
            `in_channels` (`int`):
                The number of the input channels.
            `hidden_channels` (`int`):
                The number of the hidden channels.
            `out_channels` (`int`):
                The number of the output channels.        
            `lift_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`):
                The number of hidden layers.
            `project_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the projection layer.
            `weighted_residual` (`bool`, default: `True`):
                Whether to use a linear layer in the skip connection.
                If `False`, the skip connection is a simple addition.
                Instead, a 2-layer MLP will be used after the spectral convolution, and the activation function is not applied after the residual connection.
            `bias` (`bool`, default: `True`):
                Whether to use bias terms in the subnetworks.
            `pos_enc` (`bool`, default: `False`):
                Whether to use positional encoding. If `True`, the input will be encoded with positional information.
            `activation_name` (`str`, default: `"relu"`):
                The name of the activation function.
            `activation_kwargs` (`Dict[str, object]`, default: `{}`):
                The keyword arguments for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the model parameters.
        """
        super().__init__()
        
        # Save some member variables for representation
        self.__dim_domain   = len(n_modes)
        self.__n_modes      = tuple(n_modes)
        self.__dtype = dtype = type_as_real(dtype)
        
        # Check the positional encoding
        self.__pos_enc = pos_enc
        if self.__pos_enc:  in_channels += 2*self.__dim_domain  # Add sinusoidal positional encoding

        # Define the subnetworks
        config_common   = {'bias': bias, 'activation_name': activation_name, 'activation_kwargs': activation_kwargs, 'dtype': dtype}
        config_fourier  = {'n_modes': n_modes, 'in_channels': hidden_channels, 'weighted_residual': weighted_residual}
        ## Lift
        self.network_lift   = MLP([in_channels, *lift_layer, hidden_channels], **config_common)
        ## Hidden layers
        self.network_hidden: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers<=0:
            import  warnings
            warnings.warn(f"The number of hidden layers is set to {n_layers}. Thus, the hidden layers are bypassed.", RuntimeWarning)
        for idx in range(n_layers):
            activate = True if idx<n_layers-1 else False
            self.network_hidden.append(FourierLayer(**config_common, **config_fourier, activate=activate))
        ## Projection
        self.network_projection = MLP([hidden_channels, *project_layer, out_channels], **config_common)
                        
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
            pos = positional_encoding(X.shape, 'sinusoidal', dtype=X.dtype, device=X.device)
            X = torch.cat((X, pos), dim=-1)
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def n_modes(self) -> Tuple[int, ...]:
        return self.__n_modes
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype


FNO = FourierNeuralOperator


##################################################
##################################################
# End of file