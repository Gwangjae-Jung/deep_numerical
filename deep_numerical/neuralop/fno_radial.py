from    typing              import  *
from    typing_extensions   import  Self

import  torch

from    ..layers            import  BaseModule, MLP, RadialFourierLayer
from    ..utils             import  positional_encoding


##################################################
##################################################
__all__ = [
    "RadialFourierNeuralOperator",
    "RadialFNO", "RFNO"
]


##################################################
##################################################
class RadialFourierNeuralOperator(BaseModule):
    """## RadialFourier Neural Operator (Radial-FNO)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels.
    By the convolution theorem, the kernel integration can be computed by a convolution under some mild conditions.
    Ignoring the Fourier modes of high frequencies, the Fourier Neural Operator reduces its quadratic computational complexity to quasilinear computational complexity.
    
    Reference: https://openreview.net/pdf?id=c8P9NQVtmnO
    
    -----
    ### Note
    1. Given `n_layers`, this class instantiates `n_layers` distinct `RadialFourierLayer` objects.
    2. Since both lift and projection layers act pointwise, if an input is periodic, then so is the output. Hence, the Fourier differentiation is available, provided that an input is sampled from a sufficiently smooth function.
    3. To enforce the equivariance under orthogonal transformations, the positional encoding is not implemented.
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
            
            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            pos_enc:            bool            = False,
        ) -> Self:
        """## The initializer of the class `FourierNeuralOperator`
        
        Arguments:
            `n_modes` (`Sequence[int]`): The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved. Note that the length of `n_modes` is the dimension of the domain.
                    
            `in_channels` (`int`): The number of the input channels.
            `hidden_channels` (`int`): The number of the hidden channels.
            `out_channels` (`int`): The number of the output channels.
            
            `lift_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`): The number of hidden layers.
            `project_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the projection layer.
            
            `weighted_residual` (`bool`, default: `True`): Whether to use a linear layer in the skip connection. If `False`, the skip connection is a simple addition. Instead, a 2-layer MLP will be used after the spectral convolution, and the activation function is not applied after the residual connection.
            
            `activation_name` (`str`, default: `"relu"`): The name of the activation function.
            `activation_kwargs` (`dict[str, object]`, default: `{}`): The keyword arguments for the activation function.
            
            `pos_enc` (`bool`, default: `False`): Whether to use positional encoding. If `True`, the input will be encoded with positional information.
        """        
        super().__init__()
        
        # Check the argument validity
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise ValueError(f"'n_modes[{cnt}]' is chosen {item}, which is not positive.")
        if n_layers <= 0:
            raise ValueError(f"'n_layers' is chosen {n_layers}, which is not positive.")
        
        # Save some member variables for representation
        self.__dim_domain = len(n_modes)

        # Check the positional encoding
        self.__pos_enc = pos_enc
        if self.__pos_enc:
            in_channels += 1  # Add positional encoding
        
        # Define the subnetworks
        ## Lift
        self.network_lift   = MLP(
            [in_channels, *lift_layer, hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        ## Hidden layers
        self.network_hidden: torch.nn.Sequential = torch.nn.Sequential()
        for idx in range(n_layers):
            activate = True if idx<n_layers-1 else False
            __fl_kwargs = {
                'n_modes':              n_modes,
                'in_channels':          hidden_channels,
                'weighted_residual':    weighted_residual,
                'activate':             activate,
                'activation_name':      activation_name,
                'activation_kwargs':    activation_kwargs,
            }
            self.network_hidden.append(RadialFourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels, *project_layer, out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        
        return
    
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`): The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        if self.__pos_enc:
            pos = positional_encoding(X.shape, 'radial', dtype=X.dtype, device=X.device)
            X = torch.cat((X, pos), dim=-1)
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain


##################################################
##################################################
RFNO = RadialFNO = RadialFourierNeuralOperator


##################################################
##################################################
# End of file