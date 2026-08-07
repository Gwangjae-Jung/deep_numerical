from    typing      import  Self, Sequence, Optional, override
import  torch
from    deep_numerical.utils        import  type_as_real
from    deep_numerical.neural       import  BaseModule
from    deep_numerical.neural.utils import  Activations
from    deep_numerical.neural.layer import  MLP, RadialFourierLayer
from    deep_numerical.neural.collision_op._utils   import  compute_moments_homogeneous, maxwellian_homogeneous


__all__ = ["PRFNO",]


##################################################
##################################################
class PRFNO(BaseModule):
    """## Parameterized Radial Fourier Neural Operator (RFNO)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Parameterized Radial Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels which is radially symmetric.
    
    -----
    ### Note
    1. Given `n_layers`, this class instantiates `n_layers` distinct `FourierLayer` objects. To reuse a single `FourierKernelLayer` instance for `n_layers` times, use `FourierNeuralOperatorLite`, instead.
    2. Since both lift and projection layers act pointwise, if an input is periodic, then so is the output. Hence, the Fourier differentiation is available, provided that an input is sampled from a sufficiently smooth function.
    """
    def __init__(
            self,
            dim_domain:         int,
            max_freq:           int,

            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            dim_parameters:     int = 0,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],
            pos_enc_layer:      Sequence[int]   = [256, 256],
            branch_layer:       Sequence[int]   = [256, 256],
            bias:               bool            = False,

            activation_name:    Activations         = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            dtype:              Optional[torch.dtype]   = None,
        ) -> Self:
        """## The initializer of the class `RadialFourierNeuralOperator`

        Arguments:
            `dim_domain` (`int`):
                The dimension of the domain.
            `max_freq` (`int`):
                The maximum frequency index to be preserved.
            `in_channels` (`int`):
                The number of the input channels.
            `hidden_channels` (`int`):
                The number of the hidden channels.
                Note that `hidden_channels` should be a positive *even* integer.
            `out_channels` (`int`):
                The number of the output channels.
            `dim_parameters` (`int`, default: `0`):
                The number of the hyperparameters.
            `lift_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`):
                The number of hidden layers.
            `project_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the projection layer.
            `pos_enc_layer` (`Sequence[int]`, default: `(256, 256)`):
                The numbers of channels inside the positional encoding layer.
            `branch_layer` (`Sequence[int]`, default: `(256, 256)`):
                The numbers of channels inside the branches.
            `bias` (`bool`, default: `True`):
                Whether to use bias terms in the subnetworks, *except for the parameter branches*.
            `activation_name` (`str`, default: `"relu"`):
                The name of the activation function to be used in the hidden layers.
            `activation_kwargs` (`dict[str, object]`, default: `{}`):
                The keyword arguments for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the input and output tensors.
        
        -----
        ### Note
        The argument `weighted_residual` is omitted since the hidden dimension expands when the parametric code involves.
        """
        if hidden_channels<2 or hidden_channels%2!=0:
            raise ValueError(f"The number of hidden channels must be a positive even integer, but got {hidden_channels}.")
        super().__init__()

        # Save some member variables for representation
        self.__dim_domain   = dim_domain
        self.__max_freq     = max_freq
        self.__dtype            = dtype = type_as_real(dtype)
        
        # Configure the encoder(s)
        self.network_pos_enc = MLP(
            [1, *pos_enc_layer, hidden_channels//2],    # NOTE: The input channel is 1 (radial distance)
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
            dtype               = dtype,
        )
        self.network_param = MLP(
            [dim_parameters, *branch_layer, hidden_channels//2],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
            dtype               = dtype,
        )
        
        # Define the subnetworks
        config_common   = {'bias': bias, 'activation_name': activation_name, 'activation_kwargs': activation_kwargs, 'dtype': dtype}
        config_fourier  = {'dim_domain': dim_domain, 'max_freq': max_freq}
        n_layers_prior      = n_layers // 2
        n_layers_posterior  = n_layers - n_layers_prior
        ## Lift
        self.network_lift   = MLP([in_channels, *lift_layer, hidden_channels//2], **config_common)
        ## Hidden layers
        self.network_hidden_prior: torch.nn.Sequential = torch.nn.Sequential()
        for _ in range(n_layers_prior-1):
            self.network_hidden_prior.append(RadialFourierLayer(**config_common, **config_fourier, in_channels=hidden_channels))
        self.network_hidden_prior.append(RadialFourierLayer(**config_common, **config_fourier, in_channels=hidden_channels, out_channels=hidden_channels//2))
        self.network_hidden_posterior: torch.nn.Sequential = torch.nn.Sequential()
        for _ in range(n_layers_posterior):
            self.network_hidden_posterior.append(RadialFourierLayer(**config_common, **config_fourier, in_channels=hidden_channels))
        ## Projection
        self.network_projection = MLP([hidden_channels, *project_layer, out_channels], **config_common)
        
        return
    
    
    def _compute_pos_code(self, pos: torch.Tensor) -> torch.Tensor:
        pos_rad = pos.norm(p=2, dim=-1, keepdim=True)    # <- radial position
        return self.network_pos_enc(pos_rad)
    def _compute_param_code(self, param: torch.Tensor) -> torch.Tensor:
        param_code = self.network_param.forward(param)
        b, h = param_code.size(0), param_code.size(-1)
        return param_code.reshape(b, *(1 for _ in range(self.__dim_domain)), h)
    
    
    def forward(self, X: torch.Tensor, pos: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`):
                The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
            `pos` (`torch.Tensor`):
                The position tensor of shape `(s_1, ..., s_d, d)`.
            `param` (`torch.Tensor`):
                The parameter tensor of shape `(B, dim_parameters)`.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        # Compute the deviation from the equilibrium state
        X_moments = compute_moments_homogeneous(X, pos)
        X_equi = maxwellian_homogeneous(pos, *X_moments)
        col = X-X_equi
        # Lift the input and mix with the positional code
        col = self.network_lift.forward(col)
        pos_code = self._compute_pos_code(pos)
        col = torch.cat((col, col*pos_code), dim=-1)
        # Apply the prior hidden network and mix with the parameter code
        col = self.network_hidden_prior.forward(col)
        param_code = self._compute_param_code(param)
        col = torch.cat([col, col*param_code], dim=-1)
        # Apply the posterior hidden network
        col = self.network_hidden_posterior.forward(col)
        # Project to the output space and return the corrected output
        col = self.network_projection.forward(col)
        return col
    
    
    @override
    def train(self, mode: bool=True) -> Self:
        """Customized method `train()`."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    @override
    def eval(self) -> Self:
        """Customized method `eval()`."""
        return self.train(False)
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def max_freq(self) -> int:
        return self.__max_freq
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype


##################################################
##################################################
# End of file