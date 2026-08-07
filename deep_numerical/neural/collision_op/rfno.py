from    typing      import  Literal, Sequence, Self, Optional, override
import  torch
from    deep_numerical.utils        import  type_as_real
from    deep_numerical.neural       import  BaseModule
from    deep_numerical.neural.utils import  Activations
from    deep_numerical.neural.layer import  MLP, RadialFourierLayer
from    deep_numerical.neural.collision_op._utils   import  compute_moments_homogeneous, maxwellian_homogeneous


__all__ = ["ConservativeRFNO"]


KeyRegistry = Literal['v_grid', 'pos_code', 'gaussian']


##################################################
##################################################
class ConservativeRFNO(BaseModule):
    """## Radial Fourier Neural Operator (Radial-FNO)
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
            dim_domain:         int,
            max_freq:           int,
            max_domain_value:   float,

            hidden_channels:    int,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],
            pos_enc_layer:      Sequence[int]   = [256, 256],
            bias:               bool            = True,
            
            activation_name:    Activations         = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            dtype:              Optional[torch.dtype]   = None,
        ) -> Self:
        """## The initializer of the class `FourierNeuralOperator`
        
        Arguments:
            `dim_domain` (`int`):
                The dimension of the domain.
            `max_freq` (`int`):
                The maximum frequency index to be preserved.    
            `hidden_channels` (`int`):
                The number of the hidden channels.
            `lift_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`):
                The number of hidden layers.
            `project_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the projection layer.
            `pos_enc_layer` (`Sequence[int]`, default: `[256, 256]`):
                The numbers of channels inside the positional encoding layer.
            `bias` (`bool`, default: `True`):
                Whether to use bias terms in the subnetworks.
            `activation_name` (`str`, default: `"relu"`):
                The name of the activation function.
            `activation_kwargs` (`dict[str, object]`, default: `{}`):
                The keyword arguments for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the model parameters.
        """
        if max_domain_value<=0:
            raise ValueError(f"The maximum domain value must be positive, but got {max_domain_value:.4e}.")
        if hidden_channels<2 or hidden_channels%2!=0:
            raise ValueError(f"The number of hidden channels must be a positive even integer, but got {hidden_channels}.")
        super().__init__()
        
        self.__dim_domain   = dim_domain
        self.__max_freq     = max_freq
        self.__max_domain_value = max_domain_value
        self.__dtype        = dtype = type_as_real(dtype)

        # Configure the encoder(s)
        self.__register:  dict[KeyRegistry, Optional[torch.Tensor]] = {
            'v_grid'    : None,
            'pos_code'  : None,
            'gaussian'  : None,
        }
        self.network_pos_enc = MLP(
            [1, *pos_enc_layer, hidden_channels//2],    # NOTE: The input channel is 1 (radial distance)
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
            dtype               = dtype,
        )
        
        # Define the subnetworks
        in_channels, out_channels = 1, 1
        config_common   = {'bias': bias, 'activation_name': activation_name, 'activation_kwargs': activation_kwargs, 'dtype': dtype}
        config_fourier  = {'dim_domain': dim_domain, 'max_freq': max_freq, 'in_channels': hidden_channels}
        ## Lift
        self.network_lift   = MLP([in_channels, *lift_layer, hidden_channels//2], **config_common)
        ## Hidden layers
        self.network_hidden: torch.nn.Sequential = torch.nn.Sequential()
        for _ in range(n_layers):
            self.network_hidden.append(RadialFourierLayer(**config_common, **config_fourier))
        ## Projection
        self.network_projection = MLP([hidden_channels, *project_layer, out_channels], **config_common)
        
        return
    
        
    def update_register(self, res: int) -> None:
        """Register the required data for a given resolution.
        As for `ConservativePFNO`, the positional code and the parameter code are registered.
        """
        # Set variables
        ## Data type and device
        config = {'dtype': self.__dtype, 'device': next(self.network_pos_enc.parameters()).device}
        ## Gaussian temperature
        GAUSSIAN_TEMPERATURE = 0.2
        ## Variables for computation
        dim     = self.__dim_domain
        v_max   = self.__max_domain_value
        dv      = 2*v_max/res
        _v_max  = v_max - dv/2
        v_1d    = torch.linspace(-_v_max, _v_max, res, **config)
        v_grid  = torch.stack(torch.meshgrid(*(v_1d for _ in range(dim)), indexing='ij'), dim=-1)   # Shape: (s_1, ..., s_d, d)
        # Clear previous codes and fill with new ones
        self.clear_register()
        ## `v_grid`
        self.__register['v_grid']       = v_grid
        ## `pos_code`
        pos_code    = self.network_pos_enc.forward(v_grid.norm(dim=-1, p=2, keepdim=True))
        self.__register['pos_code']     = pos_code
        ## `gaussian`
        gaussian_density    = torch.ones((1, 1), **config)
        gaussian_bulk_v     = torch.zeros((1, self.__dim_domain), **config)
        gaussian_temp       = torch.zeros((1, 1), **config) + GAUSSIAN_TEMPERATURE
        self.__register['gaussian']     = maxwellian_homogeneous(v_grid, gaussian_density, gaussian_bulk_v, gaussian_temp)
        return

    
    def clear_register(self) -> None:
        """Clear the register."""
        for key in self.__register.keys():  self.__register[key] = None
        return
    
    
    def __correct_collision_terms(self, cols: torch.Tensor) -> torch.Tensor:
        """Correct the collision terms computed by the neural operator
        
        Arguments:
            `cols` (`torch.Tensor`): The input collision terms of shape `(B, K_1, ..., K_d, 1)`, where `B` is the batch size, `K_i` is the size of the `i`-th dimension of the velocity domain.
        
        Returns:
            `torch.Tensor`: This function returns a `torch.Tensor` object of shape `(B, K_1, ..., K_d, 1)`.
        """
        # Load required variables
        dim     = self.__dim_domain
        v_grid  = self.__register['v_grid']
        
        # NOTE: To prevent division by small number, we modify the density of each collision terms to be `BASE_DENSITY`
        BASE_DENSITY = 1.0
        cols = cols * self.__register['gaussian']
        old_density, _, _ = compute_moments_homogeneous(cols, v_grid, eps=0)
        old_density = old_density.reshape(-1, *(1 for _ in range(dim)), 1)
        cols = cols + (BASE_DENSITY-old_density)*self.__register['gaussian']
        
        # Compute the signed density of the input distribution function
        moments_cols = compute_moments_homogeneous(cols, v_grid, eps=0)
        correctors   = maxwellian_homogeneous(v_grid, *moments_cols, eps=0)
        return cols-correctors
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        # Load the computational grid
        v_grid = self.__register['v_grid']
        # Compute the deviation from the equilibrium state
        X_moments = compute_moments_homogeneous(X, v_grid)
        X_equi = maxwellian_homogeneous(v_grid, *X_moments)
        col = X-X_equi
        # Lift the input and mix with the positional code
        col = self.network_lift.forward(col)
        pos_code = self.__register['pos_code']
        col = torch.cat((col, col*pos_code), dim=-1)
        # Apply the hidden layers
        col = self.network_hidden.forward(col)
        # Project to the output space and return the corrected output
        col = self.network_projection.forward(col)
        col = self.__correct_collision_terms(col)
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
    def max_domain_value(self) -> float:
        return self.__max_domain_value
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype


##################################################
##################################################
if __name__=="__main__":
    torch.set_default_dtype(torch.double)
    
    model = ConservativeRFNO(
        dim_domain          = 2,
        max_freq            = 8,
        max_domain_value    = 6.62,
        hidden_channels     = 64,
        bias                = False,
        activation_name     = "tanh",
    )
    rho = torch.ones((10, 1))*2
    u   = torch.ones((10, 2))*1.3
    T   = torch.ones((10, 1))
    res = 2**6
    model.update_register(res)
    v_grid: torch.Tensor = model._ConservativeRFNO__register['v_grid']
    x = maxwellian_homogeneous(v_grid, rho, u, T) + maxwellian_homogeneous(v_grid, rho*3.7, -u/5, T*1.5)
    y = model.forward(x)
    print('~'*20)
    print(y.norm(p=torch.inf).item(), y.shape)
    signed_density  = y.sum(dim=(1, 2)) * (2*6.62/res)**2
    signed_momentum = torch.einsum('bUVd, UVd -> bd', y, v_grid) * (2*6.62/res)**2
    signed_energy   = torch.einsum('bUVd, UVd -> b',  y, (v_grid**2).sum(dim=-1, keepdim=True)) * (2*6.62/res)**2
    print(f"* Signed density   : {signed_density.norm(torch.inf).item() :.4e}")
    print(f"* Signed momentum  : {signed_momentum.norm(torch.inf).item():.4e}")
    print(f"* Signed energy    : {signed_energy.norm(torch.inf).item()  :.4e}")
    unsigned_density    = y.abs().sum(dim=(1, 2)) * (2*6.62/res)**2
    unsigned_momentum   = torch.einsum('bUVd, UVd -> bd', y.abs(), v_grid) * (2*6.62/res)**2
    unsigned_energy     = torch.einsum('bUVd, UVd -> b',  y.abs(), (v_grid**2).sum(dim=-1, keepdim=True)) * (2*6.62/res)**2
    print(f"* Unsigned density : {unsigned_density.norm(torch.inf).item() :.4e}")
    print(f"* Unsigned momentum: {unsigned_momentum.norm(torch.inf).item():.4e}")
    print(f"* Unsigned energy  : {unsigned_energy.norm(torch.inf).item()  :.4e}")
    print('~'*20)


##################################################
##################################################
# End of file