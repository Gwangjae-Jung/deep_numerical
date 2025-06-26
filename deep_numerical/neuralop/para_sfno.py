from    typing              import  Sequence, Optional
from    typing_extensions   import  Self

from    math                import  prod
from    itertools           import  product

import  torch

from    ..layers            import  (
    BaseModule,
    MLP,
    SeparableFourierLayer, TensorizedFourierLayer
)
from    ..utils             import  warn_redundant_arguments


##################################################
##################################################
__all__ = [
    "ParameterizedSFNO",
]


##################################################
##################################################
class ParameterBranch(torch.nn.Module):
    """## Parameter branch for the parameterized neural operators.
    
    -----
    ### Note
    1. Unlike the implementation in [the paper (MIFNO)](https://www.sciencedirect.com/science/article/pii/S0021999125000968), the convolution operation is implemented using the standard `SpectralConv` class in `pytorch.layers`, which can be applied to arbitrary-dimensional tensors.
    """
    def __init__(
            self,
            n_modes:        Sequence[int],
            in_channels:    int,
            out_channels:   int,
        ) -> Self:
        super().__init__()
        self.__n_modes:         tuple[int]  = tuple(n_modes)
        self.__dim_domain:      int         = len(n_modes)
        self.__in_channels:     int         = in_channels
        self.__out_channels:    int         = out_channels
        self.__fft_dim:         tuple[int]  = tuple(range(-1-self.__dim_domain, -1))
        self.__fft_norm:        str         = "forward"
        
        num_modes: int  = prod(n_modes)
        self.__hidden_channels = int(2**self.__dim_domain)
        self.branch = torch.nn.ModuleDict(
            {
                'mlp': MLP(
                    [in_channels, num_modes*self.__hidden_channels, num_modes*self.__hidden_channels],
                    activation_name='relu',
                ),
                'cnn': ...
            }
        )
        return
    
    
    def compute_branch(self, p: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `p` (`torch.Tensor`):
                * The sequence of hyperparameters, where the shape of each tensor is `(B, in_channels)`.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, *n_modes, out_channels)`.
        """
        p = self.branch['mlp'].forward(p)
        p = p.reshape((p.size(0), ))
    


##################################################
##################################################
class ParameterizedSFNO(BaseModule):
    """## Parameterized Separable Fourier Neural Operator (SFNO)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Parameterized Separable Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels, where the weight tensors in the Fourier space are replaced with low-rank approximations.
    
    -----
    ### Note
    1. Given `n_layers`, this class instantiates `n_layers` distinct `FourierLayer` objects. To reuse a single `FourierKernelLayer` instance for `n_layers` times, use `FourierNeuralOperatorLite`, instead.
    2. Since both lift and projection layers act pointwise, if an input is periodic, then so is the output. Hence, the Fourier differentiation is available, provided that an input is sampled from a sufficiently smooth function.
    """
    def __init__(
            self,
            n_modes:            Sequence[int],

            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            rank:               int = 2,
            
            dim_parameters:     int = 0,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],

            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `SeparableFourierNeuralOperator`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
                * Note that the length of `n_modes` is the dimension of the domain.
                    
            `in_channels` (`int`): The number of the input channels.
            `hidden_channels` (`int`): The number of the hidden channels.
            `out_channels` (`int`): The number of the output channels.
            `rank` (`int`, default: `4`): The rank of the low-rank approximation.
            
            `dim_parameters` (`int`, default: `0`): The dimension of the space of the hyperparameters. This class raises an error if `dim_parameters` is not a positive integer.
            
            `lift_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`): The number of hidden layers.
            `project_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the projection layer.
        """        
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
        # Check the argument validity
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise RuntimeError(f"'n_modes[{cnt}]' is chosen {item}, which is not positive.")
        if not isinstance(dim_parameters, int) or dim_parameters<0:
            raise RuntimeError(f"'dim_parameters' is chosen {dim_parameters}, which is not a positive integer.")
        
        # Save some member variables for representation
        self.__dim_domain   = len(n_modes)
        self.__n_modes          = n_modes
        self.__hidden_channels  = hidden_channels
        
        # Define the subnetworks
        ## Lift
        self.network_lift   = MLP(
            [in_channels] + lift_layer + [hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        ## Hidden layers
        n_layers_prior      = n_layers // 2
        n_layers_posterior  = n_layers - n_layers_prior
        ### Prior
        self.network_hidden_prior: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers_prior <= 0:
            self.network_hidden_prior.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels, 'rank': rank}
            for _ in range(n_layers_prior):
                self.network_hidden_prior.append(SeparableFourierLayer(**__fl_kwargs))
        ### Posterior
        posterior_hidden_channels = 2*hidden_channels
        self.network_hidden_posterior: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers_posterior <= 0:
            self.network_hidden_posterior.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': posterior_hidden_channels, 'rank': rank}
            for _ in range(n_layers_posterior):
                self.network_hidden_posterior.append(SeparableFourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [posterior_hidden_channels] + project_layer + [out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        
        # Define the branches (the networks for the hyperparameters)
        self.branch = MLP(
            [
                dim_parameters,
                hidden_channels*2,
                hidden_channels*2,
                hidden_channels,
            ],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
                        
        return
    
        
    def forward(self, X: torch.Tensor, p: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
            `p` (`Sequence[torch.Tensor]`):
                * The sequence of hyperparameters, where the shape of each tensor is `(B, ...)` and need not be the same.
                * The length of the sequence is equal to the number of branches.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        # Prior computation
        X = self.network_lift.forward(X)
        X = self.network_hidden_prior.forward(X)
        p = self.compute_branch(p)
        X = torch.cat([X, X*p], dim=-1)
        # Posterior computation
        X = self.network_hidden_posterior.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    def compute_branch(
            self,
            p:          torch.Tensor,
        ) -> torch.Tensor:
        out = self.branch.forward(p)
        batch_size, n_channels = out.size(0), out.size(-1)
        out = out.reshape((batch_size, *(1 for _ in range(self.__dim_domain)), n_channels))
        return out
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    

##################################################
##################################################
class ParameterizedTFNO(BaseModule):
    """## Parameterized Tensorized Fourier Neural Operator (SFNO)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Parameterized Tensorized Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels, where the weight tensors in the Fourier space are replaced with low-rank approximations.
    
    -----
    ### Note
    1. Given `n_layers`, this class instantiates `n_layers` distinct `FourierLayer` objects. To reuse a single `FourierKernelLayer` instance for `n_layers` times, use `FourierNeuralOperatorLite`, instead.
    2. Since both lift and projection layers act pointwise, if an input is periodic, then so is the output. Hence, the Fourier differentiation is available, provided that an input is sampled from a sufficiently smooth function.
    """
    def __init__(
            self,
            n_modes:            Sequence[int],

            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            
            kernel_degree:      int = 2,
            kernel_rank:        Optional[int]   = None,
            
            dim_parameters:     int = 0,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],

            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `SeparableFourierNeuralOperator`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
                * Note that the length of `n_modes` is the dimension of the domain.
                    
            `in_channels` (`int`): The number of the input channels.
            `hidden_channels` (`int`): The number of the hidden channels.
            `out_channels` (`int`): The number of the output channels.
            
            `kernel_degree` (`int`, default: `2`): The degree (the number of the summands comprising each kernel) of the polynomial kernel.
            `kernel_rank` (`int`, default: `None`): The rank of the kernel. If `None`, then the rank is set to be the half of `min(in_channels, out_channels)`.
            
            `dim_parameters` (`int`, default: `0`): The dimension of the space of the hyperparameters. This class raises an error if `dim_parameters` is not a positive integer.
            
            `lift_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`): The number of hidden layers.
            `project_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the projection layer.
        """        
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
        # Check the argument validity
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise RuntimeError(f"'n_modes[{cnt}]' is chosen {item}, which is not positive.")
        if not isinstance(dim_parameters, int) or dim_parameters<0:
            raise RuntimeError(f"'dim_parameters' is chosen {dim_parameters}, which is not a positive integer.")
        
        # Save some member variables for representation
        self.__dim_domain = len(n_modes)
        
        # Define the subnetworks
        ## Lift
        self.network_lift   = MLP(
            [in_channels] + lift_layer + [hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        ## Hidden layers
        n_layers_prior      = n_layers // 2
        n_layers_posterior  = n_layers - n_layers_prior
        ### Prior
        self.network_hidden_prior: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers_prior <= 0:
            self.network_hidden_prior.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels, 'kernel_degree': kernel_degree, 'kernel_rank': kernel_rank}
            for _ in range(n_layers_prior):
                self.network_hidden_prior.append(TensorizedFourierLayer(**__fl_kwargs))
        ### Posterior
        self.network_hidden_posterior: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers_posterior <= 0:
            self.network_hidden_posterior.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': 3*hidden_channels, 'kernel_degree': kernel_degree, 'kernel_rank': kernel_rank}
            for _ in range(n_layers_posterior):
                self.network_hidden_posterior.append(SeparableFourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels] + project_layer + [out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
                        
        return
    
        
    def forward(self, X: torch.Tensor, p: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
            `p` (`Sequence[torch.Tensor]`):
                * The sequence of hyperparameters, where the shape of each tensor is `(B, ...)` and need not be the same.
                * The length of the sequence is equal to the number of branches.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        # Prior computation
        X = self.network_lift.forward(X)
        X = self.network_hidden_prior.forward(X)
        coords = self.generate_coordinates(X.shape[1:-1]).to(X.device)
        p = self.hypernet.forward(coords, p)
        X = torch.cat([X, p, X*p], dim=-1)
        # Posterior computation
        X = self.network_hidden_posterior.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain


##################################################
##################################################
pSFNO = PSFNO = ParameterizedSFNO
pTFNO = PTFNO = ParameterizedTFNO


##################################################
##################################################
# End of file