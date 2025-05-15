from    typing              import  *
from    typing_extensions   import  Self

import  torch

from    ..layers            import  BaseModule, MLP, SeparableFourierLayer
from    ..utils             import  get_activation, warn_redundant_arguments


##################################################
##################################################
__all__ = [
    "ParameterizedSFNO",
]


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
            
            n_parameters:       int = 0,

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
            
            `n_parameters` (`int`, default: `0`): The number of the hyperparameters. This class raises an error if `n_parameters` is not a positive integer.
            
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
        if not isinstance(n_parameters, int) or n_parameters<0:
            raise RuntimeError(f"'n_parameters' is chosen {n_parameters}, which is not a positive integer.")
        
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
        self.network_hidden_prior: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers_prior <= 0:
            self.network_hidden_prior.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels, 'rank': rank}
            self.network_hidden_prior.append(SeparableFourierLayer(**__fl_kwargs))
            for _ in range(n_layers_prior-1):
                self.network_hidden_prior.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden_prior.append(SeparableFourierLayer(**__fl_kwargs))
        self.network_hidden_posterior: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers_posterior <= 0:
            self.network_hidden_posterior.append(torch.nn.Identity())
        else:
            self.network_hidden_posterior.append(
                SeparableFourierLayer(
                    n_modes          = n_modes,
                    in_channels      = 2*hidden_channels,
                    out_channels     = hidden_channels,
                    rank             = rank,
                )
                # `in_channels` is the double of `hidden_channels` because the output of the previous layer is concatenated with the hyperparameters
            )
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels, 'rank': rank}
            for _ in range(n_layers_prior-1):
                self.network_hidden_posterior.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden_posterior.append(SeparableFourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels] + project_layer + [out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        
        # Define the branches (the networks for the hyperparameters)
        self.hypernet   = HyperMLP(
            channels        = [self.__dim_domain, 2*hidden_channels, hidden_channels],
            hyper_channels  = [n_parameters, 4*hidden_channels, 4*hidden_channels],
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
        coords = self.generate_coordinates(X.shape[1:-1]).to(X.device)
        p = self.hypernet.forward(coords, p)
        X = torch.cat([X, p], dim=-1)
        # Posterior computation
        X = self.network_hidden_posterior.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    def generate_coordinates(self, shape: Sequence[int]) -> torch.Tensor:
        """## Generate the coordinates for the hyperparameters
        
        Arguments:
            `shape` (`Sequence[int]`):
                * The shape of the coordinates to be generated.
                * The length of `shape` is equal to the dimension of the domain.
        
        Returns:
            `torch.Tensor`: The coordinates of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        coords = torch.meshgrid([torch.arange(s)/s for s in shape], indexing="ij")
        coords = torch.stack(coords, dim=-1)
        return coords
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    

##################################################
##################################################
pSFNO = PSFNO = ParameterizedSFNO


##################################################
##################################################
# End of file