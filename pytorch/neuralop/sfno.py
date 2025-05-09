from    typing              import  *
from    typing_extensions   import  Self

import  torch

from    ..layers            import  BaseModule, MLP, SeparableFourierLayer
from    ..utils             import  get_activation, warn_redundant_arguments


##################################################
##################################################
__all__ = [
    "SeparableFourierNeuralOperator",
    "SFNO",
]


##################################################
##################################################
class SeparableFourierNeuralOperator(BaseModule):
    """## Separable Fourier Neural Operator (SFNO)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Separable Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels, where the weight tensors in the Fourier space are replaced with low-rank approximations.
    
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
            `rank` (`int`, default: `2`): The rank of the low-rank approximation.
            
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
        self.network_hidden: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers <= 0:
            self.network_hidden.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels, 'rank': rank}
            self.network_hidden.append(SeparableFourierLayer(**__fl_kwargs))
            for _ in range(n_layers-1):
                self.network_hidden.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden.append(SeparableFourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels] + project_layer + [out_channels],
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
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain


##################################################
##################################################
SFNO    = SeparableFourierNeuralOperator


##################################################
##################################################
# End of file