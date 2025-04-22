from    typing              import  *
from    typing_extensions   import  Self

import  torch

from    ..layers            import  BaseModule, MLP, FourierLayer
from    ..utils             import  get_activation, warn_redundant_arguments


##################################################
##################################################
__all__ = [
    "FourierNeuralOperator",
    "FourierNeuralOperatorLite",
    "FNO",
    "FNOLite",
]


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

    -----
    ### Architecture
    * Input: A `torch.Tensor` object of shape `(B, domain, d_in)`
    * Output: A `torch.Tensor` object of shape `(B, domain, d_out)`
        
    1. Lift
        * The input `X` is lifted to a feature space of dimension `hidden_channels`.
    
    2. Hidden layers (Fourier layers)
        * A tensor `v_t` in the feature space passes a Fourier layer.
        * A single Fourier layer consists of the following two subnetworks:
            * Linear part
                * Given `v_t`, this subnetwork computes a linear transform of `v_t`.
                * In the implementation, it is done by `torch.einsum()`.
            * Kernel integration
                * Given `v_t`, this subnetwork computes the kernel integration with a kernel.
                * Assuming that the kernel function is a function of displacement, the kernel integration can be done by linear transform in the Fourier space.
        * After the above processes, the tensor passes an activation function, except for the last Fourier layer.
    
    3. Projection
        * The tensor which passed all Fourier layers are projected to the space of dimension `out_channels`.
    
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

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],

            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `FourierNeuralOperator`
        
        ----
        ### Arguments
        * `n_modes` (`Sequence[int]`)
            * The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
            * [Checklist]
                * Each entry should be a positive integer.
                * Let `X` be an input tensor of shape `(B, s_1, ..., s_d, d_in)`. For the FFT to be computed, it is required that `s_i // 2 >= n_modes[i]`.
            * Note that the length of `n_modes` is the dimension of the domain.
                
        * `in_channels` (`int`), `hidden_channels` (`int`), `out_channels` (`int`)
            * The number of the features in the input, hidden, and output spaces, respectively.
            * [Checklist]
                * Each value should be a positive integer.
        
        * `lift_layer` (`Sequential[int]`, default: `(256)`), `n_layers` (`int`, default: `4`), and `project_layer` (`Sequential[int]`, default: `(256)`)
            * The intermediate widths of the lift, hidden layers, and projection.
            * [Checklist]
                * Each argument should consist of positive integers.
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
            self.network_hidden.apply(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels}
            self.network_hidden.append(FourierLayer(**__fl_kwargs))
            for _ in range(n_layers - 1):
                self.network_hidden.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden.append(FourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels] + project_layer + [out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
                        
        return
    
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        ### Note
        1. This class assumes that the input tensor `X` has the shape `(B, s_1, ..., s_d, C)`.
        """
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
        
    
    def __repr__(self) -> str:
        msg: list[str] = []
        msg.append(f"FourierNeuralOperator(")
        msg.append(f"\tlift:    {self.network_lift}")
        
        msg = \
                f"FourierNeuralOperator(\n" \
                f"    lift:       {self.network_lift}\n" \
                f"    hidden:     (\n"
        for md in self.network_hidden:
            msg += \
                f"                     {md},\n"
        msg += \
                f"                ),\n" \
                f"    projection: {self.network_projection}\n" \
                f")"
        return msg
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain




class FourierNeuralOperatorLite(BaseModule):
    """## Fourier Neural Operator (FNO Lite)
    ### Integral operator via discrete Fourier transform
    
    -----
    ### Description
    The Fourier Neural Operator is an Integral Neural Operator with translation-invariant kernels.
    By the convolution theorem, the kernel integration can be computed by a convolution under some mild conditions.
    Ignoring the Fourier modes of high frequencies, the Fourier Neural Operator reduces its quadratic computational complexity to quasilinear computational complexity.
    
    Reference: https://openreview.net/pdf?id=c8P9NQVtmnO

    -----
    ### Model architecture
    * Input: A `torch.Tensor` object of shape `(B, domain, d_in)`
    * Output: A `torch.Tensor` object of shape `(B, domain, d_out)`
        
    1. Lift
        * The input `X` is lifted to a feature space of dimension `hidden_channels`.
    
    2. Hidden layers (Fourier layers)
        * A tensor `v_t` in the feature space passes a Fourier layer.
        * A single Fourier layer consists of the following two subnetworks:
            * Linear part
                * Given `v_t`, this subnetwork computes a linear transform of `v_t`.
                * In the implementation, it is done by `torch.einsum()`.
            * Kernel integration
                * Given `v_t`, this subnetwork computes the kernel integration with a kernel.
                * Assuming that the kernel function is a function of displacement, the kernel integration can be done by linear transform in the Fourier space.
        * After the above processes, the tensor passes an activation function, except for the last Fourier layer.
    
    3. Projection
        * The tensor which passed all Fourier layers are projected to the space of dimension `out_channels`.

    -----
    ### Note
    Given `n_layers`, this class instantiates a single `FourierLayer` object and reuse it for `n_layers` times.
    To use `n_layers` distinct `FourierLayer` instances, use `FourierNeuralOperator`, instead.
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

            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `FourierNeuralOperatorLite`
        
        ----
        ### Arguments
        @ `n_modes` (`Sequence[int]`)
            * The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
            * [Checklist]
                * Each entry should be a positive integer.
                * Let `X` be an input tensor of shape `(B, s_1, ..., s_d, d_in)`. For the FFT to be computed, it is required that `s_i // 2 >= n_modes[i]`.
            * Note that the length of `n_modes` is the dimension of the domain.
                
        @ `in_channels` (`int`), `hidden_channels` (`int`), `out_channels` (`int`)
            * The number of the features in the input, hidden, and output spaces, respectively.
            * [Checklist]
                * Each value should be a positive integer.
        
        @ `lift_layer` (`Sequential[int]`, default: `(256)`), `n_layers` (`int`, default: `4`), and `project_layer` (`Sequential[int]`, default: `(256)`)
            * The intermediate widths of the lift, hidden layers, and projection.
            * [Checklist]
                * Each argument should consist of positive integers.
        """        
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
        # Check the argument validity
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise RuntimeError(f"'n_modes[{cnt}]' is chosen {item}, which is not positive.")
        
        # Save some member variables for representation
        self.__dim_domain       = len(n_modes)
        self.__n_layers         = n_layers
        self.__activation_name  = activation_name
        
        # Define the subnetworks
        ## Lift
        self.network_lift   = MLP(
            [in_channels] + lift_layer + [hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        ## Hidden layers
        self.network_hidden     = torch.nn.Identity()
        if n_layers > 0:
            self.network_hidden = FourierLayer(n_modes = n_modes, channels = hidden_channels)
            self.activation     = get_activation(activation_name, activation_kwargs)
        ## Projection
        self.network_projection = MLP(
                                        [hidden_channels] + project_layer + [out_channels],
                                        activation_name     = activation_name,
                                        activation_kwargs   = activation_kwargs
                                    )
                        
        return
    
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        ### Note
        1. This class assumes that the input tensor `X` has the shape `(B, s_1, ..., s_d, C)`.
        """
        X = self.network_lift.forward(X)
        for _ in range(self.__n_layers - 1):
            X = self.network_hidden.forward(X)
            X = self.activation.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    def __repr__(self) -> str:
        return \
                f"FourierNeuralOperatorLite(\n"\
                f"    lift:       {self.network_lift}\n"\
                f"    hidden:     {self.network_hidden} x {self.__n_layers} (activation: {self.__activation_name})\n"\
                f"    projection: {self.network_projection}\n"\
                f")"
    
    
    @property
    def dim_domain(self) -> int:
        return len(self.__dim_domain)


##################################################
##################################################
FNO     = FourierNeuralOperator
FNOLite = FourierNeuralOperatorLite


##################################################
##################################################
# End of file