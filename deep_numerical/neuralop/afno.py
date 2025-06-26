from    typing              import  *
from    typing_extensions   import  Self

import  torch

from    ..layers            import  BaseModule, MLP, PatchEmbedding, FourierLayer
from    ..utils             import  Objects, get_activation, warn_redundant_arguments


##################################################
##################################################
__all__ = [
    "AdaptiveFourierNeuralOperator",
    "AFNO",
]


##################################################
##################################################
class AdaptiveFourierNeuralOperator(BaseModule):
    """## Adaptive Fourier Neural Operator (AFNO)
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

            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,
            
            patch_size:         int             = 1,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],
            
            n_blocks:           int             = 4,

            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `FourierNeuralOperator`
        
        ----
        ### Arguments
        * `dim_domain` (`Sequence[int]`)
            * The dimension of the input space.
            * This architecture assigns an affine transform for each Fourier mode, unlike in *Fourier Neural Operator*, in which only the Fourier modes of low frequencies are considered.
                
        * `in_channels` (`int`), `hidden_channels` (`int`), `out_channels` (`int`)
            * The number of the features in the input, hidden, and output spaces, respectively.
            * [Checklist]
                * Each value should be a positive integer.
        
        * `lift_layer` (`Sequential[int]`, default: `(256)`), `n_layers` (`int`, default: `4`), and `project_layer` (`Sequential[int]`, default: `(256)`)
            * The intermediate widths of the lift, hidden layers, and projection.
            * [Checklist]
                * Each argument should consist of positive integers.
        
        * `patch_size` (`int`, default: `1`)
            * The number of the patches in each direction.
            * [Checklist]
                * The value should be a positive integer, which divides the size of the input tensor in each dimension.
        
        * `n_blocks` (`int`, default: `4`)
            * The number of the blocks in the hidden space.
            * [Checklist]
                * The value should be a positive integer, which divides `hidden_channels`.
        """        
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
        # Check the argument validity
        if n_blocks <= 0 or hidden_channels%n_blocks != 0:
            raise ValueError(f"Invalid argument: `n_blocks` should be a positive integer, which divides `hidden_channels`. ({hidden_channels=}, {n_blocks=})")
        
        # Save some member variables for representation
        self.__dim_domain = dim_domain
        
        # Define the subnetworks
        ## Lift
        self.network_patch_embed   = PatchEmbedding(
            dim_domain  = dim_domain,
            in_channels = in_channels,
            patch_size  = patch_size,
        )
        ## Hidden layers
        self.network_hidden = torch.nn.Sequential()
        if n_layers <= 0:
            self.network_hidden.append(torch.nn.Identity())
        else:
            self.network_hidden.append(...)
            for _ in range(n_layers - 1):
                self.network_hidden.append(get_activation(activation_name, activation_kwargs))
                self.network_hidden.append(...)
        self.network_hidden.append(torch.nn.Flatten(start_dim=..., end_dim=...))
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
        X = self.network_patch_embed.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    

##################################################
##################################################
AFNO     = AdaptiveFourierNeuralOperator


##################################################
##################################################
# End of file