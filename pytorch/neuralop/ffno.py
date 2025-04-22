from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers        import  LiftProject, FactorizedFourierLayer
from    ..utils         import  *


##################################################
##################################################
class FFNO(nn.Module):
    """## Factorized Fourier Neural Operator (FFNO)
    -----
    The FFNO is a neural operator modelling integral operators with a skip connection, using the dimensionwise Fourier transform.
    
    Reference: https://arxiv.org/abs/2111.13802

    -----
    ### Model architecture
    
    1. A tensor-type input data `X` of shape `(B, C, shape_of_domain)` is given, where\n
        (1) `B` is the batch size\n
        (2) `C` is the number of the input channels\n
        (3) `shape_of_domain` is the shape of the domain\n
    
    2. (Lift)
        The input `X` is lifted upward to another space, whose dimension is `hidden_channels`
    
    3. (Multiple passes of Fourier layers)
        The lifted tensor `v_t` passes `n_layers` times of distinct factorized Fourier layers, each of which has the same architecture.
        A single Fourier layer consists of the following two subnetworks:
        (1) Kernel integration
            Given `v_t`, this subnetwork computes the kernel integration with a kernel.
            This shall be done as a pointwise linear transform in the frequency domain, rather than the integration in the spatial domain.
            By a pointwise map we mean that the map is defined for each point, rather than globally.
        (2) Linear part
            Given `v_t`, this subnetwork passes `v_t` as the output, acting as a skip connection.
        After the above processes, the tensor passes a nonlinearity, except for the last factorized Fourier layer.
    
    4. (Projection)
        The tensor which passed all Fourier layers are projected to the space of dimension `out_channels`.
        
    5. The projected tensor is returned as the result of the forward propagation.
    """
    
    def __init__(
                    self,
                    n_modes:                Sequence[int],
                    
                    in_channels:            int,
                    hidden_channels:        int,
                    out_channels:           int,
                    
                    n_layers:               int,
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256,
                    
                    expansion_factor:       float = 2,
        ) -> None:
        """## The initializer of the class `FFNO`
        ----
        ### Arguments
        
            1. `n_modes` (`ListData[int]`)
                This is a list or tuple of integers whose `i`-th entry denotes the number of the maximum nodes to be kept along the `i`-th variable.
                
                Checklist
                    (1) Each entry should be a positive integer.
                    (2) After the instantiation of an object of this class, the shape of the tensor `X` whose forward pass shall be coputed should be of the shape `(batch_size, num_channels, __any_shape_matching_n_modes__)` and should be large enough for sampling Fourier modes.
                
                Note that the length of `n_modes` is the dimension of the domain.
                    
            2. `in_channels` (`int`)
                The number of the channels of the input tensor, denoted by `d_a`.
                This is a required argument to determine the lifting.
                
                Checklist
                    (1) `in_channels` should be a positive integer.
            
            3. `hidden_channels` (`int`)
                The number of the channels of the tensors passed between the lifting and the projection, denoted by `d_v`.
                This is a required argument to determine each Fourier layer.
                
                Checklist
                    (1) `hidden_channels` should be a positive integer.
                    
            4. `out_channels` (`int`)
                The number of the channels of the output tensor, denoted by `d_u`.
                This is a required argument to determine the projection.
                
                Checklist
                    (1) `out_channels` should be a positive integer.
            
            5. `expansion_factor` (`float`)
                The factor in which the number of channels will be expanded in the first affine transform in the factorized Fourier layer.
                
                Checklist
                    (1) `int(expansion_factor * hidden_channels)` should be a positive integer.
            
            6. `n_layers` (`int`)
                The number of the Fourier layers instantiated in this object, denoted by `T`.
                This is a required argument to determine the depth of the entire network.
                
                Checklist
                    (1) `n_layers` should be a positive integer.
            
            7. `lifting_channels` (`int`, default: `256`)
                The number of the channels in the lift.
                
                Checklist
                    (1) `lifting_channels` should be a positive integer.
                    
            8. `projection_channels` (`int`, default: `256`)
                The number of the channels in the projection.
                
                Checklist
                    (1) `projection_channels` should be a positive integer.
        """        
        
        super().__init__()
        
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise RuntimeError(f"'k_max[{cnt}]' is chosen {item}, which is not positive.")
        
        self.n_modes            = n_modes
        self.dim_domain         = len(n_modes)
        
        self.in_channels        = in_channels
        self.hidden_channels    = hidden_channels
        self.out_channels       = out_channels
        
        self.network_lifting    = LiftProject(
                                                in_channels     = in_channels,
                                                hidden_channels = lifting_channels,
                                                out_channels    = hidden_channels,
                                                dim_domain      = self.dim_domain
                                    )
        self.network_projection = LiftProject(
                                                in_channels     = hidden_channels,
                                                hidden_channels = projection_channels,
                                                out_channels    = out_channels,
                                                dim_domain      = self.dim_domain
                                    )
        
        if n_layers <= 0:
            _network_fourier = None
        else:
            _network_fourier    = [
                                    FactorizedFourierLayer(
                                        n_modes             = n_modes,
                                        hidden_channels     = hidden_channels,
                                        expansion_factor    = expansion_factor,
                                        activation          = "gelu"
                                    )
                                    for _ in range(n_layers - 1)
                                ] + [
                                    FactorizedFourierLayer(
                                        n_modes             = n_modes,
                                        hidden_channels     = hidden_channels,
                                        expansion_factor    = expansion_factor,
                                        activation          = "identity"
                                    )
                                ]
        self.network_fourier    = nn.Sequential(*_network_fourier)
                        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.network_lifting(X)
        X = self.network_fourier(X)
        X = self.network_projection(X)
        return X
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Factorized Fourier neural operator ({self.dim_domain}D)"
        else:
            return f"FFNO ({self.dim_domain}D)"




##################################################
##################################################
