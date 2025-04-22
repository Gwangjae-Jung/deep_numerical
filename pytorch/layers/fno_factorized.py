from    typing      import  *

import  torch
from    torch       import  nn

from    ..utils     import  *


##################################################
##################################################
class FactorizedSpectralConv(nn.Module):
    def __init__(
                    self,
                    n_modes:            Sequence[int],
                    hidden_channels:    int
        ) -> None:
        super().__init__()
        
        self.n_modes        = n_modes
        self.dim_domain     = len(n_modes)
        
        if (self.dim_domain > len(EINSUM_STRING)):
            raise NotImplementedError(
                f"The spectral convolutional layer is not supported for dimensions larger than {len(EINSUM_STRING)}."
            )
        
        scale_linear = hidden_channels ** (-2)
        for cnt in range(self.dim_domain):
            setattr(
                        self,
                        f"linear{cnt}",
                        nn.Parameter(
                            scale_linear * torch.rand(
                                size = (hidden_channels, hidden_channels, n_modes[cnt]),
                                dtype = torch.cfloat
                            )
                        )
                    )
        
        self.domain_str = EINSUM_STRING[:self.dim_domain]
        
        return;
    
    
    def factorized_spectral_affine(self, fourier: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Computes the einsum of `weight` and `fourier` at dimension `dim` and the addition with the bias term.
        
        `weight`:  (channel, channel, __spatial_d__) = (C, C, sd)
        `fourier`: (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        (Note that '__spatial__' might have been reduced.)
        
        Output:  (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        
        -----
        ### Remark
        
        By implementing this function, we can not only accomplish well-posed proramming, with reducing the amount of pass of the weight.
        """
        return torch.einsum(
            f"ct{self.domain_str[dim]},bt{self.domain_str}->bc{self.domain_str}",
            (getattr(self, f"linear{dim}"), fourier)
        )
        

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(X)
        
        for d in range(self.dim_domain):
            # Discrete Fourier transform for each dimension
            fft_dim     = 2 + d
            fft_norm    = "forward"
            X_rfft_d    = torch.fft.rfft(X, dim = fft_dim, norm = fft_norm)
            
            # Instantiate a tensor which will be filled with a reduced DFT
            X_rfft_d_shape  = list(X_rfft_d.shape)
            X_spatial_shape = list(X.shape)[-self.dim_domain:]
            Y = torch.zeros(size = X_rfft_d_shape, dtype = torch.cfloat, device = X.device)
            
            # Get the range of the low-frequency parts of the discrete Fourier transform
            _range =  [slice(None), slice(None)]
            _range += [slice(None)] * d
            _range += [slice(self.n_modes[d])]
            _range += [slice(None)] * (self.dim_domain - d - 1)
            
            # Do the affine transform
            Y[_range] = self.factorized_spectral_affine(X_rfft_d[_range], dim = d)
            
            # Add to the output
            summand = torch.fft.irfft(Y, dim = fft_dim, norm = fft_norm, n = X_spatial_shape[d])
            out += summand
            
        return out




##################################################
##################################################


class FactorizedFourierLayer(nn.Module):
    """## Factorized Fourier layer
    ### Kernel integration via dimensionwise FFT with a skip connection
    
    -----
    Given a batch of uniformly discretized functions, this class computes the spectral convolution of the input with a a skip connection.
    
    -----
    ## Remark
    
    1. The input tensor should be of shape `(batch_size, hidden_channels, *shape_of_domain)`.
    2. The skip connection is implemented by adding the result of the spectral convolution with a linear transform of the input tensor. The linear transform in the skip connection is shared by all points of the domain.
    3. As for the spectral convolution, see the documentation of the class `SpectralConv`.
    """
    
    def __init__(
                    self,
                    n_modes:            Sequence[int],
                    hidden_channels:    int,
                    expansion_factor:   float,
                    activation:         str
        ) -> None:
        
        super().__init__()
        
        self.hidden_channels    = hidden_channels
        self.n_modes            = n_modes
        self.dim_domain         = len(n_modes)
        
        self.activation = getattr(nn, TORCH_ACTIVATION_DICT[activation])()
        
        conv = getattr(nn, f"Conv{self.dim_domain}d", None)
        if conv is None:
            raise NotImplementedError(
                f"The PyTorch library provides the convolutional layers up to dimenion 3, but the domain is {self.dim_domain}-dimensional."
            )
        
        self.spectral = nn.Sequential(
            FactorizedSpectralConv(
                n_modes         = n_modes,
                hidden_channels = hidden_channels
            ),
            conv(
                in_channels     = hidden_channels,
                out_channels    = int(expansion_factor * hidden_channels),
                kernel_size     = 1
            ),
            self.activation,
            conv(
                in_channels     = int(expansion_factor * hidden_channels),
                out_channels    = hidden_channels,
                kernel_size     = 1
            ),
            self.activation
        )
                
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X + self.spectral(X))
    
        


##################################################
##################################################
