from    typing      import  *

import  torch
from    torch       import  nn
from    itertools   import  product

from    custom_modules.torch.utils  import  *




##################################################
##################################################


class FactorizedSpectralConv(nn.Module):
    def __init__(
                    self,
                    n_modes:            Union[List[int], Tuple[int]],
                    hidden_channels:    int,
                    weighted_fourier_modes: bool = False
        ) -> None:
        super().__init__()
        
        self.n_modes        = n_modes
        self.dim_domain     = len(n_modes)
        
        if (self.dim_domain > len(EINSUM_FULL_STRING)):
            raise NotImplementedError(f"The spectral convolutional layer is not supported for dimensions larger than {len(EINSUM_FULL_STRING)}.")
        
        scale_linear    = hidden_channels ** (-2)
        scale_bias      = hidden_channels ** (-1)
        for cnt in range(self.dim_domain):
            setattr(
                        self,
                        f"linear{cnt}",
                        nn.Parameter(
                            scale_linear * torch.rand(
                                size = (hidden_channels, hidden_channels, n_modes[cnt]),
                                dtype = torch.cfloat,
                                requires_grad = True
                            )
                        )
                    )
            setattr(
                        self,
                        f"bias{cnt}",
                        nn.Parameter(
                            scale_bias * torch.rand(
                                size = (hidden_channels, 1, n_modes[cnt]),
                                dtype = torch.cfloat,
                                requires_grad = True
                            )
                        )
                    )
        
        if weighted_fourier_modes:
            self.fourier_weights = nn.Parameter(torch.ones(size = (self.dim_domain,), dtype = torch.float, requires_grad = True))
        
        self.domain_str = EINSUM_FULL_STRING[:self.dim_domain]
        
        return;
    
    
    def factorized_spectral_affine(self, fourier: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Computes the einsum of `weight` and `fourier` at dimension `dim` and the addition with the bias term.
        
        `weight`:  (channel, channel, __spatial__) = (C, C, s1, ..., sd)
        `fourier`: (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        (Note that '__spatial__' might have been reduced.)
        
        Output:  (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        
        -----
        ### Remark
        
        By implementing this function, we can not only accomplish well-posed proramming, with reducing the amount of pass of the weight.
        """
        
        return (
            torch.einsum(
                f"ct{self.domain_str},bt{self.domain_str}->bc{self.domain_str}",
                (getattr(self, f"linear{dim}"), fourier)
            ) + \
            getattr(self, f"bias{dim}")
        )
        

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(X)
        
        for d in range(self.dim_domain):
            # Discrete Fourier transform for each dimension
            fft_dim     = 2 + d
            fft_norm    = "forward"
            X_rfft_d = torch.fft.rfft(X, dim = fft_dim, norm = fft_norm)
            
            # Instantiate a tensor which will be filled with a reduced DFT
            X_rfft_d_shape = list(X_rfft_d.shape)
            X_spatial_shape = list(X.shape)[-self.dim_domain:]
            Y = torch.zeros(size = X_rfft_d_shape, dtype = torch.cfloat, device = X.device)
            
            # Get the range of the low-frequency parts of the discrete Fourier transform
            _range =  [slice(None), slice(None)]
            _range += [slice(None)] * d
            _range += [slice(None, self.n_modes[d])]
            _range += [slice(None)] * (self.dim_domain - d - 1)
            
            # Do the affine transform
            Y[_range] = self.factorized_spectral_affine(X_rfft_d, dim = d)
            
            # Add to the output
            out += torch.fft.irfft(Y, dim = fft_dim, norm = fft_norm, s = X_spatial_shape) * self.fourier_weights[d]
            
        return out




##################################################
##################################################


class FactorizedSpectralConv(nn.Module):
    def __init__(
                    self,
                    n_modes:            Union[List[int], Tuple[int]],
                    hidden_channels:    int
        ) -> None:
        super().__init__()
        
        self.n_modes        = n_modes
        self.dim_domain     = len(n_modes)
        
        if (self.dim_domain > len(EINSUM_FULL_STRING)):
            raise NotImplementedError(f"The spectral convolutional layer is not supported for dimensions larger than {len(EINSUM_FULL_STRING)}.")
        
        scale = hidden_channels ** (-2)
        for cnt in range(2 ** (self.dim_domain - 1)):
            setattr(
                        self,
                        f"linear{cnt}",
                        nn.Parameter(
                            scale * torch.rand(size = (hidden_channels, hidden_channels, *n_modes), dtype = torch.cfloat),
                            requires_grad = True
                        )
                    )
                
        self.domain_str = EINSUM_FULL_STRING[:self.dim_domain]
        
        return;
    
    
    def spectral_matmul(self, fourier: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Computes the einsum of `fourier` and `weight`.
        
        `fourier`: (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        `weight`:  (channel, channel, __spatial__) = (C, C, s1, ..., sd)
        (Note that '__spatial__' might have been reduced.)
        
        Output:  (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        """
        
        return torch.einsum(f"ct{self.domain_str},bt{self.domain_str}->bc{self.domain_str}", (fourier, weight))
        

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Discrete Fourier transform
        fft_dim     = list(range(-self.dim_domain, 0, 1))
        fft_norm    = "forward"
        X_rfftn     = torch.fft.rfftn(X, dim = fft_dim, norm = fft_norm)
        
        # Instantiate a tensor which will be filled with a reduced DFT
        X_rfftn_shape   = list(X_rfftn.shape)
        X_spatial_shape = list(X.shape)[-self.dim_domain:]
        Y = torch.zeros(size = X_rfftn_shape, dtype = torch.cfloat, device = X.device)
        
        # Get the range of the low-frequency parts of the discrete Fourier transform
        _list_of_slices = []
        for d in range(self.dim_domain - 1):
            _list_of_slices.append([
                                        slice(None, self.n_modes[d], None),
                                        slice(-self.n_modes[d], None, None)
                                    ])
        
        # Copy the low-frequency part of the discrete Fourier transform
        for cnt, spatial_range in enumerate(product(*_list_of_slices)):
            _range = [slice(None), slice(None)]
            _range += list(spatial_range)
            _range.append(slice(None, self.n_modes[-1], None))
            _range = tuple(_range)
            Y[_range] = self.spectral_matmul(getattr(self, f"linear{cnt}"), X_rfftn[_range])
        
        X = torch.fft.irfftn(Y, dim = fft_dim, norm = fft_norm, s = X_spatial_shape)
        
        return X



