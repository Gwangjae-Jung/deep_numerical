from    typing      import  *

import  torch
from    torch       import  nn
from    itertools   import  product

from    ..utils     import  *




##################################################
##################################################


class OrbitalSpectralConv(nn.Module):
    """## Orbital spectral convolutional layer
    ### A pointwise affine transform in the frequency domain
    
    -----
    Given a uniform discretization of a function, this class computes the spectral convolution of the input.
    
    -----
    ### Implementation
    
    See the docstring of the class `FNO`.
    
    Note that the forward method of this class entails activation.
    """
    def __init__(
                    self,
                    n_modes:            ListData[int],
                    hidden_channels:    int,
                    n_modes_as:         str = "count"
        ) -> None:
        """## The initializer of the class `OrbitalSpectralConv`
        -----
        ### Some member variables
        
        1. `__list_size__` (`list[int]`)
            The list of the size of the tensor which behaves as the integrating kernel, i.e., the weight in the Fourier space.
        
        2. `__list_range__` (`list[slice]`)
            The list of ranges to be used in the forward propagation.
        """
        super().__init__()
        
        self.n_modes        = n_modes
        self.n_modes_as     = n_modes_as
        self.dim_domain     = len(n_modes)
        
        if (self.dim_domain > len(EINSUM_FULL_STRING)):
            raise NotImplementedError(f"The spectral convolutional layer is not supported for dimensions larger than {len(EINSUM_FULL_STRING)}.")
        
        self.__list_size__  = []
        self.__list_range__ = []
        self.__set_size__()
        
        scale = hidden_channels ** (-2)
        self.linear = nn.Parameter(
            scale * torch.rand(size = (hidden_channels, hidden_channels, *self.__list_size__), dtype = torch.cfloat)
        )
        
        self.domain_str = EINSUM_FULL_STRING[:self.dim_domain]
        
        return;
    
    
    def spectral_matmul(self, fourier: torch.Tensor, _range: object) -> torch.Tensor:
        """
        Computes the spectral einsum of `self.linear[_range]` and `fourier[_range]`.
        
        `weight`:  (channel, channel, __spatial__) = (C, C, s1, ..., sd)
        `fourier`: (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        (Note that '__spatial__' might have been reduced.)
        
        Output:  (batch, channel, __spatial__) = (B, C, s1, ..., sd)
        """
        
        return torch.einsum(
            f"ct{self.domain_str},bt{self.domain_str}->bc{self.domain_str}",
            (
                self.linear[_range] - self.linear[_range].conj_physical().transpose(1, 0),
                fourier[_range]
            )
        )
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Discrete Fourier transform
        fft_dim     = list(range(-self.dim_domain, 0, 1))
        fft_norm    = "forward"
        X_rfftn     = torch.fft.rfftn(X, dim = fft_dim, norm = fft_norm)
        
        # Instantiate a tensor which will be filled with a reduced DFT
        X_rfftn_shape   = list(X_rfftn.shape)
        X_spatial_shape = list(X.shape)[-self.dim_domain:]
        Y = torch.zeros(size = X_rfftn_shape, dtype = torch.cfloat, device = X.device)
        
        # Linear transform on the Fourier modes
        for _range in product(*self.__list_range__):
            Y[_range] = self.spectral_matmul(X_rfftn, _range)
        
        # Inverse discrete Fourier transform
        X = torch.fft.irfftn(Y, dim = fft_dim, norm = fft_norm, s = X_spatial_shape)
        
        return X
    
    
    def __set_size__(self) -> None:
        # Check the type
        __supported_n_modes_as = ['max', 'count']
        if self.n_modes_as not in __supported_n_modes_as:
            raise NotImplementedError(
                f"The argument 'n_modes_as' should be one of the following:",
                __supported_n_modes_as,
                f"Passed: {self.n_modes_as}",
                end = '\n'
            )
        
        # Set `self.__list_size__`
        if (self.n_modes_as == 'max'):
            self.__list_size__      = [2 * k + 1 for k in self.n_modes]
            self.__list_size__[-1]  = self.n_modes[-1] + 1
        elif (self.n_modes_as == 'count'):
            self.__list_size__      = list(self.n_modes)
            self.__list_size__[-1]  = self.n_modes[-1] // 2 + 1
        
        # Set `self.__list_range__`
        self.__list_range__ = [ [slice(None)], [slice(None)] ]
        if (self.n_modes_as == 'max'):
            for k in range(self.dim_domain - 1):
                self.__list_range__.append([
                    slice(None, self.n_modes[k] + 1, None),
                    slice(-self.n_modes[k], None, None)
                ])
            self.__list_range__.append( [slice(None, self.n_modes[-1] + 1, None)] )
        elif (self.n_modes_as == 'count'):
            for k in range(self.dim_domain - 1):
                _left  = (self.n_modes[k] // 2) + 1
                _right = self.n_modes[k] - _left
                self.__list_range__.append([   
                        slice(None, _left, None),
                        slice(-_right, None, None)
                    ]
                )
            self.__list_range__.append( [slice(None, (self.n_modes[-1] // 2) + 1, None)] )
        
        return;




##################################################
##################################################
