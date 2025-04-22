from    typing      import  *

import  torch
from    torch       import  nn
from    itertools   import  product

from    .helper     import  *




##################################################
##################################################


FFT_NORM = 'ortho'
    

##################################################
##################################################


class LiftProject(nn.Module):
    """## Lift and projection
    ### Affine transform with an activation acting on each point of the domain
    -----
    
    ### Note
    
    This class is based on the usual `(B, C, ...)` convention.
    So it is required that the input is preprocessed to be aligned in the usual convention.
    
    -----
    
    Given a batch of discretized functions, this class computes the activation of an affine transform of the input, where the affine transform is shared by all points of the domain.
    """
    def __init__(
                    self,
                    
                    in_channels:        int,
                    hidden_channels:    int,
                    out_channels:       int,
                    
                    dim_domain:         int,
                    
                    activation:         str = "gelu"
        ) -> None:
        
        super().__init__()
        
        # Choose the convolutional layer to be used
        conv = getattr(nn, f"Conv{dim_domain}d", None)
        if conv is None:
            raise NotImplementedError(
                f"The PyTorch library provides the convolutional layers up to dimenion 3, but the domain is {dim_domain}-dimensional."
            )
        
        self.nn1 = conv(in_channels, hidden_channels, 1)
        self.nn2 = conv(hidden_channels, out_channels, 1)
        self.activation = getattr(nn, TORCH_ACTIVATION_DICT[activation])()
        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.nn1(X)
        X = self.activation(X)
        X = self.nn2(X)
        return X




##################################################
##################################################


class SpectralConv(nn.Module):
    """## Spectral convolutional layer
    -----
    
    ### Note
    
    This class is based on the usual `(B, C, ...)` convention.
    So it is required that the input is preprocessed to be aligned in the usual convention.
    
    Note also that this convolutional layer contains the activation.
    """
    def __init__(
                    self,
                    n_modes:            list[int] | tuple[int],
                    hidden_channels:    int,
                    activation:         object
        ) -> None:
        """## The initializer of the class `SpectralConv`
        -----
        ### Some member variables
        
        1. `__list_size__` (`list[int]`)
            Contains the size of the tensor which behaves as the integrating kernel, i.e., the weight tensors in the Fourier space.
        
        2. `__list_range__` (`list[slice]`)
            Contains the ranges to be used in the forward propagation.
        """
        super().__init__()
        
        self.n_modes        = n_modes
        self.dim_domain     = len(n_modes)
        self.activation     = activation
        
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
            (self.linear[_range], fourier[_range])
        )
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Discrete Fourier transform
        fft_dim     = list(range(-self.dim_domain, 0, 1))
        X_rfftn     = torch.fft.rfftn(X, dim = fft_dim, norm = FFT_NORM)
        
        # Instantiate a tensor which will be filled with a reduced DFT
        X_rfftn_shape   = list(X_rfftn.shape)
        X_spatial_shape = list(X.shape)[-self.dim_domain:]
        Y = torch.zeros(size = X_rfftn_shape, dtype = torch.cfloat, device = X.device)
        
        # Linear transform on the Fourier modes
        for _range in product(*self.__list_range__):
            Y[_range] = self.spectral_matmul(X_rfftn, _range)
        
        # Inverse discrete Fourier transform
        X = torch.fft.irfftn(Y, dim = fft_dim, norm = FFT_NORM, s = X_spatial_shape)
        
        return X
    
    
    def __set_size__(self) -> None:
        """
        ### Remark
        
        1. `n_modes`:
            * Each integer denotes the maximum absolute Fourier index.\n
            * When setting `self.__list__size__`, the addition of 1 is to achieve for both `sin` and `cos` to be contained in the approximation space. It is okay to reduce this addition.\n
        
        2. `self.__list_range__`
            * When computing the approximation of the kernel integration, the discrete Fourier transform is not fully loaded. To load partial data, one should determine the range of `self.linear` to be called.
        """
        
        # Set `self.__list_size__`
        self.__list_size__      = [2 * k + 1 for k in self.n_modes]
        self.__list_size__[-1]  = self.n_modes[-1] + 1
        
        # Set `self.__list_range__`
        self.__list_range__ = [ [slice(None)], [slice(None)] ]  # For the (batch, channel) dimensions
        for k in range(self.dim_domain - 1):
            self.__list_range__.append([
                slice(None, self.n_modes[k] + 1, None),
                slice(-self.n_modes[k], None, None)
            ])
        self.__list_range__.append( [slice(None, self.n_modes[-1] + 1, None)] )
        
        return;




##################################################
##################################################


class HamiltonianLayer(nn.Module):
    """## Hamiltonian layer    
    -----
    
    ### Note
    
    This class is based on the usual `(B, C, ...)` convention.
    So it is required that the input is preprocessed to be aligned in the usual convention.
    """
    
    def __init__(
                    self,
                    n_modes:            list[int] | tuple[int, ...],
                    hidden_channels:    int,
                    activation:         object,
        ) -> None:
        
        super().__init__()
        
        self.hidden_channels    = hidden_channels
        self.n_modes            = n_modes
        self.dim_domain         = len(n_modes)
        self.activation         = activation
        
        self.perm       = torch.randperm(n = hidden_channels)
        self.perm_inv   = inverse_permutation(self.perm)
        
        self.__space__  = [slice(None) for _ in range(self.dim_domain)]
               
        self.spectral1 = SpectralConv(
                                n_modes         = self.n_modes,
                                hidden_channels = hidden_channels // 2,
                                activation      = activation
                            )   # Halved hidden channels for the sympletic Verlet integration
        self.spectral2 = SpectralConv(
                                n_modes         = self.n_modes,
                                hidden_channels = hidden_channels // 2,
                                activation      = activation
                            )   # Halved hidden channels for the sympletic Verlet integration
        
        # self.__aux_const_for_concat = 2 if is_temporal else 1
        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        __split_idx = self.hidden_channels // 2
        channels_Y  = slice(0, __split_idx, 1)
        channels_Z  = slice(__split_idx, self.hidden_channels, 1)
        
        # Step 1. Permute and split `X` to form `Y` and `Z`
        X = X[load_channels(channels_loaded = self.perm,  dim_domain = self.dim_domain)]
        Y = X[load_channels(channels_loaded = channels_Y, dim_domain = self.dim_domain)]
        Z = X[load_channels(channels_loaded = channels_Z, dim_domain = self.dim_domain)]
                
        # Step 2. Update `Y` and `Z`
        Y = Y + self.activation(self.spectral1(Z))
        Z = Z - self.activation(self.spectral2(Y))
        
        # Step 3. Stack `Y` and `Z` to update `X` and inversely permute `X`
        X = torch.cat(
                (Y, Z),
                dim = X.ndim - self.dim_domain - 1
                # dim = X.ndim - self.dim_domain - self.__aux_const_for_concat
            )
        X = X[load_channels(channels_loaded = self.perm_inv, dim_domain = self.dim_domain)]
        
        return X
    
    
    def reverse(self, X: torch.Tensor) -> torch.Tensor:
        __split_idx = self.hidden_channels // 2
        channels_Y  = slice(0, __split_idx, 1)
        channels_Z  = slice(__split_idx, self.hidden_channels, 1)
        
        # Step 1. Permute and split `X` to form `Y` and `Z`
        X = X[load_channels(channels_loaded = self.perm,  dim_domain = self.dim_domain)]
        Y = X[load_channels(channels_loaded = channels_Y, dim_domain = self.dim_domain)]
        Z = X[load_channels(channels_loaded = channels_Z, dim_domain = self.dim_domain)]
        
        # Step 2. Revert `Y` and `Z`
        Z = Z + self.activation(self.spectral2(Y))
        Y = Y - self.activation(self.spectral1(Y))
        
        # Step 3. Stack `Y` and `Z` to update `X` and inversely permute `X`
        X = torch.cat(
                (Y, Z),
                dim = X.ndim - self.dim_domain - 1
                # dim = X.ndim - self.dim_domain - self.__aux_const_for_concat
            )
        X = X[load_channels(channels_loaded = self.perm_inv, dim_domain = self.dim_domain)]
        
        return X




##################################################
##################################################
