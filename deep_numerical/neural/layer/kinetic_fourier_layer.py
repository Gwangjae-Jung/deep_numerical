from    typing              import  Callable, Optional
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    deep_numerical.fft  import  convolve_freqs, freq_slices_low
from    deep_numerical.neural   import  BaseModule
from    deep_numerical.numerical.solver import  one_step_RK4_classic


__all__: list[str]  = ['FourierBoltzmannLayer']


##################################################
##################################################
class FourierBoltzmannLayer(BaseModule):
    def __init__(
            self,
            dimension:      int,
            degree:         int,
            n_weights:      int,
            dtype_str:      str = "double",
        ) -> Self:
        """The initializer of the class `FourierBoltzmannLayer`.
        
        Arguments:
            `dimension` (`int`):
                The dimension of the domain.
            `degree` (`int`):
                The approximation degree of the required parameters.
            `n_weights` (`int`):
                The number of weights for the layer. This corresponds to the number of the quadrature points in fast spectral methods to precompute required kernel modes.
            `dtype_str` (`str`, default: `"double"`):
                The data type of the layer parameters.
        """
        self.__check_arguments(dimension, degree, n_weights)
        n_modes = tuple((2*degree for _ in range(dimension)))
        
        super().__init__()
        self.__dimension:   int             = dimension
        self.__n_modes:     tuple[int, ...] = n_modes
        self.__n_weights:   int             = n_weights
        self.__dtype_r:     torch.dtype     = getattr(torch, dtype_str)
        self.__dtype_c:     torch.dtype     = getattr(torch, f"c{dtype_str}")
        
        self.__kernel_slices:   tuple[tuple[slice, ...], ...] = freq_slices_low(n_modes)
        
        shape_abc   = (*n_modes, 1, n_weights)
        shape_d     = (*n_modes, 1)
        # NOTE (Alignment of the dimensions)
        # (*frequency_components, channels, 
        # weights)
        self.params_scale   = nn.Parameter(torch.rand(shape_abc, dtype=self.__dtype_c))
        self.params_phase_1 = nn.Parameter(torch.rand(shape_abc, dtype=self.__dtype_c))
        self.params_phase_2 = nn.Parameter(torch.rand(shape_abc, dtype=self.__dtype_c))
        self.params_diag    = nn.Parameter(torch.rand(shape_d,   dtype=self.__dtype_c))
        
        self.__conv_dim_gain:   tuple[int, ...]  = ()
        self.__conv_dim_loss:   tuple[int, ...]  = ()
        self.__config_convolution()
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int, ...]:
        return self.__n_modes
    @property
    def n_weights(self) -> int:
        return self.__n_weights
    @property
    def n_channels(self) -> int:
        return self.__n_channels
    
    
    def __check_arguments(
            self,
            dimension:      int,
            degree:         int,
            n_weights:      int,
        ) -> None:
        assert isinstance(dimension, int) and dimension>0
        assert isinstance(degree, int) and degree>0
        assert isinstance(n_weights, int) and n_weights>0
        return
    
    
    def __config_convolution(self) -> None:
        dim = self.__dimension
        self.__conv_dim_gain:   tuple[int, ...]  = tuple(range(-2-dim, -2))  # 2 tail dimensions `ct`
        self.__conv_dim_loss:   tuple[int, ...]  = tuple(range(-1-dim, -1))  # NOTE: 1 tail dimension `c`
        return
    
    
    def compute_fft(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            X_fft:                  torch.Tensor,
        ) -> torch.Tensor:
        """## The forward propagation of `FourierBoltzmannLayer`
        
        -----
        ### Description
        This method computes the Fourier series coefficients of the collision term by calling two methods `compute_gain()` and `compute_loss()`.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, x1, ..., xd, v1, ..., vd, data)`. Here, `d` is the dimension of the domain.
        """
        gain = self.compute_gain_fft(X_fft)
        loss = self.compute_loss_fft(X_fft)
        return gain-loss
    
    
    def compute_gain_fft(self, X_fft: torch.Tensor) -> torch.Tensor:
        """Computes the Fourier series coefficients of the gain part of the collision term.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, (x1, ..., xd), v1, ..., vd, data)`. Here, `d` is the dimension of the domain, and currently, only the space-homogeneous case is considered.
        """
        # Initialize storages
        _data_init_kwargs = {
            'size':     (*X_fft.shape, self.__n_weights),
            'dtype':    X_fft.dtype,
            'device':   X_fft.device,
        }
        aX      = torch.zeros(**_data_init_kwargs)
        bX      = torch.zeros(**_data_init_kwargs)
        conv    = torch.zeros(**_data_init_kwargs)
        X_fft   = X_fft[..., None]  # Make a new dimension for the summands
        # Conduct convolution
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., *v, c, t)
            aX[*data_slice] = self.params_phase_1[*kernel_slice] * X_fft[*data_slice]
            bX[*data_slice] = self.params_phase_2[ *kernel_slice] * X_fft[*data_slice]
        aX_bX = convolve_freqs(aX, bX, dim=self.__conv_dim_gain)
        # Conduct integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., *v, c, t)
            conv[*data_slice] = self.params_scale[*kernel_slice] * aX_bX[*data_slice]
        conv = conv.sum(dim=-1, keepdim=False)
        return conv
    
    
    def compute_loss_fft(self, X_fft: torch.Tensor) -> torch.Tensor:
        """Computes the Fourier series coefficients of the loss part of the collision term.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, (x1, ..., xd), v1, ..., vd, data)`. Here, `d` is the dimension of the domain, and currently, only the space-homogeneous case is considered.
        """
        # Initialize storages
        _data_init_kwargs = {
            'size':     X_fft.shape,
            'dtype':    X_fft.dtype,
            'device':   X_fft.device,
        }
        wX_fft  = torch.zeros(**_data_init_kwargs)
        # Conduct convolution and integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., *v, c)
            wX_fft[*data_slice] = self.params_diag[*kernel_slice] * X_fft[*data_slice]
        conv = convolve_freqs(X_fft, wX_fft, self.__conv_dim_loss)
        return conv
    
    
    def forward(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            f_fft:      torch.Tensor,
            delta_t:    float,
            RK_fcn:     Callable[[float, torch.Tensor, float, Callable], torch.Tensor] = one_step_RK4_classic,
        ) -> torch.Tensor:
        return RK_fcn(0.0, f_fft, delta_t, self.compute_fft)


##################################################
##################################################
# End of file