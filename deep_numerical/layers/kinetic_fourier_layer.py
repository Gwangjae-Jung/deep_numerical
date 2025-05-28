from    typing              import  Sequence, Optional
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    ._base_module   import  BaseModule

from    ..utils     import  convolve_freqs, freq_tensor, freq_slices_low


##################################################
##################################################
__all__: list[str]  = [
    'FourierBoltzmannLayer',
    'FourierLandauLayer',
]


##################################################
##################################################
# Layers for spectral methods for solving kinetic equations
class FourierBoltzmannLayer(BaseModule):
    def __init__(
            self,
            dimension:      int,
            degree:         int,
            n_weights:      int,
            n_channels:     int = 1,
        ) -> Self:
        self.__check_arguments(dimension, degree, n_weights, n_channels)
        n_modes = tuple((2*degree for _ in range(dimension)))
        
        super().__init__()
        self.__dimension:   int         = dimension
        self.__n_modes:     tuple[int]  = n_modes
        self.__n_weights:   int         = n_weights
        self.__n_channels:  int         = n_channels
        
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        
        shape_abc   = (*n_modes, n_channels, n_weights)
        shape_d     = (*n_modes, n_channels)
        # NOTE (Alignment of the dimensions)
        # (*frequency_components, channels, weights)
        self.params_alpha   = nn.Parameter(torch.rand(shape_abc, dtype=torch.cfloat))
        self.params_beta    = nn.Parameter(torch.rand(shape_abc, dtype=torch.cfloat))
        self.params_gamma   = nn.Parameter(torch.rand(shape_abc, dtype=torch.cfloat))
        self.params_diag    = nn.Parameter(torch.rand(shape_d,   dtype=torch.cfloat))
        
        self.__conv_dim_gain:   tuple[int]  = ()
        self.__conv_dim_loss:   tuple[int]  = ()
        self.__config_convolution()
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
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
            n_channels:     int,
        ) -> None:
        assert isinstance(dimension, int) and dimension>0
        assert isinstance(degree, int) and degree>0
        assert isinstance(n_weights, int) and n_weights>0
        assert isinstance(n_channels, int) and n_channels>0
        return
    
    
    def __config_convolution(self) -> None:
        dim = self.__dimension
        self.__conv_dim_gain:   tuple[int]  = tuple(range(-2-dim, -2))  # 2 tail dimensions `ct`
        self.__conv_dim_loss:   tuple[int]  = tuple(range(-1-dim, -1))  # NOTE: 1 tail dimension `c`
        return
    
    
    def forward(self, X_fft: torch.Tensor) -> torch.Tensor:
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
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        aX      = torch.zeros(**_data_init_kwargs)
        bX      = torch.zeros(**_data_init_kwargs)
        conv    = torch.zeros(**_data_init_kwargs)
        X_fft   = X_fft[..., None]  # Make a new dimension for the summands
        # Conduct convolution
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., v, c, t)
            aX[*data_slice] = self.params_alpha[*kernel_slice] * X_fft[*data_slice]
            bX[*data_slice] = self.params_beta[ *kernel_slice] * X_fft[*data_slice]
        aX_bX = convolve_freqs(aX, bX, dim=self.__conv_dim_gain)
        # Conduct integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., v, c, t)
            conv[*data_slice] = self.params_gamma[*kernel_slice] * aX_bX[*data_slice]
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
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        wX_fft  = torch.zeros(**_data_init_kwargs)
        # Conduct convolution and integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
            wX_fft[*data_slice] = self.params_diag[*kernel_slice] * X_fft[*data_slice]
        conv = convolve_freqs(X_fft, wX_fft, self.__conv_dim_loss)
        return conv
    

##################################################
##################################################
class VariableFourierBoltzmannLayer(BaseModule):
    """## Varaible Fourier-Boltzmann layer
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
    """
    def __init__(
            self,
            dimension:      int,
            degree:         int,
            n_weights:      int,
            n_channels:     int = 1,
        ) -> Self:
        self.__check_arguments(dimension, degree, n_weights, n_channels)
        n_modes = tuple((2*degree for _ in range(dimension)))
        
        super().__init__()
        self.__dimension:   int         = dimension
        self.__n_modes:     tuple[int]  = n_modes
        self.__n_weights:   int         = n_weights
        self.__n_channels:  int         = n_channels
        
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        
        shape_abc   = (*n_modes, n_channels, n_weights)
        shape_d     = (*n_modes, n_channels)
        # NOTE (Alignment of the dimensions)
        # (*frequency_components, channels, weights)
        self.params_alpha   = nn.Parameter(torch.rand(shape_abc, dtype=torch.cfloat))
        self.params_beta    = nn.Parameter(torch.rand(shape_abc, dtype=torch.cfloat))
        self.params_gamma   = nn.Parameter(torch.rand(shape_abc, dtype=torch.cfloat))
        self.params_diag    = nn.Parameter(torch.rand(shape_d,   dtype=torch.cfloat))
        self.params_v_gamma = nn.Sequential(
            ...
        )
        self.params_v_diag  = nn.Sequential(
            ...
        )
        
        self.__conv_dim_gain:   tuple[int]  = ()
        self.__conv_dim_loss:   tuple[int]  = ()
        self.__config_convolution()
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
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
            n_channels:     int,
        ) -> None:
        assert isinstance(dimension, int) and dimension>0
        assert isinstance(degree, int) and degree>0
        assert isinstance(n_weights, int) and n_weights>0
        assert isinstance(n_channels, int) and n_channels>0
        return
    
    
    def __config_convolution(self) -> None:
        dim = self.__dimension
        self.__conv_dim_gain:   tuple[int]  = tuple(range(-2-dim, -2))  # 2 tail dimensions `ct`
        self.__conv_dim_loss:   tuple[int]  = tuple(range(-1-dim, -1))  # 1 tail dimension `c`
        return
    
    
    def forward(self, X_fft: torch.Tensor) -> torch.Tensor:
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
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        aX      = torch.zeros(**_data_init_kwargs)
        bX      = torch.zeros(**_data_init_kwargs)
        conv    = torch.zeros(**_data_init_kwargs)
        X_fft   = X_fft[..., None]
        # Conduct convolution
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., v, c, t)
            aX[*data_slice] = self.params_alpha[*kernel_slice] * X_fft[*data_slice]
            bX[*data_slice] = self.params_beta[ *kernel_slice] * X_fft[*data_slice]
        aX_bX = convolve_freqs(aX, bX, dim=self.__conv_dim_gain)
        # Conduct integration-like operation
        # NOTE: Varaible kernel
        p_gamma = self.params_gamma * self.params_v_gamma
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., v, c, t)
            conv[*data_slice] = p_gamma[*kernel_slice] * aX_bX[*data_slice]
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
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        wX_fft  = torch.zeros(**_data_init_kwargs)
        # Conduct convolution and integration-like operation
        # NOTE: Varaible kernel
        p_diag = self.params_diag * self.params_v_diag
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
            wX_fft[*data_slice] = p_diag[*kernel_slice] * X_fft[*data_slice]
        conv = convolve_freqs(X_fft, wX_fft, self.__conv_dim_loss)
        return conv


##################################################
##################################################
class FourierLandauLayer(BaseModule):
    """## Fourier-Landau layer
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
    
    -----
    ### Note
    1. So far, this class approximates the collision operator of the Fokker-Planck-Landau equation with a single species of particles. In future, the neural spectral method for the multi-species case will be implemented.
    """
    def __init__(
            self,
            dimension:      int,
            degree:         int,
        ) -> Self:
        self.__check_arguments(dimension, degree)
        n_modes = tuple((2*degree for _ in range(dimension)))
        
        super().__init__()
        self.__dimension:   int         = dimension
        self.__degree:      int         = degree
        self.__n_modes:     tuple[int]  = n_modes
        
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        self.__conv_dim:    tuple[int]  = tuple(range(-1-dimension, -1))  # 1 tail dimension `c`
        
        shape_id    = (*n_modes, 1)
        shape_proj  = (*n_modes, 1, dimension, dimension)
        # NOTE (Alignment of the dimensions)
        # (*frequency_components, channels, ...)
        self.params_id     = nn.Parameter(torch.rand(shape_id, dtype=torch.cfloat))
        self.params_proj   = nn.Parameter(torch.rand(shape_proj, dtype=torch.cfloat))
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
        return self.__n_modes
    
    
    def __check_arguments(
            self,
            dimension:      int,
            degree:         int,
        ) -> None:
        assert isinstance(dimension, int) and dimension>0
        assert isinstance(degree, int) and degree>0
        return
    
    
    def forward(self, X_fft: torch.Tensor) -> torch.Tensor:
        """## The forward propagation of `FourierLandauLayer`
        
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
        freqs = freq_tensor(self.__dimension, 2*self.__degree, keepdim=True).type(torch.float)
        freqs = freqs.to(X_fft.device)
        freq_norms = freqs.norm(p=2, dim=-1, keepdim=True)

        ##################################################
        # The positive part of the gain term
        ##################################################
        # Conduct convolution
        px1 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
        px2 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
            px1[*data_slice] = freq_norms[*kernel_slice] * X_fft[*data_slice]
            px2[*data_slice] = self.params_id[*kernel_slice] * X_fft[*data_slice]
        gain_positive = convolve_freqs(px1, px2, self.__conv_dim)
        
        ##################################################
        # The positive part of the gain term
        ##################################################
        negative_summands: list[torch.Tensor] = []
        for (i, j) in product(range(self.__dimension), range(self.__dimension)):
            nx1 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
            nx2 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
            # Conduct integration-like operation
            for kernel_slice in product(*self.__kernel_slices):
                data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
                nx1[*data_slice] = \
                    freqs[*kernel_slice, [i]] * \
                    freqs[*kernel_slice, [j]] * \
                    X_fft[*data_slice]
                nx2[*data_slice] = \
                    self.params_proj[*kernel_slice, :, i, j] * \
                    X_fft[*data_slice]
            negative_summands.append( convolve_freqs(nx1, nx2, self.__conv_dim) )
        gain_negative = torch.sum(torch.stack(negative_summands, dim=-1), dim=-1, keepdim=False)
        
        ##################################################
        # Return the subtraction
        ##################################################
        return gain_positive-gain_negative
    
    
    def compute_loss_fft(self, X_fft: torch.Tensor) -> torch.Tensor:
        """Computes the Fourier series coefficients of the loss part of the collision term.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, (x1, ..., xd), v1, ..., vd, data)`. Here, `d` is the dimension of the domain, and currently, only the space-homogeneous case is considered.
        """
        # Initialize storages
        kernel_diag = self._compute_kernel_diagonal()
        wX_fft = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
        
        # Conduct convolution and integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
            wX_fft[*data_slice] = kernel_diag[*kernel_slice] * X_fft[*data_slice]
        conv = convolve_freqs(X_fft, wX_fft, self.__conv_dim)
        return conv
    
    
    def _compute_kernel_diagonal(self) -> torch.Tensor:
        freqs = freq_tensor(self.__dimension, 2*self.__degree, keepdim=True).type(torch.cfloat)
        freqs = freqs.to(self.params_id.device)
        freq_norms = freqs.norm(p=2, dim=-1, keepdim=True)
        part1 = freq_norms.pow(2) * self.params_id
        part2 = torch.einsum(
            "...i, ...cij, ...j -> ...c",
            freqs, self.params_proj, freqs
        )
        return part1-part2


##################################################
##################################################
class RadialFourierLandauLayer(BaseModule):
    """## Radial Fourier-Landau layer: The Fourier-Landau layer with radial kernel
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
    
    -----
    ### Note
    1. So far, this class approximates the collision operator of the Fokker-Planck-Landau equation with a single species of particles. In future, the neural spectral method for the multi-species case will be implemented.
    """
    def __init__(
            self,
            dimension:      int,
            degree:         int,
        ) -> Self:
        self.__check_arguments(dimension, degree)
        n_modes = tuple((2*degree for _ in range(dimension)))
        
        super().__init__()
        self.__dimension:   int         = dimension
        self.__degree:      int         = degree
        self.__n_modes:     tuple[int]  = n_modes
        
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        self.__conv_dim:    tuple[int]  = tuple(range(-1-dimension, -1))  # 1 tail dimension `c`
        
        shape_id    = (*n_modes, 1)
        shape_proj  = (*n_modes, 1, dimension, dimension)
        # NOTE (Alignment of the dimensions)
        # (*frequency_components, channels, ...)
        self.params_id     = nn.Parameter(torch.rand(shape_id, dtype=torch.cfloat))
        self.params_proj   = nn.Parameter(torch.rand(shape_proj, dtype=torch.cfloat))
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
        return self.__n_modes
    
    
    def __check_arguments(
            self,
            dimension:      int,
            degree:         int,
        ) -> None:
        assert isinstance(dimension, int) and dimension>0
        assert isinstance(degree, int) and degree>0
        return
    
    
    def forward(self, X_fft: torch.Tensor) -> torch.Tensor:
        """## The forward propagation of `FourierLandauLayer`
        
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
        freqs = freq_tensor(self.__dimension, 2*self.__degree, keepdim=True).type(torch.float)
        freqs = freqs.to(X_fft.device)
        freq_norms = freqs.norm(p=2, dim=-1, keepdim=True)

        ##################################################
        # The positive part of the gain term
        ##################################################
        # Conduct convolution
        px1 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
        px2 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
            px1[*data_slice] = freq_norms[*kernel_slice] * X_fft[*data_slice]
            px2[*data_slice] = self.params_id[*kernel_slice] * X_fft[*data_slice]
        gain_positive = convolve_freqs(px1, px2, self.__conv_dim)
        
        ##################################################
        # The positive part of the gain term
        ##################################################
        negative_summands: list[torch.Tensor] = []
        for (i, j) in product(range(self.__dimension), range(self.__dimension)):
            nx1 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
            nx2 = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
            # Conduct integration-like operation
            for kernel_slice in product(*self.__kernel_slices):
                data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
                nx1[*data_slice] = \
                    freqs[*kernel_slice, [i]] * \
                    freqs[*kernel_slice, [j]] * \
                    X_fft[*data_slice]
                nx2[*data_slice] = \
                    self.params_proj[*kernel_slice, :, i, j] * \
                    X_fft[*data_slice]
            negative_summands.append( convolve_freqs(nx1, nx2, self.__conv_dim) )
        gain_negative = torch.sum(torch.stack(negative_summands, dim=-1), dim=-1, keepdim=False)
        
        ##################################################
        # Return the subtraction
        ##################################################
        return gain_positive-gain_negative
    
    
    def compute_loss_fft(self, X_fft: torch.Tensor) -> torch.Tensor:
        """Computes the Fourier series coefficients of the loss part of the collision term.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, (x1, ..., xd), v1, ..., vd, data)`. Here, `d` is the dimension of the domain, and currently, only the space-homogeneous case is considered.
        """
        # Initialize storages
        kernel_diag = self._compute_kernel_diagonal()
        wX_fft = torch.zeros(X_fft.shape, dtype=torch.cfloat, device=X_fft.device)
        
        # Conduct convolution and integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))  # (..., v, c)
            wX_fft[*data_slice] = kernel_diag[*kernel_slice] * X_fft[*data_slice]
        conv = convolve_freqs(X_fft, wX_fft, self.__conv_dim)
        return conv
    
    
    def _compute_kernel_diagonal(self) -> torch.Tensor:
        freqs = freq_tensor(self.__dimension, 2*self.__degree, keepdim=True).type(torch.cfloat)
        freqs = freqs.to(self.params_id.device)
        freq_norms = freqs.norm(p=2, dim=-1, keepdim=True)
        part1 = freq_norms.pow(2) * self.params_id
        part2 = torch.einsum(
            "...i, ...cij, ...j -> ...c",
            freqs, self.params_proj, freqs
        )
        return part1-part2


##################################################
##################################################
# End of file