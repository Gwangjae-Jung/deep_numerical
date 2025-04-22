from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch       import  nn
from    itertools   import  product

from    ._base_module   import  BaseModule

from    ..utils     import  EINSUM_STRING, convolve_freqs, freq_tensor, freq_slices_low, repeat, ones


##################################################
##################################################
__all__: list[str]  = [
    'FourierBoltzmannLayer',
    'ParameterizedFourierBoltzmannLayer',
    'FourierLandauLayer',
]


##################################################
##################################################
# Layers for spectral methods for solving kinetic equations
class FourierBoltzmannLayer(BaseModule):
    """## Fourier-Boltzmann layer (Fourier Spectral Network)
    
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
        1. Users should align `X_fft` in the following order of dimensions: `(batch, , v1, ..., vd, data)`. Here, `d` is the dimension of the domain, and currently, only the space-homogeneous case is considered.
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
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None), slice(None))  # (..., v, c, t)
            conv[*data_slice] = self.params_gamma[*kernel_slice] * aX_bX[*data_slice]
        conv = conv.sum(dim=-1, keepdim=False)
        return conv
    
    
    def compute_loss_fft(self, X_fft: torch.Tensor) -> torch.Tensor:
        """Computes the Fourier series coefficients of the loss part of the collision term.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, , v1, ..., vd, data)`. Here, `d` is the dimension of the domain, and currently, only the space-homogeneous case is considered.
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
class ParameterizedFourierBoltzmannLayer(BaseModule):
    """## Parameterized Fourier-Boltzmann layer (Fourier Spectral Network)
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
    """
    def __init__(
            self,
            dimension:      int,
            degree:         int,
            channels:       int,
            restitution:    float = 1.0,
            n_parameters:   int = -1,
        ) -> Self:
        self.__check_arguments(dimension, degree, channels, restitution, n_parameters)
        n_modes             = tuple((2*degree for _ in range(dimension)))
        
        super().__init__()
        self.__dimension:   int         = dimension
        self.__n_modes:     tuple[int]  = n_modes
        self.__channels:    int         = channels
        self.__restitution: float       = restitution
        
        self.__fft_dim:         tuple[int]          = tuple(range(-1-dimension, -1))
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        
        self.__freqs        = freq_tensor(dimension, 4*degree, True).type(torch.float)
        shape_phase = (*n_modes, channels)
        self.params_phase   = nn.Parameter(torch.rand(shape_phase, dtype=torch.float))
        
        self.net_alpha__freq    = nn.Sequential(
            nn.Linear(dimension, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, channels),
            nn.ReLU(),
        )
        self.net_alpha__kernel  = nn.Sequential(
            nn.Linear(n_parameters, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, channels),
            nn.ReLU(),
            nn.Flatten(),   # Do not remove this flattening layer
        )
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
        return self.__n_modes
    @property
    def channels(self) -> tuple[int]:
        return self.__channels
    @property
    def beta_alpha_arg_ratio(self) -> float:
        """In the fast spectral method for solving the Boltzmann equation, two parameters are set to be contained in the unit circle in the complex plane, and the phase of one parameter is $-c$ sith $c=(3-e)/(1+e)$ times of the phase of the other parameter, where $e$ is the coefficient of restitution. This property returns the constant $c$."""
        return (3-self.__restitution) / (1+self.__restitution)
    @property
    def batched_weight_prod_cmd(self) -> str:
        freq = EINSUM_STRING[:self.__dimension]
        return f"b{freq}t, b...{freq}t -> b...{freq}t"
    @property
    def params_alpha(self) -> torch.Tensor:
        return torch.exp(+1j * self.params_phase)
    @property
    def params_beta(self) -> torch.Tensor:
        return torch.exp(-1j * self.params_phase * self.beta_alpha_arg_ratio)
    def params_gamma(self, parameters: torch.Tensor) -> torch.Tensor:
        """Given `parameters` of shape `(batch_size, n_parameters)`, this function computes the multiplication of two tensors, one depending only on the frequency tensor, and the other one depending only on the parameters."""
        freq_part:  torch.Tensor = self.net_alpha__freq.forward(self.__freqs)
        param_part: torch.Tensor = self.net_alpha__kernel.forward(parameters)
        freq_part   = freq_part[None, ...]
        param_part  = param_part[:, *repeat(None, freq_part.ndim-2), :]
        return freq_part*param_part
    
    
    def __check_arguments(
            self,
            dimension:      int,
            degree:         int,
            channels:       int,
            restitution:    float,
            n_parameters:   int,
        ) -> None:
        assert isinstance(dimension, int) and dimension>0
        assert isinstance(degree, int) and degree>0
        assert isinstance(channels, int) and channels>0
        assert isinstance(restitution, float) and restitution<=1.0 and restitution>=0.0
        assert isinstance(n_parameters, int) and n_parameters>=1
        return
    
    
    def forward(self, X_fft: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """## The forward propagation of `FourierBoltzmannLayer`
        
        -----
        ### Description
        This method computes the Fourier series coefficients of the collision term by calling two methods `compute_gain()` and `compute_loss()`.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, ..., dimension_1, ..., dimension_d, data)`
        """
        gain = self.compute_gain(X_fft, parameters)
        loss = self.compute_loss(X_fft, parameters)
        return gain-loss
    
    
    def compute_gain(self, X_fft: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        # Initialize storages
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        aX      = torch.zeros(**_data_init_kwargs)
        bX      = torch.zeros(**_data_init_kwargs)
        conv    = torch.zeros(**_data_init_kwargs)
        # Conduct convolution
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            aX[*data_slice] = self.params_alpha[*kernel_slice] * X_fft[*data_slice]
            bX[*data_slice] = self.params_beta[ *kernel_slice] * X_fft[*data_slice]
        aX_bX = convolve_freqs(aX, bX, self.__fft_dim)
        # Conduct integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            conv[*data_slice] = torch.einsum(
                self.batched_weight_prod_cmd,
                self.params_gamma(parameters)[*data_slice], aX_bX[*data_slice]
            )
        conv = conv.sum(dim=-1, keepdim=True)
        return conv
    
    
    def compute_loss(self, X_fft: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        # Initialize storages
        fft_weight = \
            self.params_alpha * self.params_beta * \
            self.params_gamma(parameters)[:, *repeat(slice(None, None, 2), self.__dimension)]
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        wX = torch.zeros(**_data_init_kwargs)
        # Conduct convolution and integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            wX[*data_slice] = torch.einsum(
                self.batched_weight_prod_cmd,
                fft_weight[*data_slice], X_fft[*data_slice]
            )
        conv = convolve_freqs(
            X_fft.repeat(*ones(X_fft.ndim-1), self.__channels), wX,
            self.__fft_dim,
        ).sum(dim=-1, keepdim=True)
        return conv


##################################################
##################################################
class FourierLandauLayer(BaseModule):
    """## Fourier-Landau layer (Fourier Spectral Network)
    
    -----
    ### Description
    This layer is invented so that the parameters can approximate the Landau kernel modes in the spectral method for solving the Landau equation.
    """
    def __init__(
            self,
            n_modes:    Sequence[int],
            channels:   int,
        ) -> Self:
        super().__init__()
        self.__n_modes:     tuple[int]  = tuple((2*(n//2) for n in n_modes))    # Even modes for each dimension
        self.__dimension:   int         = len(n_modes)
        self.__channels:    int         = channels
        
        self.__fft_dim:         tuple[int]          = tuple(range(-1-self.__dimension, -1))
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        self.__freqs:           torch.Tensor        = freq_tensor(self.__dimension, n_modes[0], True).type(torch.float)
        
        self.params_P   = nn.Parameter(
            torch.rand(
                size    = (*n_modes, channels),
                dtype   = torch.cfloat,
            )
        )
        self.params_Q   = nn.Parameter(
            torch.rand(
                size    = (*n_modes, channels, self.__dimension, self.__dimension), 
                dtype   = torch.cfloat,
            )
        )
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
        return self.__n_modes
    @property
    def channels(self) -> tuple[int]:
        return self.__channels
    
    
    def forward(self, X_fft: torch.Tensor) -> torch.Tensor:
        """## The forward propagation of `FourierLandauLayer`
        
        -----
        ### Description
        This method computes the Fourier series coefficients of the collision term by calling two methods `compute_gain()` and `compute_loss()`.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, ..., dimension_1, ..., dimension_d, data)`
        """
        gain = self.compute_gain(X_fft)
        loss = self.compute_loss(X_fft)
        return gain-loss
    
    
    def compute_gain(self, X_fft: torch.Tensor) -> torch.Tensor:
        gain_p = self._gain_positive(X_fft)
        gain_n = self._gain_negative(X_fft)
        assert gain_p.shape==gain_n.shape, f"{gain_p.shape=}, {gain_n.shape=}"
        return gain_p-gain_n
        
        
    def _gain_positive(self, X_fft: torch.Tensor) -> torch.Tensor:
        """This part computes the positive part of the gain term by conducting the convolution of the following tensors:
        
        * `norm(freqs)**2 * X_fft
        * `params_P * X_fft`
        """
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        x1_fft = torch.zeros(**_data_init_kwargs)
        x2_fft = torch.zeros(**_data_init_kwargs)
        freq_norms_sq = self.__freqs.pow(2).sum(-1, keepdim=True)
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            x1_fft[*data_slice] = freq_norms_sq[*kernel_slice] * X_fft[*data_slice]
            x2_fft[*data_slice] = self.params_P[*kernel_slice] * X_fft[*data_slice]
        return convolve_freqs(x1_fft, x2_fft, self.__fft_dim)
        
        
    def _gain_negative(self, X_fft: torch.Tensor) -> torch.Tensor:
        """This part computes the negative part of the gain term by computing the matrix multiplications.
        Although the matrix multiplication itself cannot be computed via convolution, the expansion of the matrix multiplication (yielding `dimension**2` summands) can be computed using convolution.
        """
        summands: list[torch.Tensor] = []
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        for (i, j) in product(range(self.__dimension), range(self.__dimension)):
            x1_fft = torch.zeros(**_data_init_kwargs)
            x2_fft = torch.zeros(**_data_init_kwargs)
            for kernel_slice in product(*self.__kernel_slices):
                data_slice = (..., *kernel_slice, slice(None))
                x1_fft[*data_slice] = self.__freqs[*kernel_slice, [i]]*self.__freqs[*kernel_slice, [j]]*X_fft[*data_slice]
                x2_fft[*data_slice] = self.params_Q[*kernel_slice, :, i, j] * X_fft[*data_slice]
            summands.append(convolve_freqs(x1_fft, x2_fft, self.__fft_dim))
        return torch.stack(summands, dim=-1).sum(-1)
        
    
    def compute_loss(self, X_fft: torch.Tensor) -> torch.Tensor:
        _kernel_diag_p = torch.zeros(self.params_P.shape, dtype=torch.cfloat)
        _kernel_diag_n = torch.zeros(self.params_P.shape, dtype=torch.cfloat)
        freq_norms_sq = self.__freqs.pow(2).sum(-1, keepdim=True)
        for kernel_slice in product(*self.__kernel_slices):
            _kernel_diag_p[*kernel_slice] = \
                freq_norms_sq[*kernel_slice] * self.params_P[*kernel_slice]
            _kernel_diag_n[*kernel_slice] = \
                torch.einsum(
                    f"...i,...tij,...j->...t",
                    self.__freqs[*kernel_slice], self.params_Q[*kernel_slice], self.__freqs[*kernel_slice]
                )
        kernel_diag = _kernel_diag_p - _kernel_diag_n
        return convolve_freqs(X_fft, kernel_diag*X_fft, self.__fft_dim)
    

    def forward_once(self, X_fft: torch.Tensor) -> torch.Tensor:
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device
        }
        _data_init_kwargs_ij = {
            'size':     (*X_fft.shape[:-1], self.__channels, self.__dimension, self.__dimension),
            'dtype':    torch.cfloat,
            'device':   X_fft.device
        }
        freq_norms_sq = self.__freqs.pow(2).sum(-1, keepdim=True)
        gain_p_x1_fft = torch.zeros(**_data_init_kwargs)
        gain_p_x2_fft = torch.zeros(**_data_init_kwargs)
        gain_n_ij_x1_fft = torch.zeros(**_data_init_kwargs_ij)
        gain_n_ij_x2_fft = torch.zeros(**_data_init_kwargs_ij)
        kernel_diag_p = torch.zeros(**_data_init_kwargs)
        kernel_diag_n = torch.zeros(**_data_init_kwargs)
        
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            
            # 1. gain_positive
            gain_p_x1_fft[*data_slice] = \
                freq_norms_sq[*kernel_slice] * X_fft[*data_slice]
            gain_p_x2_fft[*data_slice] = \
                self.params_P[*kernel_slice] * X_fft[*data_slice]
            
            # 2. gain_negative
            for (i, j) in product(range(self.__dimension), range(self.__dimension)):
                gain_n_ij_x1_fft[*data_slice, i, j] = \
                    self.__freqs[*kernel_slice, i] * self.__freqs[*kernel_slice, j] * X_fft[*data_slice]
                gain_n_ij_x2_fft[*data_slice, i, j] = \
                    self.params_Q[*kernel_slice, :, i, j] * X_fft[*data_slice]
            
            # 3. loss
            kernel_diag_p[*kernel_slice] = \
                freq_norms_sq[*kernel_slice] * self.params_P[*kernel_slice]
            kernel_diag_n[*kernel_slice] = \
                torch.einsum(
                    f"...i, ...tij, ...j -> ...t",
                    self.__freqs[*kernel_slice], self.params_Q[*kernel_slice], self.__freqs[*kernel_slice]
                )
        kernel_diag = kernel_diag_p-kernel_diag_n
        
        gain_p  = \
            convolve_freqs(gain_p_x1_fft, gain_p_x2_fft, self.__fft_dim)
        gain_n  = \
            convolve_freqs(gain_n_ij_x1_fft, gain_n_ij_x2_fft, dim=(d-2 for d in self.__fft_dim)).sum((-2, -1))
        loss    = \
            convolve_freqs(X_fft, kernel_diag*X_fft, self.__fft_dim)
        
        # The reason of attaching the negative sign can be found from the spectral method
        return -(gain_p-gain_n-loss)
    
    
##################################################
##################################################
class FourierBoltzmannLayer_(BaseModule):
    """## Fourier-Boltzmann layer (Fourier Spectral Network)
    
    -----
    ### Description
    This layer is invented so that the parameters can approximate the Boltzmann kernel modes in the spectral method for solving the Boltzmann equation.
    
    -----
    ### Note
    1. Given `n_modes`, the parameters `params_alpha` and `beta_beta` are of shape `(*n_modes, channels)`, but `params_gamma` is of shape `(*n_modes_double, channels)`, where `n_modes_double[i] == 2*n_modes[i]` for all `i`, as the loss term requires dilated `params_gamma`.
    """
    def __init__(
            self,
            n_modes:    Sequence[int],
            channels:   int,
        ) -> Self:
        super().__init__()
        self.__n_modes:     tuple[int]  = tuple((2*(n//2) for n in n_modes))    # Even modes for each dimension
        self.__dimension:   int         = len(n_modes)
        self.__channels:    int         = channels
        
        self.__fft_dim:         tuple[int]          = tuple(range(-1-self.__dimension, -1))
        self.__kernel_slices:   tuple[tuple[slice]] = freq_slices_low(n_modes)
        
        n_modes_double: tuple[int] = tuple([2*n for n in n_modes])
        _params_kwargs_1 = {'size': (*n_modes, channels), 'dtype': torch.cfloat}
        _params_kwargs_2 = {'size': (*n_modes_double, channels), 'dtype': torch.cfloat}
        self.params_alpha   = nn.Parameter(torch.rand(**_params_kwargs_1))
        self.params_beta    = nn.Parameter(torch.rand(**_params_kwargs_1))
        self.params_gamma   = nn.Parameter(torch.rand(**_params_kwargs_2))
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def n_modes(self) -> tuple[int]:
        return self.__n_modes
    @property
    def channels(self) -> tuple[int]:
        return self.__channels
    
    
    def forward(self, X_fft: torch.Tensor) -> torch.Tensor:
        """## The forward propagation of `FourierBoltzmannLayer`
        
        -----
        ### Description
        This method computes the Fourier series coefficients of the collision term by calling two methods `compute_gain()` and `compute_loss()`.
        
        -----
        ### Remark
        1. Users should align `X_fft` in the following order of dimensions: `(batch, ..., dimension_1, ..., dimension_d, data)`
        """
        gain = self.compute_gain(X_fft)
        loss = self.compute_loss(X_fft)
        return gain-loss
    
    
    def compute_gain(self, X_fft: torch.Tensor) -> torch.Tensor:
        # Initialize storages
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        aX_fft  = torch.zeros(**_data_init_kwargs)
        bX_fft  = torch.zeros(**_data_init_kwargs)
        conv    = torch.zeros(**_data_init_kwargs)
        # Conduct convolution
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            aX_fft[*data_slice] = self.params_alpha[*kernel_slice] * X_fft[*data_slice]
            bX_fft[*data_slice] = self.params_beta[ *kernel_slice] * X_fft[*data_slice]
        abX_fft = convolve_freqs(aX_fft, bX_fft, self.__fft_dim)
        # Conduct integration-like operation
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            conv[*data_slice] = self.params_gamma[*kernel_slice] * abX_fft[*data_slice]
        conv = conv.sum(dim=-1, keepdim=True)
        return conv
    
    
    def compute_loss(self, X_fft: torch.Tensor) -> torch.Tensor:
        # Initialize storages
        fft_weight  = \
            self.params_alpha * self.params_beta * \
            self.params_gamma[*repeat(slice(None, None, 2), self.__dimension)]
        _data_init_kwargs = {
            'size':     (*X_fft.shape[:-1], self.__channels),
            'dtype':    torch.cfloat,
            'device':   X_fft.device,
        }
        wX_fft  = torch.zeros(**_data_init_kwargs)
        # Conduct convolution
        for kernel_slice in product(*self.__kernel_slices):
            data_slice = (..., *kernel_slice, slice(None))
            wX_fft[*data_slice] = fft_weight[*kernel_slice] * X_fft[*data_slice]
        conv = convolve_freqs(
            X_fft.repeat(*ones(X_fft.ndim-1), self.__channels), wX_fft,
            self.__fft_dim,
        ).sum(dim=-1, keepdim=True)
        return conv


##################################################
##################################################
# End of file