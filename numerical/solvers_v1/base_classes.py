import  warnings

from    typing              import  Callable, Optional, Sequence
from    typing_extensions   import  Self, override
import  numpy               as      np

from    ..utils             import  *


##################################################
##################################################
__all__ = [
    'SpectralMethodBase',
    'FastSM_Base',
]


##################################################
##################################################
class SpectralMethodBase():
    r"""The base class for the spectral methods for the kinetic equations.
    
    -----
    ### Description
    This class provides essential features which can be used to solve several kinetic equations using the spectral method (to be specific, the Fourier-Galerkin method).
    Although the inhomogeneous kinetic equation can also be solved using spectral methods by considering the transport stage, currently, this is not implemented.
    
    -----
    ### Properties
    * `dim` (`int`)
        The dimension of the domain.
    * `num_grid` (`int`)
        The number of the grid points in each dimension.
    * `v_axes` (`Sequence[int]`)
        The axes of the velocity space in a given distribution function.
    * `internal_fft_norm` (`str`)
        The normalization of the internal FFT.
    * `freq_min` (`int`)
        The minimum frequency in the Fourier space.
    * `freq_max` (`int`)
        The maximum frequency in the Fourier space.
    * `v_max` (`float`)
        The maximum velocity in the velocity space.
    * `v_ratio` (`float`)
        The ratio of the velocity space.
    * `radius` (`float`)
        The radius of the support of the density function in the velocity space.
    * `freqs` (`np.ndarray`)
        The frequencies in the Fourier space.        
    """
    def __init__(
            self,
            
            dim:        int,
            num_grid:   int,
            v_max:      float,
            
            freqs_keepdims: bool    = False,
        ) -> Self:
        """The initializer of the class `SpectralMethodBase`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `num_grid` (`int`)
            The number of the grid points in each dimension.
        * `v_max` (`float`)
            The maximum velocity in the velocity space.
        * `freqs_keepdims` (`bool`, default: `False`)
            The configuration for the initialization of `self.freqs`.
        """
        self._dim       = dim
        self._num_grid  = num_grid
        self._v_max     = v_max
        self._freqs     = freq_tensor(dim, num_grid, keepdims=freqs_keepdims)
        return
    
    
    @property
    def dim(self) -> int:
        """The dimension of the spatial domain."""
        return self._dim
    @property
    def num_grid(self) -> int:
        """The resolution of the grid in each dimension."""
        return self._num_grid
    @property
    def internal_fft_norm(self) -> str:
        """The normalization of the internal FFT, which is set to 'forward'."""
        return 'forward'
    @property
    def freq_min(self) -> int:
        """The minimum frequency in the Fourier space, determined by the resolution."""
        return -(self._num_grid // 2)
    @property
    def freq_max(self) -> int:
        """The maximum frequency in the Fourier space, determined by the resolution."""
        return (self._num_grid - 1) // 2
    @property
    def v_max(self) -> float:
        """The maximum velocity in each direction in the spectral method."""
        return self._v_max
    @property
    def v_ratio(self) -> float:
        """Returns `pi/self.v_max`, which is the inverse ratio of the extent of a period to the extent of the standard period."""
        return np.pi / self._v_max
    @property
    def v_axes(self) -> Sequence[int]:
        """The axes of the velocity space in a given distribution function, which is `tuple(range(1, 1+self.dim))`. Note that the first dimension is for the batch."""
        return tuple(range(-self.dim, 0, 1))
    @property
    def radius(self) -> float:
        """The radius of the support of the density function in the velocity space."""
        return self._v_max * LAMBDA
    @property
    def freqs(self) -> np.ndarray:
        """The tensor of allowed frequencies with `keepdims=False`."""
        return self._freqs
    
    
    def precompute(self, return_: bool=False) -> np.ndarray:
        """(`SpectralMethodBase`) The method `precompute` should be implemented in the derived classes."""
        pass
        
    
    def compute_collision_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        """(`SpectralMethodBase`) The method `compute_collision_fft` should be implemented in the derived classes."""
        pass
    
    
    def forward(
            self,
            t_curr:     float,
            f_fft:      np.ndarray,
            delta_t:    float,
            RK_fcn:     Callable[[float, np.ndarray, float, Callable], np.ndarray],
        ) -> np.ndarray:
        """Returns the Fourier coefficients of the solution at the next time step.
        
        -----
        (`SpectralMethodBase`) This method is already implemented."""
        return RK_fcn(t_curr, f_fft, delta_t, self.compute_collision_fft)
    
    
    def solve(
            self,
            t_init:     float,
            t_final:    float,
            delta_t:    float,
            f_fft:      np.ndarray,
            RK_fcn:     Callable[[None, np.ndarray, float, Callable], np.ndarray],
        ) -> np.ndarray:
        """Returns the Fourier coefficients of the solution at the final time step.
        
        -----
        (`SpectralMethodBase`) This method is already implemented."""
        t_curr = t_init
        f_fft_curr = f_fft
        while t_curr < t_final:
            f_fft_curr = self.forward(t_curr, f_fft_curr, delta_t, RK_fcn)
            t_curr += delta_t
        return f_fft_curr
    
    
    def __call__(
            self,
            t_init:     float,
            t_final:    float,
            delta_t:    float,
            f:          np.ndarray,
            RK_fcn:     Callable[[None, np.ndarray, float, Callable], np.ndarray],
        ) -> np.ndarray:
        """Returns the Fourier coefficients of the solution at the final time step.
        
        -----
        (`SpectralMethodBase`) This method is already implemented."""
        f_fft = np.fft.fftn(f, axes=self.v_axes, norm=self.internal_fft_norm)
        f_fft = self.solve(t_init, t_final, delta_t, f_fft, RK_fcn)
        return np.fft.ifftn(f_fft, axes=self.v_axes, norm=self.internal_fft_norm)
    
    
    def __repr__(self) -> str:
        return ''.join(
            [
                f"{self.__class__.__name__}(",
                f"dim={self._dim}, ",
                f"num_grid={self._num_grid}, ",
                f"v_max={self._v_max}",
                f")",
            ]
        )


##################################################
##################################################
class DiscreteVelocityMethodBase():
    r"""The base class for the discrete velocity methods for the kinetic equations.
    
    -----
    ### Description
    This class provides essential features which can be used to solve several kinetic equations using the discrete velocity method.
    
    -----
    ### Properties
    * `dim` (`int`)
        The dimension of the domain.
    * `num_grid` (`int`)
        The number of the grid points in each dimension.
    * `v_axes` (`Sequence[int]`)
        The axes of the velocity space in a given distribution function.
    * `v_max` (`float`)
        The maximum velocity in the velocity space.
    * `v_ratio` (`float`)
        The ratio of the space.
    * `radius` (`float`)
        The radius of the support of the density function in the velocity space.
    """
    def __init__(
            self,
            
            dim:        int,
            num_grid:   int,
            v_max:      float,
            
            freqs_keepdims: bool    = False,
        ) -> Self:
        """The initializer of the class `SpectralMethodBase`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `num_grid` (`int`)
            The number of the grid points in each dimension.
        * `v_max` (`float`)
            The maximum velocity in the velocity space.
        """
        self._dim       = dim
        self._num_grid  = num_grid
        self._v_max     = v_max
        return
    
    
    @property
    def dim(self) -> int:
        """The dimension of the spatial domain."""
        return self._dim
    @property
    def num_grid(self) -> int:
        """The resolution of the grid in each dimension."""
        return self._num_grid
    @property
    def v_axes(self) -> Sequence[int]:
        """The axes of the velocity space in a given distribution function, which is `tuple(range(1, 1+self.dim))`. Note that the first dimension is for the batch."""
        return tuple(range(-self.dim, 0, 1))
    @property
    def v_max(self) -> float:
        """The maximum velocity in each direction in the spectral method."""
        return self._v_max
    @property
    def v_ratio(self) -> float:
        """Returns `pi/self.v_max`, which is the inverse ratio of the extent of a period to the extent of the standard period."""
        return np.pi / self._v_max
    @property
    def radius(self) -> float:
        """The radius of the support of the density function in the velocity space."""
        return self._v_max * LAMBDA
    @property
    def input_shape(self) -> Sequence[int]:
        """Same as `repeat(num_grid, dim)`."""
        return tuple((self.num_grid for _ in range(self.dim)))
    
    
    def precompute(self, return_: bool=False) -> np.ndarray:
        """(`SpectralMethodBase`) The method `precompute` should be implemented in the derived classes."""
        pass
        
    
    def compute_collision_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        """(`SpectralMethodBase`) The method `compute_collision_fft` should be implemented in the derived classes."""
        pass
    
    
    def forward(
            self,
            t_curr:     float,
            f_fft:      np.ndarray,
            delta_t:    float,
            RK_fcn:     Callable[[float, np.ndarray, float, Callable], np.ndarray],
        ) -> np.ndarray:
        """Returns the Fourier coefficients of the solution at the next time step.
        
        -----
        (`SpectralMethodBase`) This method is already implemented."""
        return RK_fcn(t_curr, f_fft, delta_t, self.compute_collision_fft)
    
    
    def solve(
            self,
            t_init:     float,
            t_final:    float,
            delta_t:    float,
            f_fft:      np.ndarray,
            RK_fcn:     Callable[[None, np.ndarray, float, Callable], np.ndarray],
        ) -> np.ndarray:
        """Returns the Fourier coefficients of the solution at the final time step.
        
        -----
        (`SpectralMethodBase`) This method is already implemented."""
        t_curr = t_init
        f_fft_curr = f_fft
        while t_curr < t_final:
            f_fft_curr = self.forward(t_curr, f_fft_curr, delta_t, RK_fcn)
            t_curr += delta_t
        return f_fft_curr
    
    
    def __call__(
            self,
            t_init:     float,
            t_final:    float,
            delta_t:    float,
            f:          np.ndarray,
            RK_fcn:     Callable[[None, np.ndarray, float, Callable], np.ndarray],
        ) -> np.ndarray:
        """Returns the Fourier coefficients of the solution at the final time step.
        
        -----
        (`SpectralMethodBase`) This method is already implemented."""
        f_fft = np.fft.fftn(f, axes=self.v_axes, norm=self.internal_fft_norm)
        f_fft = self.solve(t_init, t_final, delta_t, f_fft, RK_fcn)
        return np.fft.ifftn(f_fft, axes=self.v_axes, norm=self.internal_fft_norm)
    
    
    def __repr__(self) -> str:
        return ''.join(
            [
                f"{self.__class__.__name__}(",
                f"dim={self._dim}, ",
                f"num_grid={self._num_grid}, ",
                f"v_max={self._v_max}",
                f")",
            ]
        )


##################################################
##################################################
class DirectSM_Base(SpectralMethodBase):
    r"""The base class for the direct spectral methods for the homogeneous Boltzmann equation.
    """
    def __init__(
            self,
            
            dim:        int,
            num_grid:   int,
            v_max:      float,
            
            order:      Optional[int]   = None,
        ) -> Self:
        """The initializer of the class `DirectSM_Base`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `num_grid` (`int`)
            The number of the grid points in each dimension.
        * `v_max` (`float`)
            The maximum velocity in the velocity space.
        * `order` (`Optional[int]`)
            The order of the quadrature rule for the computation of the kernel modes.
        """
        super().__init__(dim, num_grid, v_max)
        self._order     = order if isinstance(order, int) else num_grid
        self._kernel:   Optional[np.ndarray]    = None
        return
    

    @property
    def order(self) -> int:
        return self._order
    @property
    def kernel(self) -> Optional[np.ndarray]:
        return self._kernel
    
    
    @override
    def precompute(self, return_: bool=False) -> Optional[np.ndarray]:
        func: Callable[[np.ndarray, object], np.ndarray]
        if self._dim==1:
            func = self._precompute_integrand_1D
        elif self._dim==2:
            func = self._precompute_integrand_2D
        elif self._dim==3:
            warnings.warn(
                ' '.join(
                    [
                        f"For efficient implementation, the kernel modes for the 3-dimensional case should be referred by computing the 2-norms of the addition and subtraction of two input frequencies.",
                        f"Use the method 'compute_integral_indices' of this class."
                    ]
                ),
                Warning
            )
            func = self._precompute_integrand_3D
        else:
            raise NotImplementedError(
                ' '.join(
                    [
                        f"Currently, this function is supported from 1D to 3D.",
                        f"(dimension={self.dim})",
                    ]
                )
            )
        func_kwargs = self._precompute_integrand_kwargs
        kernel = integration_guass_legendre(
            num_roots   = self._order,
            a           = 0.0,
            b           = 1.0,
            func        = func,
            func_kwargs = func_kwargs,
        ).astype(np.complex128)
        self._kernel = kernel
        
        if return_:
            return kernel
        else:
            return None
    
    
    @property
    def _precompute_integrand_1D(self) -> Callable[[np.ndarray, object], np.ndarray]:
        pass
    @property
    def _precompute_integrand_2D(self) -> Callable[[np.ndarray, object], np.ndarray]:
        pass
    @property
    def _precompute_integrand_3D(self) -> Callable[[np.ndarray, object], np.ndarray]:
        pass
    @property
    def _precompute_integrand_kwargs(self) -> Callable[[np.ndarray, object], np.ndarray]:
        pass
    
    
    def compute_integral_indices(
            self,
            freq1:  np.ndarray,
            freq2:  np.ndarray,
        ) -> np.ndarray:
        if freq1.shape != freq2.shape:
            raise ValueError(
                '\n'.join(
                    [
                        f"Two input (collection of) frequencies 'freq1' and 'freq2' should be of the same shape.",
                        f"* {freq1.shape=}",
                        f"* {freq2.shape=}",
                    ]
                )
            )
        idx1 = np.sum(np.power(freq1+freq2, 2), axis=-1)
        idx2 = np.sum(np.power(freq1-freq2, 2), axis=-1)
        return np.stack((idx1, idx2), axis=-1)
        
    
    def compute_collision_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        # Initialize the dervatives of the Fourier coefficients
        ret = np.zeros_like(fft_curr, dtype=np.complex128)
        
        # Compute the derivatives of the Fourier coefficients using multiple loops
        for freq_target in self._freqs:
            for freq_1 in self._freqs:
                freq_2 = freq_target - freq_1
                
                # Check if the second frequency is allowed
                if np.min(freq_2) < self.freq_min or np.max(freq_2) > self.freq_max:
                    continue
                
                # Determine the indices for the computation of the kernel modes
                if self.dim <= 2:
                    kcs_12 = self._kernel[tuple(freq_1)][tuple(freq_2)]
                    kcs_22 = self._kernel[tuple(freq_2)][tuple(freq_2)]
                else:
                    _idx_a1 = int(np.sum((freq_1 + freq_2)**2))
                    _idx_a2 = int(np.sum((freq_1 - freq_2)**2))
                    _idx_b1 = int(4 * np.sum((freq_2)**2))
                    _idx_b2 = 0
                    kcs_12 = self._kernel[_idx_a1, _idx_a2]
                    kcs_22 = self._kernel[_idx_b1, _idx_b2]
                # Update the returning tensor
                ret[tuple((slice(None), *freq_target))] += \
                    (fft_curr[tuple((slice(None), *freq_1))] * fft_curr[tuple((slice(None), *freq_2))]) * \
                    (kcs_12 - kcs_22)
            
        # Return the time-derivative of the Fourier coefficients
        return ret


##################################################
##################################################
class FastSM_Base(SpectralMethodBase):
    r"""## The base class for the fast spectral method for solving homogeneous kinetic equations.
    
    -----
    ### Description
    This class provides features which can be used to solve several kinetic equations using the spectral method (to be specific, the Fourier-Galerkin method).
    The construction of this base class is inspired by the idea of rewriting of the kernel modes:
    This aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Nevertheless, alternative implementation of a fast spectral method can be done by overriding necessary methods.
    
    Reference: ["A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels" by Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu](https://epubs.siam.org/doi/10.1137/16M1096001)
    """
    def __init__(
            self,
            
            dim:        int,
            num_grid:   int,
            v_max:      float,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,
        ) -> Self:
        """The initializer of the class `FastSM_Base`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `num_grid` (`int`)
            The number of the grid points in each dimension.
        * `v_max` (`float`)
            The maximum velocity in the velocity space.
        * `quad_order_uniform` (`Optional[int]`)
            The number of the points required to do integration using uniform quadrature rule.
        * `quad_order_legendre` (`Optional[int]`)
            The number of the points required to do integration using Legendre quadrature rule.
        * `quad_order_lebedev` (`Optional[int]`)
            The number of the points required to do integration using Lebedev quadrature rule.
        """
        super().__init__(dim, num_grid, v_max)        
        self._quad_order_uniform    = \
            quad_order_uniform  if isinstance(quad_order_uniform,   int)\
            else int(1 + np.sqrt(1/2) * LAMBDA * num_grid)
        self._quad_order_legendre   = \
            quad_order_legendre if isinstance(quad_order_legendre,  int)\
            else self._num_grid
            # else int(1 + np.sqrt(dim) * LAMBDA * num_grid)
        self._quad_order_lebedev    = \
            quad_order_lebedev  if isinstance(quad_order_lebedev,   int)\
            else DEFAULT_QUAD_ORDER_LEBEDEV
        self._fsm_scale:    Optional[np.ndarray] = None
        self._fsm_phase:    Optional[np.ndarray] = None
        self._kernel_diag:  Optional[np.ndarray] = None
        return
    
    
    @property
    def quad_order_uniform(self) -> int:
        """The number of the points required to do integration on S^1."""
        return self._quad_order_uniform
    @property
    def quad_order_legendre(self) -> int:
        """The number of the points required to do radial integration."""
        return self._quad_order_legendre
    @property
    def quad_order_lebedev(self) -> int:
        """The number of the points required to do integration on S^2."""
        return self._quad_order_lebedev
    @property
    def fsm_scale(self) -> Optional[np.ndarray]:
        return np.squeeze(self._fsm_scale, axis=0)
    @property
    def fsm_phase(self) -> Optional[np.ndarray]:
        return np.squeeze(self._fsm_phase, axis=0)
    @property
    def kernel_diag(self) -> Optional[np.ndarray]:
        return np.squeeze(self._kernel_diag, axis=0)
    @property
    def approximation_level(self) -> int:
        return int(np.prod(self._fsm_phase.shape[-2:]))
    
    
    def _dimension_error(self) -> None:
        raise NotImplementedError(f"This class is supported only for 2D and 3D problems. (dim: {self._dim})")
    
    
    @override
    def precompute(self) -> None:
        """Precomputes the gain and loss parts of the kernel modes.
        
        -----
        This method computes the gain and loss parts of the kernel modes.
        Because the base class is mainly inspired by ["A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels" by Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu](https://epubs.siam.org/doi/10.1137/16M1096001), inside the implementation, there are two tensors which define the gain part of the kernel modes.
        """
        self._fsm_scale, self._fsm_phase = self._precompute_fsm_gain()
        self._kernel_diag = self._precompute_fsm_loss()
        return
    def _precompute_fsm_gain(self) -> tuple[np.ndarray, np.ndarray]:
        pass
    def _precompute_fsm_loss(self) -> np.ndarray:
        pass
    
    
    def compute_collision_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        # NOTE: Be careful to implement the truncated 'linear' convolution.
        ## Part 1. Compute the gain part
        terms1 = convolve_freqs(
            fft_curr[..., None, None] * self._fsm_phase,
            fft_curr[..., None, None] * self._fsm_phase.conj(),
            axes = self.v_axes,
        )
        gain_fft = np.sum(self._fsm_scale*terms1, axis=(-2, -1))
        
        ## Part 2. Compute the loss part
        loss_fft = convolve_freqs(
            fft_curr * self._kernel_diag,
            fft_curr,
            axes = self.v_axes,
        )
        
        ## Return the Fourier coefficients of the collision operator
        return gain_fft - loss_fft


##################################################
##################################################
# End of file