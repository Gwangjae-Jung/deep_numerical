import  warnings

from    typing              import  Callable, Optional
from    typing_extensions   import  Self, override

import  numpy               as      np
from    scipy.interpolate   import  RegularGridInterpolator

from    ..      import  utils


##################################################
##################################################
__all__: list[str] = [
    'SpectralMethodBase',
    'DirectSM_Base',
    'FastSM_Boltzmann',
    'FastSM_Landau',
]


##################################################
##################################################
class SpectralMethodBase():
    r"""The base class for the spectral methods for the kinetic equations.
    
    -----
    ### Description
    This class provides essential features which can be used to solve several kinetic equations using the spectral method (to be specific, the Fourier-Galerkin method).
    Although the inhomogeneous kinetic equation can also be solved using spectral methods by considering the transport stage, currently, this is not implemented.  
    """
    def __init__(
            self,
            
            dim:    int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            freqs_keepdims: bool = False,
        ) -> Self:
        """The initializer of the class `SpectralMethodBase`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `v_num_grid` (`int`)
            The resolution of the grid in each velocity dimension.
        * `v_max` (`float`)
            The maximum coordinate in the velocity space.
        * `x_num_grid` (`Optional[int]`, default: `None`)
            The resolution of the grid in each spatial dimeneion.
        * `x_max` (`Optional[float]`, default: `None`)
            The maximum coordinate in the spatial space.
        * `freqs_keepdims` (`bool`, default: `False`)
            The configuration for the initialization of `self.freqs`.
        
        -----
        ### Note
        `SpectralMethodBase` and all classes which inherits `SpectralMethodBase` discretizes the velocity space in such a way that both `+v_max` and `-v_max` are *not* included as entries.
        In contrast, they discretizes the space, containing both `x_max` and `-x_max` as entries.
        """
        # Dimension
        self._dim:  int = dim
        
        # Velocity space
        self._v_num_grid:   int     = v_num_grid
        self._v_max:        float   = v_max
        self._delta_v:      float   = 2*v_max / v_num_grid
        
        # Spatial space (Homogeneity)
        self._is_homogeneous:   bool    = (x_max is None) and (x_num_grid is None)
        delta_x:    Optional[float] = None
        if self._is_homogeneous:
            x_max       = 0
            x_num_grid  = 0
        else:
            delta_x = (2*x_max) / (x_num_grid-1)
        self._x_num_grid:   Optional[int]   = x_num_grid
        self._x_max:        Optional[float] = x_max
        self._delta_x:      Optional[float] = delta_x
        
        # Frequency (spectral method)
        self._freqs:    np.ndarray  = utils.freq_tensor(dim, v_num_grid, keepdims=freqs_keepdims)
        
        # Interpolation for the inhomogeneous case
        # `self._xv_pad`:   The ordered tuple of interpolating `xv` points
        # `self._xv_query`: The query points at transport stage, saved as a single tensor
        self._boundary_condition:   Optional[str]               = None
        self._xv_pad:               Optional[tuple[np.ndarray]] = None
        self._xv_query:             Optional[np.ndarray]        = None
        self._arg_bd_inflow:        Optional[np.ndarray]        = None
        self._arg_bd_inflow_spec:   Optional[np.ndarray]        = None
        
        # Done
        return
    
    
    # Properties - Dimension
    @property
    def dim(self) -> int:
        """The dimension of the spatial domain."""
        return self._dim
    @property
    def f_ndim(self) -> int:
        """The dimension of the input tensor, which is `2*(1+self.dim)`. Note that
        * the zeroth dimension is used to save multiple instances,
        * the next `self.dim` dimension(s) is(are) used to refer to the locatioon,
        * the next `self.dim` dimension(s) is(are) used to refer to the velocity,
        * and the last dimension is used to refer a specific function.
        """
        return 2*(1+self._dim)
    @property
    def f_shape(self) -> tuple[int]:
        """The shape of the input tensor."""
        n_x: int = 1 if self.is_homogeneous else self._x_num_grid
        return tuple((1, *utils.repeat(n_x, self._dim), *utils.repeat(self._v_num_grid, self._dim), -1))
    
    
    # Properties - Velocity space
    @property
    def v_num_grid(self) -> int:
        """The resolution of the grid in each velocity dimension."""
        return self._v_num_grid
    @property
    def v_max(self) -> float:
        """The maximum coordinate in the velocity space."""
        return self._v_max
    @property
    def v_axes(self) -> tuple[int]:
        """
        The axes of the velocity space in a given distribution function. It is defined as
            `tuple(range(1+self.dim, 1+2*self.dim))`.
        """
        return tuple(range(1+self.dim, 1+2*self.dim))
    
    
    # Properties - Velocity space (Supplementary)
    @property
    def v_ratio(self) -> float:
        """Returns `pi/self.v_max`, which is the inverse ratio of the extent of a period to the extent of the standard period."""
        return np.pi/self._v_max
    @property
    def v_radius(self) -> float:
        """The radius of the support of the density function in the velocity space."""
        return self._v_max * utils.LAMBDA
    
    
    # Properties - Spatial space
    @property
    def is_homogeneous(self) -> bool:
        """The homoegeneity of the problem."""
        return self._is_homogeneous
    @property
    def x_num_grid(self) -> Optional[int]:
        """The resolution of the grid in each spatial dimeneion."""
        return self._x_num_grid
    @property
    def x_max(self) -> Optional[float]:
        """The maximum coordinate in the spatial space."""
        return self._x_max
    @property
    def x_axes(self) -> tuple[int]:
        """
        The axes of the spatial space in a given distribution function. It is defined as
            `tuple(range(1, 1+self.dim))`.
        """
        return tuple(range(1, 1+self.dim))
    
    
    # Properties - Fast Fourier transform
    @property
    def internal_fft_norm(self) -> str:
        """The normalization of the internal FFT, which is set to 'forward'."""
        return 'forward'
    @property
    def freq_min(self) -> int:
        """The minimum frequency in the Fourier space, determined by the resolution."""
        return -(self._v_num_grid//2)
    @property
    def freq_max(self) -> int:
        """The maximum frequency in the Fourier space, determined by the resolution."""
        return (self._v_num_grid-1)//2
    @property
    def freqs(self) -> np.ndarray:
        """The tensor of allowed frequencies with `keepdims=False`."""
        return self._freqs

    
    # Properties - Einsum
    @property
    def einsum_string_x(self) -> str:
        return utils.EINSUM_STRING[1:1+self._dim]
    @property
    def einsum_string_v(self) -> str:
        return utils.EINSUM_STRING[1+self._dim:1+2*self._dim]
    @property
    def einsum_string_bxvd(self) -> str:
        return f"b{self.einsum_string_x}{self.einsum_string_v}d"

    
    def precompute(self, return_: bool=False) -> np.ndarray:
        """(`SpectralMethodBase`) The method `precompute` should be implemented in the derived classes."""
        pass
    
    
    @property
    def supported_boundary_conditions__dict(self) -> dict[str, tuple[str]]:
        return {
            'zero':     tuple(('zero', 'zeros')),
            'periodic': tuple(('periodic')),
        }
    @property
    def supported_boundary_conditions(self) -> set[str]:
        return set(('zero', 'zeros', 'periodic', 'specular',))
    @property
    def arg_bd_inflow(self) -> np.ndarray:
        return self._arg_bd_inflow
    @property
    def arg_bd_inflow_spec(self) -> np.ndarray:
        return self._arg_bd_inflow_spec
    

    def precompute_query(self, delta_t: float, boundary_condition: str) -> None:
        """(`SpectralMethodBase`) The method `precompute_query` is already implemented in the base class.
        Note that this method *explicitly* requires the time step with which the transport stage is conducted.
        Be sure to pass the time step which is used to conduct the collision stage.
        
        # Parameters
        `delta_t` (`float`)
            * The time step with which time marching (collision and transport stages) is conducted.
            
        `boundary_condition` (`str`)
            * The boundary condition the kinetic equation to be solved is equipped with.
            * Supported boundary conditions and corresponding keywords are given as follows:
                * Zero boundary condition: `zero`, `zeros`
                * Periodic boundary condition: `periodic`
        """
        if not self.is_homogeneous:
            # Check the boundary condition
            boundary_condition = boundary_condition.lower()
            assert boundary_condition in self.supported_boundary_conditions, \
                f"The passed 'boundary_condition' ({boundary_condition}) is not supported."
            # Compute the padded input
            padded_x_max = self._x_max + self._delta_x
            padded_x_num_grid = self._x_num_grid + 2
            self._xv_pad = tuple(
                (
                    *utils.repeat(np.linspace(-padded_x_max, padded_x_max, padded_x_num_grid), self._dim),
                    *utils.repeat(utils.velocity_grid(1, self._v_num_grid, self._v_max).flatten(), self._dim),
                )
            )
            # Compute the query points
            self._xv_query = utils.space_grid(
                2*self._dim,
                (
                    *utils.repeat(self._x_num_grid, self._dim),
                    *utils.repeat(self._v_num_grid, self._dim),
                ),
                (
                    *utils.repeat(self._x_max, self._dim),
                    *utils.repeat(self._v_max, self._dim),
                ),
                where_closed = (*utils.repeat('both', self._dim), *utils.repeat('none', self._dim)),
            )   # NOTE: Contains `pm(x_max)` but does not contain `pm(v_max)`
            self._xv_query[..., :self._dim] -= delta_t * self._xv_query[..., self._dim:]
            if boundary_condition in ("periodic",):
                self._xv_query[..., :self._dim] = \
                    (self._xv_query[..., :self._dim] + self._x_max) % (2*self._x_max) - self._x_max
            self._boundary_condition = boundary_condition
            ##### Computation of some other member variables
            self._arg_bd_inflow = utils.arg_boundary_inflow(self._xv_query)
            self._arg_bd_inflow_spec = utils.arg_specular_velocity(self._v_num_grid, self._arg_bd_inflow, dim=self._dim)
        else:
            warnings.warn(
                ' '.join([
                    "The method 'precompute_query()' is called, but no precomputation will be conducted, as the solver is set space-homogeneous.",
                    "Check your initialization again."
                ]),
                UserWarning
            )
            self._xv_pad    = None
            self._xv_query  = None
        return
        
    
    def compute_collision_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        """(`SpectralMethodBase`) This method computes `gain_fft - loss_fft`, where
        
        * `gain_fft` is the gain term in the Fourier space, computed using the method `compute_collision_gain_fft`, and
        * `loss_fft` is the loss term in the Fourier space, computed using the method `compute_collision_loss_fft`.
        """
        return (
            self.compute_collision_gain_fft(_PLACEHOLDER__t_curr, fft_curr) - \
            self.compute_collision_loss_fft(_PLACEHOLDER__t_curr, fft_curr)
        )
    def compute_collision_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        """(`SpectralMethodBase`) The method `compute_collision_gain_fft` should be implemented in the derived classes."""
        pass
    def compute_collision_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        """(`SpectralMethodBase`) The method `compute_collision_loss_fft` should be implemented in the derived classes."""
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
        homogeneous_result = RK_fcn(t_curr, f_fft, delta_t, self.compute_collision_fft)
        if self._is_homogeneous:
            return homogeneous_result
        else:
            # Frequency space -> Physical space
            homogeneous_result: np.ndarray = np.fft.ifftn(homogeneous_result, axes=self.v_axes, norm=self.internal_fft_norm).real
            
            # Enforce the boundary condition
            if self._boundary_condition in ("specular",):
                homogeneous_result[:, *[self._arg_bd_inflow[:, k] for k in range(2*self._dim)], :] = \
                    homogeneous_result[:, *[self._arg_bd_inflow_spec[:, k] for k in range(2*self._dim)], :]
            # Conduct the transport stage (by interpolation)
            homogeneous_result = homogeneous_result.transpose(
                *tuple(range(1, (2*self._dim+1)+1)), 0
            )   ## Permutation for interpolation
            homogeneous_result = np.pad(
                homogeneous_result,
                pad_width = (
                    *utils.repeat((1, 1), self._dim), # space
                    *utils.repeat((0, 0), self._dim), # velocity
                    (0, 0), # data
                    (0, 0), # *batch* (transposed)
                ),
                # mode = "wrap",
            )
            interp = RegularGridInterpolator(
                points  = self._xv_pad,
                values  = homogeneous_result,
                method          = 'linear',
                bounds_error    = False,
                fill_value      = None,
            ).__call__
            homogeneous_result = interp(self._xv_query)
            homogeneous_result = homogeneous_result.transpose(
                2*self._dim+1, *tuple(range(2*self._dim+1))
            )   ## Inverse permutation
            ### End of the transport stage
            
            # Enforce the boundary condition, again
            if self._boundary_condition in ("specular",):
                homogeneous_result[:, *[self._arg_bd_inflow[:, k] for k in range(2*self._dim)], :] = \
                    homogeneous_result[:, *[self._arg_bd_inflow_spec[:, k] for k in range(2*self._dim)], :]
            
            # Physical space -> Frequency space
            return np.fft.fftn(homogeneous_result, axes=self.v_axes, norm=self.internal_fft_norm)

        
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
                f"v_num_grid={self._v_num_grid}, ",
                f"v_max={self._v_max}",
                f"x_num_grid={self._x_num_grid}, ",
                f"x_max={self._x_max}",
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
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int],
            x_max:      Optional[float],
            
            order:      Optional[int]   = None,
        ) -> Self:
        """The initializer of the class `DirectSM_Base`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `v_num_grid` (`int`)
            The number of the grid points in each dimension.
        * `v_max` (`float`)
            The maximum velocity in the velocity space.
        * `x_num_grid` (`int`)
            The number of the grid points in each dimension.
        * `x_max` (`float`)
            The maximum velocity in the velocity space.
        * `order` (`Optional[int]`)
            The order of the quadrature rule for the computation of the kernel modes.
        """
        super().__init__(dim, v_num_grid, v_max, x_num_grid, x_max)
        self._order:    int = order if isinstance(order, int) else v_num_grid
        self._kernel:   Optional[np.ndarray]    = None
        return
    

    @property
    def order(self) -> int:
        return self._order
    @property
    def kernel(self) -> Optional[np.ndarray]:
        return self._kernel
    
    
    @override
    def precompute(self) -> None:
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
        kernel = utils.integration_guass_legendre(
            num_roots   = self._order,
            a           = 0.0,
            b           = 1.0,
            func        = func,
            func_kwargs = func_kwargs,
        ).astype(np.complex128)
        self._kernel = kernel[..., None]    # Append one extra dimension
        return
    
    
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
        
    
    def compute_collision_gain_fft(
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
                else:
                    _idx_a1 = int(np.sum((freq_1 + freq_2)**2))
                    _idx_a2 = int(np.sum((freq_1 - freq_2)**2))
                    kcs_12 = self._kernel[_idx_a1, _idx_a2]
                # Update the returning tensor
                ret[tuple((*utils.repeat(slice(None), 1+self._dim), *freq_target))] += \
                    fft_curr[tuple((*utils.repeat(slice(None), 1+self._dim), *freq_1))] * \
                    fft_curr[tuple((*utils.repeat(slice(None), 1+self._dim), *freq_2))] * \
                    kcs_12
            
        # Return the time-derivative of the Fourier coefficients
        return ret
    
    
    def compute_collision_loss_fft(
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
                    kcs_22 = self._kernel[tuple(freq_2)][tuple(freq_2)]
                else:
                    _idx_b1 = int(4 * np.sum((freq_2)**2))
                    _idx_b2 = 0
                    kcs_22 = self._kernel[_idx_b1, _idx_b2]
                # Update the returning tensor
                ret[tuple((*utils.repeat(slice(None), 1+self._dim), *freq_target))] += \
                    fft_curr[tuple((*utils.repeat(slice(None), 1+self._dim), *freq_1))] * \
                    fft_curr[tuple((*utils.repeat(slice(None), 1+self._dim), *freq_2))] * \
                    kcs_22
            
        # Return the time-derivative of the Fourier coefficients
        return ret
    

##################################################
##################################################
class FastSM_Boltzmann(SpectralMethodBase):
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
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int],
            x_max:      Optional[float],
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,
        ) -> Self:
        """The initializer of the class `FastSM_Base`.
        
        -----
        ### Arguments
        * `dim` (`int`)
            The dimension of the domain.
        * `v_num_grid` (`int`)
            The number of the grid points in each dimension.
        * `v_max` (`float`)
            The maximum velocity in the velocity space.
        * `x_num_grid` (`int`)
            The number of the grid points in each dimension.
        * `x_max` (`float`)
            The maximum velocity in the velocity space.
        * `quad_order_uniform` (`Optional[int]`)
            The number of the points required to do integration using uniform quadrature rule.
        * `quad_order_legendre` (`Optional[int]`)
            The number of the points required to do integration using Legendre quadrature rule.
        * `quad_order_lebedev` (`Optional[int]`)
            The number of the points required to do integration using Lebedev quadrature rule.
        """
        super().__init__(dim, v_num_grid, v_max, x_num_grid, x_max)        
        self._quad_order_uniform    = \
            quad_order_uniform  if isinstance(quad_order_uniform,   int)\
            else int(1 + np.sqrt(1/2) * utils.LAMBDA * v_num_grid)
        self._quad_order_legendre   = \
            quad_order_legendre if isinstance(quad_order_legendre,  int)\
            else self._v_num_grid
            # else int(1 + np.sqrt(dim) * LAMBDA * num_grid)
        self._quad_order_lebedev    = \
            quad_order_lebedev  if isinstance(quad_order_lebedev,   int)\
            else utils.DEFAULT_QUAD_ORDER_LEBEDEV
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
        return np.squeeze(self._fsm_scale, axis=tuple(range(1+self._dim)))
    @property
    def fsm_phase(self) -> Optional[np.ndarray]:
        return np.squeeze(self._fsm_phase, axis=tuple(range(1+self._dim)))
    @property
    def kernel_diag(self) -> Optional[np.ndarray]:
        return np.squeeze(self._kernel_diag, axis=tuple(range(1+self._dim)))
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
    
    
    def compute_collision_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        terms1 = utils.convolve_freqs(
            fft_curr[..., None, None] * self._fsm_phase,
            fft_curr[..., None, None] * self._fsm_phase.conj(),
            axes = self.v_axes,
        )
        gain_fft = np.sum(self._fsm_scale*terms1, axis=(-2, -1))
        return gain_fft
    
    
    def compute_collision_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        loss_fft = utils.convolve_freqs(
            fft_curr * self._kernel_diag,
            fft_curr,
            axes = self.v_axes,
        )
        return loss_fft
    

##################################################
##################################################
class FastSM_Landau(SpectralMethodBase):
    """## The class for the fast spectral method for solving the homogeneous Fokker-Planck-Landau equation.
    
    -----
    ### Description
    This class provides features for solving the Fokker-Planck-Landau equation, inspired by the [paper](https://www.sciencedirect.com/science/article/pii/S0021999100966129).
    The main components of the algorithm is to define two functions, which are labelled `fpl_character_1` and `fpl_character_2` (with an underscore at the beginning of the names), respectively.
    Here, `fpl_character_1` and `fpl_character_2` are real-valued functions defined on the set of the nonnegative integers, and
        `B(l, m) = (norm(l, 2)^2 * P[m]) - (l.T @ Q[m] @ l)`,
    where `P` and `Q` are defined as in the paper in terms of `fpl_character_1` and `fpl_character_2`.
    
    Reference: [L. Pareschi, G. Russo, G. Toscani, Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator, Journal of Computational Physics, Volume 165, Issue 1, 2000, Pages 216-236](https://www.sciencedirect.com/science/article/pii/S0021999100966129).
    
    -----
    ### Note
    1. In the implementation of the spectral method for the Fokker-Planck-Landau equation, the minimum ratio of the maximum speed in each dimension to the radius of support is 0.5, rather than `2/(3+sqrt(2))`, the value which is required for the Boltzmann equation. Hence, the following member variables are redefined:
            - `radius`: From `v_max*LAMBDA` to `v_max*LAMBDA_FPL`.
    """
    def __init__(
            self,
            
            dim:        int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            vhs_coeff:  Optional[float] = None,
            vhs_alpha:  Optional[float] = None,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
        ) -> Self:
        super().__init__(
            dim         = dim,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            x_num_grid  = x_num_grid,
            x_max       = x_max,
            freqs_keepdims  = True,
        )   # NOTE Set `freqs_keepdims=True` for the vectorized implementation
        self._vhs_coeff:    float   = vhs_coeff
        self._vhs_alpha:    float   = vhs_alpha
        self._quad_order_uniform:   int = quad_order_uniform    \
            if isinstance(quad_order_uniform,   int) else v_num_grid
        self._quad_order_legendre:  int = quad_order_legendre   \
            if isinstance(quad_order_legendre,  int) else v_num_grid
        
        # Add the batch dimension for ease of implementation
        self._freqs:        np.ndarray = np.array([])
        self._freq_norms:   np.ndarray = np.array([])
        self.__redefine_freqs()
        
        # Character functions and tensors for the gain and loss terms
        ## NOTE: The character tensors should be 1-dimensional tensors.
        ## NOTE: The gain/loss tensors should be multi-dimensional tensors of shape.
        ## Shapes of the tensors
        ## With `base_shape` defined as the property `__fpl_base_shape`
        ## * `self._fpl_gain_tensor_1`: `base_shape`
        ## * `self._fpl_gain_tensor_2`: `(*base_shape, *repeat(self._dim, 2))`
        ## * `self._fpl_loss_tensor`:   `base_shape`
        self._fpl_character_1:      Optional[np.ndarray] = None
        self._fpl_character_2:      Optional[np.ndarray] = None
        self._fpl_gain_tensor_1:    Optional[np.ndarray] = None
        self._fpl_gain_tensor_2:    Optional[np.ndarray] = None
        self._fpl_loss_tensor:      Optional[np.ndarray] = None

        # Precompute
        self.precompute()
        
        return
    
    
    @override
    @property
    def v_radius(self) -> float:
        """The radius of the support of the density function in the velocity space.
        
        -----
        (`FastSM_FPL_VHS`) Redefined."""
        return self._v_max * utils.LAMBDA_FPL
    @override
    @property
    def freqs(self) -> np.ndarray:
        """The frequency tensor for the spectral method.
        ### Note
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), self.dim)`,
        dimensions aligned as in the tensors for the distribution functions.
        -----
        (`FastSM_FPL_VHS`) Redefined."""
        return self._freqs
    @override
    @property
    def freq_norms(self) -> np.ndarray:
        """The norms of the frequency tensor for the spectral method.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`,
        dimensions aligned as in the tensors for the distribution functions.
        -----
        (`FastSM_FPL_VHS`) Redefined."""
        return self._freq_norms
    @property
    def vhs_coeff(self) -> float:
        return self._vhs_coeff
    @property
    def vhs_alpha(self) -> float:
        return self._vhs_alpha
    @property
    def fpl_character_1(self) -> Optional[np.ndarray]:
        """Returns the 1-tensor of the first characteristic function.
        ### Note
        The entry of index `k` (`k>0`) corresponds to the value of the function at `sqrt(k)`.
        In other words, the input index `k` is the square of a frequency mode.
        The shape of this property is `(1+self.dim*(self.num_grid**2),)`.
        """
        return self._fpl_character_1
    @property
    def fpl_character_2(self) -> Optional[np.ndarray]:
        """Returns the 1-tensor of the second characteristic function.
        ### Note
        The entry of index `k` (`k>0`) corresponds to the value of the function at `sqrt(k)`.
        In other words, the input index `k` is the square of a frequency mode.
        The shape of this property is `(1+self.dim*(self.num_grid**2),)`.
        """
        return self._fpl_character_2
    @property
    def fpl_base_shape(self) -> tuple[int]:
        """Returns the base shape of the tensors for the gain and loss terms, which is defined as
            `tuple((1, *ones(self._dim), *repeat(self._v_num_grid, self._dim), 1))`.
        Note that the shape of `self.fpl_gain_tensor_2` is `(*self.__fpl_base_shape, *repeat(self.dim, 2))`."""
        return tuple((1, *utils.ones(self._dim), *utils.repeat(self._v_num_grid, self._dim), 1))
    @property
    def fpl_gain_tensor_1(self) -> Optional[np.ndarray]:
        """Returns the tensor of the positive scaling part of the gain term.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`,
        dimensions aligned as in the tensors for the distribution functions.
        """
        return self._fpl_gain_tensor_1
    @property
    def fpl_gain_tensor_2(self) -> Optional[np.ndarray]:
        """Returns the tensor of the negative bilinear part of the gain term.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1, *repeat(self.dim, 2))`,
        dimensions aligned as in the tensors for the distribution functions.
        """
        return self._fpl_gain_tensor_2
    @property
    def fpl_loss_tensor(self) -> Optional[np.ndarray]:
        """Returns the tensor of the loss term.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`,
        dimensions aligned as in the tensors for the distribution functions.
        """
        return self._fpl_loss_tensor
    
    
    def __redefine_freqs(self) -> None:
        freqs = utils.freq_tensor(self._dim, self._v_num_grid, keepdims=True)
        freqs = freqs.reshape(
            (1, *utils.repeat(1, self._dim), *utils.repeat(self._v_num_grid, self._dim), self._dim)
        )
        freq_norms = np.linalg.norm(freqs, ord=2, axis=-1, keepdims=True)
        self._freqs         = freqs
        self._freq_norms    = freq_norms
        return
    
    
    @override
    def precompute(self) -> None:
        if self._dim in (2, 3):
            getattr(self, f"_precompute_fpl_character_1_{ self._dim}D")()
            getattr(self, f"_precompute_fpl_character_2_{ self._dim}D")()
            getattr(self, f"_precompute_fpl_gain_tensors_{self._dim}D")()
            self._precompute_fpl_loss_tensor()
        return
    
    
    # 2-dimensional precomputations
    def _precompute_fpl_character_1_2D(self) -> None:
        pass
    def _precompute_fpl_character_2_2D(self) -> None:
        pass
    def _precompute_fpl_gain_tensors_2D(self) -> None:
        """This function returns two tensors `P` and `Q`, which are defined in the reference.
        The shapes of `P` and `Q` are listed below.
        1. `P`: A `1+self.dim+self.dim+1`-tensor of shape `(1, *ones(self.dim), *repeat(self.num_grid, self.dim))`
        2. `Q` is a `1+self.dim+self.dim+2`-tensor of shape `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), *(2, 2))`.
        """
        pass
    # 3-dimensional precomputations
    def _precompute_fpl_character_1_3D(self) -> None:
        pass
    def _precompute_fpl_character_2_3D(self) -> None:
        pass
    def _precompute_fpl_gain_tensors_3D(self) -> None:
        pass
    
    
    def _precompute_fpl_loss_tensor(self) -> None:
        _EINSUM_XV = utils.EINSUM_STRING[:2*self._dim]
        _freq_norms_sq: np.ndarray  = np.sum(self._freqs**2, axis=-1, dtype=np.float64, keepdims=True)
        positive: np.ndarray = _freq_norms_sq * self._fpl_gain_tensor_1
        negative: np.ndarray = np.einsum(
            f"b{_EINSUM_XV}i,b{_EINSUM_XV}vij,b{_EINSUM_XV}j->b{_EINSUM_XV}v",
            self._freqs, self._fpl_gain_tensor_2, self._freqs
        )
        self._fpl_loss_tensor = positive - negative
        return
    
    
    def compute_collision_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        # NOTE: By a property of the Fourier transform and the convention of FFT libraries, the output should be negated
        # NOTE: The collision term is zero for the 1-dimensional case
        if self._dim in (2, 3):   
            gain_fft_positive: np.ndarray = \
                utils.convolve_freqs(
                    (self._freq_norms**2) * fft_curr,
                    self._fpl_gain_tensor_1 * fft_curr,
                    axes    = self.v_axes,
                )
            gain_fft_negative: np.ndarray = \
                np.sum(
                    np.stack(
                        [
                            utils.convolve_freqs(
                                self._freqs[..., [i]] * self._freqs[..., [j]] * fft_curr,
                                self._fpl_gain_tensor_2[..., i, j] * fft_curr,
                                axes    = self.v_axes,
                            )
                            for i in range(self._dim) for j in range(self._dim)
                        ], axis=-1
                    ), axis=-1
                )
            return -(gain_fft_positive - gain_fft_negative)
        elif self._dim==1:
            return np.zeros_like(fft_curr, dtype=fft_curr.dtype)


    def compute_collision_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        # NOTE: By a property of the Fourier transform and the convention of FFT libraries, the output should be negated
        # NOTE: The collision term is zero for the 1-dimensional case
        if self._dim in (2, 3):
            loss_fft:  np.ndarray = \
                utils.convolve_freqs(
                    fft_curr,
                    fft_curr * self._fpl_loss_tensor,
                    axes = self.v_axes,
                )
            return -loss_fft
        elif self._dim==1:
            return np.zeros_like(fft_curr, dtype=fft_curr.dtype)


##################################################
##################################################
# End of file