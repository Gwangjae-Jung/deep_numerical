import warnings
from   typing            import Callable, Optional, Sequence

import torch
from   typing_extensions import Self, override

from   deep_numerical                     import EINSUM_STRING, ones, repeat
from   deep_numerical.fft                 import convolve_freqs, freq_tensor
from   deep_numerical.numerical.integrate import integration_guass_legendre
from   deep_numerical.utils               import (
    arg_boundary_inflow,
    arg_specular_velocity,
    get_trange,
    space_grid,
    velocity_grid,
)
from   .constants                         import DEFAULT_QUAD_ORDER_LEBEDEV, LAMBDA, LAMBDA_FPL
from   .runge_kutta                       import one_step_RK4_classic
from   .semi_lagrangian                   import (
    moment_correction_l2,
    one_step_SL_AB3,
    one_step_SL_AB4,
    one_step_SL_Euler,
    one_step_SL_RK3,
    one_step_SL_RK4,
    one_step_SL_Strang,
    semi_lagrangian_transport,
)


__all__: list[str] = [
    'SpectralMethodBase',
    'DirectSM_Base',
    'FastSM_Boltzmann',
    'FastSM_Landau',
]


DEBUG: bool = False


class SpectralMethodBase():
    r"""The base class for the spectral methods for the kinetic equations.
    
    -----
    ### Description
    This class provides essential features which can be used to solve several kinetic equations using the spectral method (to be specific, the Fourier-Galerkin method).
    Although the inhomogeneous kinetic equation can also be solved using spectral methods by considering the transport stage, currently, this is not implemented.  
    """
    def __init__(
            self,
            
            dimension:  int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            freqs_keepdim:  bool = False,
            
            dtype:      Optional[torch.dtype]   = None,
            device:     Optional[torch.device]  = None,
        ) -> None:
        """The initializer of the class `SpectralMethodBase`.
        
        Arguments:
            `dimension` (`int`):
                The dimension of the domain.
            `v_num_grid` (`int`):
                The resolution of the grid in each velocity dimension.
            `v_max` (`float`):
                The maximum coordinate in the velocity space.
            `x_num_grid` (`Optional[int]`, default: `None`):
                The resolution of the grid in each spatial dimeneion.
            `x_max` (`Optional[float]`, default: `None`):
                The maximum coordinate in the spatial space.
            `freqs_keepdim` (`bool`, default: `False`):
                The configuration for the initialization of `self.freqs`.
            `dtype` (`torch.dtype`, default: `None`):
                The data type of the input tensor.
            `device` (`torch.device`, default: `None`):
                The device on which the input tensor is stored.

        -----
        ### Note
        `SpectralMethodBase` and all classes which inherits `SpectralMethodBase` discretizes the velocity space in such a way that both `+v_max` and `-v_max` are *not* included as entries.
        In contrast, they discretizes the space, containing both `x_max` and `-x_max` as entries.
        """
        if dimension not in (2, 3):
            raise NotImplementedError(
                ' '.join(
                    [
                        f"The spectral method for the Boltzmann equation is considered only for dimensions 2 and 3.",
                        f"(dimension={dimension})",
                    ]
                )
            )
        # Dimension
        self._dimension:    int = dimension
        
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
        
        # `torch`-related
        if dtype is None:   dtype   = torch.get_default_dtype()
        if device is None:  device  = torch.get_default_device()
        self._dtype:    torch.dtype     = dtype
        self._device:   torch.device    = device
        
        # Frequency (spectral method)
        self._freqs:    torch.Tensor  = freq_tensor(dimension, v_num_grid, keepdim=freqs_keepdim, dtype=dtype, device=device)
        
        # Interpolation for the inhomogeneous case
        # `self._xv_pad`:   The ordered tuple of interpolating `xv` points
        # `self._xv_query`: The query points at transport stage, saved as a single tensor
        self._boundary_condition:   Optional[str]                       = None
        self._xv_pad:               Optional[tuple[torch.Tensor, ...]]  = None
        self._xv_query:             Optional[torch.Tensor]              = None
        self._arg_bd_inflow:        Optional[torch.Tensor]              = None
        self._arg_bd_inflow_spec:   Optional[torch.Tensor]              = None
        
        # Done
        return
    
    
    # Properties - Dimension
    @property
    def dimension(self) -> int:
        """The dimension of the spatial domain."""
        return self._dimension
    @property
    def f_ndim(self) -> int:
        """The dimension of the input tensor, which is `2*(1+self.dimension)`. Note that
        * the zeroth dimension is used to save multiple instances,
        * the next `self.dimension` dimension(s) is(are) used to refer to the locatioon,
        * the next `self.dimension` dimension(s) is(are) used to refer to the velocity,
        * and the last dimension is used to refer a specific function.
        """
        return 2*(1+self._dimension)
    @property
    def f_shape(self) -> tuple[int, ...]:
        """The shape of the input tensor."""
        n_x: int = 1 if self.is_homogeneous else self._x_num_grid
        return tuple((1, *repeat(n_x, self._dimension), *repeat(self._v_num_grid, self._dimension), -1))
    
    
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
    def v_axes(self) -> tuple[int, ...]:
        """
        The axes of the velocity space in a given distribution function. It is defined as
            `tuple(range(1+self.dimension, 1+2*self.dimension))`.
        """
        return tuple(range(1+self._dimension, 1+2*self._dimension))
    @property
    def v_shape(self) -> tuple[int, ...]:
        """The shape of the velocity space."""
        return tuple(repeat(self._v_num_grid, self._dimension))
    
    
    # Properties - Velocity space (Supplementary)
    @property
    def v_ratio(self) -> float:
        """Returns `pi/self.v_max`, which is the inverse ratio of the extent of a period to the extent of the standard period."""
        return torch.pi/self._v_max
    @property
    def v_radius(self) -> float:
        """The radius of the support of the density function in the velocity space."""
        return self._v_max * LAMBDA
    
    
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
    def x_axes(self) -> tuple[int, ...]:
        """
        The axes of the spatial space in a given distribution function. It is defined as
            `tuple(range(1, 1+self.dimension))`.
        """
        return tuple(range(1, 1+self._dimension))
    @property
    def x_shape(self) -> tuple[int, ...]:
        """The shape of the spatial space."""
        return tuple(repeat(self._x_num_grid, self._dimension))
    
    
    # Properties - `torch`
    @property

    def dtype(self) -> torch.dtype:

        """Real floating-point data type."""
        return self._dtype

    @property


    def dtype_complex(self) -> torch.dtype:


        """Complex floating-point data type matching real dtype precision."""
        default_dtype = torch.get_default_dtype()
        if default_dtype==torch.double:
            return torch.cdouble
        elif default_dtype==torch.float:
            return torch.cfloat
        else:
            raise TypeError(f"Unsupported default dtype: {default_dtype}")

    @property


    def device(self) -> torch.device:


        """Device on which solver tensors reside."""
        return self._device

    @property


    def dtype_and_device(self) -> dict[str, object]:


        """Dictionary of dtype and device keyword arguments."""
        return {'dtype': self._dtype, 'device': self._device}
    
    
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
    def freqs(self) -> torch.Tensor:
        """The tensor of allowed frequencies with `keepdims=False`."""
        return self._freqs

    
    # Properties - Coordinates and Spacing
    @property
    def delta_v(self) -> float:
        """The grid spacing in each velocity dimension."""
        return self._delta_v

    @property
    def delta_x(self) -> Optional[float]:
        """The grid spacing in each spatial dimension."""
        return self._delta_x

    @property
    def v_coords(self) -> torch.Tensor:
        """The 1D coordinate tensor along any velocity axis."""
        return velocity_grid(1, self._v_num_grid, self._v_max, dtype=self._dtype, device=self._device).flatten()

    @property
    def x_coords(self) -> Optional[torch.Tensor]:
        """The 1D coordinate tensor along any spatial axis."""
        if self._is_homogeneous or (self._x_max is None) or (self._x_num_grid is None):
            return None
        return torch.linspace(-self._x_max, self._x_max, self._x_num_grid, dtype=self._dtype, device=self._device)

    def precompute(self, return_: bool = False) -> Optional[torch.Tensor]:
        """(`SpectralMethodBase`) The method `precompute` should be implemented in the derived classes."""
        return None
    
    
    @property
    def supported_boundary_conditions__dict(self) -> dict[str, tuple[str, ...]]:
        return {
            'zero':     tuple(('zero', 'zeros')),
            'periodic': tuple(('periodic')),
        }
    @property
    def supported_boundary_conditions(self) -> set[str]:
        return set(('zero', 'zeros', 'periodic', 'specular',))
    @property
    def arg_bd_inflow(self) -> torch.Tensor:
        return self._arg_bd_inflow
    @property
    def arg_bd_inflow_spec(self) -> torch.Tensor:
        return self._arg_bd_inflow_spec
    
    
    def precompute_query(self, delta_t: float, boundary_condition: str) -> None:
        """(`SpectralMethodBase`) The method `precompute_query` is already implemented in the base class.
        Note that this method *explicitly* requires the time step with which the transport stage is conducted.
        Be sure to pass the time step which is used to conduct the collision stage.
        
        Arguments:
            `delta_t` (`float`):
                The time step with which time marching (collision and transport stages) is conducted.
                
            `boundary_condition` (`str`):
                The boundary condition the kinetic equation to be solved is equipped with.
                Supported boundary conditions and corresponding keywords are given as follows.
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
                    *repeat(
                        torch.linspace(-padded_x_max, padded_x_max, padded_x_num_grid, dtype=self._dtype, device=self._device),
                        self._dimension
                    ),
                    *repeat(
                        velocity_grid(1, self._v_num_grid, self._v_max, dtype=self._dtype, device=self._device).flatten(),
                        self._dimension
                    ),
                )
            )
            # Compute the query points
            self._xv_query = space_grid(
                2*self._dimension,
                (
                    *repeat(self._x_num_grid, self._dimension),
                    *repeat(self._v_num_grid, self._dimension),
                ),
                (
                    *repeat(self._x_max, self._dimension),
                    *repeat(self._v_max, self._dimension),
                ),
                where_closed = (*repeat('both', self._dimension), *repeat('none', self._dimension)),
                dtype   = self._dtype,
                device  = self._device,
            )   # NOTE: Contains `pm(x_max)` but does not contain `pm(v_max)`
            self._xv_query[..., :self._dimension] -= delta_t * self._xv_query[..., self._dimension:]
            if boundary_condition in ("periodic",):
                self._xv_query[..., :self._dimension] = \
                    (self._xv_query[..., :self._dimension] + self._x_max) % (2*self._x_max) - self._x_max
            self._boundary_condition = boundary_condition
            ##### Computation of some other member variables
            self._arg_bd_inflow = arg_boundary_inflow(self._xv_query)
            self._arg_bd_inflow_spec = arg_specular_velocity(self._v_num_grid, self._arg_bd_inflow, dim=self._dimension)
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
        
    
    def compute_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        """(`SpectralMethodBase`) This method computes `gain_fft - loss_fft`, where
        
        * `gain_fft` is the gain term in the Fourier space, computed using the method `compute_gain_fft`, and
        * `loss_fft` is the loss term in the Fourier space, computed using the method `compute_loss_fft`.
        
        -----
        ### Note
        The first arguments in `self.compute_gain_fft` and `self.compute_loss_fft` are given `0.0`, which should be a real number.
        """
        return (
            self.compute_gain_fft(0.0, fft_curr) - \
            self.compute_loss_fft(0.0, fft_curr)
        )
    def compute_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        """(`SpectralMethodBase`) The method `compute_gain_fft` should be implemented in the derived classes."""
        pass
    def compute_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        """(`SpectralMethodBase`) The method `compute_loss_fft` should be implemented in the derived classes."""
        pass
    
    
    def forward(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            f_in:       torch.Tensor,
            delta_t:    float,
            RK_fcn:     Callable[[float, torch.Tensor, float, Callable], torch.Tensor] = one_step_RK4_classic,
        ) -> torch.Tensor:
        """Returns the solution at the next time step.

        -----
        (`SpectralMethodBase`) This method is already implemented.
        """
        if self._is_homogeneous:
            return self._forward_homogeneous(0.0, f_in, delta_t, RK_fcn)
        else:
            return self._forward_inhomogeneous(0.0, f_in, delta_t, RK_fcn)

    def _forward_homogeneous(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            f_fft:      torch.Tensor,
            delta_t:    float,
            RK_fcn:     Callable[[float, torch.Tensor, float, Callable], torch.Tensor],
        ) -> torch.Tensor:
        return RK_fcn(0.0, f_fft, delta_t, self.compute_fft)

    def _forward_inhomogeneous(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            f_in:       torch.Tensor,
            delta_t:    float,
            RK_fcn:     Callable[[float, torch.Tensor, float, Callable], torch.Tensor],
        ) -> torch.Tensor:
        """Inhomogeneous time stepping using conservative semi-Lagrangian transport."""
        is_fourier_input: bool = torch.is_complex(f_in)
        if is_fourier_input:
            f_phys = torch.fft.ifftn(f_in, dim=self.v_axes, norm=self.internal_fft_norm).real
        else:
            f_phys = f_in

        if hasattr(self, 'forward_semi_lagrangian'):
            f_next = getattr(self, 'forward_semi_lagrangian')(f_phys, delta_t)
        else:
            raise NotImplementedError(
                f"Inhomogeneous forward is not supported for {self.__class__.__name__} without semi-Lagrangian transport."
            )

        if is_fourier_input:
            return torch.fft.fftn(f_next, dim=self.v_axes, norm=self.internal_fft_norm)
        return f_next

    def solve(
            self,
            t_init:     float,
            t_final:    float,
            delta_t:    float,
            f_init:     torch.Tensor,
            RK_fcn:     Callable[[float, torch.Tensor, float, Callable], torch.Tensor] = one_step_RK4_classic,
        ) -> torch.Tensor:
        """Returns the numerical solution trajectory.

        Arguments:
            `t_init` (`float`):
                Initial time.
            `t_final` (`float`):
                Final time.
            `delta_t` (`float`):
                Time step size.
            `f_init` (`torch.Tensor`):
                Initial distribution function in physical space.
            `RK_fcn` (`Callable`, default: `one_step_RK4_classic`):
                Time marching algorithm for homogeneous case.

        Returns:
            `torch.Tensor`: The stacked solution trajectory along dimension 1.
        """
        trange = get_trange()
        num_updates = int((t_final - t_init) / delta_t + 0.1)
        ret: list[torch.Tensor] = [f_init]

        if self._is_homogeneous:
            f_fft_curr: torch.Tensor = torch.fft.fftn(f_init, dim=self.v_axes, norm=self.internal_fft_norm)
            for _ in trange(num_updates):
                f_fft_curr = self.forward(0.0, f_fft_curr, delta_t, RK_fcn)
                ret.append(torch.fft.ifftn(f_fft_curr, dim=self.v_axes, norm=self.internal_fft_norm).real)
        else:
            f_curr: torch.Tensor = f_init.clone()
            for _ in trange(num_updates):
                f_curr = self.forward(0.0, f_curr, delta_t, RK_fcn)
                ret.append(f_curr)

        return torch.stack(ret, dim=1)
    
    
    def __repr__(self) -> str:
        return ''.join(
            [
                f"{self.__class__.__name__}(",
                f"dimension={self._dimension}, ",
                f"v_num_grid={self._v_num_grid}, ",
                f"v_max={self._v_max}",
                f"x_num_grid={self._x_num_grid}, ",
                f"x_max={self._x_max}",
                f")",
            ]
        )



class DirectSM_Base(SpectralMethodBase):
    r"""The base class for the direct spectral methods for the homogeneous Boltzmann equation.
    """
    def __init__(
            self,
            
            dimension:  int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int],
            x_max:      Optional[float],
            
            order:      Optional[int]   = None,

            dtype:      Optional[torch.dtype]   = None,
            device:     Optional[torch.device]  = None,
        ) -> None:
        """The initializer of the class `DirectSM_Base`.
        
        -----
        Arguments:
            `dimension` (`int`):
                The dimension of the domain.
            `v_num_grid` (`int`):
                The number of the grid points in each velocity axis.
            `v_max` (`float`):
                The maximum velocity in the velocity space.
            `x_num_grid` (`int`):
                The number of the grid points in each spatial axis.
            `x_max` (`float`):
                The maximum velocity in the velocity space.
            `order` (`Optional[int]`):
                The order of the quadrature rule for the computation of the kernel modes.
            `dtype` (`torch.dtype`, default: `None`):
                The data type of the input tensor.
            `device` (`torch.device`, default: `None`):
                The device on which the input tensor is stored.
        """
        super().__init__(dimension, v_num_grid, v_max, x_num_grid, x_max, dtype=dtype, device=device)
        self._order:    int = order if isinstance(order, int) else v_num_grid
        self._kernel:   Optional[torch.Tensor]    = None
        return
    

    @property
    def order(self) -> int:
        return self._order
    @property
    def kernel(self) -> Optional[torch.Tensor]:
        return self._kernel
    
    
    @override
    def precompute(self) -> None:
        func: Callable[[torch.Tensor, object], torch.Tensor]
        if self._dimension==1:
            func = self._precompute_integrand_1D
        elif self._dimension==2:
            func = self._precompute_integrand_2D
        elif self._dimension==3:
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
                        f"(dimension={self._dimension})",
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
        ).type(self.dtype_complex)
        self._kernel = kernel[..., None]    # Append one extra dimension
        return
    
    
    @property
    def _precompute_integrand_1D(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        pass
    @property
    def _precompute_integrand_2D(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        pass
    @property
    def _precompute_integrand_3D(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        pass
    @property
    def _precompute_integrand_kwargs(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        pass
    
    
    def compute_integral_indices(
            self,
            freq1:  torch.Tensor,
            freq2:  torch.Tensor,
        ) -> torch.Tensor:
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
        idx1 = torch.sum(torch.pow(freq1+freq2, 2), dim=-1)
        idx2 = torch.sum(torch.pow(freq1-freq2, 2), dim=-1)
        return torch.stack((idx1, idx2), dim=-1)
        
    
    def compute_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        # Initialize the dervatives of the Fourier coefficients
        ret = torch.zeros_like(fft_curr, dtype=torch.complex128)
        
        # Compute the derivatives of the Fourier coefficients using multiple loops
        for freq_target in self._freqs.type(torch.long):
            for freq_1 in self._freqs.type(torch.long):
                freq_2 = freq_target - freq_1
                
                # Check if the second frequency is allowed
                if torch.min(freq_2) < self.freq_min or torch.max(freq_2) > self.freq_max:
                    continue
                
                # Determine the indices for the computation of the kernel modes
                if self._dimension <= 2:
                    kcs_12 = self._kernel[tuple(freq_1)][tuple(freq_2)]
                else:
                    _idx_a1 = int(torch.sum((freq_1 + freq_2)**2))
                    _idx_a2 = int(torch.sum((freq_1 - freq_2)**2))
                    kcs_12 = self._kernel[_idx_a1, _idx_a2]
                # Update the returning tensor
                ret[tuple((*repeat(slice(None), 1+self._dimension), *freq_target))] += \
                    fft_curr[tuple((*repeat(slice(None), 1+self._dimension), *freq_1))] * \
                    fft_curr[tuple((*repeat(slice(None), 1+self._dimension), *freq_2))] * \
                    kcs_12
            
        # Return the time-derivative of the Fourier coefficients
        return ret
    
    
    def compute_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   Optional[float],
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        # Initialize the dervatives of the Fourier coefficients
        ret = torch.zeros_like(fft_curr, dtype=torch.complex128)
        
        # Compute the derivatives of the Fourier coefficients using multiple loops
        for freq_target in self._freqs.type(torch.long):
            for freq_1 in self._freqs.type(torch.long):
                freq_2 = freq_target - freq_1
                
                # Check if the second frequency is allowed
                if torch.min(freq_2) < self.freq_min or torch.max(freq_2) > self.freq_max:
                    continue
                
                # Determine the indices for the computation of the kernel modes
                if self._dimension <= 2:
                    kcs_22 = self._kernel[tuple(freq_2)][tuple(freq_2)]
                else:
                    _idx_b1 = int(4 * torch.sum((freq_2)**2))
                    _idx_b2 = 0
                    kcs_22 = self._kernel[_idx_b1, _idx_b2]
                # Update the returning tensor
                ret[tuple((*repeat(slice(None), 1+self._dimension), *freq_target))] += \
                    fft_curr[tuple((*repeat(slice(None), 1+self._dimension), *freq_1))] * \
                    fft_curr[tuple((*repeat(slice(None), 1+self._dimension), *freq_2))] * \
                    kcs_22
            
        # Return the time-derivative of the Fourier coefficients
        return ret
    


class FastSM_Boltzmann(SpectralMethodBase):
    r"""The base class for the fast spectral method for solving the Boltzmann equation.
    
    -----
    ### Description
    This class provides features which can be used to solve the Boltzmann equation using the spectral method (to be specific, the Fourier-Galerkin method).
    The construction of this base class is inspired by the idea of rewriting of the kernel modes:
    This aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Nevertheless, alternative implementation of a fast spectral method can be done by overriding necessary methods.
    
    Reference: ["A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels" by Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu](https://epubs.siam.org/doi/10.1137/16M1096001)
    """
    def __init__(
            self,
            
            dimension:  int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int],
            x_max:      Optional[float],
            
            restitution:    float   = 1.0,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,

            dtype:      Optional[torch.dtype]   = None,
            device:     Optional[torch.device]  = None,
        ) -> None:
        """The initializer of the class `FastSM_Boltzmann`.
        
        Arguments:
            `dimension` (`int`):
                The dimension of the domain.
            `v_num_grid` (`int`):
                The number of the grid points in each velocity axis.
            `v_max` (`float`):
                The maximum velocity in the velocity space.
            `x_num_grid` (`int`):
                The number of the grid points in each spatial axis.
            `x_max` (`float`):
                The maximum velocity in the velocity space.
            `restitution` (`float`, default: `1.0`):
                The restitution coefficient of the collision. This should be in the range of `[0, 1]`.
            `quad_order_uniform` (`Optional[int]`):
                The number of the points required to do integration using uniform quadrature rule.
            `quad_order_legendre` (`Optional[int]`):
                The number of the points required to do integration using Legendre quadrature rule.
            `quad_order_lebedev` (`Optional[int]`):
                The number of the points required to do integration using Lebedev quadrature rule.
            `dtype` (`torch.dtype`, default: `None`):
                The data type of the input tensor.
            `device` (`torch.device`, default: `None`):
                The device on which the input tensor is stored.
        """
        super().__init__(dimension, v_num_grid, v_max, x_num_grid, x_max, dtype=dtype, device=device)
        self._restitution:   float = \
            restitution if isinstance(restitution, float)\
            else 1.0
        self._quad_order_uniform    = \
            quad_order_uniform  if isinstance(quad_order_uniform,   int)\
            else int(1 + LAMBDA * v_num_grid / (2**0.5))
        self._quad_order_legendre   = \
            quad_order_legendre if isinstance(quad_order_legendre,  int)\
            else self._v_num_grid
            # else int(1 + (dimension**0.5) * LAMBDA * num_grid)
            # The above value is the lower bound presented in the paper, which is
            # * about 0.641 for the 2D case
            # * about 0.785 for the 3D case
        self._quad_order_lebedev    = \
            quad_order_lebedev  if isinstance(quad_order_lebedev,   int)\
            else DEFAULT_QUAD_ORDER_LEBEDEV
        self._fsm_scale:    Optional[torch.Tensor] = None
        self._fsm_phase_1:  Optional[torch.Tensor] = None
        self._fsm_phase_2:  Optional[torch.Tensor] = None
        self._kernel_diag:  Optional[torch.Tensor] = None

        # Configuration for inhomogeneous semi-Lagrangian transport
        self._knudsen:                float = 1.0
        self._sl_scheme:              str   = 'rk4'
        self._spatial_reconstruction: str   = 'cweno35'
        self._moment_correction:      bool  = True
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
    def fsm_scale(self) -> Optional[torch.Tensor]:
        return torch.squeeze(self._fsm_scale, dim=tuple(range(1+self._dimension)))
    @property
    def fsm_phase_1(self) -> Optional[torch.Tensor]:
        return torch.squeeze(self._fsm_phase_1, dim=tuple(range(1+self._dimension)))
    @property
    def fsm_phase_2(self) -> Optional[torch.Tensor]:
        return torch.squeeze(self._fsm_phase_2, dim=tuple(range(1+self._dimension)))
    @property
    def kernel_diag(self) -> Optional[torch.Tensor]:
        return torch.squeeze(self._kernel_diag, dim=tuple(range(1+self._dimension)))
    @property
    def knudsen(self) -> float:
        """The Knudsen number epsilon in the scaled Boltzmann equation."""
        return self._knudsen
    @knudsen.setter
    def knudsen(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Knudsen number must be positive. Got {value}")
        self._knudsen = float(value)
        return
    @property
    def sl_scheme(self) -> str:
        """The time integration scheme for semi-Lagrangian transport."""
        return self._sl_scheme
    @sl_scheme.setter
    def sl_scheme(self, value: str) -> None:
        scheme = value.lower()
        valid = ('rk4', 'rk3', 'strang', 'euler', 'ab3', 'ab4')
        if scheme not in valid:
            raise ValueError(f"Unsupported semi-Lagrangian scheme: {value}. Must be one of {valid}")
        self._sl_scheme = scheme
        return
    @property
    def spatial_reconstruction(self) -> str:
        """The spatial reconstruction order for conservative semi-Lagrangian transport."""
        return self._spatial_reconstruction
    @spatial_reconstruction.setter
    def spatial_reconstruction(self, value: str) -> None:
        order = value.lower()
        valid = ('linear', 'cweno23', 'cweno35')
        if order not in valid:
            raise ValueError(f"Unsupported spatial reconstruction: {value}. Must be one of {valid}")
        self._spatial_reconstruction = order
        return
    @property
    def moment_correction(self) -> bool:
        """Whether to enforce L2 conservation of mass, momentum, and energy."""
        return self._moment_correction
    @moment_correction.setter
    def moment_correction(self, value: bool) -> None:
        self._moment_correction = bool(value)
        return

    def _dimension_error(self) -> None:
        raise NotImplementedError(f"This class is supported only for 2D and 3D problems. (dimension: {self._dimension})")
    
    
    @override
    def precompute(self) -> None:
        """Precomputes the gain and loss parts of the kernel modes.
        
        -----
        This method computes the gain and loss parts of the kernel modes.
        Because the base class is mainly inspired by ["A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels" by Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu](https://epubs.siam.org/doi/10.1137/16M1096001), inside the implementation, there are two tensors which define the gain part of the kernel modes.
        """
        self._precompute_fsm_gain()
        self._precompute_fsm_loss()
        return
    def _precompute_fsm_gain(self) -> None:
        pass
    def _precompute_fsm_loss(self) -> None:
        pass
    
    
    def compute_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        terms1 = convolve_freqs(
            fft_curr[..., None, None] * self._fsm_phase_1,
            fft_curr[..., None, None] * self._fsm_phase_2,
            dim = self.v_axes,
        )
        gain_fft = torch.sum(self._fsm_scale*terms1, dim=(-2, -1))
        return gain_fft
    
    
    def compute_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        loss_fft = convolve_freqs(
            fft_curr * self._kernel_diag,
            fft_curr,
            dim = self.v_axes,
        )
        return loss_fft
    
    
    def compute_loss_linear_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        """This method computes the FFT of the coefficient operator in the loss part of the collision operator."""
        loss_linear_fft = fft_curr * self._kernel_diag
        return loss_linear_fft

    def collision_operator_physical(
            self,
            f: torch.Tensor,
        ) -> torch.Tensor:
        """Computes the collision operator Q(f) in physical space.

        ## Description
        Applies FFT along velocity axes, evaluates the fast spectral collision operator
        in Fourier space, and transforms back via IFFT with real projection.

        Arguments:
            `f` (`torch.Tensor`):
                The distribution function in physical space, shaped `(batch, *space, *velocity, 1)`.

        Returns:
            `torch.Tensor`: The collision operator Q(f) in physical space of identical shape.
        """
        f_fft = torch.fft.fftn(f, dim=self.v_axes, norm=self.internal_fft_norm)
        q_fft = self.compute_fft(0.0, f_fft)
        q_phys = torch.fft.ifftn(q_fft, dim=self.v_axes, norm=self.internal_fft_norm).real
        return q_phys

    def forward_semi_lagrangian(
            self,
            f_curr:     torch.Tensor,
            delta_t:    float,
        ) -> torch.Tensor:
        """Advances the inhomogeneous Boltzmann equation by one time step delta_t using conservative semi-Lagrangian method.

        ## Description
        Following Boscarino, Cho, and Russo (2024), this solves the equation:
            \\partial_t f + v \\cdot \\nabla_x f = \\frac{1}{\\varepsilon} Q(f)
        coupled with conservative semi-Lagrangian advection and fast spectral collision.
        If moment_correction is enabled, mass, momentum, and energy are conserved to machine precision.

        Arguments:
            `f_curr` (`torch.Tensor`):
                The current distribution function in physical space, shaped `(batch, *space, *velocity, 1)`.
            `delta_t` (`float`):
                The time step size.

        Returns:
            `torch.Tensor`: The updated distribution function at t + delta_t.
        """
        v_coords = self.v_coords
        knudsen  = self._knudsen
        order    = self._spatial_reconstruction

        def collision_fn(f_state: torch.Tensor) -> torch.Tensor:
            q_raw = self.collision_operator_physical(f_state)
            if self._moment_correction:
                q_raw = moment_correction_l2(q_raw, v_coords, self._delta_v)
            return q_raw

        scheme = self._sl_scheme
        if scheme == 'rk4':
            f_next = one_step_SL_RK4(
                f_curr       = f_curr,
                delta_t      = delta_t,
                collision_fn = collision_fn,
                v_coords     = v_coords,
                x_max        = self._x_max,
                knudsen      = knudsen,
                order        = order,
            )
        elif scheme == 'rk3':
            f_next = one_step_SL_RK3(
                f_curr       = f_curr,
                delta_t      = delta_t,
                collision_fn = collision_fn,
                v_coords     = v_coords,
                x_max        = self._x_max,
                knudsen      = knudsen,
                order        = order,
            )
        elif scheme == 'strang':
            f_next = one_step_SL_Strang(
                f_curr       = f_curr,
                delta_t      = delta_t,
                collision_fn = collision_fn,
                v_coords     = v_coords,
                x_max        = self._x_max,
                knudsen      = knudsen,
                order        = order,
            )
        elif scheme == 'euler':
            f_next = one_step_SL_Euler(
                f_curr       = f_curr,
                delta_t      = delta_t,
                collision_fn = collision_fn,
                v_coords     = v_coords,
                x_max        = self._x_max,
                knudsen      = knudsen,
                order        = order,
            )
        else:
            raise ValueError(f"Unknown semi-Lagrangian scheme: {scheme}")

        return f_next
    
    def to(self, device: torch.device) -> Self:
        if self._fsm_scale is not None:
            self._fsm_scale = self._fsm_scale.to(device)
        if self._fsm_phase_1 is not None:
            self._fsm_phase_1 = self._fsm_phase_1.to(device)
        if self._fsm_phase_2 is not None:
            self._fsm_phase_2 = self._fsm_phase_2.to(device)
        if self._kernel_diag is not None:
            self._kernel_diag = self._kernel_diag.to(device)
        self._freqs  = self._freqs.to(device)
        self._device = device
        return self
    


class FastSM_Landau(SpectralMethodBase):
    """The class for the fast spectral method for solving the Fokker-Planck-Landau equation with the VHS model.
    
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
            
            dimension:  int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            
            dtype:  Optional[torch.dtype]   = None,
            device: Optional[torch.device]  = None,
        ) -> None:
        """The initializer of the class `FastSM_Boltzmann`.
        
        Arguments:
            `dimension` (`int`):
                The dimension of the domain.
            `v_num_grid` (`int`):
                The number of the grid points in each velocity axis.
            `v_max` (`float`):
                The maximum velocity in the velocity space.
            `x_num_grid` (`int`):
                The number of the grid points in each spatial axis.
            `x_max` (`float`):
                The maximum velocity in the velocity space.
            `quad_order_uniform` (`Optional[int]`):
                The number of the points required to do integration using uniform quadrature rule.
            `quad_order_legendre` (`Optional[int]`):
                The number of the points required to do integration using Legendre quadrature rule.
            `quad_order_lebedev` (`Optional[int]`):
                The number of the points required to do integration using Lebedev quadrature rule.
            `dtype` (`torch.dtype`, default: `None`):
                The data type of the input tensor.
            `device` (`torch.device`, default: `None`):
                The device on which the input tensor is stored.
        """
        super().__init__(
            dimension       = dimension,
            v_num_grid      = v_num_grid,
            v_max           = v_max,
            x_num_grid      = x_num_grid,
            x_max           = x_max,
            freqs_keepdim   = True,
            dtype           = dtype,
            device          = device,
        )   # NOTE Set `freqs_keepdims=True` for the vectorized implementation
        self._quad_order_uniform:   int = quad_order_uniform    \
            if isinstance(quad_order_uniform,   int) else v_num_grid
        self._quad_order_legendre:  int = quad_order_legendre   \
            if isinstance(quad_order_legendre,  int) else v_num_grid
        
        # Add the batch dimension for ease of implementation
        self._freqs:        torch.Tensor = torch.tensor([], **self.dtype_and_device)
        self._freq_norms:   torch.Tensor = torch.tensor([], **self.dtype_and_device)
        self.__redefine_freqs()
        
        # Character functions and tensors for the gain and loss terms
        ## NOTE: The character tensors should be 1-dimensional tensors.
        ## NOTE: The gain/loss tensors should be multi-dimensional tensors of shape.
        ## Shapes of the tensors
        ## With `base_shape` defined as the property `__fpl_base_shape`
        ## * `self._fpl_gain_tensor_1`: `base_shape`
        ## * `self._fpl_gain_tensor_2`: `(*base_shape, *repeat(self._dimension, 2))`
        ## * `self._fpl_loss_tensor`:   `base_shape`
        self._fpl_character_1:      Optional[torch.Tensor] = None
        self._fpl_character_2:      Optional[torch.Tensor] = None
        self._fpl_gain_tensor_1:    Optional[torch.Tensor] = None
        self._fpl_gain_tensor_2:    Optional[torch.Tensor] = None
        self._fpl_loss_tensor:      Optional[torch.Tensor] = None
        
        return
    
    
    @override
    @property
    def v_radius(self) -> float:
        """The radius of the support of the density function in the velocity space.
        
        -----
        (`FastSM_FPL_VHS`) Redefined."""
        return self._v_max * LAMBDA_FPL
    @override
    @property
    def freqs(self) -> torch.Tensor:
        """The frequency tensor for the spectral method.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), self.dim)`,
        dimensions aligned as in the tensors for the distribution functions.
        
        -----
        (`FastSM_FPL_VHS`) Redefined."""
        return self._freqs
    @override
    @property
    def freq_norms(self) -> torch.Tensor:
        """The norms of the frequency tensor for the spectral method.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`,
        dimensions aligned as in the tensors for the distribution functions.
        
        -----
        (`FastSM_FPL_VHS`) Redefined."""
        return self._freq_norms
    @property
    def fpl_character_1(self) -> Optional[torch.Tensor]:
        """Returns the 1-tensor of the first characteristic function.
        ### Note
        The entry of index `k` (`k>0`) corresponds to the value of the function at `sqrt(k)`.
        In other words, the input index `k` is the square of a frequency mode.
        The shape of this property is `(1+self.dim*(self.num_grid**2),)`.
        """
        return self._fpl_character_1
    @property
    def fpl_character_2(self) -> Optional[torch.Tensor]:
        """Returns the 1-tensor of the second characteristic function.
        ### Note
        The entry of index `k` (`k>0`) corresponds to the value of the function at `sqrt(k)`.
        In other words, the input index `k` is the square of a frequency mode.
        The shape of this property is `(1+self.dim*(self.num_grid**2),)`.
        """
        return self._fpl_character_2
    @property
    def fpl_base_shape(self) -> tuple[int, ...]:
        """Returns the base shape of the tensors for the gain and loss terms, which is defined as
            `tuple((1, *ones(self._dimension), *repeat(self._v_num_grid, self._dimension), 1))`.
        Note that the shape of `self.fpl_gain_tensor_2` is `(*self.__fpl_base_shape, *repeat(self.dim, 2))`."""
        return tuple((1, *ones(self._dimension), *repeat(self._v_num_grid, self._dimension), 1))
    @property
    def fpl_gain_tensor_1(self) -> Optional[torch.Tensor]:
        """Returns the tensor of the positive scaling part of the gain term.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`,
        dimensions aligned as in the tensors for the distribution functions.
        """
        return self._fpl_gain_tensor_1
    @property
    def fpl_gain_tensor_2(self) -> Optional[torch.Tensor]:
        """Returns the tensor of the negative bilinear part of the gain term.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1, *repeat(self.dim, 2))`,
        dimensions aligned as in the tensors for the distribution functions.
        """
        return self._fpl_gain_tensor_2
    @property
    def fpl_loss_tensor(self) -> Optional[torch.Tensor]:
        """Returns the tensor of the loss term.
        ### Note
        This tensor is of shape
            `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`,
        dimensions aligned as in the tensors for the distribution functions.
        """
        return self._fpl_loss_tensor
    
    
    def __redefine_freqs(self) -> None:
        freqs = freq_tensor(
            self._dimension, self._v_num_grid, keepdim=True,
            dtype = torch.long, device=self._device     # This tensor is an integer-valued tensor
        )
        freqs = torch.reshape(
            freqs,
            (
                1,
                *repeat(1, self._dimension),
                *repeat(self._v_num_grid, self._dimension),
                self._dimension,
            )
        )
        freq_norms  = torch.norm(freqs.type(self._dtype), p=2, dim=-1, keepdim=True)
        self._freqs         = freqs
        self._freq_norms    = freq_norms
        return
    
    
    @override
    def precompute(self) -> None:
        if self._dimension in (2, 3):
            getattr(self, f"_precompute_fpl_character_1_{ self._dimension}D")()
            getattr(self, f"_precompute_fpl_character_2_{ self._dimension}D")()
            getattr(self, f"_precompute_fpl_gain_tensors_{self._dimension}D")()
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
        1. `P`: A `(1+self.dim+self.dim+1)`-tensor of shape `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), 1)`
        2. `Q` is a `(1+self.dim+self.dim+2)`-tensor of shape `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), *(self.dim, self.dim))`.
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
        _EINSUM_XV = EINSUM_STRING[:2*self._dimension]
        _freq_norms_sq: torch.Tensor  = torch.sum(self._freqs**2, dim=-1, keepdim=True).type(self._dtype)
        positive: torch.Tensor = _freq_norms_sq * self._fpl_gain_tensor_1
        negative: torch.Tensor = torch.einsum(
            f"b{_EINSUM_XV}i,b{_EINSUM_XV}vij,b{_EINSUM_XV}j->b{_EINSUM_XV}v",
            self._freqs.type(self._dtype), self._fpl_gain_tensor_2, self._freqs.type(self._dtype)
        )
        self._fpl_loss_tensor = positive - negative
        return
    
    
    def compute_gain_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        # NOTE: By a property of the Fourier transform and the convention of FFT libraries, the output should be negated
        # NOTE: The collision term is zero for the 1-dimensional case
        if self._dimension in (2, 3):   
            gain_fft_positive = convolve_freqs(
                (self._freq_norms**2) * fft_curr,
                self._fpl_gain_tensor_1 * fft_curr,
                dim    = self.v_axes,
            )
            gain_fft_negative: torch.Tensor = \
                torch.sum(
                    torch.stack(
                        [
                            convolve_freqs(
                                self._freqs[..., [i]] * self._freqs[..., [j]] * fft_curr,
                                self._fpl_gain_tensor_2[..., i, j] * fft_curr,
                                dim    = self.v_axes,
                            )
                            for i in range(self._dimension) for j in range(self._dimension)
                        ], dim=-1
                    ), dim=-1
                )
            return -(gain_fft_positive - gain_fft_negative)
        elif self._dimension==1:
            return torch.zeros_like(fft_curr, dtype=fft_curr.dtype, device=fft_curr.device)
    
    
    def compute_loss_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        # NOTE: Be careful to implement the truncated 'linear' convolution.     
        # NOTE: The collision term is zero for the 1-dimensional case
        if self._dimension in (2, 3):
            loss_fft = convolve_freqs(
                fft_curr,
                fft_curr * self._fpl_loss_tensor,
                dim = self.v_axes,
            )
            # NOTE: The negation is due to the differentiation of the Fourier transform of translated functions.
            return -loss_fft
        elif self._dimension==1:
            return torch.zeros_like(fft_curr, dtype=fft_curr.dtype, device=fft_curr.device)

    
    def compute_loss_linear_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:   torch.Tensor,
        ) -> torch.Tensor:
        # NOTE: Be careful to implement the truncated 'linear' convolution.     
        # NOTE: The collision term is zero for the 1-dimensional case
        if self._dimension in (2, 3):
            loss_linear_fft = fft_curr * self._fpl_loss_tensor
            # NOTE: The negation is due to the differentiation of the Fourier transform of translated functions.
            return -loss_linear_fft
        elif self._dimension==1:
            return torch.zeros_like(fft_curr, dtype=fft_curr.dtype, device=fft_curr.device)
    
    
    def to(self, device: torch.device) -> Self:
        self._freqs      = self._freqs.to(device)
        self._freq_norms = self._freq_norms.to(device)
        if self._fpl_character_1 is not None:
            self._fpl_character_1   = self._fpl_character_1.to(device)
        if self._fpl_character_2 is not None:
            self._fpl_character_2   = self._fpl_character_2.to(device)
        if self._fpl_gain_tensor_1 is not None:
            self._fpl_gain_tensor_1 = self._fpl_gain_tensor_1.to(device)
        if self._fpl_gain_tensor_2 is not None:
            self._fpl_gain_tensor_2 = self._fpl_gain_tensor_2.to(device)
        if self._fpl_loss_tensor is not None:
            self._fpl_loss_tensor   = self._fpl_loss_tensor.to(device)
        self._device = device
        return self


##################################################
def main() -> None:
    return


if __name__ == '__main__':
    main()