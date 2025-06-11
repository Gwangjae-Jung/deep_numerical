r"""## The implementation of the fast spectral method for solving the homogeneous Boltzmann equation

-----
### Description
Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.

Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
"""
from    typing              import  Optional
from    typing_extensions   import  Self, override

import  torch
from    torch.special       import  bessel_j0   as  j0

from    .base_classes       import  FastSM_Boltzmann
from    ...                 import  utils
from    ._kernel_modes.boltzmann_VHS    import  Boltzmann_VHS_kernel_modes


##################################################
##################################################
__all__ = ['FastSM_Boltzmann_VHS', 'FastSM_Boltzmann_VSS']
   
    
##################################################
##################################################
class FastSM_Boltzmann_VHS(FastSM_Boltzmann):
    r"""## The class for the fast spectral method for solving the homogeneous Boltzmann equation with the VHS model.
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
    """
    def __init__(
            self,
            
            dimension:  int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            vhs_coeff:  Optional[float] = None,
            vhs_alpha:  Optional[float] = None,
            
            restitution:    float   = 1.0,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,
            
            dtype:      torch.device    = torch.float64,
            device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
        ) -> Self:
        super().__init__(
            dimension   = dimension,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            x_num_grid  = x_num_grid,
            x_max       = x_max,
            restitution = restitution,
            quad_order_uniform  = quad_order_uniform,
            quad_order_legendre = quad_order_legendre,
            quad_order_lebedev  = quad_order_lebedev,
            dtype   = dtype,
            device  = device,
        )
        if vhs_coeff is None or vhs_alpha is None:
            raise ValueError(f"'vhs_coeff' and 'vhs_alpha' should be given. ({vhs_coeff=:.2e}, {vhs_alpha=:.2f})")     
        self._vhs_coeff     = vhs_coeff
        self._vhs_alpha     = vhs_alpha
        self.precompute()
        return
    
    
    @property
    def vhs_coeff(self) -> float:
        return self._vhs_coeff
    @property
    def vhs_alpha(self) -> float:
        return self._vhs_alpha


    # Override reshaping properties for precomputation
    @override
    @property
    def _freqs_reshape(self) -> tuple[int]:
        """Reshape the frequency tensor to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*repeat(resolution, dimension)`
        - Data dimension: `dimension`
        - Integral dimension: `*ones(2)`
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.repeat(self._v_num_grid, self._dimension),
            self._dimension,
            *utils.ones(2)
        )
    @override
    @property
    def _r_roots_reshape(self) -> tuple[int]:
        """Reshape the tensor of the Legendre quadrature points to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(quad_order_legendre, 1)`
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(self._quad_order_legendre, 1)
        )
    @override
    @property
    def _r_weights_reshape(self) -> tuple[int]:
        """Reshape the tensor of the Legendre quadrature weights to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(quad_order_legendre, 1)`
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(self._quad_order_legendre, 1)
        )
    @override
    @property
    def _s_roots_reshape(self) -> tuple[int]:
        """Reshape the tensor of the quadrature points on $S^{d-1}$ to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `dimension`
        - Integral dimension: `(1, -1)` (which is `(1, quad_order_uniform)` for `dimension==2`, and `(1, quad_order_lebedev)` for `dimension==3`)
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            self._dimension,
            *(1, -1)
        )
    @override
    @property
    def _s_weights_reshape(self) -> tuple[int]:
        """Reshape the tensor of the quadrature weights on $S^{d-1}$ to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(1, -1)` (which is `(1, quad_order_uniform)` for `dimension==2`, and `(1, quad_order_lebedev)` for `dimension==3`)
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(1, -1)
        )
    
    
    # Override precomputation methods
    @override
    def _precompute_fsm_gain(self) -> None:
        fsm_scale, fsm_phase_1, fsm_phase_2 = getattr(self, f'_precompute_fsm_gain_{self._dimension}D')()
        self._fsm_scale    = fsm_scale
        self._fsm_phase_1  = fsm_phase_1
        self._fsm_phase_2  = fsm_phase_2
        return
    @override
    def _precompute_fsm_loss(self) -> None:
        self._kernel_diag = Boltzmann_VHS_kernel_modes(
            dimension       = self._dimension,
            num_grid        = self._v_num_grid,
            v_max           = self._v_max,
            vhs_coeff       = self._vhs_coeff,
            vhs_alpha       = self._vhs_alpha,
            num_roots       = self._quad_order_legendre,
            diagonal_only   = True,
            dtype           = self._dtype,
            device          = self._device,
        )[*utils.repeat(None, 1+self._dimension), ..., None]
        # Append one extra dimension
        return
    
    
    def _precompute_fsm_gain_2D(self) -> tuple[torch.Tensor]:
        """
        ### Note
        Here, we order data in the following order:
            `(batch, space, velocity, data, *quadrature_rules)`
        """
        """
        Commnets
        
        1. The tensor of the points sampled uniformly on $S^1$ is initially saved in a tensor of shape `(quad_order_uniform, dim)`. As this tensor has to be reshaped to a tensor of shape `(1, *ones(dim), *ones(dim), dim, *(1, quad_order_uniform))`, transposition is required.
        """
        # Get the frequency tensor
        freqs = utils.freq_tensor(self._dimension, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.reshape(self._freqs_reshape)

        # Get the reshaped arrays of the values of `radius, circle`
        r_roots, r_weights  = utils.roots_legendre_shifted(self._quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights  = utils.roots_circle(self.quad_order_uniform, dtype=self._dtype, device=self._device)
        s_roots = s_roots.T     # See `Comments 1`.
        r_roots:   torch.Tensor = r_roots.reshape(self._r_roots_reshape)
        s_roots:   torch.Tensor = s_roots.reshape(self._s_roots_reshape)
        r_weights: torch.Tensor = r_weights.reshape(self._r_weights_reshape)
        s_weights: torch.Tensor = s_weights.reshape(self._s_weights_reshape)

        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_scale_coeff    = (2*torch.pi) * self.vhs_coeff
        _fsm_scale_fcn      = \
            torch.pow(r_roots, self.vhs_alpha+self._dimension-1) * \
            j0(
                (0.5 * self.v_ratio) * \
                r_roots * \
                torch.sqrt((freqs**2).sum(-3, keepdim=True))
            )
        fsm_scale = _fsm_scale_coeff * _fsm_scale_fcn * r_weights
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        _fsm_arg = \
            (0.5 * self.v_ratio) * \
            (freqs * r_roots * s_roots).sum(-3, keepdim=True)
        _coeff_1 = (1+self._restitution)/2
        _coeff_2 = (3-self._restitution)/2
        _phase_weight = torch.sqrt(s_weights)
        fsm_phase_1 = torch.exp(+1j * _fsm_arg * _coeff_1) * _phase_weight
        fsm_phase_2 = torch.exp(-1j * _fsm_arg * _coeff_2) * _phase_weight
               
        # Return the output tensors
        return (fsm_scale, fsm_phase_1, fsm_phase_2)
    
    
    def _precompute_fsm_gain_3D(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Here, we order data in the following order:
            `(batch, space, velocity, data, *quadrature_rules)`
        """
        """
        Comments
        
        1. The tensor of the Lebedev quarature points on $S^2$ is initially saved in a tensor of shape `(-1, dim)`, so it should be transposed to a tensor of shape `(dim, -1)`.
        """
        # Get the frequency tensor
        freqs = utils.freq_tensor(self._dimension, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.reshape(self._freqs_reshape)

        # Get the reshaped arrays of the values of `radius, phi, theta`
        r_roots, r_weights  = utils.roots_legendre_shifted(self.quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights  = utils.roots_lebedev(self.quad_order_lebedev, dtype=self._dtype, device=self._device) # See `Comments 1`.
        s_roots = s_roots.transpose(0,1)
        r_roots:   torch.Tensor = r_roots.reshape(self._r_roots_reshape)
        s_roots:   torch.Tensor = s_roots.reshape(self._s_roots_reshape)
        r_weights: torch.Tensor = r_weights.reshape(self._r_weights_reshape)
        s_weights: torch.Tensor = s_weights.reshape(self._s_weights_reshape)

        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_scale_coeff    = (4*torch.pi) * self.vhs_coeff
        _fsm_scale_fcn      = \
            torch.pow(r_roots, self.vhs_alpha+self._dimension-1) * \
            utils.sinc(
                (0.5 * self.v_ratio) * \
                r_roots * \
                torch.norm(freqs, p=2, dim=-3, keepdim=True)
            )
        fsm_scale = _fsm_scale_coeff * _fsm_scale_fcn * r_weights
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        ## NOTE: Some weights in the Lebedev rule might be negative.
        _fsm_arg = \
            (0.5 * self.v_ratio) * \
            (freqs * r_roots * s_roots).sum(-3, keepdim=True)
        _scale_1 = (1+self._restitution)/2
        _scale_2 = (3-self._restitution)/2
        _phase_weight = torch.sqrt(s_weights.type(self.dtype_complex))
        fsm_phase_1 = torch.exp(+1j * _fsm_arg * _scale_1) * _phase_weight
        fsm_phase_2 = torch.exp(-1j * _fsm_arg * _scale_2) * _phase_weight
                
        # Reshape and return the result
        return (fsm_scale, fsm_phase_1, fsm_phase_2)


    # Override miscellaneous methods
    @override
    @property
    def approximation_level(self) -> int:
        """Return the approximation level of the fast spectral method."""
        return int(self._fsm_phase_1.size(-2) * self._fsm_phase_1.size(-1))
    
    
##################################################
##################################################
class FastSM_Boltzmann_VSS(FastSM_Boltzmann):
    r"""## The class for the fast spectral method for solving the homogeneous Boltzmann equation with the VSS model.
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
    """
    def __init__(
            self,
            
            dimension:  int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            vss_coeff:      Optional[float] = None,
            vss_exp_speed:  Optional[float] = None,
            vss_exp_angle:  Optional[float] = None,
            
            restitution:    float   = 1.0,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,
            
            dtype:      torch.device    = torch.float64,
            device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
        ) -> Self:
        super().__init__(
            dimension   = dimension,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            x_num_grid  = x_num_grid,
            x_max       = x_max,
            restitution = restitution,
            quad_order_uniform  = quad_order_uniform,
            quad_order_legendre = quad_order_legendre,
            quad_order_lebedev  = quad_order_lebedev,
            dtype   = dtype,
            device  = device,
        )
        if vss_coeff is None or vss_exp_speed is None or vss_exp_angle is None:
            raise ValueError(f"'vss_coeff', 'vss_exp_speed', and 'vss_exp_angle' should be given. ({vss_coeff=:.2e}, {vss_exp_speed=:.2f}, {vss_exp_angle=:.2f})")
        self._vss_coeff     = vss_coeff
        self._vss_exp_speed = vss_exp_speed
        self._vss_exp_angle = vss_exp_angle
        self.precompute()
        return
    
    
    @property
    def vss_coeff(self) -> float:
        return self._vss_coeff
    @property
    def vss_exp_speed(self) -> float:
        return self._vss_exp_speed
    @property
    def vss_exp_angle(self) -> float:
        return self._vss_exp_angle
    
    
    # Override reshaping properties for precomputation
    # Reshaping properties for precomputation
    @override
    @property
    def _freqs_reshape(self) -> tuple[int]:
        """Reshape the frequency tensor to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*repeat(resolution, dimension)`
        - Data dimension: `dimension`
        - Integral dimension: `*ones(3)`
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.repeat(self._v_num_grid, self._dimension),
            self._dimension,
            *utils.ones(3)
        )
    @override
    @property
    def _r_roots_reshape(self) -> tuple[int]:
        """Reshape the tensor of the Legendre quadrature points to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(quad_order_legendre, 1, 1)`
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(self._quad_order_legendre, 1, 1)
        )
    @override
    @property
    def _r_weights_reshape(self) -> tuple[int]:
        """Reshape the tensor of the Legendre quadrature weights to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(quad_order_legendre, 1, 1)`
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(self._quad_order_legendre, 1, 1)
        )
    @override
    @property
    def _s_roots_reshape(self) -> tuple[int]:
        """Reshape the tensor of the quadrature points on $S^{d-1}$ to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `dimension`
        - Integral dimension: `(1, -1, 1)` (which is `(1, quad_order_uniform, 1)` for `dimension==2`, and `(1, quad_order_lebedev, 1)` for `dimension==3`)
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            self._dimension,
            *(1, -1, 1)
        )
    @override
    @property
    def _s_weights_reshape(self) -> tuple[int]:
        """Reshape the tensor of the quadrature weights on $S^{d-1}$ to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(1, -1, 1)` (which is `(1, quad_order_uniform, 1)` for `dimension==2`, and `(1, quad_order_lebedev, 1)` for `dimension==3`)
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(1, -1, 1)
        )
    @override
    @property
    def _g_roots_reshape(self) -> tuple[int]:
        """Reshape the tensor of the quadrature points on $S^{d-1}$ to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `dimension`
        - Integral dimension: `(1, 1, -1)` (which is `(1, 1, quad_order_uniform)` for `dimension==2`, and `(1, 1, quad_order_lebedev)` for `dimension==3`)
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            self._dimension,
            *(1, 1, -1)
        )
    @override
    @property
    def _g_weights_reshape(self) -> tuple[int]:
        """Reshape the tensor of the quadrature weights on $S^{d-1}$ to the following shape:
        - Batch dimension: `1`
        - Space dimension: `*ones(dimension)`
        - Velocity dimension: `*ones(dimension)`
        - Data dimension: `1`
        - Integral dimension: `(1, 1, -1)` (which is `(1, 1, quad_order_uniform)` for `dimension==2`, and `(1, 1, quad_order_lebedev)` for `dimension==3`)
        """
        return (
            1,
            *utils.ones(self._dimension),
            *utils.ones(self._dimension),
            1,
            *(1, 1, -1)
        )
    
    
    # Override precomputation methods
    @override
    def _precompute_fsm_gain(self) -> None:
        fsm_scale, fsm_phase_1, fsm_phase_2 = getattr(self, f'_precompute_fsm_gain_{self._dimension}D')()
        self._fsm_scale    = fsm_scale
        self._fsm_phase_1  = fsm_phase_1
        self._fsm_phase_2  = fsm_phase_2
        return
    @override
    def _precompute_fsm_loss(self) -> None:
        from    scipy.special   import  gamma
        from    ...utils        import  area_of_unit_sphere
        
        coeff_kernel_modes: float   = self._vss_coeff / area_of_unit_sphere(self._dimension)
        # Division by the area of the unit sphere in the velocity space should be done
        # as `Boltzmann_VHS_kernel_modes` multiplies the output by this area
        _scale: float
        if self._dimension==2:
            _scale = 2**(self._vss_exp_angle+1) * (torch.pi**0.5) * (
                gamma(self._vss_exp_angle+0.5) / \
                gamma(self._vss_exp_angle+1.0)
            )
        elif self._dimension==3:
            _scale = 2**(self._vss_exp_angle+2) * torch.pi / (
                self._vss_exp_angle+1
            )
        coeff_kernel_modes *= _scale
        
        self._kernel_diag = Boltzmann_VHS_kernel_modes(
            dimension       = self._dimension,
            num_grid        = self._v_num_grid,
            v_max           = self._v_max,
            vhs_coeff       = coeff_kernel_modes,
            vhs_alpha       = self._vss_exp_speed,
            num_roots       = self._quad_order_legendre,
            diagonal_only   = True,
            dtype           = self._dtype,
            device          = self._device,
        )[*utils.repeat(None, 1+self._dimension), ..., None]
        # Append one extra dimension
        
        return
    
    
    def _precompute_fsm_gain_2D(self) -> tuple[torch.Tensor]:
        """
        ### Note
        Here, we order data in the following order:
            `(batch, space, velocity, data, *quadrature_rules)`
        """
        """
        Commnets
        
        1. The tensor of the points sampled uniformly on $S^1$ is initially saved in a tensor of shape `(quad_order_uniform, dim)`. As this tensor has to be reshaped to a tensor of shape `(1, *ones(dim), *ones(dim), dim, *(1, quad_order_uniform))`, transposition is required.
        """
        # Get the frequency tensor
        freqs = utils.freq_tensor(self._dimension, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.reshape(self._freqs_reshape)

        # Get the reshaped arrays of the values of `radius, circle`
        r_roots, r_weights  = utils.roots_legendre_shifted(self._quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights  = utils.roots_circle(self.quad_order_uniform, dtype=self._dtype, device=self._device)
        g_roots, g_weights  = utils.roots_circle(self.quad_order_uniform, dtype=self._dtype, device=self._device)
        s_roots = s_roots.T     # See `Comments 1`.
        g_roots = g_roots.T     # See `Comments 1`.
        r_roots:   torch.Tensor = r_roots.reshape(self._r_roots_reshape)
        s_roots:   torch.Tensor = s_roots.reshape(self._s_roots_reshape)
        g_roots:   torch.Tensor = g_roots.reshape(self._g_roots_reshape)
        r_weights: torch.Tensor = r_weights.reshape(self._r_weights_reshape)
        s_weights: torch.Tensor = s_weights.reshape(self._s_weights_reshape)
        g_weights: torch.Tensor = g_weights.reshape(self._g_weights_reshape)
        
        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_power      = \
            self.vss_coeff * \
            torch.pow(r_roots, self.vss_exp_speed+self._dimension-1)
        _fsm_integral   = \
            torch.sum(
                torch.pow(
                    1 + torch.sum(s_roots*g_roots, dim=-4, keepdim=True),
                    self.vss_exp_angle
                ) * \
                torch.exp(
                    -0.5j * self.v_ratio * r_roots * \
                    torch.sum(freqs*g_roots, dim=-4, keepdim=True)
                ) * \
                g_weights,
                dim=-1, keepdim=True
            )
        fsm_scale = _fsm_power * _fsm_integral * r_weights
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        _fsm_arg = \
            (0.5 * self.v_ratio) * \
            (freqs * r_roots * s_roots).sum(-4, keepdim=True)
        _coeff_1 = (1+self._restitution)/2
        _coeff_2 = (3-self._restitution)/2
        _phase_weight = torch.sqrt(s_weights)
        fsm_phase_1 = torch.exp(+1j * _fsm_arg * _coeff_1) * _phase_weight
        fsm_phase_2 = torch.exp(-1j * _fsm_arg * _coeff_2) * _phase_weight
               
        # Return the output tensors
        return map(lambda tensor: torch.squeeze(tensor, dim=-1), (fsm_scale, fsm_phase_1, fsm_phase_2))
    
    
    def _precompute_fsm_gain_3D(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Here, we order data in the following order:
            `(batch, space, velocity, data, *quadrature_rules)`
        """
        """
        Comments
        
        1. The tensor of the Lebedev quarature points on $S^2$ is initially saved in a tensor of shape `(-1, dim)`, so it should be transposed to a tensor of shape `(dim, -1)`.
        """
        # Get the frequency tensor
        freqs = utils.freq_tensor(self._dimension, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.reshape(self._freqs_reshape)

        # Get the reshaped arrays of the values of `radius, phi, theta`
        r_roots, r_weights  = utils.roots_legendre_shifted(self.quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights  = utils.roots_lebedev(self.quad_order_lebedev, dtype=self._dtype, device=self._device) # See `Comments 1`.
        g_roots, g_weights  = utils.roots_lebedev(self.quad_order_lebedev, dtype=self._dtype, device=self._device) # See `Comments 1`.
        s_roots = s_roots.transpose(0,1)
        g_roots = g_roots.transpose(0,1)
        r_roots:   torch.Tensor = r_roots.reshape(self._r_roots_reshape)
        s_roots:   torch.Tensor = s_roots.reshape(self._s_roots_reshape)
        g_roots:   torch.Tensor = g_roots.reshape(self._g_roots_reshape)
        r_weights: torch.Tensor = r_weights.reshape(self._r_weights_reshape)
        s_weights: torch.Tensor = s_weights.reshape(self._s_weights_reshape)
        g_weights: torch.Tensor = g_weights.reshape(self._g_weights_reshape)

        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_power      = \
            self.vss_coeff * \
            torch.pow(r_roots, self.vss_exp_speed+self._dimension-1)
        _fsm_integral   = \
            torch.sum(
                torch.pow(
                    1 + torch.sum(s_roots*g_roots, dim=-4, keepdim=True),
                    self.vss_exp_angle
                ) * \
                torch.exp(
                    -0.5j * self.v_ratio * r_roots * \
                    torch.sum(freqs*g_roots, dim=-4, keepdim=True)
                ) * \
                g_weights,
                dim=-1, keepdim=True
            )
        fsm_scale = _fsm_power * _fsm_integral * r_weights
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        ## NOTE: Some weights in the Lebedev rule might be negative.
        _fsm_arg = \
            (0.5 * self.v_ratio) * \
            (freqs * r_roots * s_roots).sum(-4, keepdim=True)
        _scale_1 = (1+self._restitution)/2
        _scale_2 = (3-self._restitution)/2
        _phase_weight = torch.sqrt(s_weights.type(self.dtype_complex))
        fsm_phase_1 = torch.exp(+1j * _fsm_arg * _scale_1) * _phase_weight
        fsm_phase_2 = torch.exp(-1j * _fsm_arg * _scale_2) * _phase_weight
                
        # Reshape and return the result
        return map(lambda tensor: torch.squeeze(tensor, dim=-1), (fsm_scale, fsm_phase_1, fsm_phase_2))

    
    def reset_VHS_parameters(self, new_vhs_coeff: float, new_vhs_alpha: float) -> None:
        """Set a new value for the exponent of the VHS model."""
        self._vhs_coeff = new_vhs_coeff
        self._vhs_alpha = new_vhs_alpha
        self.precompute()
        return


    # Override miscellaneous methods
    @override
    @property
    def approximation_level(self) -> int:
        """Return the approximation level of the fast spectral method."""
        return int(self._fsm_phase_1.size(-2) * self._fsm_phase_1.size(-1))
    
    
##################################################
##################################################
# End of file