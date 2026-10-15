r"""The implementation of the fast spectral method for solving the homogeneous Boltzmann equation

-----
### Description
Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.

Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
"""
from   typing            import Optional

from   scipy.special     import gamma
import torch
from   torch.special     import bessel_j0 as j0
from   typing_extensions import override

from   deep_numerical                     import ones, repeat
from   deep_numerical.fft                 import freq_tensor
from   deep_numerical.numerical.integrate import roots_circle, roots_lebedev, roots_legendre_shifted
from   deep_numerical.numerical.math      import area_of_unit_sphere, sinc
from   ._kernel_modes.boltzmann_VHS       import Boltzmann_VHS_kernel_modes
from   .base_classes                      import FastSM_Boltzmann


__all__: list[str] = ['FastSM_Boltzmann_VHS', 'FastSM_Boltzmann_VSS']


class FastSM_Boltzmann_VHS(FastSM_Boltzmann):
    r"""The class for the fast spectral method for solving the homogeneous Boltzmann equation with the VHS model.
    
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

            dtype:  Optional[torch.dtype]   = None,
            device: Optional[torch.device]  = None,
        ) -> None:
        """The initializer of `FastSM_Boltzmann_VHS`
        
        ## Description
        Initializes the fast spectral method solver for the homogeneous Boltzmann equation with the variable hard sphere (VHS) model.
        
        ## Arguments
        `dimension` (`int`): Dimension of the domain.
        `v_num_grid` (`int`): Number of grid points along each velocity dimension.
        `v_max` (`float`): Maximum velocity value defining the velocity truncation domain `[-v_max, v_max]^d`.
        `x_num_grid` (`Optional[int]`, default: `None`): Number of grid points along each spatial dimension.
        `x_max` (`Optional[float]`, default: `None`): Maximum spatial boundary coordinate.
        `vhs_coeff` (`Optional[float]`, default: `None`): Coefficient `C_gamma` of the VHS collision kernel.
        `vhs_alpha` (`Optional[float]`, default: `None`): Exponent `gamma` (or `alpha`) of relative velocity in the VHS collision kernel.
        `restitution` (`float`, default: `1.0`): Coefficient of restitution for elastic or inelastic collisions.
        `quad_order_uniform` (`Optional[int]`, default: `None`): Quadrature order for the radial/uniform angle quadrature.
        `quad_order_legendre` (`Optional[int]`, default: `None`): Quadrature order for the Legendre quadrature.
        `quad_order_lebedev` (`Optional[int]`, default: `None`): Quadrature order for the spherical Lebedev quadrature rule.
        `dtype` (`Optional[torch.dtype]`, default: `None`): Data type for tensors.
        `device` (`Optional[torch.device]`, default: `None`): Target computing device.
        
        ## Returns
        `None`: None.
        """
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
        """The coefficient of the VHS collision model."""
        return self._vhs_coeff
    @property
    def vhs_alpha(self) -> float:
        """The exponent of the relative velocity in the VHS collision model."""
        return self._vhs_alpha


    # Precomputation methods
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
        )[*repeat(None, 1+self._dimension), ..., None]
        return

    def _precompute_fsm_gain_2D(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precomputes 2D gain kernel modes for the fast spectral method.

        ## Description
        Computes the scaling and phase factor tensors using polar quadrature on `S^1`.
        Data ordering: `(batch, space, velocity, data, *quadrature_rules)`.

        ## Returns
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tensors `(fsm_scale, fsm_phase_1, fsm_phase_2)`.
        """
        dim = self._dimension

        # Frequency tensor: (1, 1, 1, v_num_grid, v_num_grid, 2, 1, 1)
        freqs = freq_tensor(dim, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.view(1, *ones(dim), *repeat(self._v_num_grid, dim), dim, 1, 1)

        # Quadrature roots and weights
        r_roots, r_weights = roots_legendre_shifted(self._quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights = roots_circle(self.quad_order_uniform, dtype=self._dtype, device=self._device)
        s_roots = s_roots.T

        r_roots   = r_roots.view(1, *ones(2*dim), 1, -1, 1)
        r_weights = r_weights.view(1, *ones(2*dim), 1, -1, 1)
        s_roots   = s_roots.view(1, *ones(2*dim), dim, 1, -1)
        s_weights = s_weights.view(1, *ones(2*dim), 1, 1, -1)

        # Scaling part (fsm_scale)
        scale_coeff = (2 * torch.pi) * self.vhs_coeff
        scale_fcn   = (
            torch.pow(r_roots, self.vhs_alpha + dim - 1) *
            j0(
                (0.5 * self.v_ratio) *
                r_roots *
                torch.sqrt((freqs**2).sum(-3, keepdim=True))
            )
        )
        fsm_scale = scale_coeff * scale_fcn * r_weights

        # Phase factor (fsm_phase)
        fsm_arg      = (0.5 * self.v_ratio) * (freqs * r_roots * s_roots).sum(-3, keepdim=True)
        coeff_1      = (1 + self._restitution) / 2
        coeff_2      = (3 - self._restitution) / 2
        phase_weight = torch.sqrt(s_weights)
        fsm_phase_1  = torch.exp(+1j * fsm_arg * coeff_1) * phase_weight
        fsm_phase_2  = torch.exp(-1j * fsm_arg * coeff_2) * phase_weight

        return (fsm_scale, fsm_phase_1, fsm_phase_2)

    def _precompute_fsm_gain_3D(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precomputes 3D gain kernel modes for the fast spectral method.

        ## Description
        Computes the scaling and phase factor tensors using spherical Lebedev quadrature on `S^2`.
        Data ordering: `(batch, space, velocity, data, *quadrature_rules)`.

        ## Returns
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tensors `(fsm_scale, fsm_phase_1, fsm_phase_2)`.
        """
        dim = self._dimension

        # Frequency tensor: (1, 1, 1, 1, v_num_grid, v_num_grid, v_num_grid, 3, 1, 1)
        freqs = freq_tensor(dim, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.view(1, *ones(dim), *repeat(self._v_num_grid, dim), dim, 1, 1)

        # Quadrature roots and weights
        r_roots, r_weights = roots_legendre_shifted(self.quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights = roots_lebedev(self.quad_order_lebedev, dtype=self._dtype, device=self._device)
        s_roots = s_roots.transpose(0, 1)

        r_roots   = r_roots.view(1, *ones(2*dim), 1, -1, 1)
        r_weights = r_weights.view(1, *ones(2*dim), 1, -1, 1)
        s_roots   = s_roots.view(1, *ones(2*dim), dim, 1, -1)
        s_weights = s_weights.view(1, *ones(2*dim), 1, 1, -1)

        # Scaling part (fsm_scale)
        scale_coeff = (4 * torch.pi) * self.vhs_coeff
        scale_fcn   = (
            torch.pow(r_roots, self.vhs_alpha + dim - 1) *
            sinc(
                (0.5 * self.v_ratio) *
                r_roots *
                torch.norm(freqs, p=2, dim=-3, keepdim=True)
            )
        )
        fsm_scale = scale_coeff * scale_fcn * r_weights

        # Phase factor (fsm_phase)
        fsm_arg      = (0.5 * self.v_ratio) * (freqs * r_roots * s_roots).sum(-3, keepdim=True)
        scale_1      = (1 + self._restitution) / 2
        scale_2      = (3 - self._restitution) / 2
        phase_weight = torch.sqrt(s_weights.type(self.dtype_complex))
        fsm_phase_1  = torch.exp(+1j * fsm_arg * scale_1) * phase_weight
        fsm_phase_2  = torch.exp(-1j * fsm_arg * scale_2) * phase_weight

        return (fsm_scale, fsm_phase_1, fsm_phase_2)


    # Override miscellaneous methods
    @override
    @property
    def approximation_level(self) -> int:
        """Return the approximation level of the fast spectral method."""
        return int(self._fsm_phase_1.size(-2) * self._fsm_phase_1.size(-1))
    
    

class FastSM_Boltzmann_VSS(FastSM_Boltzmann):
    r"""The class for the fast spectral method for solving the homogeneous Boltzmann equation with the VSS model.
    
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
            
            dtype:  Optional[torch.dtype]   = None,
            device: Optional[torch.device]  = None,
        ) -> None:
        """The initializer of `FastSM_Boltzmann_VSS`
        
        ## Description
        Initializes the fast spectral method solver for the homogeneous Boltzmann equation with the variable soft sphere (VSS) model.
        
        ## Arguments
        `dimension` (`int`): Dimension of the domain.
        `v_num_grid` (`int`): Number of grid points along each velocity dimension.
        `v_max` (`float`): Maximum velocity value defining the velocity truncation domain `[-v_max, v_max]^d`.
        `x_num_grid` (`Optional[int]`, default: `None`): Number of grid points along each spatial dimension.
        `x_max` (`Optional[float]`, default: `None`): Maximum spatial boundary coordinate.
        `vss_coeff` (`Optional[float]`, default: `None`): Coefficient `C_gamma` of the VSS collision kernel.
        `vss_exp_speed` (`Optional[float]`, default: `None`): Exponent `gamma` (or `alpha`) of relative velocity in the VSS kernel.
        `vss_exp_angle` (`Optional[float]`, default: `None`): Exponent `eta` of the scattering angle in the VSS kernel.
        `restitution` (`float`, default: `1.0`): Coefficient of restitution.
        `quad_order_uniform` (`Optional[int]`, default: `None`): Quadrature order for the radial/uniform angle quadrature.
        `quad_order_legendre` (`Optional[int]`, default: `None`): Quadrature order for the Legendre quadrature.
        `quad_order_lebedev` (`Optional[int]`, default: `None`): Quadrature order for the spherical Lebedev quadrature rule.
        `dtype` (`Optional[torch.dtype]`, default: `None`): Data type for tensors.
        `device` (`Optional[torch.device]`, default: `None`): Target computing device.
        
        ## Returns
        `None`: None.
        """
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
        """The coefficient of the VSS collision model."""
        return self._vss_coeff
    @property
    def vss_exp_speed(self) -> float:
        """The exponent of the relative velocity in the VSS collision model."""
        return self._vss_exp_speed
    @property
    def vss_exp_angle(self) -> float:
        """The exponent of the scattering angle in the VSS collision model."""
        return self._vss_exp_angle
    
    
    # Precomputation methods
    @override
    def _precompute_fsm_gain(self) -> None:
        fsm_scale, fsm_phase_1, fsm_phase_2 = getattr(self, f'_precompute_fsm_gain_{self._dimension}D')()
        self._fsm_scale    = fsm_scale
        self._fsm_phase_1  = fsm_phase_1
        self._fsm_phase_2  = fsm_phase_2
        return

    @override
    def _precompute_fsm_loss(self) -> None:
        coeff_kernel_modes: float = self._vss_coeff / area_of_unit_sphere(self._dimension)
        _scale: float
        if self._dimension == 2:
            _scale = 2**(self._vss_exp_angle + 1) * (torch.pi**0.5) * (
                gamma(self._vss_exp_angle + 0.5) /
                gamma(self._vss_exp_angle + 1.0)
            )
        elif self._dimension == 3:
            _scale = 2**(self._vss_exp_angle + 2) * torch.pi / (
                self._vss_exp_angle + 1
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
        )[*repeat(None, 1+self._dimension), ..., None]
        return

    def _precompute_fsm_gain_2D(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precomputes 2D gain kernel modes for the fast spectral method.

        ## Description
        Computes the scaling and phase factor tensors using polar quadrature on `S^1`.
        Data ordering: `(batch, space, velocity, data, *quadrature_rules)`.

        ## Returns
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tensors `(fsm_scale, fsm_phase_1, fsm_phase_2)`.
        """
        dim = self._dimension

        # Frequency tensor: (1, 1, 1, v_num_grid, v_num_grid, 2, 1, 1, 1)
        freqs = freq_tensor(dim, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.view(1, *ones(dim), *repeat(self._v_num_grid, dim), dim, 1, 1, 1)

        # Quadrature roots and weights
        r_roots, r_weights = roots_legendre_shifted(self._quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights = roots_circle(self.quad_order_uniform, dtype=self._dtype, device=self._device)
        g_roots, g_weights = roots_circle(self.quad_order_uniform, dtype=self._dtype, device=self._device)
        s_roots = s_roots.T
        g_roots = g_roots.T

        r_roots   = r_roots.view(1, *ones(2*dim), 1, -1, 1, 1)
        r_weights = r_weights.view(1, *ones(2*dim), 1, -1, 1, 1)
        s_roots   = s_roots.view(1, *ones(2*dim), dim, 1, -1, 1)
        s_weights = s_weights.view(1, *ones(2*dim), 1, 1, -1, 1)
        g_roots   = g_roots.view(1, *ones(2*dim), dim, 1, 1, -1)
        g_weights = g_weights.view(1, *ones(2*dim), 1, 1, 1, -1)

        # Part 1: Compute the scaling part (fsm_scale)
        fsm_power    = self.vss_coeff * torch.pow(r_roots, self.vss_exp_speed + dim - 1)
        fsm_integral = torch.sum(
            torch.pow(
                1 + torch.sum(s_roots * g_roots, dim=-4, keepdim=True),
                self.vss_exp_angle
            ) *
            torch.exp(
                -0.5j * self.v_ratio * r_roots *
                torch.sum(freqs * g_roots, dim=-4, keepdim=True)
            ) *
            g_weights,
            dim=-1, keepdim=True
        )
        fsm_scale = fsm_power * fsm_integral * r_weights

        # Part 2: Compute the multiplicative part (fsm_phase)
        fsm_arg      = (0.5 * self.v_ratio) * (freqs * r_roots * s_roots).sum(-4, keepdim=True)
        coeff_1      = (1 + self._restitution) / 2
        coeff_2      = (3 - self._restitution) / 2
        phase_weight = torch.sqrt(s_weights)
        fsm_phase_1  = torch.exp(+1j * fsm_arg * coeff_1) * phase_weight
        fsm_phase_2  = torch.exp(-1j * fsm_arg * coeff_2) * phase_weight

        scale_2d   = torch.squeeze(fsm_scale,   dim=-1)
        phase_1_2d = torch.squeeze(fsm_phase_1, dim=-1)
        phase_2_2d = torch.squeeze(fsm_phase_2, dim=-1)
        return (scale_2d, phase_1_2d, phase_2_2d)

    def _precompute_fsm_gain_3D(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precomputes 3D gain kernel modes for the fast spectral method.

        ## Description
        Computes the scaling and phase factor tensors using spherical Lebedev quadrature on `S^2`.
        Data ordering: `(batch, space, velocity, data, *quadrature_rules)`.

        ## Returns
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tensors `(fsm_scale, fsm_phase_1, fsm_phase_2)`.
        """
        dim = self._dimension

        # Frequency tensor: (1, 1, 1, 1, v_num_grid, v_num_grid, v_num_grid, 3, 1, 1, 1)
        freqs = freq_tensor(dim, self._v_num_grid, True, dtype=self._dtype, device=self._device)
        freqs = freqs.view(1, *ones(dim), *repeat(self._v_num_grid, dim), dim, 1, 1, 1)

        # Quadrature roots and weights
        r_roots, r_weights = roots_legendre_shifted(self.quad_order_legendre, 0, 2*self.v_radius, dtype=self._dtype, device=self._device)
        s_roots, s_weights = roots_lebedev(self.quad_order_lebedev, dtype=self._dtype, device=self._device)
        g_roots, g_weights = roots_lebedev(self.quad_order_lebedev, dtype=self._dtype, device=self._device)
        s_roots = s_roots.transpose(0, 1)
        g_roots = g_roots.transpose(0, 1)

        r_roots   = r_roots.view(1, *ones(2*dim), 1, -1, 1, 1)
        r_weights = r_weights.view(1, *ones(2*dim), 1, -1, 1, 1)
        s_roots   = s_roots.view(1, *ones(2*dim), dim, 1, -1, 1)
        s_weights = s_weights.view(1, *ones(2*dim), 1, 1, -1, 1)
        g_roots   = g_roots.view(1, *ones(2*dim), dim, 1, 1, -1)
        g_weights = g_weights.view(1, *ones(2*dim), 1, 1, 1, -1)

        # Part 1: Compute the scaling part (fsm_scale)
        fsm_power    = self.vss_coeff * torch.pow(r_roots, self.vss_exp_speed + dim - 1)
        fsm_integral = torch.sum(
            torch.pow(
                1 + torch.sum(s_roots * g_roots, dim=-4, keepdim=True),
                self.vss_exp_angle
            ) *
            torch.exp(
                -0.5j * self.v_ratio * r_roots *
                torch.sum(freqs * g_roots, dim=-4, keepdim=True)
            ) *
            g_weights,
            dim=-1, keepdim=True
        )
        fsm_scale = fsm_power * fsm_integral * r_weights

        # Part 2: Compute the multiplicative part (fsm_phase)
        fsm_arg      = (0.5 * self.v_ratio) * (freqs * r_roots * s_roots).sum(-4, keepdim=True)
        scale_1      = (1 + self._restitution) / 2
        scale_2      = (3 - self._restitution) / 2
        phase_weight = torch.sqrt(s_weights.type(self.dtype_complex))
        fsm_phase_1  = torch.exp(+1j * fsm_arg * scale_1) * phase_weight
        fsm_phase_2  = torch.exp(-1j * fsm_arg * scale_2) * phase_weight

        scale_3d   = torch.squeeze(fsm_scale,   dim=-1)
        phase_1_3d = torch.squeeze(fsm_phase_1, dim=-1)
        phase_2_3d = torch.squeeze(fsm_phase_2, dim=-1)
        return (scale_3d, phase_1_3d, phase_2_3d)

    def reset_VHS_parameters(self, new_vhs_coeff: float, new_vhs_alpha: float) -> None:
        """Set a new value for the exponent of the VHS model."""
        self._vhs_coeff = new_vhs_coeff
        self._vhs_alpha = new_vhs_alpha
        self.precompute()
        return

    @override
    @property
    def approximation_level(self) -> int:
        """Return the approximation level of the fast spectral method."""
        return int(self._fsm_phase_1.size(-2) * self._fsm_phase_1.size(-1))


##################################################
def main() -> None:
    return


if __name__ == '__main__':
    main()