r"""## The module for the implementation of the direct spectral method

-----
### Description
This module provides the classes which are used to solve the homogeneous Boltzmann equation using the direct spectral method (which is based on the Fourier-Galerkin method).
To this end, we assume that the solution is compactly supported (so that it can be periodized in a bounded computational domain) and the Fourier coefficients of the solution depends on time; then we construct the system of ordinary differential equations of the Fourier coefficients.
"""
from    typing              import  Callable, Optional
from    typing_extensions   import  Self, override
import  torch

from    .base_classes       import  DirectSM_Base
from    ._kernel_modes.boltzmann_VHS      import  *
from    ..utils             import  TORCH_DEFAULT_DEVICE


##################################################
##################################################
__all__ = [
    'DirectSM_Boltzmann_VHS',
]
        

##################################################
##################################################
class DirectSM_Boltzmann_VHS(DirectSM_Base):
    r"""## The class for the direct spectral method for solving the homogeneous Boltzmann equation with the VHS model.
    
    -----
    ### Description
    This class provides features which can be used to solve the Boltzmann equation with the various hard sphere (VHS) model.
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
            
            order:      Optional[int]   = None,
            
            dtype:      torch.dtype     = torch.float64,
            device:     torch.device    = TORCH_DEFAULT_DEVICE,
        ) -> Self:
        super().__init__(
            dim         = dim,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            x_num_grid  = x_num_grid,
            x_max       = x_max,
            order       = order,
            dtype       = dtype,
            device      = device,
        )
        if vhs_alpha is None or vhs_alpha is None:
            raise ValueError(f"'vhs_coeff' and 'vhs_alpha' should be given. ({vhs_alpha=:.2e}, {vhs_coeff=:.2f})")
        self._vhs_coeff = vhs_coeff
        self._vhs_alpha = vhs_alpha
        self.precompute()
        return

    
    @property
    def vhs_coeff(self) -> float:
        return self._vhs_coeff
    @property
    def vhs_alpha(self) -> float:
        return self._vhs_alpha


    @override
    @property
    def _precompute_integrand_1D(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        return Boltzmann_VHS_kernel_modes_1D_integrand
    @override
    @property
    def _precompute_integrand_2D(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        return Boltzmann_VHS_kernel_modes_2D_integrand
    @override
    @property
    def _precompute_integrand_3D(self) -> Callable[[torch.Tensor, object], torch.Tensor]:
        return Boltzmann_VHS_kernel_modes_3D_integrand
    @override
    @property
    def _precompute_integrand_kwargs(self) -> dict[str, object]:
        return {
            'num_grid':     self.v_num_grid,
            'v_max':        self.v_max,
            'vhs_coeff':    self.vhs_coeff,
            'vhs_alpha':    self.vhs_alpha,
        }


##################################################
##################################################
# End of file