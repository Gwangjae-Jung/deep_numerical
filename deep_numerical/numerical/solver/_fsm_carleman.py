r"""## The implementation of the fast spectral method for solving the homogeneous Boltzmann equation

### Description
TBD
"""
from    typing              import  *
import  numpy               as      np
from    scipy.interpolate   import  CubicSpline
from    scipy.special       import  roots_legendre

from    ...utils            import  *


##################################################
##################################################
__all__ = [
    'fsm_decoupled_factor',
    'fsm_kernel_modes_2D',
]


##################################################
##################################################
_NUM_GL_ROOTS   = 25
_NUM_S_DATA     = 250


##################################################
##################################################
# Utility functions
def _unity(placeholder: object) -> float:
    """Returns `1.0`."""
    return 1.0


def decoupled_factor_for_unity_2D(v_max: float) -> Callable[[np.ndarray], np.ndarray]:
    R = v_max * LAMBDA
    return lambda s: (4*R) * sinc((2*R)*s)


##################################################
##################################################
# Decoupling models
def fsm_decoupled_factor(
        dim:        int,
        v_max:      float,
        coeff_fcn:  Callable[[np.ndarray], np.ndarray] = _unity,
        s_min:      float = None,
        s_max:      float = None,
        num_s:      int = 100,
        
        BETA:       bool = False,
    ) -> Callable[[np.ndarray], np.ndarray]:
    r"""Returns the decoupled factor for the collision kernels satisfying the decoupling assumption.
    
    -----
    ### Definition
    This function returns the function whose input is an array `s` and the corresponding output is given as follows:
        \int_{0}^{2R} {
            2 \rho^{d-2}
            \times
            a(\rho)
            \times
            \cos( (\pi/T) \rho s)
            ds
        }.
    """
    if s_min is None or s_max is None:
        raise ValueError(
            f"'s_min' and 's_max' should be given. "\
            f"(s_min: {s_min:.4e}, s_max: {s_max:.4e})"
        )
    s_data = np.linspace(s_min, s_max, num_s)[:, None]
    
    def integrand(rho: np.ndarray) -> np.ndarray:
        rho = rho.reshape(1, -1)
        return (
            2*np.power(rho, dim-2) * \
            coeff_fcn(rho) * \
            np.cos((np.pi/v_max) * s_data * rho)
        )
    
    phi_data = integration_guass_legendre(
                    num_roots   = _NUM_GL_ROOTS,
                    a           = 0.0,
                    b           = 2*v_max*LAMBDA if not BETA else v_max*LAMBDA,
                    func        = integrand,
                )
    s_data = s_data.flatten()
    
    return CubicSpline(s_data, phi_data)


def fsm_kernel_modes_2D(
        num_grid:   int,
        v_max:      float,
        order:      int,
        num_s_data: int = _NUM_S_DATA,
        coeff_fcn1: Callable[[np.ndarray], np.ndarray]  = _unity,
        coeff_fcn2: Callable[[np.ndarray], np.ndarray]  = _unity,
    ) -> tuple[ArrayData, ArrayData]:
    freqs   = freq_tensor(2, num_grid, keepdims=True).astype(np.float64)
    s_bound = float(num_grid) / np.sqrt(2)
    phi1    = fsm_decoupled_factor(2, v_max, coeff_fcn1, num_s_data, s_min=-s_bound, s_max=s_bound)
    phi2    = fsm_decoupled_factor(2, v_max, coeff_fcn2, num_s_data, s_min=-s_bound, s_max=s_bound)
    
    thetas, weights = roots_legendre(order)
    thetas  = (np.pi/2) * (1 + thetas)
    weights = (np.pi/2) * weights
    # thetas  = np.arange(order) * np.pi / order
    # weights = np.pi / order
    
    _c, _s  = np.cos(thetas), np.sin(thetas)
    units1  = np.stack((+_c, _s), axis=0)
    units2  = np.stack((-_s, _c), axis=0)
    roots1  = np.einsum("...k,kj->...j", freqs, units1)
    roots2  = np.einsum("...k,kj->...j", freqs, units2)
    
    arr_phi1 = phi1(roots1) * np.sqrt(weights)
    arr_phi2 = phi2(roots2) * np.sqrt(weights)
    
    return (arr_phi1, arr_phi2)
    

##################################################
##################################################
# End of file