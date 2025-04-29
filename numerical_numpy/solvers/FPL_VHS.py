from    typing              import  Callable

import  numpy               as      np
from    scipy.special       import  j0

from    ..      import  utils


##################################################
##################################################
__all__: list[str] = [
    # 2-dimensional precomputations
    'precompute_fpl_character_1_2D',
    'precompute_fpl_character_2_2D',
    
    # 3-dimensional precomputations
    'precompute_fpl_character_1_3D',
    'precompute_fpl_character_2_3D',
    'precompute_fpl_gain_tensors_3D',
    'precompute_fpl_gain_tensors_3D__positive',
    'precompute_fpl_gain_tensors_3D__negative',
]


##################################################
##################################################
# 2-dimensional precomputations
def precompute_fpl_character_1_2D(
        v_num_grid:     int,
        v_max:          float,
        
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        quad_order_legendre:    int,
    ) -> np.ndarray:
    dim = 2
    v_ratio = np.pi/v_max
    
    # Define variables
    _len_norms  = int(dim * (v_num_grid//2)**2 + 1)
    norms       = np.sqrt(np.arange(_len_norms))[:, None]
    r, w = utils.roots_legendre_shifted(quad_order_legendre, 0, np.pi)
    r: np.ndarray = r.reshape(1, -1)
    w: np.ndarray = w.reshape(1, -1)
    
    # Compute the integrand
    _scale  = float((2*np.pi*vhs_coeff) / np.power(v_ratio, 2+vhs_alpha))
    _power: np.ndarray  = np.power(r, 3+vhs_alpha)
    _fcn:   np.ndarray  = j0(norms * r)
    integrand: np.ndarray = _scale * _power * _fcn
    
    # Compute the integral
    fpl_character_1 = np.sum(w*integrand, axis=-1)
    return fpl_character_1


def precompute_fpl_character_2_2D(
        v_num_grid:     int,
        v_max:          float,
        
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        quad_order_legendre:    int,
        quad_order_uniform:     int,
    ) -> np.ndarray:
    dim = 2
    v_ratio = np.pi/v_max
    
    # Define variables
    _len_norms = int(dim * (v_num_grid//2)**2 + 1)
    norms   = np.sqrt(np.arange(_len_norms))[:, None]
    r, w    = utils.roots_legendre_shifted(quad_order_legendre, 0, np.pi)
    r       = r.reshape((1, -1))
    w       = w.reshape((1, -1))
    
    # Compute the integrand
    scale = float(vhs_coeff / np.power(v_ratio, 2+vhs_alpha))
    _power: np.ndarray = r**(3+vhs_alpha)
    _func:  np.ndarray = _fpl_character_2__weight_C(r * norms, quad_order_uniform)
    integrand: np.ndarray = _power * _func
    
    # Compute the integral
    fpl_character_2 = scale * np.sum(integrand*w, axis=-1)
    return fpl_character_2


# 3-dimensional precomputations
def precompute_fpl_character_1_3D() -> None:
    # No implementation
    return


def precompute_fpl_character_2_3D() -> None:
    # No implementation
    return


def precompute_fpl_gain_tensors_3D() -> tuple[np.ndarray, np.ndarray]:
    return (
        precompute_fpl_gain_tensors_3D__positive(),
        precompute_fpl_gain_tensors_3D__negative(),
    )
    


def precompute_fpl_gain_tensors_3D__positive(
        v_num_grid:     int,
        v_max:          float,
        
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        quad_order_legendre:    int,
    ) -> None:
    dim = 3
    v_ratio = np.pi/v_max
    freqs = utils.freq_tensor(dim, v_num_grid, keepdims=True)
    
    # Define variables
    _len_norms = int(1 + dim * ((v_num_grid//2)**2))
    norms: np.ndarray = np.sqrt(np.arange(_len_norms))[:, None]
    r, w    = utils.roots_legendre_shifted(quad_order_legendre, 0, np.pi)
    r: np.ndarray = r.reshape((1, -1))
    w: np.ndarray = w.reshape((1, -1))
    
    # Compute the integrand
    _coeff = float((4*np.pi*vhs_coeff) / np.power(v_ratio, 3+vhs_alpha))
    _power: np.ndarray = np.power(r, 4+vhs_alpha)
    _func:  np.ndarray = utils.sinc(norms * r)
    integrand: np.ndarray = _coeff * _power * _func
    
    # Compute the integral
    gain_tensor_1 = np.sum(integrand*w, axis=-1)
    _freq_norms_sq: np.ndarray = np.sum(freqs**2, axis=-1, keepdims=True)
    fpl_gain_tensor_1 = gain_tensor_1[_freq_norms_sq]
    return fpl_gain_tensor_1


def precompute_fpl_gain_tensors_3D__negative(
        v_num_grid:     int,
        v_max:          float,
        
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        quad_order_legendre:    int,
        quad_order_uniform:     int,
    ) -> np.ndarray:
    dim = 3
    v_ratio = np.pi/v_max
    freqs = utils.freq_tensor(dim, v_num_grid, keepdims=True)
    
    # Initialize the tensor
    fpl_gain_tensor_2: np.ndarray = \
        np.zeros((1, *utils.ones(dim), *utils.repeat(v_num_grid, dim), 1, 3, 3))
    
    # Define variables
    freqs_xy:       np.ndarray = freqs[..., :2]
    freqs_z:        np.ndarray = freqs[..., [2]]
    freq_norms_xy:  np.ndarray = np.linalg.norm(freqs_xy, ord=2, axis=-1, keepdims=True)
    r, w = utils.roots_legendre_shifted(quad_order_legendre, 0, np.pi)
    ## Reshape the variables
    freqs_xy:       np.ndarray = freqs_xy[..., None]
    freqs_z:        np.ndarray = freqs_z[..., None]
    freq_norms_xy:  np.ndarray = freq_norms_xy[..., None]
    r:      np.ndarray = r.reshape((*utils.ones(freqs_xy.ndim-1), -1))
    w:      np.ndarray = w.reshape((*utils.ones(freqs_xy.ndim-1), -1))
    
    
    # Computation begins
    _coeff_common = float(vhs_coeff / np.power(v_ratio, 3+vhs_alpha))
    _power: np.ndarray = np.power(r, 4+vhs_alpha)
    
    ## <<< (0, 1) entry (nondiagonal) >>> 
    _entry_01_scale: np.ndarray
    with np.errstate(divide='ignore', invalid='ignore'):
        _entry_01_scale = \
            _coeff_common * \
            freqs_xy.prod(axis=-1, keepdims=True) / \
            np.power(freq_norms_xy, 2)
    _entry_01_scale[utils.zeros(2*dim)] = 0     # (batch, x, y, z, v_x, v_y))
    _entry_01_weight: np.ndarray = _fpl_character_2_3D__entry01_weight(
        a   = freqs_z,
        b   = freq_norms_xy,
        scale_factor    = r,
        quad_order_uniform = quad_order_uniform,
    )
    entry_01_integrand: np.ndarray = \
        _entry_01_scale * _power * _entry_01_weight
    del(_entry_01_scale, _entry_01_weight)
    entry_01: np.ndarray = np.sum(entry_01_integrand*w, axis=-1)    # Drop the `r`-dimension
    
    ## <<< (2, 2) entry (diagonal) >>>
    _entry_22_scale:    float = _coeff_common * (2*np.pi)
    _entry_22_weight:   np.ndarray = _fpl_character_2_3D__entry22_weight(
        a   = freqs_z,
        b   = freq_norms_xy,
        scale_factor    = r,
        quad_order_uniform = quad_order_uniform,
    )
    entry_22_integrand: np.ndarray = _entry_22_scale * _power * _entry_22_weight
    del(_power, _entry_22_weight)
    entry_22: np.ndarray = np.sum(entry_22_integrand*w, axis=-1)    # Drop the `r`-dimension
    
    ## Reshape the tensors
    _reduce = tuple((0, *utils.zeros(dim), ..., 0)) # (batch, space, value)
    entry_01 = entry_01[*_reduce]
    entry_22 = entry_22[*_reduce]
    
    ## Fill the main entries
    fpl_gain_tensor_2[*_reduce, 0, 1] = entry_01
    fpl_gain_tensor_2[*_reduce, 2, 2] = entry_22
    
    ## Fill the remaining entries using symmetry (diagonal)
    fpl_gain_tensor_2[*_reduce, 0, 0] = entry_22.transpose((1,2,0))
    fpl_gain_tensor_2[*_reduce, 1, 1] = entry_22.transpose((2,0,1))
    
    ## Fill the remaining entries using symmetry (nondiagonal)
    fpl_gain_tensor_2[*_reduce, 0, 2] = entry_01.transpose((2,0,1))
    fpl_gain_tensor_2[*_reduce, 1, 2] = entry_01.transpose((1,2,0))
    fpl_gain_tensor_2[*_reduce, 1, 0] = fpl_gain_tensor_2[*_reduce, 0, 1]
    fpl_gain_tensor_2[*_reduce, 2, 0] = fpl_gain_tensor_2[*_reduce, 0, 2]
    fpl_gain_tensor_2[*_reduce, 2, 1] = fpl_gain_tensor_2[*_reduce, 1, 2]
    
    return fpl_gain_tensor_2
   

##################################################
##################################################
# Computation of several weight functions
def _fpl_character_2__weight_C(
        x:  np.ndarray,
        quad_order_uniform: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
    r"""Returns the weight function defined as below:
    
        $x \mapsto \int_{0}{2\pi} \cos(2t) \cos(x\cos(t)) \, dt$
    """
    r, w = utils.roots_legendre_shifted(quad_order_uniform, 0, 2*np.pi)
    r = r.reshape(*utils.ones(x.ndim), -1)
    w = w.reshape(*utils.ones(x.ndim), -1)
    integrand = np.cos(2*r) * np.cos(x[..., None]*np.cos(r))
    return np.sum(integrand*w, axis=-1)


def _fpl_character_2_3D__entry01_weight(
        a:              np.ndarray,
        b:              np.ndarray,
        scale_factor:   np.ndarray,
        quad_order_uniform: int,
    ) -> np.ndarray:
    r"""Returns the weight function for computing the `(0, 1)` entry, which is defined as follows:
        Given two real numbers $a$ and $b$ which are multiplied by `scale_factor`, the output is the integration of the product of the following functions (in $t$) on $[0, \pi]$:
        * $\cos( a \cos(t) )$,
        * $\sin^3(t)$,
        * $C( b\sin(t) )$.
    """
    assert a.shape==b.shape, \
        f"The input variables must have the same shape. ({a.shape=}, {b.shape=})"
    assert a.ndim==b.ndim and b.ndim==scale_factor.ndim, \
        f"The input variables must have the same number of dimensions. ({a.ndim=}, {b.ndim=}, {scale_factor.ndim=})"
    _ndim = scale_factor.ndim
    
    # Define the integrated variables
    t, w = utils.roots_legendre_shifted(quad_order_uniform, 0, np.pi)
    t = t.reshape(*utils.ones(_ndim), -1)
    w = w.reshape(*utils.ones(_ndim), -1)
    
    # Compute the integrand (Scaling is necessary)
    a: np.ndarray = scale_factor * a
    _prod_1: np.ndarray = \
        np.cos(a[..., None]*np.cos(t))
    del(a)
    _prod_2: np.ndarray = \
        np.sin(t)**3
    del()
    b: np.ndarray = scale_factor * b
    _prod_3: np.ndarray = \
        _fpl_character_2__weight_C(b[..., None]*np.sin(t), quad_order_uniform)
    del(b)
    integrand = _prod_1 * _prod_2 * _prod_3
    
    # Compute the integral
    return np.sum(integrand*w, axis=-1)


def _fpl_character_2_3D__entry22_weight(
        a:              np.ndarray,
        b:              np.ndarray,
        scale_factor:   np.ndarray,
        quad_order_uniform: int
    ) -> np.ndarray:
    r"""Returns the weight function for computing the `(2, 2)` entry, which is defined as follows:
        Given two real numbers $a$ and $b$ which are multiplied by `scale_factor`, the output is the integration of the product of the following functions (in $t$) on $[0, \pi]$:
        * $\cos( a \cos(t) )$,
        * $\cos^2(t) \sin(t)$,
        * $J_0( b\sin(t) )$.
    """
    assert a.shape==b.shape, \
        f"The input variables must have the same shape. ({a.shape=}, {b.shape=})"
    assert a.ndim==b.ndim and b.ndim==scale_factor.ndim, \
        f"The input variables must have the same number of dimensions. ({a.ndim=}, {b.ndim=}, {scale_factor.ndim=})"
    _ndim = scale_factor.ndim
    
    # Define the integrated variables
    t, w = utils.roots_legendre_shifted(quad_order_uniform, 0, np.pi)
    t = t.reshape(*utils.ones(_ndim), -1)
    w = w.reshape(*utils.ones(_ndim), -1)
    
    # Compute the integrand (Scaling is necessary)
    a: np.ndarray = scale_factor * a
    _prod_1: np.ndarray = \
        np.cos(a[..., None]*np.cos(t))
    del(a)
    _prod_2: np.ndarray = \
        (np.cos(t)**2) * np.sin(t)
    b: np.ndarray = scale_factor * b
    _prod_3: np.ndarray = \
        j0(b[..., None]*np.sin(t))
    del(b)
    integrand = _prod_1 * _prod_2 * _prod_3
    
    # Compute the integral
    return np.sum(integrand*w, axis=-1)


##################################################
##################################################
# End of file