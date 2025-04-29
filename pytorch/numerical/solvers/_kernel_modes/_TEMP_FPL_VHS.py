from    typing              import  Callable

import  torch
from    torch.special       import  bessel_j0   as  j0

from    ....    import  utils


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
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    dimension = 2
    v_ratio = torch.pi/v_max
    
    # Define variables
    _len_norms  = int(dimension * (v_num_grid//2)**2 + 1)
    norms       = torch.sqrt(torch.arange(_len_norms))[:, None]
    r, w = utils.roots_legendre_shifted(quad_order_legendre, 0, torch.pi)
    r: torch.Tensor = r.reshape(1, -1)
    w: torch.Tensor = w.reshape(1, -1)
    
    # Compute the integrand
    _scale  = float((2*torch.pi*vhs_coeff) / torch.pow(v_ratio, 2+vhs_alpha))
    _power: torch.Tensor  = torch.pow(r, 3+vhs_alpha)
    _fcn:   torch.Tensor  = j0(norms * r)
    integrand: torch.Tensor = _scale * _power * _fcn
    
    # Compute the integral
    fpl_character_1 = torch.sum(w*integrand, dim=-1)
    return fpl_character_1


def precompute_fpl_character_2_2D(
        v_num_grid:     int,
        v_max:          float,
        
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        quad_order_legendre:    int,
        quad_order_uniform:     int,
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    dimension = 2
    v_ratio = torch.pi/v_max
    
    # Define variables
    _len_norms = int(dimension * (v_num_grid//2)**2 + 1)
    norms   = torch.sqrt(torch.arange(_len_norms))[:, None]
    r, w    = utils.roots_legendre_shifted(quad_order_legendre, 0, torch.pi)
    r       = r.reshape((1, -1))
    w       = w.reshape((1, -1))
    
    # Compute the integrand
    scale = float(vhs_coeff / torch.pow(v_ratio, 2+vhs_alpha))
    _power: torch.Tensor = r**(3+vhs_alpha)
    _func:  torch.Tensor = _fpl_character_2__weight_C(r * norms, quad_order_uniform)
    integrand: torch.Tensor = _power * _func
    
    # Compute the integral
    fpl_character_2 = scale * torch.sum(integrand*w, dim=-1)
    return fpl_character_2


# 3-dimensional precomputations
def precompute_fpl_character_1_3D() -> None:
    # No implementation
    return


def precompute_fpl_character_2_3D() -> None:
    # No implementation
    return


def precompute_fpl_gain_tensors_3D() -> tuple[torch.Tensor, torch.Tensor]:
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
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> None:
    dimension = 3
    v_ratio = torch.pi/v_max
    freqs = utils.freq_tensor(dimension, v_num_grid, keepdims=True)
    
    # Define variables
    _len_norms = int(1 + dimension * ((v_num_grid//2)**2))
    norms: torch.Tensor = torch.sqrt(torch.arange(_len_norms))[:, None]
    r, w    = utils.roots_legendre_shifted(quad_order_legendre, 0, torch.pi)
    r: torch.Tensor = r.reshape((1, -1))
    w: torch.Tensor = w.reshape((1, -1))
    
    # Compute the integrand
    _coeff = float((4*torch.pi*vhs_coeff) / torch.pow(v_ratio, 3+vhs_alpha))
    _power: torch.Tensor = torch.pow(r, 4+vhs_alpha)
    _func:  torch.Tensor = utils.sinc(norms * r)
    integrand: torch.Tensor = _coeff * _power * _func
    
    # Compute the integral
    gain_tensor_1 = torch.sum(integrand*w, dim=-1)
    _freq_norms_sq: torch.Tensor = torch.sum(freqs**2, dim=-1, keepdims=True)
    fpl_gain_tensor_1 = gain_tensor_1[_freq_norms_sq]
    return fpl_gain_tensor_1


def precompute_fpl_gain_tensors_3D__negative(
        v_num_grid:     int,
        v_max:          float,
        
        vhs_coeff:      float,
        vhs_alpha:      float,
        
        quad_order_legendre:    int,
        quad_order_uniform:     int,
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    dimension = 3
    v_ratio = torch.pi/v_max
    freqs = utils.freq_tensor(dimension, v_num_grid, keepdims=True)
    
    # Initialize the tensor
    fpl_gain_tensor_2: torch.Tensor = \
        torch.zeros((1, *utils.ones(dimension), *utils.repeat(v_num_grid, dimension), 1, 3, 3))
    
    # Define variables
    freqs_xy:       torch.Tensor = freqs[..., :2]
    freqs_z:        torch.Tensor = freqs[..., [2]]
    freq_norms_xy:  torch.Tensor = torch.linalg.norm(freqs_xy, ord=2, dim=-1, keepdims=True)
    r, w = utils.roots_legendre_shifted(quad_order_legendre, 0, torch.pi)
    ## Reshape the variables
    freqs_xy:       torch.Tensor = freqs_xy[..., None]
    freqs_z:        torch.Tensor = freqs_z[..., None]
    freq_norms_xy:  torch.Tensor = freq_norms_xy[..., None]
    r:      torch.Tensor = r.reshape((*utils.ones(freqs_xy.ndim-1), -1))
    w:      torch.Tensor = w.reshape((*utils.ones(freqs_xy.ndim-1), -1))
    
    
    # Computation begins
    _coeff_common = float(vhs_coeff / torch.pow(v_ratio, 3+vhs_alpha))
    _power: torch.Tensor = torch.pow(r, 4+vhs_alpha)
    
    ## <<< (0, 1) entry (nondiagonal) >>> 
    _entry_01_scale: torch.Tensor
    with torch.errstate(divide='ignore', invalid='ignore'):
        _entry_01_scale = \
            _coeff_common * \
            freqs_xy.prod(dim=-1, keepdims=True) / \
            torch.pow(freq_norms_xy, 2)
    _entry_01_scale[utils.zeros(2*dimension)] = 0     # (batch, x, y, z, v_x, v_y))
    _entry_01_weight: torch.Tensor = _fpl_character_2_3D__entry01_weight(
        a   = freqs_z,
        b   = freq_norms_xy,
        scale_factor    = r,
        quad_order_uniform = quad_order_uniform,
    )
    entry_01_integrand: torch.Tensor = \
        _entry_01_scale * _power * _entry_01_weight
    del(_entry_01_scale, _entry_01_weight)
    entry_01: torch.Tensor = torch.sum(entry_01_integrand*w, dim=-1)    # Drop the `r`-dimension
    
    ## <<< (2, 2) entry (diagonal) >>>
    _entry_22_scale:    float = _coeff_common * (2*torch.pi)
    _entry_22_weight:   torch.Tensor = _fpl_character_2_3D__entry22_weight(
        a   = freqs_z,
        b   = freq_norms_xy,
        scale_factor    = r,
        quad_order_uniform = quad_order_uniform,
    )
    entry_22_integrand: torch.Tensor = _entry_22_scale * _power * _entry_22_weight
    del(_power, _entry_22_weight)
    entry_22: torch.Tensor = torch.sum(entry_22_integrand*w, dim=-1)    # Drop the `r`-dimension
    
    ## Reshape the tensors
    _reduce = tuple((0, *utils.zeros(dimension), ..., 0)) # (batch, space, value)
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
        x:  torch.Tensor,
        quad_order_uniform: int,
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Returns the weight function defined as below:
    
        $x \mapsto \int_{0}{2\pi} \cos(2t) \cos(x\cos(t)) \, dt$
    """
    r, w = utils.roots_legendre_shifted(quad_order_uniform, 0, 2*torch.pi)
    r = r.reshape(*utils.ones(x.ndim), -1)
    w = w.reshape(*utils.ones(x.ndim), -1)
    integrand = torch.cos(2*r) * torch.cos(x[..., None]*torch.cos(r))
    return torch.sum(integrand*w, dim=-1)


def _fpl_character_2_3D__entry01_weight(
        a:              torch.Tensor,
        b:              torch.Tensor,
        scale_factor:   torch.Tensor,
        quad_order_uniform: int,
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
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
    t, w = utils.roots_legendre_shifted(quad_order_uniform, 0, torch.pi)
    t = t.reshape(*utils.ones(_ndim), -1)
    w = w.reshape(*utils.ones(_ndim), -1)
    
    # Compute the integrand (Scaling is necessary)
    a: torch.Tensor = scale_factor * a
    _prod_1: torch.Tensor = \
        torch.cos(a[..., None]*torch.cos(t))
    del(a)
    _prod_2: torch.Tensor = \
        torch.sin(t)**3
    del()
    b: torch.Tensor = scale_factor * b
    _prod_3: torch.Tensor = \
        _fpl_character_2__weight_C(b[..., None]*torch.sin(t), quad_order_uniform)
    del(b)
    integrand = _prod_1 * _prod_2 * _prod_3
    
    # Compute the integral
    return torch.sum(integrand*w, dim=-1)


def _fpl_character_2_3D__entry22_weight(
        a:              torch.Tensor,
        b:              torch.Tensor,
        scale_factor:   torch.Tensor,
        quad_order_uniform: int,
        
        dtype:      torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device:     torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
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
    t, w = utils.roots_legendre_shifted(quad_order_uniform, 0, torch.pi)
    t = t.reshape(*utils.ones(_ndim), -1)
    w = w.reshape(*utils.ones(_ndim), -1)
    
    # Compute the integrand (Scaling is necessary)
    a: torch.Tensor = scale_factor * a
    _prod_1: torch.Tensor = \
        torch.cos(a[..., None]*torch.cos(t))
    del(a)
    _prod_2: torch.Tensor = \
        (torch.cos(t)**2) * torch.sin(t)
    b: torch.Tensor = scale_factor * b
    _prod_3: torch.Tensor = \
        j0(b[..., None]*torch.sin(t))
    del(b)
    integrand = _prod_1 * _prod_2 * _prod_3
    
    # Compute the integral
    return torch.sum(integrand*w, dim=-1)


##################################################
##################################################
# End of file