from    typing              import  Optional, Callable
from    typing_extensions   import  Self, override

import  torch
from    torch.special       import  bessel_j0   as  j0

from    .base_classes       import  FastSM_Landau
from    ...                 import  utils


##################################################
##################################################
__all__: list[str] = ['FastSM_Landau_VHS']


##################################################
##################################################
class FastSM_Landau_VHS(FastSM_Landau):
    """## The class for the fast spectral method for solving the Fokker-Planck-Landau equation with the VHS model.
    
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
            
            vhs_coeff:  Optional[float] = None,
            vhs_alpha:  Optional[float] = None,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            
            dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
            device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
        ) -> Self:
        super().__init__(
            dimension       = dimension,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            x_num_grid  = x_num_grid,
            x_max       = x_max,
            vhs_coeff   = vhs_coeff,
            vhs_alpha   = vhs_alpha,
            quad_order_uniform  = quad_order_uniform,
            quad_order_legendre = quad_order_legendre,
            dtype           = dtype,
            device          = device,
        )
        return
    
    
    # 2-dimensional precomputations
    def _precompute_fpl_character_1_2D(self) -> None:
        # Define variables
        _len_norms  = int(self._dimension * (self._v_num_grid//2)**2 + 1)
        norms       = torch.sqrt(torch.arange(_len_norms, **self.dtype_and_device))[:, None]
        r, w = utils.roots_legendre_shifted(self._quad_order_legendre, 0, torch.pi, **self.dtype_and_device)
        r: torch.Tensor = r.reshape(1, -1)
        w: torch.Tensor = w.reshape(1, -1)
        
        # Compute the integrand
        _scale  = float(
            (2*torch.pi*self._vhs_coeff) / (self.v_ratio**(2+self._vhs_alpha))
        )
        _power: torch.Tensor  = torch.pow(r, 3+self._vhs_alpha)
        _fcn:   torch.Tensor  = j0(norms * r)
        integrand: torch.Tensor = _scale * _power * _fcn
        
        # Compute the integral
        self._fpl_character_1 = torch.sum(w*integrand, dim=-1)
        return
    
    
    def _precompute_fpl_character_2_2D(self) -> None:
        # Define variables
        _len_norms = int(self._dimension * (self._v_num_grid//2)**2 + 1)
        norms   = torch.sqrt(torch.arange(_len_norms, **self.dtype_and_device))[:, None]
        r, w    = utils.roots_legendre_shifted(self._quad_order_legendre, 0, torch.pi, **self.dtype_and_device)
        r       = r.reshape((1, -1))
        w       = w.reshape((1, -1))
        
        # Compute the integrand
        scale = float(
            self._vhs_coeff / (self.v_ratio**(2+self._vhs_alpha))
        )
        _power: torch.Tensor = r**(3+self._vhs_alpha)
        _func:  torch.Tensor = _fpl_character_2__weight_C(r * norms, self._quad_order_uniform, **self.dtype_and_device)
        integrand: torch.Tensor = _power * _func
        
        # Compute the integral
        self._fpl_character_2 = scale * torch.sum(integrand*w, dim=-1)
        return
        
    
    def _precompute_fpl_gain_tensors_2D(self) -> None:
        """This function returns two tensors `P` and `Q`, which are defined in the reference.
        The shapes of `P` and `Q` are listed below.
        1. `P`: A `1+self.dim+self.dim+1`-tensor of shape `(1, *ones(self.dim), *repeat(self.num_grid, self.dim))`
        2. `Q` is a `1+self.dim+self.dim+2`-tensor of shape `(1, *ones(self.dim), *repeat(self.num_grid, self.dim), *(2, 2))`."""
        _freq_norms_sq: torch.Tensor  = torch.sum(self._freqs**2, dim=-1, keepdim=True)
        F = self._fpl_character_1[_freq_norms_sq]   # -> self._fpl_gain_tensor_1
        G = self._fpl_character_2[_freq_norms_sq]
        # NOTE: `F` and `G` are of shape `(1, *ones(self.dim), *repeat(self.num_grid, 1))`
        _freq_norms_sq = _freq_norms_sq.type(self._dtype)  # Typecasting
        
        Q = torch.zeros((*self.fpl_base_shape, *utils.repeat(self._dimension, 2)), **self.dtype_and_device)
        _i = self._freqs[..., [0]].type(self._dtype)
        _j = self._freqs[..., [1]].type(self._dtype)
        double_cos: torch.Tensor = (_i**2 - _j**2) / _freq_norms_sq
        double_sin: torch.Tensor = (2 * _i*_j)     / _freq_norms_sq
        # The first `1+self._dimension` dimensions are batch/spatial dimensions
        double_cos[utils.zeros(1+2*self._dimension)] = 0
        double_sin[utils.zeros(1+2*self._dimension)] = 0
        Q[..., 0, 0] = 0.5 * (F + double_cos * G)
        Q[..., 1, 1] = 0.5 * (F - double_cos * G)
        Q[..., 0, 1] = 0.5 * double_sin * G
        Q[..., 1, 0] = Q[..., 0, 1]
        
        self._fpl_gain_tensor_1 = F
        self._fpl_gain_tensor_2 = Q
        return
    
    
    # 3-dimensional precomputations
    def _precompute_fpl_character_1_3D(self) -> None:
        # No implementation
        return
    
    
    def _precompute_fpl_character_2_3D(self) -> None:
        # No implementation
        return
    
    
    def _precompute_fpl_gain_tensors_3D(self) -> None:
        self.__precompute_fpl_gain_tensors_3D__positive()
        self.__precompute_fpl_gain_tensors_3D__negative()
        return
        
    
    def __precompute_fpl_gain_tensors_3D__positive(self) -> None:
        # Define variables
        _len_norms = int(1 + self._dimension * ((self._v_num_grid//2)**2))
        norms: torch.Tensor = torch.sqrt(torch.arange(_len_norms, **self.dtype_and_device))[:, None]
        r, w = utils.roots_legendre_shifted(self._quad_order_legendre, 0, torch.pi, **self.dtype_and_device)
        r: torch.Tensor = r.reshape((1, -1))
        w: torch.Tensor = w.reshape((1, -1))
        
        # Compute the integrand
        _coeff = float(
            (4*torch.pi*self._vhs_coeff) / (self.v_ratio**(3+self._vhs_alpha))
        )
        _power: torch.Tensor = torch.pow(r, 4+self._vhs_alpha)
        _func:  torch.Tensor = utils.sinc(norms * r)
        integrand: torch.Tensor = _coeff * _power * _func
        
        # Compute the integral
        gain_tensor_1 = torch.sum(integrand*w, dim=-1)
        _freq_norms_sq: torch.Tensor = torch.sum(self._freqs**2, dim=-1, keepdim=True)
        self._fpl_gain_tensor_1 = gain_tensor_1[_freq_norms_sq]
        return
    
    
    def __precompute_fpl_gain_tensors_3D__negative(self) -> None:
        # Initialize the tensor
        self._fpl_gain_tensor_2: torch.Tensor = \
            torch.zeros(
                (1, *utils.ones(self._dimension), *utils.repeat(self._v_num_grid, self._dimension), 1, 3, 3),
                **self.dtype_and_device,
            )
        
        # Define variables
        freqs_xy:       torch.Tensor = self._freqs[...,  :2].type(self._dtype)
        freqs_z:        torch.Tensor = self._freqs[..., [2]].type(self._dtype)
        freq_norms_xy:  torch.Tensor = torch.norm(freqs_xy, p=2, dim=-1, keepdim=True)
        r, w = utils.roots_legendre_shifted(self._quad_order_legendre, 0, torch.pi, **self.dtype_and_device)
        ## Reshape the variables
        freqs_xy:       torch.Tensor = freqs_xy[..., None]
        freqs_z:        torch.Tensor = freqs_z[ ..., None]
        freq_norms_xy:  torch.Tensor = freq_norms_xy[..., None]
        r:      torch.Tensor = r.reshape((*utils.ones(freqs_xy.ndim-1), -1))
        w:      torch.Tensor = w.reshape((*utils.ones(freqs_xy.ndim-1), -1))
        
        
        # Computation begins
        _coeff_common = float(
            self._vhs_coeff / (self.v_ratio**(3+self._vhs_alpha))
        )
        _power: torch.Tensor = torch.pow(r, 4+self._vhs_alpha)
        
        ## <<< (0, 1) entry (nondiagonal) >>> 
        _entry_01_scale: torch.Tensor = \
            _coeff_common * \
            freqs_xy.prod(dim=-1, keepdim=True) / \
            torch.pow(freq_norms_xy, 2)
        _entry_01_scale[utils.zeros(2*self._dimension)] = 0     # (batch, x, y, z, v_x, v_y))
        _entry_01_weight: torch.Tensor = _fpl_character_2_3D__entry01_weight(
            a   = freqs_z,
            b   = freq_norms_xy,
            scale_factor    = r,
            quad_order_uniform = self._quad_order_uniform,
            **self.dtype_and_device,
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
            quad_order_uniform = self._quad_order_uniform,
            **self.dtype_and_device,
        )
        entry_22_integrand: torch.Tensor = _entry_22_scale * _power * _entry_22_weight
        del(_power, _entry_22_weight)
        entry_22: torch.Tensor = torch.sum(entry_22_integrand*w, dim=-1)    # Drop the `r`-dimension
        
        ## Reshape the tensors
        _reduce = tuple((0, *utils.zeros(self._dimension), ..., 0)) # (batch, space, value)
        entry_01 = entry_01[*_reduce]
        entry_22 = entry_22[*_reduce]
        
        ## Fill the main entries
        self._fpl_gain_tensor_2[*_reduce, 0, 1] = entry_01
        self._fpl_gain_tensor_2[*_reduce, 2, 2] = entry_22
        
        ## Fill the remaining entries using symmetry (diagonal)
        self._fpl_gain_tensor_2[*_reduce, 0, 0] = entry_22.permute((1,2,0))
        self._fpl_gain_tensor_2[*_reduce, 1, 1] = entry_22.permute((2,0,1))
        
        ## Fill the remaining entries using symmetry (nondiagonal)
        self._fpl_gain_tensor_2[*_reduce, 0, 2] = entry_01.permute((2,0,1))
        self._fpl_gain_tensor_2[*_reduce, 1, 2] = entry_01.permute((1,2,0))
        self._fpl_gain_tensor_2[*_reduce, 1, 0] = self._fpl_gain_tensor_2[*_reduce, 0, 1]
        self._fpl_gain_tensor_2[*_reduce, 2, 0] = self._fpl_gain_tensor_2[*_reduce, 0, 2]
        self._fpl_gain_tensor_2[*_reduce, 2, 1] = self._fpl_gain_tensor_2[*_reduce, 1, 2]
        
        return


##################################################
##################################################
# Computation of several weight functions
def _fpl_character_2__weight_C(
        x:  torch.Tensor,
        quad_order_uniform: int,
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Returns the weight function defined as below:
    
        $x \mapsto \int_{0}{2\pi} \cos(2t) \cos(x\cos(t)) \, dt$
    """
    r, w = utils.roots_legendre_shifted(quad_order_uniform, 0, 2*torch.pi, dtype=dtype, device=device)
    r = r.reshape(*utils.ones(x.ndim), -1)
    w = w.reshape(*utils.ones(x.ndim), -1)
    integrand = torch.cos(2*r) * torch.cos(x[..., None]*torch.cos(r))
    return torch.sum(integrand*w, dim=-1)


def _fpl_character_2_3D__entry01_weight(
        a:              torch.Tensor,
        b:              torch.Tensor,
        scale_factor:   torch.Tensor,
        quad_order_uniform: int,
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
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
    t, w = utils.roots_legendre_shifted(quad_order_uniform, 0, torch.pi, dtype=dtype, device=device)
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
        _fpl_character_2__weight_C(b[..., None]*torch.sin(t), quad_order_uniform, dtype=dtype, device=device)
    del(b)
    integrand = _prod_1 * _prod_2 * _prod_3
    
    # Compute the integral
    return torch.sum(integrand*w, dim=-1)


def _fpl_character_2_3D__entry22_weight(
        a:              torch.Tensor,
        b:              torch.Tensor,
        scale_factor:   torch.Tensor,
        quad_order_uniform: int,
        dtype:  torch.dtype     = utils.TORCH_DEFAULT_DTYPE,
        device: torch.device    = utils.TORCH_DEFAULT_DEVICE,
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
    t, w = utils.roots_legendre_shifted(quad_order_uniform, 0, torch.pi, dtype=dtype, device=device)
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