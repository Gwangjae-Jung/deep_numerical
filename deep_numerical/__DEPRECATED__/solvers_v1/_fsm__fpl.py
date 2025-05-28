from    typing              import  Optional, Callable
from    typing_extensions   import  Self, override

import  numpy               as      np
from    scipy.special       import  j0

from    .base_classes       import  SpectralMethodBase
from    ..utils             import  *


##################################################
##################################################
__all__: list[str] = ['FastSM_FPL_VHS']


##################################################
##################################################
class FastSM_FPL_VHS(SpectralMethodBase):
    """## The class for the fast spectral method for solving the homogeneous Fokker-Planck-Landau equation with the VHS model.
    
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
            num_grid:   int,
            v_max:      float,
            
            vhs_coeff:  float,
            vhs_alpha:  float,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,
        ) -> Self:
        super().__init__(
            dim         = dim,
            num_grid    = num_grid,
            v_max       = v_max,
            freqs_keepdims  = True,
        )   # NOTE Set `freqs_keepdims=True` for the vectorized implementation
        self._vhs_coeff:    float   = vhs_coeff
        self._vhs_alpha:    float   = vhs_alpha
        self._quad_order_uniform:   int = quad_order_uniform    \
            if isinstance(quad_order_uniform,   int) else num_grid
        self._quad_order_legendre:  int = quad_order_legendre   \
            if isinstance(quad_order_legendre,  int) else num_grid
        self._quad_order_lebedev:   int = quad_order_lebedev    \
            if isinstance(quad_order_lebedev,   int) else num_grid
        self._fpl_character_1:  Optional[np.ndarray] = None
        self._fpl_character_2:  Optional[np.ndarray] = None
        self._fpl_gain_neagative:   Optional[np.ndarray] = None
        
        # Add the batch dimension for ease of implementation
        self._freqs:        np.ndarray = self._freqs[None, ...]
        self._freq_norms:   np.ndarray = np.linalg.norm(self._freqs, ord=2, axis=-1)
        
        # Character functions and tensors for the gain and loss terms
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
    def radius(self) -> float:
        return self._v_max * LAMBDA_FPL
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
    def fpl_gain_tensor_1(self) -> Optional[np.ndarray]:
        """Returns the tensor of the positive scaling part of the gain term.
        ### Note
        This tensor is a `self.dim`-tensor of shape `repeat(self.num_grid, seld.dim)`
        For each frequency mode `k`, the entry of index `k` corresponds to the entry of the first gain tensor of index `k`.
        """
        return self._fpl_gain_tensor_1[0]   # Do not remove the indexing `[0]`
    @property
    def fpl_gain_tensor_2(self) -> Optional[np.ndarray]:
        """Returns the tensor of the negative bilinear part of the gain term.
        ### Note
        This tensor is a `self.dim+2`-tensor of shape `(*repeat(self.num_grid, seld.dim), 2, 2)`.
        For each frequency mode `k`, the entry of index `k` corresponds to the entry of the second gain tensor of index `k`.
        """
        return self._fpl_gain_tensor_2[0]   # Do not remove the indexing `[0]`
    @property
    def fpl_loss_tensor(self) -> Optional[np.ndarray]:
        """Returns the tensor of the loss term.
        ### Note
        The tensor is `self.dim`-dimensional of shape `repeat(self.num_grid, seld.dim)`.
        For each frequency mode `k`, the entry of index `k` corresponds to the entry of the loss tensor of index `k`.
        """
        return self._fpl_loss_tensor[0]     # Do not remove the indexing `[0]`
    
    
    @override
    def precompute(self) -> None:
        getattr(self, f"_precompute_fpl_character_1_{ self._dim}D")()
        getattr(self, f"_precompute_fpl_character_2_{ self._dim}D")()
        getattr(self, f"_precompute_fpl_gain_tensors_{self._dim}D")()
        self._precompute_fpl_loss_tensor()
        return
    
    
    # 2-dimensional precomputations
    def _precompute_fpl_character_1_2D(self) -> None:
        # Define variables
        _len_norms  = int(self._dim * (self._num_grid//2)**2 + 1)
        norms       = np.sqrt(np.arange(_len_norms))[:, None]
        r, w = roots_legendre_shifted(self._quad_order_legendre, 0, np.pi)
        r: np.ndarray = r.reshape(1, -1)
        w: np.ndarray = w.reshape(1, -1)
        
        # Compute the integrand
        _scale  = float((2*np.pi*self._vhs_coeff) / np.power(self.v_ratio, 2+self._vhs_alpha))
        _power: np.ndarray  = np.power(r, 3+self._vhs_alpha)
        _fcn:   np.ndarray  = j0(norms * r)
        integrand: np.ndarray = _scale * _power * _fcn
        
        # Compute the integral
        self._fpl_character_1 = np.sum(w*integrand, axis=-1)
        return
    
    
    def _precompute_fpl_character_2_2D(self) -> None:
        scale = float(self._vhs_coeff / np.power(self.v_ratio, 2+self._vhs_alpha))
        
        _len_norms = int(self._dim * (self._num_grid//2)**2 + 1)
        norms   = np.sqrt(np.arange(_len_norms))[:, None]
        r, w    = roots_legendre_shifted(self._quad_order_legendre, 0, np.pi)
        r       = r.reshape((1, -1))
        w       = w.reshape((1, -1))
        _power: np.ndarray = r**(3+self._vhs_alpha)
        _func:  np.ndarray = _fpl_character_2__weight_C(r * norms, self._quad_order_uniform)
        integrand: np.ndarray = _power * _func
        
        self._fpl_character_2 = scale * np.sum(integrand*w, axis=-1)
        return
        
    
    def _precompute_fpl_gain_tensors_2D(self) -> None:
        """This function returns two tensors `P` and `Q`, which are defined in the reference.
        Remark that `P` is a `1+self.dim`-tensor of shape `(1, *repeat(self.num_grid, self.dim))`, while `Q` is a `1+self.dim+2`-tensor of shape `(1, *repeat(self.num_grid, self.dim), 2, 2)`."""
        _freq_norms_sq: np.ndarray  = np.sum(self._freqs**2, axis=-1)
        F = self._fpl_character_1[_freq_norms_sq]
        G = self._fpl_character_2[_freq_norms_sq]
        
        Q = np.zeros((1, *repeat(self._num_grid, self._dim), *repeat(self._dim, 2)))
        double_cos: np.ndarray
        double_sin: np.ndarray
        with np.errstate(divide='ignore', invalid='ignore'):
            _i = self._freqs[..., 0]
            _j = self._freqs[..., 1]
            double_cos = (_i**2 - _j**2) / _freq_norms_sq
            double_sin = (2 * _i*_j)     / _freq_norms_sq
            double_cos[zeros(1+self._dim)] = 0  # 1 for the batch dimension
            double_sin[zeros(1+self._dim)] = 0  # 1 for the batch dimension
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
        coeff = float((4*np.pi*self._vhs_coeff) / np.power(self.v_ratio, 3+self._vhs_alpha))
        
        _len_norms = int(1 + self._dim * ((self._num_grid//2)**2))
        norms: np.ndarray = np.sqrt(np.arange(_len_norms))[:, None]
        r, w    = roots_legendre_shifted(self._quad_order_legendre, 0, np.pi)
        r: np.ndarray = r.reshape((1, -1))
        w: np.ndarray = w.reshape((1, -1))
        _power: np.ndarray = np.power(r, 4+self._vhs_alpha)
        _func:  np.ndarray = sinc(norms * r)
        integrand: np.ndarray = _power * _func
        
        gain_tensor_1 = coeff * np.sum(integrand*w, axis=-1)
        
        _freq_norms_sq: np.ndarray = np.sum(self._freqs**2, axis=-1)
        self._fpl_gain_tensor_1 = gain_tensor_1[_freq_norms_sq]
        return
    
    
    def __precompute_fpl_gain_tensors_3D__negative(self) -> None:
        self._fpl_gain_tensor_2 = np.zeros((1, *repeat(self._num_grid, self._dim), 3, 3))
        
        _coeff_common = float(self._vhs_coeff / np.power(self.v_ratio, 3+self._vhs_alpha))
        freqs_xy:       np.ndarray = self._freqs[..., :2]
        freqs_z:        np.ndarray = self._freqs[..., [2]]
        freq_norms_xy:  np.ndarray = np.linalg.norm(freqs_xy, ord=2, axis=-1, keepdims=True)
        
        r, w = roots_legendre_shifted(self._quad_order_legendre, 0, np.pi)
        r:      np.ndarray = r.reshape((*ones(freqs_xy.ndim), -1))
        w:      np.ndarray = w.reshape((*ones(freqs_xy.ndim), -1))
        _power: np.ndarray = np.power(r, 4+self._vhs_alpha)
        
        # <<< (0, 1) entry (nondiagonal) >>> 
        _entry_01_scale: np.ndarray
        with np.errstate(divide='ignore', invalid='ignore'):
            _entry_01_scale = \
                _coeff_common * \
                freqs_xy.prod(axis=-1, keepdims=True) / \
                np.power(freq_norms_xy, 2)
        _entry_01_scale[zeros(self._dim)] = 0  # 1 for the batch dimension
        _entry_01_weight: np.ndarray = _fpl_character_2_3D__entry01_weight(
            a   = freqs_z,
            b   = freq_norms_xy,
            scale_factor    = r,
            quad_order_uniform = self._quad_order_uniform,
        )
        entry_01_integrand: np.ndarray = \
            _entry_01_scale * _power * _entry_01_weight
        del(_entry_01_scale, _entry_01_weight)
        entry_01: np.ndarray = np.sum(entry_01_integrand*w, axis=-1)
        
        # <<< (2, 2) entry (diagonal) >>>
        _entry_22_scale:     float = _coeff_common * (2*np.pi)
        _entry_22_weight:   np.ndarray = _fpl_character_2_3D__entry22_weight(
            a   = freqs_z,
            b   = freq_norms_xy,
            scale_factor    = r,
            quad_order_uniform = self._quad_order_uniform,
        )
        entry_22_integrand: np.ndarray = _entry_22_scale * _power * _entry_22_weight
        del(_power, _entry_22_weight)
        entry_22: np.ndarray = np.sum(entry_22_integrand*w, axis=-1)
        
        # Reshape the tensors
        _reduce = tuple((0, ...))
        entry_01 = entry_01[_reduce]
        entry_22 = entry_22[_reduce]
        
        # Fill the main entries
        self._fpl_gain_tensor_2[_reduce, 0, 1] = entry_01
        self._fpl_gain_tensor_2[_reduce, 2, 2] = entry_22
        
        # Fill the remaining entries using symmetry (diagonal)
        self._fpl_gain_tensor_2[_reduce, 0, 0] = entry_22.transpose((1,2,0))
        self._fpl_gain_tensor_2[_reduce, 1, 1] = entry_22.transpose((2,0,1))
        
        # Fill the remaining entries using symmetry (nondiagonal)
        self._fpl_gain_tensor_2[_reduce, 0, 2] = entry_01.transpose((2,0,1))
        self._fpl_gain_tensor_2[_reduce, 1, 2] = entry_01.transpose((1,2,0))
        self._fpl_gain_tensor_2[_reduce, 1, 0] = self._fpl_gain_tensor_2[_reduce, 0, 1]
        self._fpl_gain_tensor_2[_reduce, 2, 0] = self._fpl_gain_tensor_2[_reduce, 0, 2]
        self._fpl_gain_tensor_2[_reduce, 2, 1] = self._fpl_gain_tensor_2[_reduce, 1, 2]
        
        return
    
    
    def _precompute_fpl_loss_tensor(self) -> None:
        _EINSUM_V = EINSUM_STRING[:self._dim]
        positive: np.ndarray = np.power(self._freq_norms, 2) * self._fpl_gain_tensor_1
        negative: np.ndarray = np.einsum(
            f"b{_EINSUM_V}i,b{_EINSUM_V}ij,b{_EINSUM_V}j->b{_EINSUM_V}",
            self._freqs, self._fpl_gain_tensor_2, self._freqs
        )
        self._fpl_loss_tensor = positive - negative
        return
    
    
    @override
    def compute_collision_fft(
            self,
            _PLACEHOLDER__t_curr:   float,
            fft_curr:               np.ndarray,
        ) -> np.ndarray:
        # NOTE: Be careful to implement the truncated 'linear' convolution.        
        gain_fft_positive: np.ndarray = \
            convolve_freqs(
                (self._freq_norms**2) * fft_curr,
                self._fpl_gain_tensor_1 * fft_curr,
                axes    = self.v_axes,
            )
        gain_fft_negative: np.ndarray = \
            np.sum(
                np.stack(
                    [
                        convolve_freqs(
                            self._freqs[..., i] * self._freqs[..., j] * fft_curr,
                            self._fpl_gain_tensor_2[..., i, j] * fft_curr,
                            axes    = self.v_axes,
                        )
                        for i in range(self._dim) for j in range(self._dim)
                    ], axis=-1
                ), axis=-1
            )
        loss_fft:  np.ndarray = \
            convolve_freqs(
                fft_curr,
                fft_curr * self._fpl_loss_tensor,
                axes = self.v_axes,
            )
        return -(gain_fft_positive - gain_fft_negative - loss_fft)


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
    r, w = roots_legendre_shifted(quad_order_uniform, 0, 2*np.pi)
    r = r.reshape(*ones(x.ndim), -1)
    w = w.reshape(*ones(x.ndim), -1)
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
    t, w = roots_legendre_shifted(quad_order_uniform, 0, np.pi)
    t = t.reshape(*ones(_ndim), -1)
    w = w.reshape(*ones(_ndim), -1)
    
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
    t, w = roots_legendre_shifted(quad_order_uniform, 0, np.pi)
    t = t.reshape(*ones(_ndim), -1)
    w = w.reshape(*ones(_ndim), -1)
    
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