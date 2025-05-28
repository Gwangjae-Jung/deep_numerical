r"""## The implementation of the fast spectral method for solving the homogeneous Boltzmann equation

-----
### Description
Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.

Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
"""
from    typing              import  Optional
from    typing_extensions   import  Self, override

import  numpy               as      np
from    scipy.special       import  j0
from    scipy.integrate     import  lebedev_rule

from    .base_classes       import  FastSM_Base
from    .Boltzmann_VHS      import  Boltzmann_VHS_kernel_modes
from    ..utils             import  freq_tensor, roots_circle, roots_legendre_shifted, sinc, ones


##################################################
##################################################
__all__ = [
    'FastSM_Boltzmann_VHS',
]
   
    
##################################################
##################################################
class FastSM_Boltzmann_VHS(FastSM_Base):
    r"""## The class for the fast spectral method for solving the homogeneous Boltzmann equation with the VHS model.
    
    -----
    ### Description
    Based on the rewriting of the expression of the kernel modes, this module aims to implement the fast spectral method by exploiting a proper quadrature rule for the double integral on `[0, 2R] \times S^{d-1}`, where `R` is the radius of the support of the density function in the velocity space and `d` is the dimension of the domain.
    
    Reference: [Irene M. Gamba, Jeffrey R. Haack, Cory D. Hauck, and Jingwei Hu, A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels, SIAM Journal on Scientific Computing, Volume 39, Issue 1, 2017, Pages B658-B674](https://epubs.siam.org/doi/10.1137/16M1096001)
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
            quad_order_uniform  = quad_order_uniform,
            quad_order_legendre = quad_order_legendre,
            quad_order_lebedev  = quad_order_lebedev,
        )        
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
    
    
    @override
    def _precompute_fsm_gain(self) -> tuple[np.ndarray, np.ndarray]:
        return getattr(self, f'_precompute_fsm_gain_{self.dim}D')()
    @override
    def _precompute_fsm_loss(self) -> np.ndarray:
        return Boltzmann_VHS_kernel_modes(
            dim             = self._dim,
            num_grid        = self._num_grid,
            v_max           = self._v_max,
            vhs_coeff       = self._vhs_coeff,
            vhs_alpha       = self._vhs_alpha,
            num_roots       = self._quad_order_legendre,
            diagonal_only   = True,
        ).astype(np.complex128)[None, ...]
    
    
    def _precompute_fsm_gain_2D(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ### Note
        In this implementation, the tensors are reshaped to satisfy the following ordering:
            * `*freq`: 3 dimensions, size 1 or `num_grid`, where the latter size is to consider the sum or difference of two frequencies.
            * `radius`: 1 dimension, size 1 or `quad_order_legendre`.
            * `*circle`: 1 dimension, size 1 or `quad_order_uniform`.
            * value: 1 dimension, variable length depending on the type of data to be saved (size 1 is allowed).
        Do not forget to set `keepdims=True` when computing, for example, summations.
        """
        # Get the frequency tensor
        _freqs_shape_new = (*(self._num_grid for _ in range(self.dim)), *ones(2), self.dim)
        freqs = freq_tensor(self._dim, self._num_grid, True)
        freqs = freqs.reshape(_freqs_shape_new)
        
        # Get the reshaped arrays of the values of `radius, circle`
        r_roots, r_weights  = roots_legendre_shifted(self._quad_order_legendre, 0, 2*self.radius)
        s_roots, s_weights  = roots_circle(self.quad_order_uniform)
        r_roots = r_roots.reshape((*ones(self.dim), -1, 1, 1))
        s_roots = s_roots.reshape((*ones(self.dim), 1, -1, 2))
        r_weights = r_weights.reshape((*ones(self.dim), -1, 1, 1))
        s_weights = s_weights.reshape((*ones(self.dim), 1, -1, 1))
        
        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_scale_coeff    = (2*np.pi) * self.vhs_coeff
        _fsm_scale_power    = np.power(r_roots, self.vhs_alpha+self.dim-1)
        _fsm_scale_fcn      = j0(
                                    (0.5 * np.pi / self.v_max) * \
                                    r_roots * \
                                    np.sqrt((freqs**2).sum(-1, keepdims=True))
                                )
        fsm_scale = _fsm_scale_coeff * _fsm_scale_power * _fsm_scale_fcn
        fsm_scale = fsm_scale * r_weights
        fsm_scale = np.squeeze(fsm_scale, axis=-1)
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        _fsm_arg = \
                    (0.5 * np.pi / self.v_max) * \
                    (freqs * r_roots * s_roots).sum(-1, keepdims=True)
        fsm_phase = np.exp(1j * _fsm_arg) * np.sqrt(s_weights)
        fsm_phase = np.squeeze(fsm_phase, axis=-1)
        
        # Reshape and return the output tensors
        fsm_scale = fsm_scale[None, ...]
        fsm_phase = fsm_phase[None, ...]
        return (fsm_scale, fsm_phase)
    
    
    def _precompute_fsm_gain_3D(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ### Note
        In this implementation, the tensors are reshaped to satisfy the following ordering:
        * `*freq`: 3 dimensions, size 1 or `num_grid`.
        * `radius`: 1 dimension, size 1 or `quad_order_legendre`.
        * `*sphere`: 1 dimension, size 1 or `quad_order_lebedev`.
        * value: 1 dimension, variable length depending on the type of data to be saved (size 1 is allowed).
        """
        # Get the frequency tensor
        _n = self.num_grid
        _freqs_shape_new = (*(_n for _ in range(self.dim)), *ones(2), self.dim)
        freqs = freq_tensor(self.dim, _n, True)
        freqs = freqs.reshape(_freqs_shape_new)
        
        # Get the reshaped arrays of the values of `radius, phi, theta`
        r_roots, r_weights  = roots_legendre_shifted(self.quad_order_legendre, 0, 2*self.radius)
        s_roots, s_weights  = lebedev_rule(self.quad_order_lebedev)
        s_roots = s_roots.transpose(1, 0)
        s_weights:  np.ndarray
        r_roots = r_roots.reshape((*ones(self.dim), -1, 1, 1))
        s_roots = s_roots.reshape((*ones(self.dim), 1, -1, 3))
        r_weights = r_weights.reshape((*ones(self.dim), -1, 1, 1))
        s_weights = s_weights.reshape((*ones(self.dim), 1, -1, 1))
                
        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_scale_coeff    = (4*np.pi) * self.vhs_coeff
        _fsm_scale_power    = np.power(r_roots, self.vhs_alpha+self.dim-1)
        _fsm_scale_fcn      = sinc(
                                    (0.5 * np.pi / self.v_max) * \
                                    r_roots * \
                                    np.linalg.norm(freqs, ord=2, axis=-1, keepdims=True)
                                )
        fsm_scale = _fsm_scale_coeff * _fsm_scale_power * _fsm_scale_fcn
        fsm_scale = np.squeeze(fsm_scale * r_weights, axis=-1)
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        ## NOTE: Some weights in the Lebedev rule might be negative.
        _fsm_arg = \
                    (0.5 * np.pi / self.v_max) * \
                    (freqs * r_roots * s_roots).sum(-1, keepdims=True)
        fsm_phase = np.exp(1j * _fsm_arg) * np.sqrt(s_weights, dtype=np.complex128)
        fsm_phase = np.squeeze(fsm_phase, axis=-1)
        
        # Reshape and return the result
        fsm_scale = fsm_scale[None, ...]
        fsm_phase = fsm_phase[None, ...]
        return (fsm_scale, fsm_phase)

    
##################################################
##################################################
# End of file