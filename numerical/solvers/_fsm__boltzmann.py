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

from    .base_classes       import  FastSM_Boltzmann
from    .Boltzmann_VHS      import  Boltzmann_VHS_kernel_modes
from    ..                  import  utils


##################################################
##################################################
__all__ = [
    'FastSM_Boltzmann_VHS',
]
   
    
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
            
            dim:        int,
            
            v_num_grid: int,
            v_max:      float,
            
            x_num_grid: Optional[int]   = None,
            x_max:      Optional[float] = None,
            
            vhs_coeff:  Optional[float] = None,
            vhs_alpha:  Optional[float] = None,
            
            quad_order_uniform:     Optional[int] = None,
            quad_order_legendre:    Optional[int] = None,
            quad_order_lebedev:     Optional[int] = None,
        ) -> Self:
        super().__init__(
            dim         = dim,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            x_num_grid  = x_num_grid,
            x_max       = x_max,
            quad_order_uniform  = quad_order_uniform,
            quad_order_legendre = quad_order_legendre,
            quad_order_lebedev  = quad_order_lebedev,
        )
        if vhs_alpha is None or vhs_alpha is None:
            raise ValueError(f"'vhs_coeff' and 'vhs_alpha' should be given. ({vhs_alpha=:.2e}, {vhs_coeff=:.2f})")     
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
            num_grid        = self._v_num_grid,
            v_max           = self._v_max,
            vhs_coeff       = self._vhs_coeff,
            vhs_alpha       = self._vhs_alpha,
            num_roots       = self._quad_order_legendre,
            diagonal_only   = True,
        ).astype(np.complex128)[*utils.repeat(None, 1+self._dim), ..., None]  # Append one extra dimension
    
    
    def _precompute_fsm_gain_2D(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ### Note
        Here, we order data in the following order:
            `(batch, space, velocity, data, *quadrature_rules)`
        """
        """
        Commnets
        
        1. The tensor of the points sampled uniformly on $S^1$ is initially saved in a tensor of shape `(quad_order_uniform, dim)`. As this tensor has to be reshaped to a tensor of shape `(1, *ones(dim), *ones(dim), dim, *(1, quad_order_uniform))`, transposition is required.
        """
        # Define the shapes
        _freqs_reshape:     tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.repeat(self._v_num_grid, self._dim), self._dim,
                *utils.ones(2)  )
        _r_roots_reshape:   tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), 1,
                *(self._quad_order_legendre, 1) )
        _r_weights_reshape: tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), 1,
                *(self._quad_order_legendre, 1) )
        _s_roots_reshape:   tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), self._dim,
                *(1, self._quad_order_uniform)  )
        _s_weights_reshape: tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), 1,
                *(1, self._quad_order_uniform)  )
        
        # Get the frequency tensor
        freqs = utils.freq_tensor(self._dim, self._v_num_grid, True)
        freqs = freqs.reshape(_freqs_reshape)
        
        # Get the reshaped arrays of the values of `radius, circle`
        r_roots, r_weights  = utils.roots_legendre_shifted(self._quad_order_legendre, 0, 2*self.v_radius)
        s_roots, s_weights  = utils.roots_circle(self.quad_order_uniform)
        s_roots = s_roots.T     # See `Comments 1`.
        r_roots:   np.ndarray = r_roots.reshape(_r_roots_reshape)
        s_roots:   np.ndarray = s_roots.reshape(_s_roots_reshape)
        r_weights: np.ndarray = r_weights.reshape(_r_weights_reshape)
        s_weights: np.ndarray = s_weights.reshape(_s_weights_reshape)
        
        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_scale_coeff    = (2*np.pi) * self.vhs_coeff
        _fsm_scale_power    = np.power(r_roots, self.vhs_alpha+self.dim-1)
        _fsm_scale_fcn      = j0(
                                    (0.5 * self.v_ratio) * \
                                    r_roots * \
                                    np.sqrt((freqs**2).sum(-3, keepdims=True))
                                )
        fsm_scale = _fsm_scale_coeff * _fsm_scale_power * _fsm_scale_fcn
        fsm_scale = fsm_scale * r_weights
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        _fsm_arg = \
                    (0.5 * self.v_ratio) * \
                    (freqs * r_roots * s_roots).sum(-3, keepdims=True)
        fsm_phase = np.exp(1j * _fsm_arg) * np.sqrt(s_weights)
               
        # Reshape and return the output tensors
        return (fsm_scale, fsm_phase)
    
    
    def _precompute_fsm_gain_3D(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Here, we order data in the following order:
            `(batch, space, velocity, data, *quadrature_rules)`
        """
        """
        Comments
        
        1. The tensor of the Lebedev quarature points on $S^2$ is initially saved in a tensor of shape `(dim, -1)`, unlike the tensor of the quadrature points on `S^1` in the 2-dimensioal case. Thus, no transposition is required in this case.
        """
        # Define the shapes
        _freqs_reshape:     tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.repeat(self._v_num_grid, self._dim), self.dim,
                *utils.ones(2)  )
        _r_roots_reshape:   tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), 1,
                *(self._quad_order_legendre, 1) )
        _r_weights_reshape: tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), 1,
                *(self._quad_order_legendre, 1) )
        _s_roots_reshape:   tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), self._dim,
                *(1, -1)  )
        _s_weights_reshape: tuple[int] = \
            (   1, *utils.ones(self._dim), *utils.ones(self._dim), 1,
                *(1, -1)  )
        
        # Get the frequency tensor
        freqs = utils.freq_tensor(self.dim, self._v_num_grid, True)
        freqs = freqs.reshape(_freqs_reshape)
        
        # Get the reshaped arrays of the values of `radius, phi, theta`
        r_roots, r_weights  = utils.roots_legendre_shifted(self.quad_order_legendre, 0, 2*self.v_radius)
        s_roots, s_weights  = lebedev_rule(self.quad_order_lebedev) # See `Comments 1`.
        r_roots:   np.ndarray = r_roots.reshape(_r_roots_reshape)
        s_roots:   np.ndarray = s_roots.reshape(_s_roots_reshape)
        r_weights: np.ndarray = r_weights.reshape(_r_weights_reshape)
        s_weights: np.ndarray = s_weights.reshape(_s_weights_reshape)
        
        # Part 1: Compute the scaling part (`fsm_scale`)
        _fsm_scale_coeff    = (4*np.pi) * self.vhs_coeff
        _fsm_scale_power    = np.power(r_roots, self.vhs_alpha+self.dim-1)
        _fsm_scale_fcn      = utils.sinc(
                                    (0.5 * self.v_ratio) * \
                                    r_roots * \
                                    np.linalg.norm(freqs, ord=2, axis=-3, keepdims=True)
                                )
        fsm_scale = _fsm_scale_coeff * _fsm_scale_power * _fsm_scale_fcn * r_weights
        
        # Part 2: Compute the multiplicative part (`fsm_phase`)
        ## NOTE: Some weights in the Lebedev rule might be negative.
        _fsm_arg = \
                    (0.5 * self.v_ratio) * \
                    (freqs * r_roots * s_roots).sum(-3, keepdims=True)
        fsm_phase = np.exp(1j * _fsm_arg) * np.sqrt(s_weights, dtype=np.complex128)
                
        # Reshape and return the result
        # np.savez("numpy_fsm.npz", fsm_scale=fsm_scale, fsm_phase=fsm_phase)
        return (fsm_scale, fsm_phase)

    
##################################################
##################################################
# End of file