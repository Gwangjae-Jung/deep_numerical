"""Conservative semi-Lagrangian transport and time discretization schemes for kinetic equations.

-----
### Description
This module implements the high-order conservative semi-Lagrangian (SL) method for the
inhomogeneous Boltzmann equation and related kinetic transport equations.

Key components:
* Conservative spatial reconstruction using QCWENO polynomials (CWENO35, CWENO23, and linear).
* Multi-dimensional dimension-by-dimension characteristic tracing.
* Weighted and unweighted L2-minimization for exact moment conservation.
* High-order Runge-Kutta (RK3, RK4), Adams-Bashforth (AB3, AB4), Strang splitting, and Euler SL schemes.

-----
### References
[1] S. Boscarino, S. Y. Cho, and G. Russo, A conservative semi-Lagrangian method for inhomogeneous
    Boltzmann equation, Journal of Computational Physics, 498 (2024), 112633.
[2] S. Y. Cho, S. Boscarino, G. Russo, and S.-B. Yun, Conservative semi-Lagrangian schemes for
    kinetic equations part I: reconstruction, Journal of Computational Physics, 432 (2021), 110159.
[3] S. Boscarino, S. Y. Cho, and G. Russo, A local velocity grid conservative semi-Lagrangian
    scheme for BGK model, Journal of Computational Physics, 460 (2022), 111178.
"""
from   typing import Callable, Literal, Optional, Sequence, Tuple
import torch

from   deep_numerical import ones, repeat, zeros


__all__: list[str] = [
    'reconstruct_1d_conservative',
    'semi_lagrangian_transport',
    'moment_correction_l2',
    'compute_fluid_moments',
    'one_step_SL_Euler',
    'one_step_SL_RK3',
    'one_step_SL_RK4',
    'one_step_SL_Strang',
    'one_step_SL_AB3',
    'one_step_SL_AB4',
]

# Numerical tolerances and constants
EPSILON_WENO: float = 1e-6
"""Numerical epsilon to prevent division by zero in WENO weights."""

# CWENO linear weights for central and directional stencils
CWENO_C1:     float = 0.125
CWENO_C2:     float = 0.250
CWENO_C3:     float = 0.125
CWENO_CC:     float = 0.500


##################################################
def reconstruct_1d_conservative(
    u:     torch.Tensor,
    xi:    torch.Tensor,
    order: Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> torch.Tensor:
    """Evaluates the 1D conservative reconstruction of cell-averaged quantities along dimension 1.

    ## Description
    Given a tensor `u` whose target spatial dimension is located at `dim=1` with $N_x$ grid cells,
    and normalized characteristic foot coordinates $\\xi = (x_{foot} - x_{min}) / \\Delta x$,
    this function computes the high-order sliding-average conservative reconstruction
    guaranteeing global discrete conservation of $\\sum_i u_i$.

    ## Arguments
    `u` (`torch.Tensor`): Input tensor with the target spatial axis at dimension 1.
    `xi` (`torch.Tensor`): Normalized coordinates along the target spatial axis, broadcastable to `u`.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Order of reconstruction.

    ## Returns
    `torch.Tensor`: The conservatively reconstructed tensor of the same shape as `u`.
    """
    num_grid_x: int          = u.size(1)
    icell:      torch.Tensor = (torch.floor(xi) % num_grid_x).long()
    icell_p1:   torch.Tensor = (icell + 1) % num_grid_x
    theta:      torch.Tensor = xi - torch.floor(xi)

    if order == 'linear':
        u_icell:    torch.Tensor = torch.take_along_dim(u, icell.expand_as(u),    dim=1)
        u_icell_p1: torch.Tensor = torch.take_along_dim(u, icell_p1.expand_as(u), dim=1)
        return (1.0 - theta) * u_icell + theta * u_icell_p1

    im2: torch.Tensor = torch.roll(u, shifts=2,  dims=1)
    im1: torch.Tensor = torch.roll(u, shifts=1,  dims=1)
    i0:  torch.Tensor = u
    ip1: torch.Tensor = torch.roll(u, shifts=-1, dims=1)
    ip2: torch.Tensor = torch.roll(u, shifts=-2, dims=1)

    # Smoothness indicators for quadratic stencils P1, P2, P3
    beta1: torch.Tensor = (13.0 / 12.0) * (im2 - 2.0 * im1 + i0)**2  + (0.25) * (im2 - 4.0 * im1 + 3.0 * i0)**2
    beta2: torch.Tensor = (13.0 / 12.0) * (im1 - 2.0 * i0  + ip1)**2 + (0.25) * (im1 - ip1)**2
    beta3: torch.Tensor = (13.0 / 12.0) * (i0  - 2.0 * ip1 + ip2)**2 + (0.25) * (3.0 * i0 - 4.0 * ip1 + ip2)**2

    t:      torch.Tensor = 2.0 * theta - 1.0
    alpha0: torch.Tensor = (1.0 - t) / 2.0
    beta_0: torch.Tensor = (t + 1.0) / 2.0
    alpha1: torch.Tensor = (1.0 - t**2) / 8.0
    beta_1: torch.Tensor = -alpha1
    alpha2: torch.Tensor = (1.0 - t**3) / 48.0
    beta_2: torch.Tensor = (t**3 + 1.0) / 48.0

    if order == 'cweno23':
        # 3rd order CWENO reconstruction (CWENO23)
        c1_w23:     float        = 0.25
        c2_w23:     float        = 0.50
        c3_w23:     float        = 0.25
        a1:         torch.Tensor = c1_w23 / (EPSILON_WENO + beta1)**2
        a2:         torch.Tensor = c2_w23 / (EPSILON_WENO + beta2)**2
        a3:         torch.Tensor = c3_w23 / (EPSILON_WENO + beta3)**2
        sum_a:      torch.Tensor = a1 + a2 + a3
        w1:         torch.Tensor = a1 / sum_a
        w2:         torch.Tensor = a2 / sum_a
        w3:         torch.Tensor = a3 / sum_a

        r0: torch.Tensor = w1 * ((23.0 / 24.0) * i0 + (1.0 / 12.0) * im1 - (1.0 / 24.0) * im2) \
                         + w2 * ((13.0 / 12.0) * i0 - (1.0 / 24.0) * (im1 + ip1)) \
                         + w3 * ((23.0 / 24.0) * i0 + (1.0 / 12.0) * ip1 - (1.0 / 24.0) * ip2)
        r1: torch.Tensor = w1 * ((3.0 * i0 - 4.0 * im1 + im2) / 2.0) \
                         + w2 * ((ip1 - im1) / 2.0) \
                         + w3 * ((3.0 * i0 - 4.0 * ip1 + ip2) / (-2.0))
        r2: torch.Tensor = 2.0 * (w1 * ((i0 - 2.0 * im1 + im2) / 2.0)
                                + w2 * ((ip1 - 2.0 * i0 + im1) / 2.0)
                                + w3 * ((i0 - 2.0 * ip1 + ip2) / 2.0))

        r0_i:   torch.Tensor = torch.take_along_dim(r0, icell.expand_as(r0),    dim=1)
        r0_ip1: torch.Tensor = torch.take_along_dim(r0, icell_p1.expand_as(r0), dim=1)
        r1_i:   torch.Tensor = torch.take_along_dim(r1, icell.expand_as(r1),    dim=1)
        r1_ip1: torch.Tensor = torch.take_along_dim(r1, icell_p1.expand_as(r1), dim=1)
        r2_i:   torch.Tensor = torch.take_along_dim(r2, icell.expand_as(r2),    dim=1)
        r2_ip1: torch.Tensor = torch.take_along_dim(r2, icell_p1.expand_as(r2), dim=1)

        return (alpha0 * r0_i + beta_0 * r0_ip1) \
             + (alpha1 * r1_i + beta_1 * r1_ip1) \
             + (alpha2 * r2_i + beta_2 * r2_ip1)

    # CWENO35: 5th-order conservative reconstruction (Boscarino, Cho, Russo 2024, Appendix A)
    u_tilde_1: torch.Tensor = (17.0 / 24.0) * (ip1 - im1) + (5.0 / 48.0) * (im2 - ip2)
    u_tilde_2: torch.Tensor = (-1.0 / 16.0) * (im2 + 22.0 * i0 + ip2) + (0.75) * (ip1 + im1)
    u_tilde_3: torch.Tensor = (-1.0 / 6.0)  * (ip1 - im1) - (1.0 / 12.0) * (im2 - ip2)
    u_tilde_4: torch.Tensor = (1.0 / 24.0)  * (im2 + 6.0 * i0 + ip2) - (1.0 / 6.0) * (ip1 + im1)

    a11: torch.Tensor = (3.0 * i0 - 4.0 * im1 + im2) / 2.0
    a12: torch.Tensor = (i0 - 2.0 * im1 + im2) / 2.0
    a21: torch.Tensor = (ip1 - im1) / 2.0
    a22: torch.Tensor = (ip1 - 2.0 * i0 + im1) / 2.0
    a31: torch.Tensor = (3.0 * i0 - 4.0 * ip1 + ip2) / (-2.0)
    a32: torch.Tensor = (i0 - 2.0 * ip1 + ip2) / 2.0

    a_c1: torch.Tensor = (u_tilde_1 - CWENO_C1 * a11 - CWENO_C2 * a21 - CWENO_C3 * a31) / CWENO_CC
    a_c2: torch.Tensor = (u_tilde_2 - CWENO_C1 * a12 - CWENO_C2 * a22 - CWENO_C3 * a32) / CWENO_CC
    a_c3: torch.Tensor = u_tilde_3 / CWENO_CC
    a_c4: torch.Tensor = u_tilde_4 / CWENO_CC

    beta_c: torch.Tensor = (a_c1**2
                           + (13.0 / 3.0) * a_c2**2
                           + (3129.0 / 80.0) * a_c3**2
                           + (87617.0 / 140.0) * a_c4**2
                           + 0.5 * a_c1 * a_c3
                           + 4.2 * a_c2 * a_c4)

    alpha_1: torch.Tensor = CWENO_C1 / (EPSILON_WENO + beta1)**2
    alpha_2: torch.Tensor = CWENO_C2 / (EPSILON_WENO + beta2)**2
    alpha_3: torch.Tensor = CWENO_C3 / (EPSILON_WENO + beta3)**2
    alpha_c: torch.Tensor = CWENO_CC / (EPSILON_WENO + beta_c)**2
    sum_alp: torch.Tensor = alpha_1 + alpha_2 + alpha_3 + alpha_c

    w_1: torch.Tensor = alpha_1 / sum_alp
    w_2: torch.Tensor = alpha_2 / sum_alp
    w_3: torch.Tensor = alpha_3 / sum_alp
    w_c: torch.Tensor = alpha_c / sum_alp

    # Moments R_i^(l)
    r_0: torch.Tensor = w_c * ((577.0 / 480.0) * i0 - (29.0 / 240.0) * im1 + (19.0 / 960.0) * im2
                              - (29.0 / 240.0) * ip1 + (19.0 / 960.0) * ip2) \
                      - w_2 * ((im1 - 26.0 * i0 + ip1) / 24.0) \
                      + w_1 * ((23.0 / 24.0) * i0 + (1.0 / 12.0) * im1 - (1.0 / 24.0) * im2) \
                      + w_3 * ((23.0 / 24.0) * i0 + (1.0 / 12.0) * ip1 - (1.0 / 24.0) * ip2)

    r_1: torch.Tensor = -w_c * ((8.0 * im1 - im2 - 8.0 * ip1 + ip2) / 12.0) \
                      + w_1 * ((3.0 * i0 - 4.0 * im1 + im2) / 2.0) \
                      - w_3 * ((3.0 * i0 - 4.0 * ip1 + ip2) / 2.0) \
                      - w_2 * ((im1 - ip1) / 2.0)

    r_2: torch.Tensor = 2.0 * (w_1 * ((i0 - 2.0 * im1 + im2) / 2.0)
                             + w_2 * ((im1 - 2.0 * i0 + ip1) / 2.0)
                             + w_3 * ((i0 - 2.0 * ip1 + ip2) / 2.0)) \
                      - 2.0 * w_c * ((10.0 * i0 - 6.0 * im1 + im2 - 6.0 * ip1 + ip2) / 4.0)

    r_3: torch.Tensor = 6.0 * w_c * ((im1 - ip1) / 3.0 - (im2 - ip2) / 6.0)
    r_4: torch.Tensor = 24.0 * w_c * ((im2 + 6.0 * i0 + ip2) / 12.0 - (im1 + ip1) / 3.0)

    alpha3 = (1.0 - t**4) / 384.0
    beta_3 = -alpha3
    alpha4 = (1.0 - t**5) / 3840.0
    beta_4 = (t**5 + 1.0) / 3840.0

    r0_i   = torch.take_along_dim(r_0, icell.expand_as(r_0),    dim=1)
    r0_ip1 = torch.take_along_dim(r_0, icell_p1.expand_as(r_0), dim=1)
    r1_i   = torch.take_along_dim(r_1, icell.expand_as(r_1),    dim=1)
    r1_ip1 = torch.take_along_dim(r_1, icell_p1.expand_as(r_1), dim=1)
    r2_i   = torch.take_along_dim(r_2, icell.expand_as(r_2),    dim=1)
    r2_ip1 = torch.take_along_dim(r_2, icell_p1.expand_as(r_2), dim=1)
    r3_i:   torch.Tensor = torch.take_along_dim(r_3, icell.expand_as(r_3),    dim=1)
    r3_ip1: torch.Tensor = torch.take_along_dim(r_3, icell_p1.expand_as(r_3), dim=1)
    r4_i:   torch.Tensor = torch.take_along_dim(r_4, icell.expand_as(r_4),    dim=1)
    r4_ip1: torch.Tensor = torch.take_along_dim(r_4, icell_p1.expand_as(r_4), dim=1)

    return (alpha0 * r0_i + beta_0 * r0_ip1) \
         + (alpha1 * r1_i + beta_1 * r1_ip1) \
         + (alpha2 * r2_i + beta_2 * r2_ip1) \
         + (alpha3 * r3_i + beta_3 * r3_ip1) \
         + (alpha4 * r4_i + beta_4 * r4_ip1)


##################################################
def semi_lagrangian_transport(
    f:        torch.Tensor,
    v_coords: Sequence[torch.Tensor],
    delta_t:  float,
    x_max:    float,
    x_min:    Optional[float]                          = None,
    order:    Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> torch.Tensor:
    """Computes the conservative semi-Lagrangian transport of a distribution function.

    ## Description
    Solves the free-streaming kinetic advection equation $\\partial_t f + v \\cdot \\nabla_x f = 0$
    for time increment `delta_t` along straight-line characteristics $x(t) = x_0 + v t$
    using dimension-by-dimension conservative reconstruction with periodic boundary conditions.

    ## Arguments
    `f` (`torch.Tensor`): Input distribution tensor of shape `(batch, *res_x, *res_v, channels)`.
    `v_coords` (`Sequence[torch.Tensor]`): List of 1D velocity coordinate tensors along each velocity dimension.
    `delta_t` (`float`): Time step increment for transport.
    `x_max` (`float`): Spatial upper boundary coordinate.
    `x_min` (`Optional[float]`, default: `None`): Spatial lower boundary coordinate. Defaults to `-x_max`.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Reconstruction order.

    ## Returns
    `torch.Tensor`: Transported distribution function with preserved discrete mass.
    """
    if abs(delta_t) < 1e-15:
        return f

    if isinstance(v_coords, torch.Tensor):
        dim: int = (f.ndim - 2) // 2
        coords_list: list[torch.Tensor] = [v_coords] * dim
    else:
        dim = len(v_coords)
        coords_list = list(v_coords)

    if x_min is None:
        x_min = -x_max
    dom_len: float = x_max - x_min

    f_curr: torch.Tensor = f
    for k in range(dim):
        num_grid_x: int = f_curr.size(1 + k)
        if num_grid_x <= 1:
            continue

        delta_x:  float        = dom_len / num_grid_x
        v_1d:     torch.Tensor = coords_list[k].to(device=f_curr.device, dtype=f_curr.dtype)
        num_v:    int          = v_1d.size(0)
        x_1d:     torch.Tensor = torch.linspace(x_min, x_max - delta_x, num_grid_x,
                                              device=f_curr.device, dtype=f_curr.dtype)

        # Characteristic foot: x_foot = x - v * delta_t (periodic wrap into [x_min, x_max))
        x_foot:   torch.Tensor = x_1d[:, None] - v_1d[None, :] * delta_t
        x_wrap:   torch.Tensor = (x_foot - x_min) % dom_len + x_min
        xi:       torch.Tensor = (x_wrap - x_min) / delta_x

        # Align active spatial and velocity dimensions to dimensions 1 and 2
        s_axis:   int          = 1 + k
        v_axis:   int          = 1 + dim + k
        f_perm:   torch.Tensor = f_curr.movedim((s_axis, v_axis), (1, 2))

        # Reshape xi to broadcast over remaining trailing dimensions
        trailing: tuple[int, ...] = tuple(ones(f_perm.ndim - 3))
        xi_resh:  torch.Tensor    = xi.reshape(1, num_grid_x, num_v, *trailing)

        f_recons: torch.Tensor = reconstruct_1d_conservative(f_perm, xi_resh, order=order)
        f_curr                 = f_recons.movedim((1, 2), (s_axis, v_axis))

    return f_curr


##################################################
def moment_correction_l2(
    q_physical: torch.Tensor,
    v_coords:   Sequence[torch.Tensor],
    delta_v:    float,
    weight:     Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Enforces exact conservation of mass, momentum, and energy on collision operator values.

    ## Description
    Given collision operator values `q_physical` in physical space, computes the constrained
    weighted L2-projection ensuring $(\\Delta v)^d \\sum_j Q_{i, j} \\phi_j = 0$
    for collision invariants $\\phi_j = [1, v_j, |v_j|^2/2]^\\top$.

    ## Arguments
    `q_physical` (`torch.Tensor`): Evaluated collision operator tensor in physical space.
    `v_coords` (`Sequence[torch.Tensor]`): 1D coordinate tensors of velocity grids.
    `delta_v` (`float`): Uniform velocity grid spacing.
    `weight` (`Optional[torch.Tensor]`, default: `None`): Optional positive weighting tensor.

    ## Returns
    `torch.Tensor`: Corrected collision operator tensor preserving all collision invariants.
    """
    if isinstance(v_coords, torch.Tensor):
        dim: int = (q_physical.ndim - 2) // 2
        coords_list: list[torch.Tensor] = [v_coords] * dim
    else:
        dim = len(v_coords)
        coords_list = list(v_coords)

    v_meshes: list[torch.Tensor] = torch.meshgrid(*coords_list, indexing='ij')
    v_stack:  torch.Tensor = torch.stack(v_meshes, dim=-1).to(device=q_physical.device, dtype=q_physical.dtype)
    v_flat:   torch.Tensor = v_stack.reshape(-1, dim)
    v_sq:     torch.Tensor = torch.sum(v_flat**2, dim=-1)
    num_pts:  int          = v_flat.size(0)

    dv_pow:   float        = float(delta_v ** dim)
    row_mass: torch.Tensor = torch.ones(num_pts, device=q_physical.device, dtype=q_physical.dtype) * dv_pow
    row_mom:  torch.Tensor = v_flat.T * dv_pow
    row_eng:  torch.Tensor = (0.5 * v_sq) * dv_pow
    c_mat:    torch.Tensor = torch.cat([row_mass[None, :], row_mom, row_eng[None, :]], dim=0)

    if weight is not None:
        w_flat:   torch.Tensor = weight.reshape(-1).to(device=q_physical.device, dtype=q_physical.dtype)
        c_mat                  = c_mat / (w_flat[None, :] + 1e-15)

    gram:     torch.Tensor = c_mat @ c_mat.T
    inv_gram: torch.Tensor = torch.linalg.pinv(gram)

    # Flatten velocity dimensions: (batch, *space, Nv_total, channels)
    v_axes_start: int          = 1 + dim
    v_axes_end:   int          = 1 + 2 * dim
    orig_shape:   torch.Size   = q_physical.shape
    batch_space:  int          = int(torch.prod(torch.tensor(orig_shape[:v_axes_start])))
    channels:     int          = orig_shape[-1]
    q_reshaped:   torch.Tensor = q_physical.reshape(batch_space, num_pts, channels)

    if weight is not None:
        q_proj:   torch.Tensor = q_reshaped * w_flat[None, :, None]
    else:
        q_proj                 = q_reshaped

    # c_mat: (d+2, num_pts), q_proj: (BS, num_pts, C)
    # moments: (BS, d+2, C) = einsum('mn, bnc -> bmc')
    moments:  torch.Tensor = torch.einsum('mn,bnc->bmc', c_mat, q_proj)
    lambda_v: torch.Tensor = torch.einsum('lm,bmc->blc', inv_gram, moments)
    corr:     torch.Tensor = torch.einsum('ml,blc->bmc', c_mat.T, lambda_v)

    if weight is not None:
        q_corr:   torch.Tensor = q_reshaped - corr / (w_flat[None, :, None] + 1e-15)
    else:
        q_corr                 = q_reshaped - corr

    return q_corr.reshape(orig_shape)


##################################################
def compute_fluid_moments(
    f:        torch.Tensor,
    v_coords: Sequence[torch.Tensor] | torch.Tensor,
    delta_v:  float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes macroscopic fluid moments (density, momentum, total energy) from distribution function f.

    ## Description
    Calculates the first three moments:
        \\rho(x)      = \\int f(x, v) dv
        (\\rho u)(x)  = \\int v f(x, v) dv
        E(x)          = 0.5 * \\int |v|^2 f(x, v) dv

    Arguments:
        `f` (`torch.Tensor`):
            Distribution function tensor shaped `(batch, *space, *velocity, channels)`.
        `v_coords` (`Sequence[torch.Tensor] | torch.Tensor`):
            Collection of 1D velocity coordinate tensors along each velocity dimension, or 1D velocity tensor if symmetric.
        `delta_v` (`float`):
            Uniform velocity grid spacing.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
            Tuple `(density, momentum, total_energy)` evaluated over spatial grid.
    """
    if isinstance(v_coords, torch.Tensor):
        dim: int = (f.ndim - 2) // 2
        coords_list: list[torch.Tensor] = [v_coords] * dim
    else:
        dim = len(v_coords)
        coords_list = list(v_coords)

    v_meshes = torch.meshgrid(*coords_list, indexing='ij')
    v_stack  = torch.stack(v_meshes, dim=-1).to(device=f.device, dtype=f.dtype)
    v_flat   = v_stack.reshape(-1, dim)
    v_sq     = torch.sum(v_flat**2, dim=-1)
    num_pts  = v_flat.size(0)
    dv_pow   = float(delta_v ** dim)

    v_axes_start = 1 + dim
    orig_shape   = f.shape
    bs_shape     = orig_shape[:v_axes_start]
    channels     = orig_shape[-1]
    f_flat_v     = f.reshape(-1, num_pts, channels)

    density  = (torch.sum(f_flat_v, dim=1) * dv_pow).reshape(*bs_shape, channels)
    momentum = (torch.einsum('nd,bnc->bdc', v_flat, f_flat_v) * dv_pow).reshape(*bs_shape, dim, channels)
    energy   = (0.5 * torch.einsum('n,bnc->bc', v_sq, f_flat_v) * dv_pow).reshape(*bs_shape, channels)

    return (density, momentum, energy)


##################################################
def one_step_SL_Euler(
    f_curr:       torch.Tensor,
    delta_t:      float,
    collision_fn: Callable[[torch.Tensor], torch.Tensor],
    v_coords:     Sequence[torch.Tensor],
    x_max:        float,
    x_min:        Optional[float]                          = None,
    knudsen:      float                                    = 1.0,
    order:        Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> torch.Tensor:
    """Performs a single first-order semi-Lagrangian explicit Euler time advancement step.

    ## Description
    Advances the inhomogeneous kinetic transport equation by $\\Delta t$ using first-order
    semi-Lagrangian characteristic tracking: $f^{n+1} = \\tilde{f}^n + \\Delta t \\tilde{K}^n$.

    ## Arguments
    `f_curr` (`torch.Tensor`): Current distribution function tensor in physical space.
    `delta_t` (`float`): Time advancement step size $\\Delta t$.
    `collision_fn` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing physical collision operator $Q(f)$.
    `v_coords` (`Sequence[torch.Tensor]`): 1D velocity coordinate tensors.
    `x_max` (`float`): Maximum spatial boundary coordinate.
    `x_min` (`Optional[float]`, default: `None`): Minimum spatial boundary coordinate.
    `knudsen` (`float`, default: `1.0`): Knudsen number $\\text{Kn}$.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Spatial reconstruction order.

    ## Returns
    `torch.Tensor`: Solution tensor $f^{n+1}$ at next time level.
    """
    f_tilde: torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, delta_t, x_max, x_min=x_min, order=order)
    k_curr:  torch.Tensor = collision_fn(f_curr) / knudsen
    k_tilde: torch.Tensor = semi_lagrangian_transport(k_curr, v_coords, delta_t, x_max, x_min=x_min, order=order)
    return f_tilde + delta_t * k_tilde


def one_step_SL_RK3(
    f_curr:       torch.Tensor,
    delta_t:      float,
    collision_fn: Callable[[torch.Tensor], torch.Tensor],
    v_coords:     Sequence[torch.Tensor],
    x_max:        float,
    x_min:        Optional[float]                          = None,
    knudsen:      float                                    = 1.0,
    order:        Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> torch.Tensor:
    """Performs a single 3rd-order semi-Lagrangian Runge-Kutta (Sanderse & Veldman) time advancement step.

    ## Description
    Advances the inhomogeneous kinetic equation by $\\Delta t$ using the 3rd-order non-splitting
    semi-Lagrangian RK scheme proposed in Boscarino, Cho, and Russo (2024), Section 2.3.

    ## Arguments
    `f_curr` (`torch.Tensor`): Current distribution function tensor in physical space.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `collision_fn` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing physical collision operator $Q(f)$.
    `v_coords` (`Sequence[torch.Tensor]`): 1D velocity coordinate tensors.
    `x_max` (`float`): Maximum spatial boundary coordinate.
    `x_min` (`Optional[float]`, default: `None`): Minimum spatial boundary coordinate.
    `knudsen` (`float`, default: `1.0`): Knudsen number $\\text{Kn}$.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Spatial reconstruction order.

    ## Returns
    `torch.Tensor`: Solution tensor $f^{n+1}$ at next time level.
    """
    # Stage 1: (c1 = 0)
    k_11:    torch.Tensor = collision_fn(f_curr) / knudsen

    # Stage 2: (c2 = 1/3)
    dt_stg2: float        = delta_t / 3.0
    f_20:    torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, dt_stg2, x_max, x_min=x_min, order=order)
    k_21:    torch.Tensor = semi_lagrangian_transport(k_11,   v_coords, dt_stg2, x_max, x_min=x_min, order=order)
    f_22:    torch.Tensor = f_20 + dt_stg2 * k_21
    k_22:    torch.Tensor = collision_fn(f_22) / knudsen

    # Stage 3: (c3 = 1)
    f_30:    torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, delta_t,             x_max, x_min=x_min, order=order)
    k_31:    torch.Tensor = semi_lagrangian_transport(k_11,   v_coords, delta_t,             x_max, x_min=x_min, order=order)
    k_32:    torch.Tensor = semi_lagrangian_transport(k_22,   v_coords, 2.0 * delta_t / 3.0, x_max, x_min=x_min, order=order)
    f_33:    torch.Tensor = f_30 - delta_t * k_31 + 2.0 * delta_t * k_32
    k_33:    torch.Tensor = collision_fn(f_33) / knudsen

    # Numerical solution update (b1 = 0, b2 = 3/4, b3 = 1/4)
    return f_30 + (0.75 * delta_t) * k_32 + (0.25 * delta_t) * k_33


def one_step_SL_RK4(
    f_curr:       torch.Tensor,
    delta_t:      float,
    collision_fn: Callable[[torch.Tensor], torch.Tensor],
    v_coords:     Sequence[torch.Tensor],
    x_max:        float,
    x_min:        Optional[float]                          = None,
    knudsen:      float                                    = 1.0,
    order:        Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> torch.Tensor:
    """Performs a single 4th-order semi-Lagrangian Runge-Kutta time advancement step.

    ## Description
    Advances the inhomogeneous kinetic equation by $\\Delta t$ using the classical 4th-order non-splitting
    semi-Lagrangian RK scheme proposed in Boscarino, Cho, and Russo (2024), Section 2.3.

    ## Arguments
    `f_curr` (`torch.Tensor`): Current distribution function tensor in physical space.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `collision_fn` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing physical collision operator $Q(f)$.
    `v_coords` (`Sequence[torch.Tensor]`): 1D velocity coordinate tensors.
    `x_max` (`float`): Maximum spatial boundary coordinate.
    `x_min` (`Optional[float]`, default: `None`): Minimum spatial boundary coordinate.
    `knudsen` (`float`, default: `1.0`): Knudsen number $\\text{Kn}$.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Spatial reconstruction order.

    ## Returns
    `torch.Tensor`: Solution tensor $f^{n+1}$ at next time level.
    """
    dt_half: float        = 0.5 * delta_t

    # Stage 1: (c1 = 0)
    k_11:    torch.Tensor = collision_fn(f_curr) / knudsen

    # Stage 2: (c2 = 1/2)
    f_20:    torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, dt_half, x_max, x_min=x_min, order=order)
    k_21:    torch.Tensor = semi_lagrangian_transport(k_11,   v_coords, dt_half, x_max, x_min=x_min, order=order)
    f_22:    torch.Tensor = f_20 + dt_half * k_21
    k_22:    torch.Tensor = collision_fn(f_22) / knudsen

    # Stage 3: (c3 = 1/2, a31 = 0, a32 = 1/2 -> c3 - c2 = 0)
    f_30:    torch.Tensor = f_20
    k_32:    torch.Tensor = k_22
    f_33:    torch.Tensor = f_30 + dt_half * k_32
    k_33:    torch.Tensor = collision_fn(f_33) / knudsen

    # Stage 4: (c4 = 1, a41 = a42 = 0, a43 = 1 -> c4 - c3 = 1/2)
    f_40:    torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, delta_t, x_max, x_min=x_min, order=order)
    k_43:    torch.Tensor = semi_lagrangian_transport(k_33,   v_coords, dt_half, x_max, x_min=x_min, order=order)
    f_44:    torch.Tensor = f_40 + delta_t * k_43
    k_44:    torch.Tensor = collision_fn(f_44) / knudsen

    # Fluxes for numerical solution update
    k_41:    torch.Tensor = semi_lagrangian_transport(k_11, v_coords, delta_t, x_max, x_min=x_min, order=order)
    k_42:    torch.Tensor = semi_lagrangian_transport(k_22, v_coords, dt_half, x_max, x_min=x_min, order=order)

    return f_40 + (delta_t / 6.0) * k_41 + (delta_t / 3.0) * k_42 + (delta_t / 3.0) * k_43 + (delta_t / 6.0) * k_44


def one_step_SL_Strang(
    f_curr:       torch.Tensor,
    delta_t:      float,
    collision_fn: Callable[[torch.Tensor], torch.Tensor],
    v_coords:     Sequence[torch.Tensor],
    x_max:        float,
    x_min:        Optional[float]                          = None,
    knudsen:      float                                    = 1.0,
    order:        Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> torch.Tensor:
    """Performs a single second-order Strang splitting semi-Lagrangian time advancement step.

    ## Description
    Splits transport and collision into a symmetrical sequence:
    1. Transport for $\\Delta t / 2$.
    2. RK4 collision integration for $\\Delta t$.
    3. Transport for $\\Delta t / 2$.

    ## Arguments
    `f_curr` (`torch.Tensor`): Current distribution function tensor in physical space.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `collision_fn` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing physical collision operator $Q(f)$.
    `v_coords` (`Sequence[torch.Tensor]`): 1D velocity coordinate tensors.
    `x_max` (`float`): Maximum spatial boundary coordinate.
    `x_min` (`Optional[float]`, default: `None`): Minimum spatial boundary coordinate.
    `knudsen` (`float`, default: `1.0`): Knudsen number $\\text{Kn}$.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Spatial reconstruction order.

    ## Returns
    `torch.Tensor`: Solution tensor $f^{n+1}$ at next time level.
    """
    dt_half: float        = 0.5 * delta_t
    f_half1: torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, dt_half, x_max, x_min=x_min, order=order)

    # Collision step via classical RK4
    k1:      torch.Tensor = collision_fn(f_half1) / knudsen
    k2:      torch.Tensor = collision_fn(f_half1 + dt_half * k1) / knudsen
    k3:      torch.Tensor = collision_fn(f_half1 + dt_half * k2) / knudsen
    k4:      torch.Tensor = collision_fn(f_half1 + delta_t * k3) / knudsen
    f_coll:  torch.Tensor = f_half1 + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return semi_lagrangian_transport(f_coll, v_coords, dt_half, x_max, x_min=x_min, order=order)


def one_step_SL_AB3(
    f_curr:       torch.Tensor,
    delta_t:      float,
    collision_fn: Callable[[torch.Tensor], torch.Tensor],
    v_coords:     Sequence[torch.Tensor],
    x_max:        float,
    history_K:    Sequence[torch.Tensor],
    x_min:        Optional[float]                          = None,
    knudsen:      float                                    = 1.0,
    order:        Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs a single 3-step Adams-Bashforth semi-Lagrangian time advancement step.

    ## Description
    Advances the kinetic transport equation using the 3-step explicit Adams-Bashforth SL scheme
    (Boscarino, Cho, and Russo 2024, Section 2.4). Requires past collision stages in `history_K`.

    ## Arguments
    `f_curr` (`torch.Tensor`): Current distribution function tensor in physical space.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `collision_fn` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing physical collision operator $Q(f)$.
    `v_coords` (`Sequence[torch.Tensor]`): 1D velocity coordinate tensors.
    `x_max` (`float`): Maximum spatial boundary coordinate.
    `history_K` (`Sequence[torch.Tensor]`): List of past collision operators `[K^n, K^{n-1}, K^{n-2}]`.
    `x_min` (`Optional[float]`, default: `None`): Minimum spatial boundary coordinate.
    `knudsen` (`float`, default: `1.0`): Knudsen number $\\text{Kn}$.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Spatial reconstruction order.

    ## Returns
    `Tuple[torch.Tensor, torch.Tensor]`: Pair `(f_next, k_next)` where `k_next` is the evaluated collision operator at $t_{n+1}$.
    """
    f_tilde: torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, delta_t, x_max, x_min=x_min, order=order)
    k_n1:    torch.Tensor = semi_lagrangian_transport(history_K[0], v_coords, delta_t,       x_max, x_min=x_min, order=order)
    k_n2:    torch.Tensor = semi_lagrangian_transport(history_K[1], v_coords, 2.0 * delta_t, x_max, x_min=x_min, order=order)
    k_n3:    torch.Tensor = semi_lagrangian_transport(history_K[2], v_coords, 3.0 * delta_t, x_max, x_min=x_min, order=order)

    f_next:  torch.Tensor = f_tilde + delta_t * ((23.0 / 12.0) * k_n1 - (16.0 / 12.0) * k_n2 + (5.0 / 12.0) * k_n3)
    k_next:  torch.Tensor = collision_fn(f_next) / knudsen
    return f_next, k_next


def one_step_SL_AB4(
    f_curr:       torch.Tensor,
    delta_t:      float,
    collision_fn: Callable[[torch.Tensor], torch.Tensor],
    v_coords:     Sequence[torch.Tensor],
    x_max:        float,
    history_K:    Sequence[torch.Tensor],
    x_min:        Optional[float]                          = None,
    knudsen:      float                                    = 1.0,
    order:        Literal['linear', 'cweno23', 'cweno35'] = 'cweno35',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs a single 4-step Adams-Bashforth semi-Lagrangian time advancement step.

    ## Description
    Advances the kinetic transport equation using the 4-step explicit Adams-Bashforth SL scheme
    (Boscarino, Cho, and Russo 2024, Section 2.4). Requires past collision stages in `history_K`.

    ## Arguments
    `f_curr` (`torch.Tensor`): Current distribution function tensor in physical space.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `collision_fn` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing physical collision operator $Q(f)$.
    `v_coords` (`Sequence[torch.Tensor]`): 1D velocity coordinate tensors.
    `x_max` (`float`): Maximum spatial boundary coordinate.
    `history_K` (`Sequence[torch.Tensor]`): List of past collision operators `[K^n, K^{n-1}, K^{n-2}, K^{n-3}]`.
    `x_min` (`Optional[float]`, default: `None`): Minimum spatial boundary coordinate.
    `knudsen` (`float`, default: `1.0`): Knudsen number $\\text{Kn}$.
    `order` (`Literal['linear', 'cweno23', 'cweno35']`, default: `'cweno35'`): Spatial reconstruction order.

    ## Returns
    `Tuple[torch.Tensor, torch.Tensor]`: Pair `(f_next, k_next)` where `k_next` is the evaluated collision operator at $t_{n+1}$.
    """
    f_tilde: torch.Tensor = semi_lagrangian_transport(f_curr, v_coords, delta_t, x_max, x_min=x_min, order=order)
    k_n1:    torch.Tensor = semi_lagrangian_transport(history_K[0], v_coords, delta_t,       x_max, x_min=x_min, order=order)
    k_n2:    torch.Tensor = semi_lagrangian_transport(history_K[1], v_coords, 2.0 * delta_t, x_max, x_min=x_min, order=order)
    k_n3:    torch.Tensor = semi_lagrangian_transport(history_K[2], v_coords, 3.0 * delta_t, x_max, x_min=x_min, order=order)
    k_n4:    torch.Tensor = semi_lagrangian_transport(history_K[3], v_coords, 4.0 * delta_t, x_max, x_min=x_min, order=order)

    f_next:  torch.Tensor = f_tilde + delta_t * ((55.0 / 24.0) * k_n1 - (59.0 / 24.0) * k_n2
                                               + (37.0 / 24.0) * k_n3 - (9.0 / 24.0)  * k_n4)
    k_next:  torch.Tensor = collision_fn(f_next) / knudsen
    return f_next, k_next


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()
