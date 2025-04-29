# %%
import  time
from    tqdm        import  tqdm

import  numpy                   as      np

import  os

import  find_library
from    numerical_numpy.config          import  SpectralMethodArguments
from    numerical_numpy.utils           import  *
from    numerical_numpy.distribution    import  *
from    numerical_numpy.solvers         import  *


# %%
args = SpectralMethodArguments(equation='boltzmann', is_fast=False)
print(args)

arr_t = np.linspace(args.min_t, args.max_t, args.num_t)
grid_v = velocity_grid(args.dimension, args.resolution, args.max_v)

# %%
print(f"[ Initialization ]")
print(f"* Setting the initial value.")
arr_f:  np.ndarray
"""A space-homogeneous initial condition"""
if args.problem_type == 'bkw':
    bkw_sol = bkw(
                    arr_t, grid_v,
                    coeff_ext   = args.bkw_coeff_ext,
                    vhs_coeff   = args.vhs_coeff,
                    equation    = args.equation,
                )
    arr_f = bkw_sol[:, 0]
elif args.problem_type == 'maxwellian':
    arr_f = maxwellian_homogeneous(
                grid_v,
                args.maxwellian_mean_density,
                args.maxwellian_mean_velocity,
                args.maxwellian_mean_temperature,
            )
elif args.problem_type == 'multimodal':
    _modes = [
            args.multimodal_weights[k] * \
                maxwellian_homogeneous(
                    grid_v,
                    args.multimodal_mean_density[k],
                    args.multimodal_mean_velocity[k],
                    args.multimodal_mean_temperature[k],
                )
            for k in range(args.multimodal_num_modes)
    ]
    arr_f = np.sum(np.stack(_modes, axis=-1), axis=-1)
else:
    ValueError(f"Got an unexpected 'problem_type': {args.problem_type}")

print(f"* Computing the kernel components.")
sm_object = DirectSM_Boltzmann_VHS(
    dim         = args.dimension,
    v_num_grid  = args.resolution,
    v_max       = args.max_v,
    vhs_coeff   = args.vhs_coeff,
    vhs_alpha   = args.vhs_alpha,
    order       = args.quad_order_legendre,
)
arr_f_fft = np.fft.fftn(arr_f, axes=sm_object.v_axes, norm=args.fft_norm)
print(f"\tDone.")


# %%
print(f"[ Computation ]")
print(f"* Computing the numerical solution.")
arr_distribution        = [arr_f]
arr_mean_density        = []
arr_mean_velocity       = []
arr_mean_temperature    = []
arr_kinetic_energy      = []
arr_rel_error           = []

elapsed_time = time.time()

for cnt, t_curr in enumerate(tqdm(arr_t)):
    # Update the function
    arr_f_fft = sm_object.forward(t_curr, arr_f_fft, args.delta_t, one_step_RK4_classic)
    arr_f = np.fft.ifftn(arr_f_fft, axes=sm_object.v_axes, norm=args.fft_norm).real
    arr_distribution.append(arr_f)

elapsed_time = time.time() - elapsed_time


print(f"* Computing physical quantities and the relative error of order {args.metric_order:.1f}.")
_target:    np.ndarray = None
for cnt, data in tqdm(enumerate(arr_distribution)):
    rho, u, T = compute_moments_homogeneous(arr_f, grid_v)
    arr_mean_density.append(rho)
    arr_mean_velocity.append(u)
    arr_mean_temperature.append(T)
    arr_kinetic_energy.append(compute_energy_homogeneous(data, grid_v))
       
    if args.problem_type == 'bkw':
        _target = bkw_sol[:, cnt]
    elif args.problem_type == 'maxwellian' and _target is None:
        _target = arr_distribution[[0]]
    elif args.problem_type == 'multimodal' and _target is None:
        _target = maxwellian_homogeneous(
            grid_v,
            arr_mean_density[0],
            arr_mean_velocity[0],
            arr_mean_temperature[0],
        )
    arr_rel_error.append( metric(data, _target) )


arr_distribution        = np.stack(arr_distribution, axis=1)
arr_mean_density        = np.stack(arr_mean_density, axis=1)
arr_mean_velocity       = np.stack(arr_mean_velocity, axis=1)
arr_mean_temperature    = np.stack(arr_mean_temperature, axis=1)
arr_rel_error           = np.array(arr_rel_error)
print(f"\tDone.")


# %%
if os.path.exists(args.save_dir) == False:
    print(f"* As the saving directory does not exist, create a new one.")
    os.makedirs(args.save_dir)
np.savez(
    file = '/'.join([args.save_dir, args.file_name]),
    
    args            = args,
    elapsed_time    = elapsed_time,
    
    arr_distribution        = arr_distribution,
    arr_mean_density        = arr_mean_density,
    arr_mean_velocity       = arr_mean_velocity,
    arr_mean_temperature    = arr_mean_temperature,
    arr_kinetic_energy      = arr_kinetic_energy,
    arr_rel_error           = arr_rel_error,
    
    metric_order    = args.metric_order,
)

# %% [markdown]
# End of file


