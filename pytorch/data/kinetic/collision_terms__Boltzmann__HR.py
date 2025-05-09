# %%
from    typing      import  Callable

import  numpy               as      np
import  torch
import  matplotlib.pyplot   as      plt

from    tqdm    import  tqdm

from    kinetic_distribtutions      import  *

from    pathlib             import  Path
root_dir    = r"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE"
path_root   = Path(root_dir)
path_lib    = path_root / "python_deep_numerical"
path_data   = path_root / "datasets"

from    sys         import  path
path.append( str(path_lib) )
from    pytorch     import  utils
from    pytorch.numerical   import  distribution
from    pytorch.numerical.solvers     import  FastSM_Boltzmann_VHS

dtype:  torch.dtype     = torch.float64
device: torch.device    = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device: torch.device    = torch.device('cpu')

dtype_and_device = {'dtype': dtype, 'device': device}
__dtype_str = str(dtype).split('.')[-1]

# %%
PART_INIT:  int     = 1
N_REPEAT:   int     = 4
NUM_INST:   int     = 5

DELTA_T:    float   = 0.1
MAX_T:      float   = 5.0
NUM_T:      int     = 1 + int(MAX_T/DELTA_T + 0.1)
DATA_SIZE:  int     = NUM_INST * NUM_T

T1__n_init  = T2__n_init    = T3__n_init    = NUM_INST
T1__size    = T2__size      = T3__size      = DATA_SIZE

DIMENSION:  int     = 3
RESOLUTION: int     = 2**5
V_MAX:      float   = 3.0/utils.LAMBDA
DELTA_V:    float   = (2*V_MAX) / RESOLUTION
V_WHERE_CLOSED: str = 'left'

v_grid = utils.velocity_grid(DIMENSION, RESOLUTION, V_MAX, where_closed=V_WHERE_CLOSED, **dtype_and_device)

FFT_AXES:   tuple[int] = tuple(range(-(1+DIMENSION), -1))
FFT_NORM:   str  = 'forward'

sample_q: Callable[[int], tuple[torch.Tensor]] = \
    lambda batch_size: sample_quantities(DIMENSION, batch_size, **dtype_and_device)

VHS_COEFF = 1 / utils.area_of_unit_sphere(DIMENSION)
for part in range(PART_INIT, PART_INIT+N_REPEAT):
    print('+' + '='*30 + ' +')
    print(f"# part: {str(part).zfill(len(str(N_REPEAT)))}")
    for VHS_ALPHA in [0.0]:
        torch.cuda.empty_cache()
        print('+' + '-'*30 + ' +')
        print(f"* vhs_alpha: {VHS_ALPHA:.2f}")
        # %%
        solver = FastSM_Boltzmann_VHS(
            dimension   = DIMENSION,
            v_num_grid  = RESOLUTION,
            v_max       = V_MAX,
            
            vhs_coeff   = VHS_COEFF,
            vhs_alpha   = VHS_ALPHA,
            
            **dtype_and_device,
        )
        FFT_CONFIG: dict[str, object] = {'s': solver.v_shape, 'dim': solver.v_axes, 'norm': FFT_NORM}

        # %% [markdown]
        # Type 1. Maxwellian distribution

        # %%
        print(f"# Type 1. Maxwellian distribution")
        T1__data:   list[torch.Tensor]  = []
        T1__gain:   list[torch.Tensor]  = []
        T1__loss:   list[torch.Tensor]  = []

        T1__init: torch.Tensor = distribution.maxwellian_homogeneous(v_grid, *sample_q(T1__n_init))
        arr_f_1 = T1__init = normalize_density(T1__init, DELTA_V)
        arr_f_1_fft: torch.Tensor = torch.fft.fftn(arr_f_1, **FFT_CONFIG)

        for cnt in tqdm(range(NUM_T)):
            ##### 1. Save the distribution at the previous time step
            T1__data.append(arr_f_1)
            ##### 2. Save the collision term at the previous time step
            _gain_1_fft = solver.compute_gain_fft(None, arr_f_1_fft)
            _loss_1_fft = solver.compute_loss_fft(None, arr_f_1_fft)
            gain_1 = torch.real(torch.fft.ifftn(_gain_1_fft, **FFT_CONFIG))
            loss_1 = torch.real(torch.fft.ifftn(_loss_1_fft, **FFT_CONFIG))
            T1__gain.append(gain_1)
            T1__loss.append(loss_1)
            ##### 3. Compute the distribution at the current time step
            if cnt<NUM_T-1:
                arr_f_1_fft = solver.forward(0.0, arr_f_1_fft, DELTA_T, utils.one_step_RK4_classic)
                arr_f_1 = torch.real(torch.fft.ifftn(arr_f_1_fft, **FFT_CONFIG))
            
        T1__data:   torch.Tensor    = torch.stack(T1__data, dim=1).cpu()
        T1__gain:   torch.Tensor    = torch.stack(T1__gain, dim=1).cpu()
        T1__loss:   torch.Tensor    = torch.stack(T1__loss, dim=1).cpu()


        print(f"The shape of the input data")
        print(f">>> {T1__data.shape}")
        print(f"The shape of the output data")
        print(f">>> {T1__gain.shape}, {T1__loss.shape}")

        # %% [markdown]
        # Type 2. Sum of two Maxwellian distributions

        # %%
        print(f"# Type 2. Sum of two Maxwellian distributions")
        T2__data:   list[torch.Tensor]  = []
        T2__gain:   list[torch.Tensor]  = []
        T2__loss:   list[torch.Tensor]  = []

        T2__init: torch.Tensor = \
            0.5 * (
                distribution.maxwellian_homogeneous(v_grid, *sample_q(T2__n_init))
                +
                distribution.maxwellian_homogeneous(v_grid, *sample_q(T2__n_init))
            )
        arr_f_2 = T2__init = normalize_density(T2__init, DELTA_V)
        arr_f_2_fft: torch.Tensor = torch.fft.fftn(arr_f_2, **FFT_CONFIG)

        for cnt in tqdm(range(NUM_T)):
            ##### 1. Save the distribution at the previous time step
            T2__data.append(arr_f_2)
            ##### 2. Save the collision term at the previous time step
            _gain_2_fft = solver.compute_gain_fft(None, arr_f_2_fft)
            _loss_2_fft = solver.compute_loss_fft(None, arr_f_2_fft)
            gain_2 = torch.real(torch.fft.ifftn(_gain_2_fft, **FFT_CONFIG))
            loss_2 = torch.real(torch.fft.ifftn(_loss_2_fft, **FFT_CONFIG))
            T2__gain.append(gain_2)
            T2__loss.append(loss_2)
            ##### 3. Compute the distribution at the current time step
            if cnt<NUM_T-1:
                arr_f_2_fft = solver.forward(0.0, arr_f_2_fft, DELTA_T, utils.one_step_RK4_classic)
                arr_f_2 = torch.real(torch.fft.ifftn(arr_f_2_fft, **FFT_CONFIG))
            
        T2__data:   torch.Tensor    = torch.stack(T2__data, dim=1).cpu()
        T2__gain:   torch.Tensor    = torch.stack(T2__gain, dim=1).cpu()
        T2__loss:   torch.Tensor    = torch.stack(T2__loss, dim=1).cpu()

        print(f"The shape of the input data")
        print(f">>> {T2__data.shape}")
        print(f"The shape of the output data")
        print(f">>> {T2__gain.shape}, {T2__loss.shape}")

        # %% [markdown]
        # Type 3. Perturbed Maxwellian distributions

        # %%
        print(f"# Type 3. Perturbed Maxwellian distributions")
        T3__data:   list[torch.Tensor] = []
        T3__gain:   list[torch.Tensor] = []
        T3__loss:   list[torch.Tensor] = []

        coeffs = sample_noise_quadratic(DIMENSION, V_MAX, T3__n_init, **dtype_and_device)
        quad = compute_quadratic_polynomial(v_grid, coeffs)
        quad = quad.reshape(T3__n_init, *utils.ones(DIMENSION), *utils.repeat(RESOLUTION, DIMENSION), 1)

        T3__init: torch.Tensor = \
            distribution.maxwellian_homogeneous(v_grid, *sample_q(T3__n_init)) * \
            (1 + quad)
        arr_f_3 = T3__init = normalize_density(T3__init, DELTA_V)
        arr_f_3_fft: torch.Tensor = torch.fft.fftn(arr_f_3, **FFT_CONFIG)

        for cnt in tqdm(range(NUM_T)):
            ##### 1. Save the distribution at the previous time step
            T3__data.append(arr_f_3)
            ##### 2. Save the collision term at the previous time step
            _gain_3_fft = solver.compute_gain_fft(None, arr_f_3_fft)
            _loss_3_fft = solver.compute_loss_fft(None, arr_f_3_fft)
            gain_3 = torch.real(torch.fft.ifftn(_gain_3_fft, **FFT_CONFIG))
            loss_3 = torch.real(torch.fft.ifftn(_loss_3_fft, **FFT_CONFIG))
            T3__gain.append(gain_3)
            T3__loss.append(loss_3)
            ##### 3. Compute the distribution at the current time step
            if cnt<NUM_T-1:
                arr_f_3_fft = solver.forward(0.0, arr_f_3_fft, DELTA_T, utils.one_step_RK4_classic)
                arr_f_3 = torch.real(torch.fft.ifftn(arr_f_3_fft, **FFT_CONFIG))
            
        T3__data:   torch.Tensor    = torch.stack(T3__data, dim=1).cpu()
        T3__gain:   torch.Tensor    = torch.stack(T3__gain, dim=1).cpu()
        T3__loss:   torch.Tensor    = torch.stack(T3__loss, dim=1).cpu()

        print(f"The shape of the input data")
        print(f">>> {T3__data.shape}")
        print(f"The shape of the output data")
        print(f">>> {T3__gain.shape}, {T3__loss.shape}")

        # %% [markdown]
        # Merge the data

        # %%
        print(f"# Merge the data")
        data    = \
            torch.concatenate((T1__data, T2__data, T3__data), dim=0)
        gain    = \
            torch.concatenate((T1__gain, T2__gain, T3__gain), dim=0)
        loss    = \
            torch.concatenate((T1__loss, T2__loss, T3__loss), dim=0)

        print(f"The shape of the input data")
        print(f">>> {data.shape}")
        print(f"The shape of the output data")
        print(f">>> {gain.shape}, {loss.shape}")

        # %%
        saved_data: dict[str, object] = {
            'input_distribution':   data,
            'collision_gain':       gain,
            'collision_loss':       loss,
            
            'n_init':           3*NUM_INST,
            
            'max_t':            MAX_T,
            'delta_t':          DELTA_T,
            
            'resolution':       RESOLUTION,
            'v_max':            V_MAX,
            'v_where_closed':   V_WHERE_CLOSED,
            
            'vhs_coeff':    VHS_COEFF,
            'vhs_alpha':    VHS_ALPHA,
            
            'equation':     'Boltzmann',
            'dtype_str':    __dtype_str,
        }
        file_dir = path_data / __dtype_str
        file_name = f"Boltzmann__{DIMENSION}D__res{str(RESOLUTION).zfill(3)}__alpha{float(VHS_ALPHA):.1e}__part{str(part).zfill(len(str(N_REPEAT)))}.pth"
        if not Path.exists(file_dir):
            Path.mkdir(file_dir, parents=True)
        torch.save(saved_data, file_dir/file_name)
        del(
            T1__data, T1__gain, T1__loss,
            T2__data, T2__gain, T2__loss,
            T3__data, T3__gain, T3__loss,
            data, gain, loss,
            saved_data,
        )

        print('\n'*2)
        # %% [markdown]
        # End of file


