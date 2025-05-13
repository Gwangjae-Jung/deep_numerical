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

from    sys         import  path
path.append( str(path_lib) )
from    pytorch     import  utils
from    pytorch.numerical   import  distribution
from    pytorch.numerical.solvers     import  FastSM_Boltzmann_VHS

dtype:  torch.dtype     = torch.float64
# device: torch.device    = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device: torch.device    = torch.device('cpu')

dtype_and_device = {'dtype': dtype, 'device': device}
__dtype_str = str(dtype).split('.')[-1]

# %%
NUM_INST:   int     = 1

DELTA_T:    float   = 0.1
MAX_T:      float   = 5.0
NUM_T:      int     = 1 + int(MAX_T/DELTA_T + 0.1)
DATA_SIZE:  int     = NUM_INST * NUM_T

DIMENSION:  int     = 3
RESOLUTION: int     = 2**6
V_MAX:      float   = 3.0/utils.LAMBDA
DELTA_V:    float   = (2*V_MAX) / RESOLUTION
V_WHERE_CLOSED: str = 'left'

v_grid = utils.velocity_grid(DIMENSION, RESOLUTION, V_MAX, where_closed=V_WHERE_CLOSED, **dtype_and_device)

FFT_AXES:   tuple[int] = tuple(range(-(1+DIMENSION), -1))
FFT_NORM:   str  = 'forward'

sample_q: Callable[[int], tuple[torch.Tensor]] = \
    lambda batch_size: sample_quantities(DIMENSION, batch_size, **dtype_and_device)

VHS_COEFF = 1 / utils.area_of_unit_sphere(DIMENSION)

for part in range(1, 10+1):
    print('-'*30)
    print(f"Part {part}")
    # for VHS_ALPHA in [-2.0, -1.0, 0.0, 1.0]:
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
        # Type 2. Sum of two Maxwellian distributions

        # %%
        print(f"# Type 1. Sum of two Maxwellian distributions")
        T2__data:   list[torch.Tensor]  = []
        T2__gain:   list[torch.Tensor]  = []
        T2__loss:   list[torch.Tensor]  = []

        T2__init: torch.Tensor = \
            0.5 * (
                distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
                +
                distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
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
        # Merge the data

        # %%
        print(f"# Merge the data")
        data    = T2__data
        gain    = T2__gain
        loss    = T2__loss

        print(f"The shape of the input data")
        print(f">>> {data.shape}")
        print(f"The shape of the output data")
        print(f">>> {gain.shape}, {loss.shape}")

        # %%
        saved_data: dict[str, object] = {
            'input_distribution':   data,
            'collision_gain':       gain,
            'collision_loss':       loss,
            
            'n_init':           NUM_INST,
            
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
        
        path_data   = path_root / "datasets" / f"Boltzmann_{DIMENSION}D" / "biMaxwellian"
        file_dir = path_data / f"coeff{VHS_COEFF:.2e}"
        file_name = f"res{str(RESOLUTION).zfill(3)}__alpha{float(VHS_ALPHA):.1e}__part{str(part).zfill(2)}.pth"
        if not Path.exists(file_dir):
            Path.mkdir(file_dir, parents=True)
        torch.save(saved_data, file_dir/file_name)

        print('\n'*2)
        # %% [markdown]
        # End of file


