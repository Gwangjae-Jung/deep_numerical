# %%
from    typing      import  Callable
from    itertools   import  product
from    time        import  time
import  torch
from    tqdm        import  tqdm
from    kinetic_distribtutions      import  *

from    pathlib             import  Path
root_dir    = r"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE"
path_root   = Path(root_dir)
path_lib    = path_root / "deep_numerical"

from    sys         import  path
path.append( str(path_lib) )
from    deep_numerical     import  utils
from    deep_numerical.numerical   import  distribution
from    deep_numerical.numerical.solvers     import  FastSM_Boltzmann_VSS

dtype:  torch.dtype     = torch.double
device: torch.device    = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device: torch.device    = torch.device('cpu')

dtype_and_device = {'dtype': dtype, 'device': device}
__dtype_str = str(dtype).split('.')[-1]

# %%
PART_INIT:  int     = 1
N_REPEAT:   int     = 18
PART_LAST:  int     = PART_INIT + N_REPEAT - 1
NUM_INST:   int     = 10

DELTA_T:    float   = 0.1
MAX_T:      float   = 10.0
NUM_T:      int     = 1 + int(MAX_T/DELTA_T + 0.1)
DATA_SIZE:  int     = NUM_INST * NUM_T

DIMENSION:  int     = 2
RESOLUTION: int     = 2**6
V_MAX:      float   = 3.0/utils.LAMBDA
DELTA_V:    float   = (2*V_MAX) / RESOLUTION
V_WHERE_CLOSED: str = 'none'

v_grid = utils.velocity_grid(
    DIMENSION, RESOLUTION, V_MAX,
    where_closed=V_WHERE_CLOSED,
    **dtype_and_device
)

FFT_AXES:   tuple[int] = tuple(range(-(1+DIMENSION), -1))
FFT_NORM:   str  = 'forward'

sample_q: Callable[[int], tuple[torch.Tensor]] = \
    lambda batch_size: sample_quantities(DIMENSION, batch_size, **dtype_and_device)

VSS_COEFF = 1.0 / utils.area_of_unit_sphere(DIMENSION)
for part in range(PART_INIT, PART_LAST+1):
    print('+' + '='*30 + ' +')
    print(f"# part: {str(part).zfill(len(str(PART_LAST)))}")
    for VSS_EXP_SPEED, VSS_EXP_ANGLE in product(
            [-2.0, -1.0, 0.0, 1.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ):
        torch.cuda.empty_cache()
        print('+' + '-'*30 + ' +')
        print(f"* VSS model (coeff, alpha, beta): {VSS_COEFF:.2f}, {VSS_EXP_SPEED:.2f}, {VSS_EXP_ANGLE:.2f}")
        # %%
        elapsed_time: float = time()
        solver = FastSM_Boltzmann_VSS(
            dimension   = DIMENSION,
            v_num_grid  = RESOLUTION,
            v_max       = V_MAX,

            vss_coeff       = VSS_COEFF,
            vss_exp_speed   = VSS_EXP_SPEED,
            vss_exp_angle   = VSS_EXP_ANGLE,

            quad_order_uniform = 75,

            **dtype_and_device,
        )
        FFT_CONFIG: dict[str, object] = {'s': solver.v_shape, 'dim': solver.v_axes, 'norm': FFT_NORM}

        # %%
        data:   list[torch.Tensor]  = []
        arr_f: torch.Tensor = \
            0.5 * (
                distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
                +
                distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
            )
        arr_f = normalize_density(arr_f, DELTA_V)
        print(f"The shape of the initial distribution")
        print(f">>> {arr_f.shape}")
        arr_f_fft: torch.Tensor = torch.fft.fftn(arr_f, **FFT_CONFIG)

        for cnt in tqdm(range(NUM_T)):
            ##### 1. Save the distribution at the previous time step
            data.append(arr_f.cpu())
            ##### 2. Compute the distribution at the current time step
            if cnt<NUM_T-1:
                arr_f_fft = solver.forward(0.0, arr_f_fft, DELTA_T, utils.one_step_RK4_classic)
                arr_f = torch.real(torch.fft.ifftn(arr_f_fft, **FFT_CONFIG))
        data:   torch.Tensor    = torch.stack(data, dim=1)
        elapsed_time = time() - elapsed_time

        print(f"The shape of the data")
        print(f">>> {data.shape}")
        print(f"Elapsed time")
        print(f">>> {elapsed_time:.2f} sec")

        # %%
        saved_data: dict[str, object] = {
            'data':             data,
            
            'n_init':           NUM_INST,
            
            'max_t':            MAX_T,
            'delta_t':          DELTA_T,
            
            'resolution':       RESOLUTION,
            'v_max':            V_MAX,
            'v_where_closed':   V_WHERE_CLOSED,

            'vss_coeff':    VSS_COEFF,
            'vss_exp_speed':    VSS_EXP_SPEED,
            'vss_exp_angle':    VSS_EXP_ANGLE,

            'equation':     'Boltzmann',
            'dtype_str':    __dtype_str,
            
            'elapsed_time': elapsed_time,
        }

        file_dir   = path_root / "datasets" / f"Boltzmann_{DIMENSION}D" / "biMaxwellian" / f"coeff{VSS_COEFF:.2e}"
        file_name = f"res{str(RESOLUTION).zfill(3)}__alpha{float(VSS_EXP_SPEED):.1e}__beta{float(VSS_EXP_ANGLE):.1e}__part{str(part).zfill(len(str(PART_LAST)))}.pth"
        if not Path.exists(file_dir):
            Path.mkdir(file_dir, parents=True)
        print(f"Saving data as follows:")
        print(f"* Directory: {file_dir}")
        print(f"* File name: {file_name}")
        torch.save(saved_data, file_dir/file_name)
        del(data, saved_data)

        print('\n'*2)
        # %% [markdown]
        # End of file


