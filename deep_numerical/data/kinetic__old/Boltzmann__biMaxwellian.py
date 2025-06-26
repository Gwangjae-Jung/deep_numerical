# %%
from    typing      import  Callable
from    time        import  time
import  torch
from    tqdm        import  tqdm
import  yaml
from    kinetic_distribtutions      import  *

from    pathlib             import  Path
root_dir    = r"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE"
path_root   = Path(root_dir)
"""Directory `/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE`"""
path_lib    = path_root / "deep_numerical"

from    sys         import  path
path.append( str(path_lib) )
from    deep_numerical     import  utils
from    deep_numerical.numerical   import  distribution
from    deep_numerical.numerical.solvers     import  FastSM_Boltzmann_VHS


# %%
print(f"Loading the configuration file...")
with open("config_generation.yaml") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
print(f"\tDone.")


# %%
config_computation = config['computation']
dtype:  torch.dtype     = getattr(torch, config_computation['dtype'])
device: torch.device    = torch.device(
    f"cuda:{config_computation['cuda_index']}" if torch.cuda.is_available()
    else 'cpu'
)

dtype_and_device = {'dtype': dtype, 'device': device}
__dtype_str = str(dtype).split('.')[-1]

# %%
config_dataset = config['dataset']
PART_INIT:  int     = config_dataset['part_init']
N_REPEAT:   int     = config_dataset['n_repeat']
PART_LAST:  int     = PART_INIT + N_REPEAT - 1
NUM_INST:   int     = config_dataset['n_inst']

DELTA_T:    float   = config_dataset['delta_t']
MAX_T:      float   = config_dataset['max_t']
NUM_T:      int     = 1 + int(MAX_T/DELTA_T + 0.1)
DATA_SIZE:  int     = NUM_INST * NUM_T

DIMENSION:  int     = config_dataset['dimension']
RESOLUTION: int     = config_dataset['resolution']
V_MAX:      float   = 3.0/utils.LAMBDA
DELTA_V:    float   = (2*V_MAX) / RESOLUTION
V_WHERE_CLOSED: str = config_dataset['v_where_closed']

v_grid = utils.velocity_grid(
    DIMENSION, RESOLUTION, V_MAX,
    where_closed=V_WHERE_CLOSED,
    **dtype_and_device
)

FFT_AXES:   tuple[int] = tuple(range(-(1+DIMENSION), -1))
FFT_NORM:   str  = 'forward'

sample_q: Callable[[int], tuple[torch.Tensor]] = \
    lambda batch_size: sample_quantities(
        DIMENSION, batch_size,
        v_max=V_MAX,
        **dtype_and_device,
    )

VHS_COEFF = 1.0 / utils.area_of_unit_sphere(DIMENSION)
for part in range(PART_INIT, PART_LAST+1):
    print('+' + '='*30 + ' +')
    print(f"# part: {str(part).zfill(len(str(PART_LAST)))}")
    # Generate the initial conditions
    # The initial conditions are shared for all collision kernels
    # and are independently generated for each part.
    init_f: torch.Tensor = \
        0.5 * (
            distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
            +
            distribution.maxwellian_homogeneous(v_grid, *sample_q(NUM_INST))
        )
    init_f = normalize_density(init_f, DELTA_V)
    
    # Compute the solutions for various collision kernels
    for VHS_ALPHA in config_dataset['exp_speed']:
        torch.cuda.empty_cache()
        print('+' + '-'*30 + ' +')
        print(f"* VHS model (coeff, alpha): {VHS_COEFF:.2f}, {VHS_ALPHA:.2f}")
        # %%
        elapsed_time: float = time()
        solver = FastSM_Boltzmann_VHS(
            dimension   = DIMENSION,
            v_num_grid  = RESOLUTION,
            v_max       = V_MAX,
            
            vhs_coeff   = VHS_COEFF,
            vhs_alpha   = VHS_ALPHA,
            
            **dtype_and_device,
        )
        FFT_CONFIG: dict[str, object] = {'s': solver.v_shape, 'dim': solver.v_axes, 'norm': FFT_NORM}

        # %%
        data:   list[torch.Tensor]  = []
        arr_f = init_f.clone()
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
            
            'vhs_coeff':    VHS_COEFF,
            'vhs_alpha':    VHS_ALPHA,
            
            'equation':     'Boltzmann',
            'dtype_str':    __dtype_str,
            
            'elapsed_time': elapsed_time,
        }
        
        file_dir   = path_root / config_dataset['dir_name'] / f"Boltzmann_{DIMENSION}D" / "biMaxwellian" / f"coeff{VHS_COEFF:.2e}"
        file_name = f"res{str(RESOLUTION).zfill(3)}__alpha{float(VHS_ALPHA):.1e}__part{str(part).zfill(len(str(PART_LAST)))}.pth"
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


