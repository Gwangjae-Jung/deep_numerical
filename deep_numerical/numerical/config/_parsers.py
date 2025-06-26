import  argparse
import  json

from    typing              import  *
from    typing_extensions   import  Self

import  torch

from    ...utils            import  \
    DEFAULT_QUAD_ORDER_UNIFORM, DEFAULT_QUAD_ORDER_LEGENDRE, LAMBDA, \
    area_of_unit_sphere
from    ..distribution       import  \
    get_bkw_coeff_int, get_bkw_coeff_ext


##################################################
##################################################
__all__: list[str] = [
    'parse_arguments',
    'SpectralMethodArguments',
]


##################################################
##################################################
DEFAULT_MIN_T_2D: float = 0.0
DEFAULT_MIN_T_3D: float = 5.5


##################################################
##################################################
def parse_arguments() -> argparse.Namespace:
    # Argument parser
    parser = argparse.ArgumentParser(description='Solving the Boltzmann equation using the spectral method.')

    # Dimension
    parser.add_argument(
        '-dim', '--dimension',
        type        = int,
        required    = True,
        help        = 'The spatial dimension.',
    )
    
    # Time
    parser.add_argument(
        '--delta_t',
        type    = float,
        help    = 'The timestep.',
        default = 0.05,
    )
    parser.add_argument(
        '--min_t',
        type    = float,
        help    = 'The initial time.',
        default = None, # 0.0 for 2D, 5.5 for 3D
    )
    parser.add_argument(
        '--max_t',
        type    = float,
        help    = 'The terminal time.',
        default = 10.0,
    )

    # Velocity
    parser.add_argument(
        '-res', '--resolution',
        type        = int,
        required    = True,
        help        = 'The number of the grids in each dimension.',
    )
    parser.add_argument(
        '--max_v',
        type        = float,
        help        = 'The maximum speed in each dimension.',
        default     = 3.0 / LAMBDA,
    )
    
    # Orders of quadrature rules
    parser.add_argument(
        '--quad_order_uniform',
        type        = int,
        help        = 'The order of the uniform quadrature rule.',
        default     = DEFAULT_QUAD_ORDER_UNIFORM,
    )
    parser.add_argument(
        '--quad_order_legendre',
        type        = int,
        help        = 'The order of the Legendre quadrature rule.',
        default     = DEFAULT_QUAD_ORDER_LEGENDRE,
    )
    parser.add_argument(
        '--quad_order_lebedev',
        type        = int,
        help        = 'The order of the Lebedev quadrature rule.',
        default     = DEFAULT_QUAD_ORDER_UNIFORM,
    )
        
    # VHS model
    parser.add_argument(
        '--vhs_alpha',
        type    = float,
        help    ='The exponent of the relative speed in the VHS model.',
        default = None,
    )
    parser.add_argument(
        '--vhs_coeff',
        type    = float,
        help    = 'The coefficient of the power of the relative speed in the VHS model.',
        default = None
    )
    
    # Metric
    parser.add_argument(
        '--metric_order',
        type    = float,
        help    = 'The order for which the relative error is computed.',
        default = 1
    )
    
    # Problem type
    parser.add_argument(
        '-prob', '--problem_type',
        type        = str,
        required    = True,
        help        = 'The type of the problem to be solved. (\'bkw\', \'maxwellian\', \'multimodal\')'
    )
    default_config_path = __file__.replace('_parsers.py', 'problem_details.json')
    parser.add_argument(
        '-prob_config_path', '--problem_config_path',
        type    = str,
        help    = 'The path to the configuration file for the problem.',
        default = default_config_path
    )
    
    # Saving result
    parser.add_argument(
        '--save_dir',
        type    = str,
        help    = 'The path to which the result is saved.',
        default = './data'
    )
    
    # Return the argument parser
    return parser.parse_args()


##################################################
##################################################
class SpectralMethodArguments():
    def __init__(
            self,
            equation:   str,
            is_fast:    bool,
        ) -> Self:
        args = parse_arguments()
        
        ##### Configurations #####
        # Configurations - Equation
        self._args__equation:   str = equation.lower()
        
        # Configurations - Dimension
        self._args__dimension:  int = args.dimension
        assert self._args__dimension in (2, 3), \
            f"The spatial dimension must be 2 or 3, but got {self._args__dimension}."
        
        # Configurations - Time
        self._args__delta_t:  float   = args.delta_t
        self._args__min_t:    float   = args.min_t
        self._args__max_t:    float   = args.max_t
        if self._args__min_t is None:
            if self._args__dimension == 2:
                self._args__min_t = DEFAULT_MIN_T_2D
            else:
                self._args__min_t = DEFAULT_MIN_T_3D
        assert self._args__min_t < self._args__max_t, \
            f"The initial time must be less than the terminal time, but got min_t={self._args__min_t:.2f} and max_t={self._args__max_t:.2f}."

        # Configurations - Velocity
        self._args__resolution:   int     = args.resolution
        self._args__max_v:        float   = args.max_v

        # Configurations - Orders of quadrature rules
        self._args__quad_order_uniform:   int     = args.quad_order_uniform
        self._args__quad_order_legendre:  int     = args.quad_order_legendre
        self._args__quad_order_lebedev:   int     = args.quad_order_lebedev
        
        # Configurations - Algorithm
        self._args__is_fast:      bool = is_fast

        # Configurations - FFT
        self._args__fft_norm:     str = 'forward'

        # Configurations - VHS model
        self._args__vhs_alpha:    float   = args.vhs_alpha
        self._args__vhs_coeff:    float   = args.vhs_coeff
        if self._args__vhs_alpha is None:
            self._args__vhs_alpha = 0.0
        if self._args__vhs_coeff is None:
            self._args__vhs_coeff = 1/area_of_unit_sphere(self._args__dimension)

        # Configurations - metric
        self._args__metric_order: float   = args.metric_order

        # Configurations - Problem type
        self._args__problem_type:         str = args.problem_type.lower()
        self._args__problem_config_path:  str = args.problem_config_path
        self._args__bkw_coeff_int:                Optional[float]         = None
        self._args__bkw_coeff_ext:                Optional[float]         = None
        self._args__maxwellian_mean_density:      Optional[float]         = None
        self._args__maxwellian_mean_velocity:     Optional[torch.Tensor]    = None
        self._args__maxwellian_mean_temperature:  Optional[float]         = None
        self._args__multimodal_num_modes:         Optional[int]           = None
        self._args__multimodal_mean_density:      Optional[torch.Tensor]    = None
        self._args__multimodal_mean_velocity:     Optional[torch.Tensor]    = None
        self._args__multimodal_mean_temperature:  Optional[torch.Tensor]    = None
        self._args__multimodal_weights:           Optional[torch.Tensor]    = None
        self._set_problem_details()
        
        # Configurations - Save
        self._args__file_name:    str = ""
        self._args__file_title:   str = ""
        self._set_file_info()
        self._args__save_dir:     str = args.save_dir
        
        ##### Return #####
        return
    
    
    # Properties - Equation
    @property
    def equation(self) -> str:
        return self._args__equation
    
    # Properties - Dimension
    @property
    def dimension(self) -> float:
        return self._args__dimension
    
    # Properties - Time
    @property
    def delta_t(self) -> float:
        return self._args__delta_t
    @property
    def min_t(self) -> float:
        return self._args__min_t
    @property
    def max_t(self) -> float:
        return self._args__max_t
    @property
    def num_t(self) -> int:
        return int((self.max_t - self.min_t) / self.delta_t + 1.1)
    
    # Properties - Velocity
    @property
    def resolution(self) -> int:
        return self._args__resolution
    @property
    def max_v(self) -> float:
        return self._args__max_v
    
    # Properties - Orders of quadrature rules
    @property
    def quad_order_uniform(self) -> int:
        return self._args__quad_order_uniform
    @property
    def quad_order_legendre(self) -> int:
        return self._args__quad_order_legendre
    @property
    def quad_order_lebedev(self) -> int:
        return self._args__quad_order_lebedev
    
    # Properties - Algorithm
    @property
    def is_fast(self) -> bool:
        return self._args__is_fast
    
    # Properties - FFT
    @property
    def fft_norm(self) -> str:
        return self._args__fft_norm
    
    # Properties - VHS model
    @property
    def vhs_alpha(self) -> float:
        return self._args__vhs_alpha
    @property
    def vhs_coeff(self) -> float:
        return self._args__vhs_coeff

    # Properties - Metric
    @property
    def metric_order(self) -> float:
        return self._args__metric_order

    # Configurations - Problem type
    @property
    def problem_type(self) -> str:
        return self._args__problem_type
    @property
    def problem_config_path(self) -> str:
        return self._args__problem_config_path
    ## BKW solution
    @property
    def bkw_coeff_int(self) -> float:
        return self._args__bkw_coeff_int
    @property
    def bkw_coeff_ext(self) -> float:
        return self._args__bkw_coeff_ext
    ## Maxwellian distribution
    @property
    def maxwellian_mean_density(self) -> float:
        return self._args__maxwellian_mean_density
    @property
    def maxwellian_mean_velocity(self) -> float:
        return self._args__maxwellian_mean_velocity
    @property
    def maxwellian_mean_temperature(self) -> float:
        return self._args__maxwellian_mean_temperature
    ## Multimodal distribution
    @property
    def multimodal_num_modes(self) -> int:
        return self._args__multimodal_num_modes
    @property
    def multimodal_mean_density(self) -> torch.Tensor:
        return self._args__multimodal_mean_density
    @property
    def multimodal_mean_velocity(self) -> torch.Tensor:
        return self._args__multimodal_mean_velocity
    @property
    def multimodal_mean_temperature(self) -> torch.Tensor:
        return self._args__multimodal_mean_temperature
    @property
    def multimodal_weights(self) -> torch.Tensor:
        return self._args__multimodal_weights
    
    # Configurations - Save
    @property
    def file_name(self) -> str:
        return self._args__file_name
    @property
    def file_title(self) -> str:
        return self._args__file_title
    @property
    def save_dir(self) -> str:
        return self._args__save_dir
    
    
    # Set the details of the problem
    def _set_problem_details(self) -> None:
        with open(self._args__problem_config_path, "r") as f:
            prob_details = json.load(f)[self.problem_type]
            
        ### BKW ###
        if self._args__problem_type == 'bkw':
            assert self._args__vhs_alpha == 0.0, \
                f"The BKW solution is an analytical solution for the VHS model for the Maxwellian gas (alpha=0), but got alpha={self._args__vhs_alpha}."
            self._args__bkw_coeff_int:    float = get_bkw_coeff_int(self._args__dimension, self._args__vhs_coeff, self._args__equation)
            self._args__bkw_coeff_ext:    float = get_bkw_coeff_ext(self._args__dimension)
        
        ### Maxwellian ###
        elif self._args__problem_type == 'maxwellian':
            self._args__maxwellian_mean_density:      float       = \
                float(prob_details['initial_condition']['density'])
            self._args__maxwellian_mean_velocity:     torch.Tensor  = \
                torch.tensor(prob_details['initial_condition']['velocity'][:self.dimension], dtype=torch.float)
            self._args__maxwellian_mean_temperature:  float       = \
                float(prob_details['initial_condition']['temperature'])
        
        ### multimodal ###
        elif self._args__problem_type == 'multimodal':
            self._args__multimodal_num_modes = prob_details['num_modes']
            self._args__multimodal_mean_density:         torch.Tensor  = \
                torch.tensor(
                    [
                        prob_details[f'mode{k+1}']['density']
                        for k in range(self._args__multimodal_num_modes)
                    ],
                    dtype=torch.float
                )
            self._args__multimodal_mean_velocity:        torch.Tensor  = \
                torch.tensor(
                    [
                        prob_details[f'mode{k+1}']['velocity'][:self.dimension]
                        for k in range(self._args__multimodal_num_modes)
                    ],
                    dtype=torch.float
                )
            self._args__multimodal_mean_temperature:     torch.Tensor  = \
                torch.tensor(
                    [
                        prob_details[f'mode{k+1}']['temperature']
                        for k in range(self._args__multimodal_num_modes)
                    ],
                    dtype=torch.float
                )
            self._args__multimodal_weights:              torch.Tensor  = \
                torch.tensor(
                    [prob_details[f'mode{k+1}']['weight'] for k in range(self._args__multimodal_num_modes)],
                    dtype=torch.float
                )
            print(f"*** {self._args__multimodal_num_modes=}")
            print(f"*** {self._args__multimodal_mean_density=}")
            print(f"*** {self._args__multimodal_mean_velocity=}")
            print(f"*** {self._args__multimodal_mean_temperature=}")
            print(f"*** {self._args__multimodal_weights=}")
        
        else:
            raise ValueError(f"The problem type must be 'bkw', 'maxwellian', or 'multimodal', but got {self._args__problem_type}.")
        
        return            
    
    
    # Set the title of the problem
    def _set_file_info(self) -> None:
        _algorithm_prefix = "F" if self.is_fast else "D"
        file_name_prefix      = _algorithm_prefix + f"SM_{self.dimension}D"
        file_name_appendix    = f"res{self.resolution}_Tmin{self.min_t:.2f}_Tmax{self.max_t:.2f}_dt{self.delta_t:.2f}_alpha{self.vhs_alpha:.2f}"
        
        distribution_in_file_name:  str
        initialized_distribution:   str
        if self.problem_type == 'bkw':
            distribution_in_file_name   = "BKW"
            initialized_distribution    = "a BKW solution"
        elif self.problem_type == 'maxwellian':
            distribution_in_file_name   = "Maxwellian"
            initialized_distribution    = "a Maxwellian distribution"
        elif self.problem_type == 'multimodal':
            distribution_in_file_name   = "multimodal"
            initialized_distribution    = "a multimodal distribution"
        else:
            distribution_in_file_name   = ""
            initialized_distribution    = ""
        self._args__file_name     = '__'.join((file_name_prefix, distribution_in_file_name, file_name_appendix)) + ".npz"
        self._args__file_title    = ' '.join( ("Initialized with", initialized_distribution))
        return
    
    
    def _show(self) -> None:
        _line   = '=' * 20
        _front  = _line + "[ Arguments ]" + _line
        _back   = '=' * len(_front)
        print(_front)
        _prefix = '_args__'
        _len_prefix = len(_prefix)
        for obj in dir(self):
            if not obj.startswith(_prefix):
                continue
            print(f"| * {obj[_len_prefix:]}: {getattr(self, obj)}")
        print(_back)
    
    
    def __str__(self) -> str:
        msg = []
        _line   = '=' * 20
        _front  = _line + "[ Arguments ]" + _line
        _back   = '=' * len(_front)
        _prefix = '_args__'
        _len_prefix = len(_prefix)
        
        msg.append(_front)
        for obj in dir(self):
            if not obj.startswith(_prefix):
                continue
            msg.append(f"| * {obj[_len_prefix:]}: {getattr(self, obj)}")
        msg.append(_back)
        
        return '\n'.join(msg)
    
        
##################################################
##################################################
# End of file