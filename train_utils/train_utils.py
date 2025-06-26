import  warnings

from    typing      import  Callable, Sequence, Union, Self
from    pathlib     import  Path
import  torch

from    pathlib     import  Path
from    sys         import  path    as  sys_path
sys_path.append(str(Path(__file__).parent))
from    deep_numerical.utils        import  velocity_grid, relative_error


##################################################
##################################################
__all__: list[str] = [
    'load_data', 'augment_data_2D',
    'exponential_cosine',
    'LossFunctions',
]


##################################################
##################################################
def load_data(
        directory:  Union[str, Path],
        resolution: int,
        alpha:      Sequence[float],
        part_index: Sequence[int],
        dtype:      torch.dtype = torch.float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """
    Arguments:
        `directory` (`Union[str, Path]`):
            The directory where the data files are stored.
        `resolution` (`int`):
            The resolution of the data.
        `alpha` (`Sequence[float]`):
            The values of alpha of the data to be loaded.
        `part_index` (`Sequence[int]`):
            The parts of the data to be loaded.
        `dtype` (`torch.dtype`, default: `torch.float`):
            The data type of the loaded tensors.
    
    Returns:
        `dict[str, torch.Tensor]`:
            A dictionary containing the concatenated `data`, `vhs_coeff`, and `vhs_alpha`.
        `dict[str, object]`:
            A dictionary containing shared attributes such as `v_max` and `where_closed`.
    """
    from    itertools       import  product
    
    if not isinstance(directory, Path):
        directory = Path(directory)

    def _get_file_name(res: int, a: float, p: int) -> str:
        res = str(res).zfill(3)
        p   = str(p).zfill(1)
        return f"res{res}__alpha{a:.1e}__part{p}.pth"

    files = [
        torch.load(
            directory/_get_file_name(resolution, a, p),
            weights_only = False,
        )
        for a, p in product(alpha, part_index)
    ]
    
    data: torch.Tensor = torch.cat([f['data'].type(dtype) for f in files], dim=0)
    batch_size = data.size(0)
    ndim = (data.ndim-3)//2
    data = data[:, :, *(0 for _ in range(ndim)), ...]   # Remove the space dimensions
    vhs_coeff: torch.Tensor = \
        torch.cat(
            [
                float(f['vhs_coeff']) * torch.ones((batch_size, 1), dtype=dtype)
                for f in files
            ],
            dim=0,
        )
    vhs_alpha: torch.Tensor = \
        torch.cat(
            [
                float(f['vhs_alpha']) * torch.ones((batch_size, 1), dtype=dtype)
                for f in files
            ],
            dim=0,
        )

    return (
        {
            # Gathered attributes
            'data':         data,
            'vhs_coeff':    vhs_coeff,
            'vhs_alpha':    vhs_alpha,
        },
        {
            # Shared attributes
            'v_max':            float(files[0]['v_max']),
            'v_where_closed':   str(files[0]['v_where_closed']),
        }
    )


def augment_data_2D(data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Augment the data for 2D by flipping and rotating.
    
    Arguments:
        `data_dict` (`dict[str, torch.Tensor]`):
            A dictionary containing the data to augment. The dictionary should have the following keys:
            - `data`: A tensor of shape `(B, T, H, W, C)`.
            - `vhs_coeff`: A tensor of shape `(B,)`.
            - `vhs_alpha`: A tensor of shape `(B,)`.
    """
    data        = data_dict['data']
    assert data.ndim==5
    vhs_coeff   = data_dict['vhs_coeff']
    vhs_alpha   = data_dict['vhs_alpha']
    
    # Rotate data
    data        = torch.cat([torch.rot90(data, k, dims=(-3,-2)) for k in range(4)], dim=0)
    vhs_coeff   = torch.cat([vhs_coeff for _ in range(4)], dim=0)
    vhs_alpha   = torch.cat([vhs_alpha for _ in range(4)], dim=0)
    
    # Flip data
    data        = torch.cat([data, torch.flip(data, dims=(-2,))], dim=0)
    vhs_coeff   = torch.cat([vhs_coeff, vhs_coeff], dim=0)
    vhs_alpha   = torch.cat([vhs_alpha, vhs_alpha], dim=0)
    
    return {
        'data':         data,
        'vhs_coeff':    vhs_coeff,
        'vhs_alpha':    vhs_alpha,
    }


##################################################
##################################################
# Learning rate scheduler
def exponential_cosine(
        period:     float,
        half_life:  float,
    ) -> Callable[[int], float]:
    from    math    import  cos, exp, log, pi
    assert period > 0
    assert half_life > 0
    omega   = 2*pi/period
    lambda_lr: Callable[[int], float] = \
        lambda epoch: 0.5 * (1+cos(omega*epoch)) * exp(-log(2) * epoch/half_life)
    return lambda_lr


##################################################
##################################################
class LossFunctions():
    def __init__(
        self,
        dimension:      int,
        resolution:     int,
        v_max:          float,
        v_where_closed: str='none',
        dtype:         torch.dtype=torch.float,
        device:         torch.device=torch.device('cpu')
    ) -> Self:
        """
        Initialize the loss functions for the Boltzmann equation.
        
        Arguments:
            `dimension` (`int`):
                The dimension of the velocity space (2 or 3).
            `resolution` (`int`):
                The resolution of the velocity grid.
            `v_max` (`float`):
                The maximum velocity.
            `v_where_closed` (`str`, default: 'none'):
                Where the velocity grid is closed ('none', 'periodic', 'dirichlet').
            `dtype` (`torch.dtype`, default: `torch.float`):
                The data type of the tensors.
            `device` (`torch.device`, default: `torch.device('cpu')`):
                The device to use for the tensors.
        """
        self.__dimension = dimension
        self.__resolution = resolution
        self.__v_max = v_max
        self.__v_where_closed = v_where_closed
        self.__dtype = dtype
        self.__device = device
        self.__v_axes = tuple(range(-1-dimension, -1))
        
        self.__v_grid = velocity_grid(dimension, resolution, v_max, where_closed=v_where_closed, dtype=dtype, device=device)
        self.__weight_mass: torch.Tensor = \
            torch.ones((*(resolution for _ in range(dimension)), 1), device=device)
        self.__weight_momentum: torch.Tensor = \
            self.__v_grid.clone()
        self.__weight_energy: torch.Tensor = \
            self.__v_grid.pow(2).sum(dim=-1, keepdim=True)
        
        return

    
    def compute_loss_data(
            self,
            preds:      Sequence[torch.Tensor],
            targets:    Sequence[torch.Tensor],
            p:          float=2.0,
        ) -> torch.Tensor:
        """
        Compute the data loss.
        
        Arguments:
            `preds` (`Sequence[torch.Tensor]`):
                A sequence of predicted tensors.
            `targets` (`Sequence[torch.Tensor]`):
                A sequence of target tensors.
            `p` (`float`, default: 2.0):
                The p-norm to use for the relative error.
        
        Returns:
            `torch.Tensor`:
                The computed data loss.
        """
        assert len(preds)==len(targets)
        loss_data = torch.zeros((1,), device=self.__device)
        for pred, target in zip(preds, targets):
            loss_data += relative_error(pred, target, p=p)
        return loss_data
        
    
    def compute_loss_cons(
            self,
            preds:      Sequence[torch.Tensor],
            targets:    Sequence[torch.Tensor],
            p:          float=2.0,
        ) -> torch.Tensor:
        assert len(preds)==len(targets)
        loss_mass       = torch.zeros((1,), device=self.__device)
        loss_momentum   = torch.zeros((1,), device=self.__device)
        loss_energy     = torch.zeros((1,), device=self.__device)
        for pred, target in zip(preds, targets):
            loss_mass       += \
                relative_error(
                    (pred   * self.__weight_mass).mean(dim=self.__v_axes),
                    (target * self.__weight_mass).mean(dim=self.__v_axes),
                    p=p,
                ).sum()
            loss_momentum   += \
                relative_error(
                    (pred   * self.__weight_momentum).mean(dim=self.__v_axes),
                    (target * self.__weight_momentum).mean(dim=self.__v_axes),
                    p=p,
                ).sum()
            loss_energy     += \
                relative_error(
                    (pred   * self.__weight_energy).mean(dim=self.__v_axes),
                    (target * self.__weight_energy).mean(dim=self.__v_axes),
                    p=p,
                ).sum()
        return loss_mass+loss_momentum+loss_energy
    
    
    def compute_loss(
            self,
            preds:      Sequence[torch.Tensor],
            targets:    Sequence[torch.Tensor],
            p:          float=2.0,
        ) -> torch.Tensor:
        """
        Compute the total loss.
        
        Arguments:
            `preds` (`Sequence[torch.Tensor]`):
                A sequence of predicted tensors.
            `targets` (`Sequence[torch.Tensor]`):
                A sequence of target tensors.
            `p` (`float`, default: 2.0):
                The p-norm to use for the relative error.
        
        Returns:
            `torch.Tensor`:
                The computed total loss.
        """
        loss_data = self.compute_loss_data(preds, targets, p=p)
        loss_cons = self.compute_loss_cons(preds, targets, p=p)
        return loss_data + loss_cons


##################################################
##################################################
# End of file