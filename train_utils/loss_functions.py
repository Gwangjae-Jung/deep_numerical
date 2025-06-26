from    typing      import  Sequence, Self
import  torch

from    sys         import  path    as  sys_path
from    pathlib     import  Path
sys_path.append(str(Path(__file__).parent))
from    deep_numerical.utils        import  velocity_grid, relative_error


##################################################
##################################################
__all__: list[str] = [
    'LossFunctions',
]


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