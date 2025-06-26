from    typing      import  Optional, Sequence, Union, Self
from    pathlib     import  Path
import  torch


##################################################
##################################################
__all__: list[str] = [
    'load_data', 'augment_data_2D',
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
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            directory:  Union[str, Path],
            resolution: int,
            alpha:      Sequence[float],
            part_index: Sequence[int],
            dtype:      torch.dtype = torch.float,
            
            is_parameterized:       bool    = False,
            
            weighting_criterion:    Optional[Union[float, str]] = None,
            num_bins:               int = 30,
        ) -> Self:
        # Check the input arguments
        if not (
            (isinstance(weighting_criterion, float) and weighting_criterion>=1) or
            (isinstance(weighting_criterion, str) and weighting_criterion.lower() in ['entropy'])
        ):
            raise ValueError(
                f"Invalid weighting criterion: {weighting_criterion}. "
                "It should be a float >= 1 or 'entropy'."
            )
        
        # Initialize the basic configurations
        super().__init__()
        self.__directory    = Path(directory) if not isinstance(directory, Path) else directory
        self.__resolution   = resolution
        self.__alpha        = alpha
        self.__part_index   = part_index
        self.__dtype_r      = dtype
        self.__is_parameterized     = is_parameterized
        self.__weighting_criterion  = weighting_criterion
        self.__num_bins             = num_bins
        
        # Initialize auxiliary attributes
        self.__dimension:       Optional[int]   = None
        self.__v_max:           Optional[float] = None
        self.__v_where_closed:  Optional[str]   = None
        self.__scores:          Optional[torch.Tensor] = None

        # Load the data and precompute the weights
        self.__data:    Optional[torch.Tensor] = None
        self.__params:  Optional[torch.Tensor] = None
        self.__weights: Optional[torch.Tensor] = None
        self.__load_data()
        self.__precompute_weights()
        
        # Done
        return
    
    
    def _get_file_name(self, a: float, p: int) -> Path:
        res = str(self.__resolution).zfill(3)
        return self.__directory / f"res{res}__alpha{a:.1e}__part{p}.pth"


    def __len__(self) -> int:
        return self.__data.size(0)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.__is_parameterized:
            return self.__data[index], self.__params[index], self.__weights[index]
        else:
            return self.__data[index], self.__weights[index]

    
    def __load_data(self) -> None:
        from    itertools   import  product
        
        file_list = [
            torch.load(self._get_file_name(a, p), weights_only=False)
            for a, p in product(self.__alpha, self.__part_index)
        ]
        _list_of_v_max  = [f['v_max'] for f in file_list]
        _list_of_v_wc   = [f['v_where_closed'] for f in file_list]
        assert len(set(_list_of_v_max))==1, \
            "All files must have the same values of 'v_max'."
        assert len(set(_list_of_v_wc))==1, \
            "All files must have the same values of 'v_where_closed'."
        
        self.__data = \
            torch.cat([f['data'].type(self.__dtype_r) for f in file_list], dim=0)
        if self.__is_parameterized:
            self.__params = \
                torch.cat(
                    [
                        f['vhs_alpha'] * \
                        torch.ones((len(f['data']), 1), dtype=self.__dtype_r)
                        for f in file_list
                    ],
                    dim = 0,
                )
        
        self.__dimension        = (self.__data.ndim-3)//2
        self.__resolution       = self.__data.size(-2)
        self.__v_max            = float(_list_of_v_max[0])
        self.__v_where_closed   = str(_list_of_v_wc[0])
        
        return
    
    
    def __precompute_weights(self) -> None:
        if self.__weighting_criterion is None:
            return
        
        import  numpy   as  np
        from    deep_numerical.utils        import  velocity_grid, relative_error
        from    deep_numerical.numerical    import  distribution    as  D
        
        old_shape = tuple(self.__data.shape)
        num_traj, num_steps = old_shape[:2]
        new_shape = tuple((num_traj*num_steps, *old_shape[2:]))
        
        v_grid = velocity_grid(
            self.__dimension,
            self.__resolution,
            self.__v_max,
            where_closed    = self.__v_where_closed,
        )
        dist = self.__data.reshape(new_shape)
        equi = D.maxwellian_homogeneous(v_grid, *D.compute_moments_homogeneous(dist, v_grid))
        
        scores: Optional[Union[np.ndarray, torch.Tensor]] = None
        if isinstance(self.__weighting_criterion, float):
            """Compute the relative L^p error."""
            scores = relative_error(dist, equi, p=self.__weighting_criterion)
        elif isinstance(self.__weighting_criterion, str) and self.__weighting_criterion.lower()=='entropy':
            """Compute the entropy."""
            dv = 2*self.__v_max/self.__resolution
            ent_dist = D.compute_entropy_homogeneous(dist, dv)
            ent_equi = D.compute_entropy_homogeneous(equi, dv)
            scores = relative_error(ent_dist, ent_equi) # Here, the order does not matter.
        scores = scores.reshape((num_traj, num_steps)).log10()
        # self.__scores = scores
        
        bins = torch.linspace(scores.min(), scores.max(), 1+self.__num_bins)
        torch.searchsorted(bins, scores)
        
        return
    

##################################################
##################################################
# End of file