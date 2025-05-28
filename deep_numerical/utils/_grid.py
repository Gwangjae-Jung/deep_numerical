from    typing          import  Optional, Union
import  torch
from    ._dtype         import  Objects, TORCH_DEFAULT_DTYPE, TORCH_DEFAULT_DEVICE
from    ._helper        import  zeros, ones, repeat



##################################################
##################################################
__all__: list[str] =  [
    'space_grid',
    'space_index',
    'space_index_tensor',
    'space_index_pair_tensor',
        
    'velocity_grid',
    'velocity_index',
    'velocity_index_tensor',
    
    'compute_conservative_pairs',
    'arg_boundary',
    'arg_boundary_inflow',
    'arg_boundary_outflow',
    
    'arg_specular_velocity',
    'specular_velocity',
]


##################################################
##################################################
# Spatial and velocity grid
def space_grid(
        dimension:      int,
        num_grids:      Objects[int],
        max_values:     Objects[float],
        min_values:     Optional[Objects[float]] = None,
        where_closed:   Optional[Objects[str  ]] = None,
        
        dtype:          torch.dtype     = TORCH_DEFAULT_DTYPE,
        device:         torch.device    = TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Generates the spatial grid.
    
    Arguments:
        `dimension` (`int`):
            The spatial dimension.
        `num_grids` (`Objects[int]`):
            The number of the grids in each dimension.
        `max_values` (`float`):
            The maximum value in each direction.
        `min_values` (`Optional[Objects[float]]`, default: `None`):
            The minimum value in each direction, which is set `-max_value` by default.
            An error is raised if `max_values[i] < min_values[i]` for some index `i`.
        `where_closed` (`Optional[Objects[str]]`, default: `None`):
            Determines which endpoint of each dimension is closed, i.e., contained in the grid. The value of this argument should be given as a single or a sequence of the following strings: `"both"`, `"left"`, `"right"`, or `"none"`. If `where_closed` is not a sequence, then the given configuration is applied for all dimensions.
        `dtype` (`torch.dtype`, default: `torch.float32`):
            The data type of the grid.
        `device` (`torch.device`, default: `torch.device('cpu')`):
            The device on which the grid is created.
    
    Returns:
        `torch.Tensor`: The generated grid of shape `(*num_grids, dimension)`.
        
    ### Note
    This function is designed to generate a grid for the periodic case.
    """
    ##### Redefine the arguments
    # Redefine `num_grids`
    if isinstance(num_grids, int):
        num_grids = tuple(repeat(num_grids, dimension))
    
    # Redefine `max_values`
    if isinstance(max_values, float) or isinstance(max_values, int):
        max_values = tuple(repeat(max_values, dimension))
    
    # Redefine `min_values`
    if min_values is None:
        min_values = tuple((-max_values[i] for i in range(dimension)))
    elif isinstance(min_values, float) or isinstance(min_values, int):
        min_values = tuple(repeat(min_values, dimension))
    
    # Check the region of discretization
    if len(max_values) != len(min_values):
        raise ValueError(
            '\n'.join(
                [
                    f"Shape mismatch:",
                    f"* {dimension=}"
                    f"* {len(min_values)=}"
                    f"* {len(max_values)=}"
                ]
            )
        )
    for idx in range(dimension):
        if max_values[idx] < min_values[idx]:
            raise ValueError(
                '\n'.join(
                    [
                        f"'max_values[idx]' should not be less than 'min_value[idx]'.",
                        f"* {idx=}",
                        f"* {min_values[idx]=}",
                        f"* {max_values[idx]=}",
                    ]
                )
            )
    
    # Redefine `where_closed`
    _where_closed_permitted = ('both', 'left', 'right', 'none')
    if where_closed is None:
        where_closed = 'none'
    if isinstance(where_closed, str):
        where_closed = tuple(repeat(where_closed, dimension))
    where_closed = tuple((x.lower() for x in where_closed))
    
    # Check if every entry of `where_closed` belongs to `_where_closed_permitted`
    for idx in range(dimension):
        if where_closed[idx] not in _where_closed_permitted:
            raise ValueError(f"For the index {idx}, the configuration is {where_closed[idx]}, which is not in {_where_closed_permitted}.")
    
    ##### Create the grid for each case
    list_of_grid = []
    for d in range(dimension):
        __left:     float
        __right:    float
        __num_d = num_grids[d]
        __max_d = max_values[d]
        __min_d = min_values[d]
        __dx_d_not0 = (__max_d - __min_d) / __num_d
        # Both endpoints are included
        if where_closed[d] == _where_closed_permitted[0]:
            __left  = __min_d
            __right = __max_d
        # Only the left endpoint is included
        elif where_closed[d] == _where_closed_permitted[1]:
            __left  = __min_d
            __right = __max_d - __dx_d_not0
        # Only the right endpoint is included
        elif where_closed[d] == _where_closed_permitted[2]:
            __left  = __min_d + __dx_d_not0
            __right = __max_d
        # Both endpoints are excluded
        elif where_closed[d] == _where_closed_permitted[3]:
            __left  = __min_d + __dx_d_not0 / 2
            __right = __max_d - __dx_d_not0 / 2
        else:
            ValueError(f"Unexpected configuration {where_closed[d]=} with {d=} is encountered.")
        list_of_grid.append(
            torch.linspace(__left, __right, __num_d, dtype=dtype, device=device)
        )
    
    return torch.stack(torch.meshgrid(*list_of_grid, indexing='ij'), dim=-1)


def space_index(
        n:      int,
        device: torch.device=TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Same as `torch.arange(n, device=device)`."""
    return torch.arange(n, device=device)


def space_index_tensor(
        dimension:  int,
        num_grids:  Objects[int],
        keepdim:    bool = False,
        device:     torch.device=TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return the collection of all possible spatial indices."""
    indices: torch.Tensor = torch.stack(
        torch.meshgrid(
            *repeat(space_index(num_grids, device), dimension),
            indexing='ij',
        ),
        dim = -1
    )
    if keepdim:
        return indices
    else:
        return indices.reshape(-1, dimension)


def space_index_pair_tensor(
        dimension:  int,
        num_grids:  Objects[int],
        keepdim:    bool = False,
        device:     torch.device=TORCH_DEFAULT_DEVICE,
    ) -> torch.Tensor:
    """Return the collection of all possible pairs of spatial indices."""
    if isinstance(num_grids, int):
        num_grids = repeat(num_grids, dimension)
    elif len(num_grids)!=dimension:
        raise ValueError(f"The length of 'num_grids' should be equal to 'dimension'.")
    _list_of_grids = [space_index(_num_grid, device) for _num_grid in num_grids]
    indices: torch.Tensor = torch.stack(
        torch.meshgrid(*(2*_list_of_grids), indexing='ij'),
        dim = -1,
    )
    if keepdim:
        return indices
    else:
        return indices.reshape(-1, 2*dimension)


velocity_grid           = space_grid
velocity_index          = space_index
velocity_index_tensor   = space_index_tensor


##################################################
##################################################
# Indices of the boundary points
def arg_boundary(
        points:             torch.Tensor,
        contains_velocity:  bool    = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Returns the indices of the boundary points.
    
    -----
    ### Note
    1. This function also works for the spatio-velocity grid. If the velocity space is also discretized, then pass `contains_velocity=True`; otherwise, pass `contains_velocity=False`.
    2. So far, this function only works for cubic grids.
    """
    dim = points.shape[-1]//2 if contains_velocity else points.shape[-1]
    grid_x = points[..., :dim]
    x_max:          float = torch.max(torch.abs(grid_x)).item()
    delta_x_min:    float = torch.min(grid_x[*ones(dim)] - grid_x[*zeros(dim)]).item()
    LHS = torch.max( torch.abs(points[..., :dim]), dim=-1 )
    RHS = x_max - 0.5*delta_x_min
    return torch.argwhere(LHS > RHS)


def arg_boundary_inflow(
        xv:             torch.Tensor,
        return_normals: bool    = False,
        eps:            float   = 1e-12,
    ) -> Objects[torch.Tensor]:
    """Returns the indices of the inflow boundary points.
    
    -----
    ### Description
    Given a spatio-velocity grid `xv`, this function computes all indices of the outflow boundary points, i.e., the points `[*x, *v]` for which `dot(x, v) > 0`.
    
    -----
    ### Parameters
    * `xv` (`torch.Tensor`)
        * The input spatio-velocity grid of shape `(*repeat(resolution_x, dimension), *repeat(resolution_v, dimension), 2*dimension)`.
    * `return_normals` (`bool`, default: `False`)
        * If `True`, then this function also returns the tensor of the unit normal vectors computed at each boundary point. Else, this function returns only the indicies explained above.
    
    -----
    ### Note
    1. So far, this function only works for cubic grids.
    """
    # Precompute several variables
    dim: int = xv.shape[-1]//2
    x_max:          float = \
        torch.max(torch.abs(xv[..., *zeros(dim), :dim])).item()
    delta_x_min:    float = \
        torch.min(xv[*ones(dim), *zeros(dim), :dim] - xv[*zeros(dim), *zeros(dim), :dim]).item()
    
    # Find the boundary points
    arg_bd = arg_boundary(xv, contains_velocity=True)
    bd = xv[*(arg_bd[:, d] for d in range(2*dim))]
    bd = bd.reshape(-1, 2*dim)  # The list of boundary points

    # Extract the outward vectors
    normals = bd[..., :dim] # Outward normal vectors
    normals = torch.sign(normals) * torch.where(torch.abs(normals) > x_max-0.5*delta_x_min, 1, 0)
    normals = normals / torch.norm(normals, p=2, dim=-1, keepdims=True)  # Normalization
    
    # Compute the dot product with the normal vector and the velocity
    dot_n_v: torch.Tensor = torch.einsum("...i, ...i -> ...", normals, bd[..., dim:])
    arg_inflow = torch.argwhere(dot_n_v < -eps)[..., 0]
    assert arg_inflow.ndim==1
    arg_inflow = arg_bd[arg_inflow]
    
    if return_normals:
        return (arg_inflow, normals)
    else:
        return arg_inflow


def arg_boundary_outflow(
        xv:             torch.Tensor,
        return_normals: bool    = False,
        eps:            float   = 1e-12,
    ) -> Objects[torch.Tensor]:
    """Returns the indices of the outflow boundary points.
    
    -----
    ### Description
    Given a spatio-velocity grid `xv`, this function computes all indices of the outflow boundary points, i.e., the points `[*x, *v]` for which `dot(x, v) > 0`.
    
    -----
    ### Parameters
    * `xv` (`torch.Tensor`)
        * The input spatio-velocity grid of shape `(*repeat(resolution_x, dimension), *repeat(resolution_v, dimension), 2*dimension)`.
    * `return_normals` (`bool`, default: `False`)
        * If `True`, then this function also returns the tensor of the unit normal vectors computed at each boundary point. Else, this function returns only the indicies explained above.
    
    -----
    ### Note
    1. So far, this function only works for cubic grids.
    """
    # Precompute several variables
    dim: int = xv.shape[-1]//2
    x_max:          float = \
        torch.max(torch.abs(xv[..., *zeros(dim), :dim])).item()
    delta_x_min:    float = \
        torch.min(xv[*ones(dim), *zeros(dim), :dim] - xv[*zeros(dim), *zeros(dim), :dim]).item()
    
    # Find the boundary points
    arg_bd = arg_boundary(xv, contains_velocity=True)
    bd = xv[*(arg_bd[:, d] for d in range(2*dim))]
    bd = bd.reshape(-1, 2*dim)  # The list of boundary points

    # Extract the outward vectors
    normals = bd[..., :dim] # Outward normal vectors
    normals = torch.sign(normals) * torch.where(torch.abs(normals) > x_max-0.5*delta_x_min, 1, 0)
    normals = normals / torch.norm(normals, p=2, dim=-1, keepdims=True)  # Normalization
    
    # Compute the dot product with the normal vector and the velocity
    dot_n_v: torch.Tensor = torch.einsum("...i, ...i -> ...", normals, bd[..., dim:])
    arg_inflow = torch.argwhere(dot_n_v > eps)[..., 0]
    assert arg_inflow.ndim==1
    arg_inflow = arg_bd[arg_inflow]
    
    if return_normals:
        return (arg_inflow, normals)
    else:
        return arg_inflow
    
    
def arg_specular_velocity(
        resolution_v:   int,
        idx:            torch.Tensor,
        dim:            Optional[int]   = None,
    ) -> torch.Tensor:
    """This function returns the indices of the velocity-specular points.
    
    -----
    # Note
    1. This function is not suggested to be used in practice. If you already have a tensor of points to be reflected in the velocity space, use `specular_velocity` instead.
    2. So far, this function only works for cubic grids.
    """
    assert idx.ndim==2
    if dim is not None:
        assert idx.shape[-1]%2==0 and idx.shape[-1]//2==dim, \
            f"The passed argument 'idx' has shape {idx.shape}, and the last dimension should be of dimension 2*'dim', but 'dim'={dim}."
    else:
        assert idx.shape[-1]%2==0, \
            f"The passed argument 'idx' has shape {idx.shape}, and the last dimension should be of positive and even dimension."
        dim = idx.shape[-1]//2
    idx_specular = idx.clone()
    idx_specular[..., dim:] = (resolution_v-1) - idx_specular[..., dim:]
    return idx_specular


def specular_velocity(xv: torch.Tensor) -> torch.Tensor:
    """This function returns the velocity-specular points.
    """
    assert xv.shape[-1]%2 == 0
    dim = xv.shape[-1]//2
    specular_velocity = torch.flip(xv[..., dim:], dim=range(-1-dim, -1))
    ret = xv.clone()
    ret[..., dim:] = specular_velocity
    return ret
    
    
##################################################
##################################################
def compute_conservative_pairs(
        dimension:  int,
        num_grids:  Objects[int],
        
        verbose:    bool = False,
    ) -> dict[tuple[int], torch.Tensor]:
    """Computes the pairs of the velocity indices for which the conservation laws are satisfied.
    
    -----
    ### Note
    This function assumes that the velocity space is discretized using the same step size throughout all dimensions.
    """    
    indices     = space_index_tensor(dimension, num_grids)
    idx_pairs   = space_index_pair_tensor(dimension, num_grids)
    ret: dict[tuple[int], torch.Tensor] = {}
    
    if verbose:
        from    tqdm.notebook       import  tqdm
        it = tqdm(idx_pairs, desc='Computing the pairs for which the conserative laws are satisfied')
    else:
        it = idx_pairs
    
    def _compute_momentum_and_energy(pair: torch.Tensor) -> tuple[torch.Tensor, int]:
        a1, a2 = pair[:dimension], pair[dimension:]
        momentum    = a1 + a2
        energy      = int(torch.sum(pair**2))
        return momentum, energy
    
    for pair in it:
        m, e = _compute_momentum_and_energy(pair)
        ret_at_pair = []
        for j1 in indices:
            j2 = m - j1
            if torch.any(torch.max(j2) >= num_grids) or torch.any(torch.min(j2) < 0):
                continue
            e_ = int(torch.sum(j1**2 + j2**2))
            if e != e_:
                continue
            ret_at_pair.append(torch.concatenate((j1, j2)))
        ret_at_pair = torch.array(ret_at_pair, dtype=torch.long)
        ret[tuple(pair)] = ret_at_pair
    
    return ret


##################################################
##################################################
# End of file