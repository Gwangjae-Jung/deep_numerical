from   typing         import Callable, Literal, Optional, Sequence, Union
import torch

from   deep_numerical import Objects, ones, repeat, zeros


__all__: list[str] = [
    'cartesian_grid',
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
def cartesian_grid(*tensors: torch.Tensor) -> torch.Tensor:
    """Computes a Cartesian grid from 1D coordinate tensors.

    ## Description
    Given multiple 1D coordinate tensors, this function generates the multi-dimensional
    Cartesian product grid using meshgrid indexing `'ij'` and stacks the coordinates along the last dimension.

    ## Arguments
    `*tensors` (`torch.Tensor`): 1D coordinate tensors for each axis.

    ## Returns
    `torch.Tensor`: Multi-dimensional grid tensor of shape `(*shapes, len(tensors))`.
    """
    return torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)


def space_grid(
    dimension:    int,
    num_grids:    Objects[int],
    max_values:   Objects[float],
    min_values:   Optional[Objects[float]]                          = None,
    where_closed: Optional[Literal['both', 'left', 'right', 'none']] = None,
    dtype:        Optional[torch.dtype]                             = None,
    device:       Optional[torch.device]                            = None,
) -> torch.Tensor:
    """Generates the spatial grid.

    ## Description
    Generates a uniform spatial discretization grid across `dimension` dimensions with custom domain boundaries and closure configurations.

    ## Arguments
    `dimension` (`int`): The spatial dimension.
    `num_grids` (`Objects[int]`): The number of grids in each dimension.
    `max_values` (`Objects[float]`): The maximum value in each direction.
    `min_values` (`Optional[Objects[float]]`, default: `None`): The minimum value in each direction. If `None`, set to `-max_values`.
    `where_closed` (`Optional[Literal['both', 'left', 'right', 'none']]`, default: `None`): Determines which endpoint of each dimension is closed.
    `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the grid.
    `device` (`Optional[torch.device]`, default: `None`): The device on which the grid is created.

    ## Returns
    `torch.Tensor`: The generated grid of shape `(*num_grids, dimension)`.
    """
    if isinstance(num_grids, int):
        num_grids_seq: tuple[int, ...] = tuple(repeat(num_grids, dimension))
    else:
        num_grids_seq = tuple(num_grids)

    if isinstance(max_values, (float, int)):
        max_values_seq: tuple[float, ...] = tuple(repeat(float(max_values), dimension))
    else:
        max_values_seq = tuple(max_values)

    if min_values is None:
        min_values_seq: tuple[float, ...] = tuple((-max_values_seq[i] for i in range(dimension)))
    elif isinstance(min_values, (float, int)):
        min_values_seq = tuple(repeat(float(min_values), dimension))
    else:
        min_values_seq = tuple(min_values)

    if len(max_values_seq) != len(min_values_seq):
        raise ValueError(
            f"Shape mismatch: dimension={dimension}, len(min_values)={len(min_values_seq)}, len(max_values)={len(max_values_seq)}"
        )
    for idx in range(dimension):
        if max_values_seq[idx] < min_values_seq[idx]:
            raise ValueError(
                f"'max_values[idx]' should not be less than 'min_values[idx]': idx={idx}, min={min_values_seq[idx]}, max={max_values_seq[idx]}"
            )

    _where_closed_permitted = ('both', 'left', 'right', 'none')
    if where_closed is None:
        where_closed_val: tuple[str, ...] = tuple(repeat('none', dimension))
    elif isinstance(where_closed, str):
        where_closed_val = tuple(repeat(where_closed.lower(), dimension))
    else:
        where_closed_val = tuple((x.lower() for x in where_closed))

    for idx in range(dimension):
        if where_closed_val[idx] not in _where_closed_permitted:
            raise ValueError(f"For index {idx}, configuration '{where_closed_val[idx]}' is not in {_where_closed_permitted}.")

    list_of_grid: list[torch.Tensor] = []
    for d in range(dimension):
        __left:      float
        __right:     float
        __num_d:     int   = num_grids_seq[d]
        __max_d:     float = max_values_seq[d]
        __min_d:     float = min_values_seq[d]
        __dx_d_not0: float = (__max_d - __min_d) / __num_d

        if where_closed_val[d] == 'both':
            __left  = __min_d
            __right = __max_d
        elif where_closed_val[d] == 'left':
            __left  = __min_d
            __right = __max_d - __dx_d_not0
        elif where_closed_val[d] == 'right':
            __left  = __min_d + __dx_d_not0
            __right = __max_d
        elif where_closed_val[d] == 'none':
            __left  = __min_d + __dx_d_not0 / 2
            __right = __max_d - __dx_d_not0 / 2
        else:
            raise ValueError(f"Unexpected configuration where_closed={where_closed_val[d]} at d={d}.")

        list_of_grid.append(
            torch.linspace(__left, __right, __num_d, dtype=dtype, device=device)
        )

    return torch.stack(torch.meshgrid(*list_of_grid, indexing='ij'), dim=-1)


def space_index(
    n:      int,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """Generates 1D spatial coordinate indices.

    ## Description
    Returns a 1D tensor of sequential indices from `0` to `n - 1`. Equivalent to `torch.arange(n, device=device)`.

    ## Arguments
    `n` (`int`): Number of grid points.
    `device` (`Optional[torch.device]`, default: `None`): Device on which the tensor is created.

    ## Returns
    `torch.LongTensor`: 1D index tensor of shape `(n,)`.
    """
    return torch.arange(n, device=device)


def space_index_tensor(
    dimension: int,
    num_grids: Objects[int],
    keepdim:   bool                   = False,
    device:    Optional[torch.device] = None,
) -> torch.LongTensor:
    """Generates multi-dimensional spatial grid indices.

    ## Description
    Returns the collection of all multi-index coordinate tuples across a `dimension`-dimensional grid.

    ## Arguments
    `dimension` (`int`): The spatial dimension.
    `num_grids` (`Objects[int]`): Number of grid points in each dimension.
    `keepdim` (`bool`, default: `False`): If `True`, returns tensor of shape `(*num_grids, dimension)`. Otherwise, flattens to `(-1, dimension)`.
    `device` (`Optional[torch.device]`, default: `None`): Device on which the tensor is created.

    ## Returns
    `torch.LongTensor`: Index tensor.
    """
    indices: torch.LongTensor = torch.stack(
        torch.meshgrid(
            *repeat(space_index(num_grids, device), dimension),
            indexing = 'ij',
        ),
        dim = -1,
    )
    if keepdim:
        return indices
    else:
        return indices.reshape(-1, dimension)


def space_index_pair_tensor(
    dimension: int,
    num_grids: Objects[int],
    keepdim:   bool                   = False,
    device:    Optional[torch.device] = None,
) -> torch.LongTensor:
    """Generates all pairs of multi-dimensional spatial grid indices.

    ## Description
    Returns the collection of all possible pairs of coordinate indices across a `dimension`-dimensional grid.

    ## Arguments
    `dimension` (`int`): The spatial dimension.
    `num_grids` (`Objects[int]`): Number of grid points in each dimension.
    `keepdim` (`bool`, default: `False`): If `True`, returns shape `(*(2*num_grids), 2*dimension)`. Otherwise, flattens to `(-1, 2*dimension)`.
    `device` (`Optional[torch.device]`, default: `None`): Device on which the tensor is created.

    ## Returns
    `torch.LongTensor`: Paired index tensor.
    """
    if isinstance(num_grids, int):
        num_grids_seq: tuple[int, ...] = tuple(repeat(num_grids, dimension))
    elif len(num_grids) != dimension:
        raise ValueError("The length of 'num_grids' should be equal to 'dimension'.")
    else:
        num_grids_seq = tuple(num_grids)

    _list_of_grids: list[torch.LongTensor] = [space_index(_num_grid, device) for _num_grid in num_grids_seq]
    indices: torch.LongTensor = torch.stack(
        torch.meshgrid(*(2 * _list_of_grids), indexing='ij'),
        dim = -1,
    )
    if keepdim:
        return indices
    else:
        return indices.reshape(-1, 2 * dimension)


def arg_boundary(
    points:            torch.Tensor,
    contains_velocity: bool = False,
) -> torch.LongTensor:
    """Returns the indices of the boundary points.

    ## Description
    Finds and returns indices of points on the boundary of a cubic spatial (or spatio-velocity) grid.

    ## Arguments
    `points` (`torch.Tensor`): Input coordinate grid tensor.
    `contains_velocity` (`bool`, default: `False`): Whether `points` contains both spatial and velocity components.

    ## Returns
    `torch.LongTensor`: Indices of boundary points.
    """
    dim:         int          = points.shape[-1] // 2 if contains_velocity else points.shape[-1]
    grid_x:      torch.Tensor = points[..., :dim]
    x_max:       float        = torch.max(torch.abs(grid_x)).item()
    delta_x_min: float        = torch.min(grid_x[*ones(dim)] - grid_x[*zeros(dim)]).item()
    lhs:         torch.Tensor = torch.max(torch.abs(points[..., :dim]), dim=-1)
    rhs:         float        = x_max - 0.5 * delta_x_min
    return torch.argwhere(lhs > rhs)


def arg_boundary_inflow(
    xv:             torch.Tensor,
    return_normals: bool  = False,
    eps:            float = 1e-12,
) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.Tensor]]:
    """Returns the indices of the inflow boundary points.

    ## Description
    Given a spatio-velocity grid `xv`, computes all indices of inflow boundary points, i.e., points where `dot(normal, v) < -eps`.

    ## Arguments
    `xv` (`torch.Tensor`): Spatio-velocity grid of shape `(*resolution_x, *resolution_v, 2*dimension)`.
    `return_normals` (`bool`, default: `False`): Whether to also return the unit normal vectors.
    `eps` (`float`, default: `1e-12`): Threshold tolerance for inner product.

    ## Returns
    `Union[torch.LongTensor, tuple[torch.LongTensor, torch.Tensor]]`: Inflow indices, or tuple with normal vectors.
    """
    dim:         int          = xv.shape[-1] // 2
    x_max:       float        = torch.max(torch.abs(xv[..., *zeros(dim), :dim])).item()
    delta_x_min: float        = torch.min(xv[*ones(dim), *zeros(dim), :dim] - xv[*zeros(dim), *zeros(dim), :dim]).item()

    arg_bd: torch.LongTensor = arg_boundary(xv, contains_velocity=True)
    bd:     torch.Tensor     = xv[*(arg_bd[:, d] for d in range(2 * dim))].reshape(-1, 2 * dim)

    normals: torch.Tensor = bd[..., :dim]
    normals = torch.sign(normals) * torch.where(torch.abs(normals) > x_max - 0.5 * delta_x_min, 1, 0)
    normals = normals / torch.norm(normals, p=2, dim=-1, keepdims=True)

    dot_n_v:    torch.Tensor   = torch.einsum("...i, ...i -> ...", normals, bd[..., dim:])
    arg_inflow: torch.Tensor   = torch.argwhere(dot_n_v < -eps)[..., 0]
    arg_inflow_final: torch.LongTensor = arg_bd[arg_inflow]

    if return_normals:
        return (arg_inflow_final, normals)
    else:
        return arg_inflow_final


def arg_boundary_outflow(
    xv:             torch.Tensor,
    return_normals: bool  = False,
    eps:            float = 1e-12,
) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.Tensor]]:
    """Returns the indices of the outflow boundary points.

    ## Description
    Given a spatio-velocity grid `xv`, computes all indices of outflow boundary points, i.e., points where `dot(normal, v) > eps`.

    ## Arguments
    `xv` (`torch.Tensor`): Spatio-velocity grid of shape `(*resolution_x, *resolution_v, 2*dimension)`.
    `return_normals` (`bool`, default: `False`): Whether to also return the unit normal vectors.
    `eps` (`float`, default: `1e-12`): Threshold tolerance for inner product.

    ## Returns
    `Union[torch.LongTensor, tuple[torch.LongTensor, torch.Tensor]]`: Outflow indices, or tuple with normal vectors.
    """
    dim:         int          = xv.shape[-1] // 2
    x_max:       float        = torch.max(torch.abs(xv[..., *zeros(dim), :dim])).item()
    delta_x_min: float        = torch.min(xv[*ones(dim), *zeros(dim), :dim] - xv[*zeros(dim), *zeros(dim), :dim]).item()

    arg_bd: torch.LongTensor = arg_boundary(xv, contains_velocity=True)
    bd:     torch.Tensor     = xv[*(arg_bd[:, d] for d in range(2 * dim))].reshape(-1, 2 * dim)

    normals: torch.Tensor = bd[..., :dim]
    normals = torch.sign(normals) * torch.where(torch.abs(normals) > x_max - 0.5 * delta_x_min, 1, 0)
    normals = normals / torch.norm(normals, p=2, dim=-1, keepdims=True)

    dot_n_v:     torch.Tensor   = torch.einsum("...i, ...i -> ...", normals, bd[..., dim:])
    arg_outflow: torch.Tensor   = torch.argwhere(dot_n_v > eps)[..., 0]
    arg_outflow_final: torch.LongTensor = arg_bd[arg_outflow]

    if return_normals:
        return (arg_outflow_final, normals)
    else:
        return arg_outflow_final


def arg_specular_velocity(
    resolution_v: int,
    idx:          torch.LongTensor,
    dim:          Optional[int] = None,
) -> torch.LongTensor:
    """Returns the indices of the velocity-specular points.

    ## Description
    Computes reflected indices in velocity space across axes.

    ## Arguments
    `resolution_v` (`int`): Grid resolution in velocity space.
    `idx` (`torch.LongTensor`): Tensor of coordinate index pairs.
    `dim` (`Optional[int]`, default: `None`): Velocity dimension.

    ## Returns
    `torch.LongTensor`: Specularly reflected index tensor.
    """
    assert idx.ndim == 2
    if dim is not None:
        assert idx.shape[-1] % 2 == 0 and idx.shape[-1] // 2 == dim
    else:
        assert idx.shape[-1] % 2 == 0
        dim = idx.shape[-1] // 2
    idx_specular: torch.LongTensor = idx.clone()
    idx_specular[..., dim:] = (resolution_v - 1) - idx_specular[..., dim:]
    return idx_specular


def specular_velocity(xv: torch.Tensor) -> torch.Tensor:
    """Computes specular velocity points.

    ## Description
    Given a spatio-velocity coordinate tensor `xv`, this function inverts / reflects
    the velocity components across the velocity grid axes.

    ## Arguments
    `xv` (`torch.Tensor`): Tensor of points whose last dimension has even length `2 * dim`.

    ## Returns
    `torch.Tensor`: Tensor with specularly reflected velocities.
    """
    assert xv.shape[-1] % 2 == 0
    dim: int = xv.shape[-1] // 2
    specular_vel: torch.Tensor = torch.flip(xv[..., dim:], dim=range(-1 - dim, -1))
    ret:          torch.Tensor = xv.clone()
    ret[..., dim:] = specular_vel
    return ret


def compute_conservative_pairs(
    dimension: int,
    num_grids: Objects[int],
    verbose:   bool         = False,
) -> dict[tuple[int, ...], torch.LongTensor]:
    """Computes pairs of velocity indices for which conservation laws are satisfied.

    ## Description
    Computes all pairs of velocity indices `(j1, j2)` which preserve total momentum and energy from an initial pair.

    ## Arguments
    `dimension` (`int`): Dimension of velocity space.
    `num_grids` (`Objects[int]`): Number of grids in each dimension.
    `verbose` (`bool`, default: `False`): Whether to display progress bar.

    ## Returns
    `dict[tuple[int, ...], torch.LongTensor]`: Mapping from index pairs to satisfying collision pairs.
    """
    indices:   torch.LongTensor = space_index_tensor(dimension, num_grids)
    idx_pairs: torch.LongTensor = space_index_pair_tensor(dimension, num_grids)
    ret:       dict[tuple[int, ...], torch.LongTensor] = {}

    if verbose:
        from tqdm.notebook import tqdm
        it: Any = tqdm(idx_pairs, desc='Computing pairs satisfying conservation laws')
    else:
        it = idx_pairs

    def _compute_momentum_and_energy(pair: torch.LongTensor) -> tuple[torch.LongTensor, int]:
        a1:     torch.LongTensor = pair[:dimension]
        a2:     torch.LongTensor = pair[dimension:]
        mom:    torch.LongTensor = a1 + a2
        energy: int              = int(torch.sum(pair ** 2))
        return mom, energy

    for pair in it:
        m, e = _compute_momentum_and_energy(pair)
        ret_at_pair: list[torch.Tensor] = []
        for j1 in indices:
            j2: torch.LongTensor = m - j1
            if torch.any(torch.max(j2) >= num_grids) or torch.any(torch.min(j2) < 0):
                continue
            e_: int = int(torch.sum(j1 ** 2 + j2 ** 2))
            if e != e_:
                continue
            ret_at_pair.append(torch.concatenate((j1, j2)))
        ret_at_pair_tensor: torch.LongTensor = (
            torch.stack(ret_at_pair) if len(ret_at_pair) > 0 else torch.empty((0, 2 * dimension), dtype=torch.long)
        )
        ret[tuple(pair.tolist())] = ret_at_pair_tensor

    return ret


# Velocity aliases
velocity_grid:         Callable[..., torch.Tensor]     = space_grid
velocity_index:        Callable[..., torch.LongTensor] = space_index
velocity_index_tensor: Callable[..., torch.LongTensor] = space_index_tensor


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()