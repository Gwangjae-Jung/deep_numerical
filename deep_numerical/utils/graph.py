import warnings
from   typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from   deep_numerical import ArrayData, Objects


__all__: list[str] = [
    "GridGenerator",
    "node_idx_to_arr_idx",
    "generate_grid",
    "generate_radius_graph",
    "generate_k_nearest_neighbor_graph",
]

_DEFAULT_RADIUS_RANGE:     tuple[float, float] = (0.0, float('inf'))
_DEFAULT_RADIUS_INCLUSION: tuple[bool, bool]   = (True, True)
_DEFAULT_MAX_NEIGHBORS:    Optional[int]       = None
_DEFAULT_ALLOW_LOOP:       bool                = False


class GridGenerator():
    """The base grid generator.

    ## Description
    This class is a base class for other types of grid-generating classes.
    Given `domain` and `grid_size`, it generates a uniform grid and provides methods to construct radius graphs and k-nearest-neighbor graphs.

    ## Arguments
    `domain` (`Sequence[Sequence[float]]`): Domain bounding box as min/max pairs for each dimension.
    `grid_size` (`Sequence[int]`): Number of grid points along each dimension.
    `radius_range` (`Optional[Sequence[float]]`, default: `_DEFAULT_RADIUS_RANGE`): Min and max radius for radius graph connections.
    `radius_inclusion` (`Optional[Sequence[bool]]`, default: `_DEFAULT_RADIUS_INCLUSION`): Whether boundary radii are inclusive.
    `max_neighbors` (`Optional[int]`, default: `_DEFAULT_MAX_NEIGHBORS`): Maximum number of nearest neighbors.
    `allow_loop` (`Optional[bool]`, default: `_DEFAULT_ALLOW_LOOP`): Whether to allow self-loops in generated graphs.
    """

    def __init__(
        self,
        domain:           Sequence[Sequence[float]],
        grid_size:        Sequence[int],
        radius_range:     Optional[Sequence[float]] = _DEFAULT_RADIUS_RANGE,
        radius_inclusion: Optional[Sequence[bool]]  = _DEFAULT_RADIUS_INCLUSION,
        max_neighbors:    Optional[int]             = _DEFAULT_MAX_NEIGHBORS,
        allow_loop:       Optional[bool]            = _DEFAULT_ALLOW_LOOP,
    ) -> None:
        """Initializes the `GridGenerator`.

        ## Description
        Generates a uniform grid on `domain` with shape `grid_size`, and prepares parameters
        for radius and k-nearest-neighbor subgraph generation.

        ## Arguments
        `domain` (`Sequence[Sequence[float]]`): Domain bounding box as min/max pairs for each dimension.
        `grid_size` (`Sequence[int]`): Number of grid points along each dimension.
        `radius_range` (`Optional[Sequence[float]]`, default: `_DEFAULT_RADIUS_RANGE`): Min and max radius for radius graph connections.
        `radius_inclusion` (`Optional[Sequence[bool]]`, default: `_DEFAULT_RADIUS_INCLUSION`): Whether boundary radii are inclusive.
        `max_neighbors` (`Optional[int]`, default: `_DEFAULT_MAX_NEIGHBORS`): Maximum number of nearest neighbors.
        `allow_loop` (`Optional[bool]`, default: `_DEFAULT_ALLOW_LOOP`): Whether to allow self-loops in generated graphs.

        ## Returns
        `None`: None.
        """
        self.__grid:      torch.Tensor  = generate_grid(domain, grid_size, keep_shape=False)
        self.__grid_size: Sequence[int] = grid_size

        self.__radius_range:     tuple[float, float] = tuple(radius_range) if radius_range is not None else _DEFAULT_RADIUS_RANGE
        self.__radius_inclusion: tuple[bool, bool]   = tuple(radius_inclusion) if radius_inclusion is not None else _DEFAULT_RADIUS_INCLUSION

        self.__max_neighbors: Optional[int] = max_neighbors
        self.__allow_loop:    Optional[bool] = allow_loop
        return None

    @property
    def radius_range(self) -> tuple[float, float]:
        """The range of radius with respect to which the radius graph will be generated."""
        return self.__radius_range

    @property
    def radius_inclusion(self) -> tuple[bool, bool]:
        """The sequence which saves the inclusion of the marginal radii."""
        return self.__radius_inclusion

    @property
    def allow_loop(self) -> Optional[bool]:
        """The boolean option `allow_loop`."""
        return self.__allow_loop

    @property
    def max_neighbors(self) -> Optional[int]:
        """The maximum number of the neighbors."""
        return self.__max_neighbors

    @property
    def grid_size(self) -> Sequence[int]:
        """The size of the grid."""
        return self.__grid_size

    @property
    def grid(self) -> torch.Tensor:
        """The clone of the grid coordinates."""
        return self.__grid.clone()

    @property
    def num_nodes(self) -> int:
        """The number of the nodes in the grid."""
        return self.__grid.size(0)

    @property
    def dim_domain(self) -> int:
        """The dimension of the domain."""
        return self.__grid.size(1)

    @property
    def node_index(self) -> torch.LongTensor:
        """The tensor of the indices of the nodes in the grid."""
        return torch.arange(self.num_nodes)

    def construct_graph(
        self,
        subgrid_index:    Optional[torch.LongTensor] = None,
        point_cloud:      Optional[torch.Tensor]     = None,
        radius_range:     Optional[Sequence[float]]  = None,
        radius_inclusion: Optional[Sequence[bool]]   = None,
        max_neighbors:    Optional[int]              = None,
        allow_loop:       Optional[bool]             = None,
        p:                float                      = 2.0,
    ) -> torch.LongTensor:
        """Generates the radius graph for the (sub)grid or point cloud.

        ## Description
        Constructs the radius graph for the passed subgraph or default configuration.

        ## Arguments
        `subgrid_index` (`Optional[torch.LongTensor]`, default: `None`): Subgrid node index selection.
        `point_cloud` (`Optional[torch.Tensor]`, default: `None`): Point cloud coordinates.
        `radius_range` (`Optional[Sequence[float]]`, default: `None`): Minimum and maximum radius.
        `radius_inclusion` (`Optional[Sequence[bool]]`, default: `None`): Boundary radius inclusion.
        `max_neighbors` (`Optional[int]`, default: `None`): Maximum neighbors per node.
        `allow_loop` (`Optional[bool]`, default: `None`): Whether self-loops are allowed.
        `p` (`float`, default: `2.0`): Distance Minkowski norm order.

        ## Returns
        `torch.LongTensor`: Edge index tensor of shape `(2, num_edges)`.
        """
        coords: torch.Tensor
        if subgrid_index is not None:
            coords = self.grid[subgrid_index]
        elif point_cloud is not None:
            coords = point_cloud
        else:
            coords = self.grid

        if radius_range is None:
            radius_range = self.radius_range
        if radius_inclusion is None:
            radius_inclusion = self.radius_inclusion
        if max_neighbors is None:
            max_neighbors = self.max_neighbors
        if allow_loop is None:
            allow_loop = self.allow_loop

        return generate_radius_graph(
            coords           = coords,
            radius_range     = radius_range,
            radius_inclusion = radius_inclusion,
            max_neighbors    = max_neighbors,
            allow_loop       = bool(allow_loop),
            p                = p,
        )


##################################################
def node_idx_to_arr_idx(
    index:     Union[Objects[int], ArrayData],
    num_grids: List[int],
) -> np.ndarray:
    """Converts 1D node indices to multi-dimensional grid indices.

    ## Description
    Given node indices and the number of grids per axis, returns the corresponding coordinate indices on the grid.

    ## Arguments
    `index` (`Union[Objects[int], ArrayData]`): 1D sequence of flattened node indices.
    `num_grids` (`List[int]`): List of grid resolution per dimension.

    ## Returns
    `np.ndarray`: 2D array of coordinate indices of shape `(len(index), len(num_grids))`.
    """
    if not isinstance(index, np.ndarray) and not isinstance(index, torch.Tensor):
        index_arr: np.ndarray = np.array(index)
    elif isinstance(index, torch.Tensor):
        index_arr = index.detach().cpu().numpy()
    else:
        index_arr = index
    assert index_arr.ndim == 1

    ret: np.ndarray = np.zeros(shape=(len(index_arr), len(num_grids)), dtype=np.int32)
    cnt: int        = 1
    idx_copy: np.ndarray = index_arr.copy()
    for n_grid in list(reversed(num_grids)):
        ret[:, -cnt] = idx_copy % n_grid
        idx_copy     = idx_copy // n_grid
        cnt         += 1

    return ret


def generate_grid(
    domain:     Sequence[Sequence[float]],
    grid_size:  Sequence[int],
    keep_shape: bool = False,
) -> torch.Tensor:
    """Generates the uniform grid on the passed domain and grid size.

    ## Description
    Generates a uniform grid on the given box domain. Note that this function is pending deprecation in favor of `space_grid`.

    ## Arguments
    `domain` (`Sequence[Sequence[float]]`): Bounding domain endpoints for each dimension.
    `grid_size` (`Sequence[int]`): Number of grids for each dimension.
    `keep_shape` (`bool`, default: `False`): Whether to keep the spatial shape or flatten.

    ## Returns
    `torch.Tensor`: Multi-dimensional coordinate grid tensor.
    """
    warnings.warn(
        "This function will be deprecated in the future. Use `custom_module.numerical.utils.space_grid` instead.",
        PendingDeprecationWarning,
        stacklevel=2,
    )
    assert len(domain) == len(grid_size)
    grid_list: list[torch.Tensor] = torch.meshgrid(
        [
            torch.linspace(domain[idx][0], domain[idx][1], grid_size[idx])
            for idx in range(len(domain))
        ],
        indexing = 'ij',
    )
    grid: torch.Tensor = torch.hstack([g.reshape(-1, 1) for g in grid_list])
    if keep_shape:
        grid = grid.reshape(*grid_size, len(grid_size))
        assert grid.ndim == 1 + len(grid_size)
    else:
        assert grid.ndim == 2
    return grid


def generate_radius_graph(
    coords:           torch.Tensor,
    radius_range:     Sequence[float],
    radius_inclusion: Sequence[bool] = (True, True),
    max_neighbors:    Optional[int]  = None,
    allow_loop:       bool           = False,
    p:                float          = 2.0,
) -> torch.LongTensor:
    """Generates the radius graph on the passed point cloud.

    ## Description
    Constructs a radius graph given coordinate points and a radius interval `[r_min, r_max]`.
    The output edge connectivity follows the PyTorch Geometric `edge_index` convention.

    ## Arguments
    `coords` (`torch.Tensor`): Point cloud tensor of shape `(N, d)`.
    `radius_range` (`Sequence[float]`): Pair `(r, s)` with `r < s`.
    `radius_inclusion` (`Sequence[bool]`, default: `(True, True)`): Inclusiveness of interval endpoints.
    `max_neighbors` (`Optional[int]`, default: `None`): Maximum neighbors per node.
    `allow_loop` (`bool`, default: `False`): Whether self-loops are allowed.
    `p` (`float`, default: `2.0`): Distance Minkowski norm order.

    ## Returns
    `torch.LongTensor`: Edge index tensor of shape `(2, num_edges)`.
    """
    _cdist: torch.Tensor = torch.cdist(coords, coords, p=p)
    if max_neighbors is None or max_neighbors >= len(_cdist):
        max_neighbors = len(_cdist) - 1

    _mask: torch.Tensor = (_cdist >= radius_range[0]) if radius_inclusion[0] else (_cdist > radius_range[0])
    _mask = _mask & ((_cdist <= radius_range[1]) if radius_inclusion[1] else (_cdist < radius_range[1]))

    _temp: torch.Tensor      = torch.zeros_like(_cdist, dtype=torch.bool)
    knn:   torch.LongTensor  = torch.topk(_cdist, k=1 + max_neighbors, largest=False)[-1]
    _r0:   torch.Tensor      = torch.arange(len(knn)).repeat_interleave(1 + max_neighbors)
    _r1:   torch.Tensor      = knn.flatten()
    _temp[_r0, _r1]          = True
    _mask                    = _mask & _temp

    if not allow_loop:
        _mask = _mask & (_cdist != 0)

    target, source = torch.where(_mask)
    return torch.stack([source, target])


def generate_k_nearest_neighbor_graph(
    coords:        torch.Tensor,
    max_neighbors: int,
    allow_loop:    bool = False,
) -> torch.LongTensor:
    """Generates the k-nearest-neighbor graph on the passed point cloud.

    ## Description
    Constructs a k-nearest-neighbor graph given a point cloud and maximum neighbor count.
    The output edge connectivity follows the PyTorch Geometric `edge_index` convention.

    ## Arguments
    `coords` (`torch.Tensor`): Point cloud tensor of shape `(N, d)`.
    `max_neighbors` (`int`): The maximum number of neighbors for each node.
    `allow_loop` (`bool`, default: `False`): Whether self-loops are allowed.

    ## Returns
    `torch.LongTensor`: Edge index tensor of shape `(2, num_edges)`.
    """
    knn: torch.LongTensor = torch.topk(torch.cdist(coords, coords), k=max_neighbors + 1, dim=-1, largest=False)[-1]
    if allow_loop:
        knn_flat: torch.Tensor = knn.flatten()
        target:   torch.Tensor = torch.arange(len(coords)).repeat_interleave(1 + max_neighbors)
    else:
        knn_flat = knn[..., 1:].flatten()
        target   = torch.arange(len(coords)).repeat_interleave(max_neighbors)
    return torch.stack([knn_flat, target])


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()