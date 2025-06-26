import  warnings

from    typing              import  *
from    typing_extensions   import  Self

import  numpy       as      np
import  torch

from    ...utils_main._dtype    import  ArrayData, Objects


##################################################
##################################################
__all__ = [
    "node_idx_to_arr_idx",
    
    "generate_grid",
    "generate_radius_graph",
    "generate_k_nearest_neighbor_graph",
    
    "GridGenerator",
    
    "_DEFAULT_RADIUS_RANGE",
    "_DEFAULT_RADIUS_INCLUSION",
    "_DEFAULT_MAX_NEIGHBORS",
    "_DEFAULT_ALLOW_LOOP",
]


##################################################
##################################################
_DEFAULT_RADIUS_RANGE       = tuple((0, torch.inf))
_DEFAULT_RADIUS_INCLUSION   = tuple((True, True))
_DEFAULT_MAX_NEIGHBORS      = None
_DEFAULT_ALLOW_LOOP         = False


##################################################
##################################################
def node_idx_to_arr_idx(
        index:      Union[Objects[int], ArrayData],
        num_grids:  list[int]
    ) -> np.ndarray:
    """
    ### Description
    Given node indices and the number of the grids, this function returns the indices corresponding to the grid.
    
    ### Example
    >>> index = [0, 1, 20, 123]
    >>> num_grids = [11, 5, 4]
    >>> node_idx_to_arr_idx(index, num_grids)
    array([[0, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [6, 0, 3]])
    """
    if not isinstance(index, ArrayData):
        index = np.array(index)
    assert index.ndim == 1
    
    ret = np.zeros(shape = (len(index), len(num_grids))).astype(np.int32)
    cnt = 1
    for n_grid in list(reversed(num_grids)):
        ret[:, -cnt] = index % n_grid
        index = index // n_grid
        cnt += 1
    
    return ret


##################################################
##################################################
def generate_grid(
        domain:     Sequence[Sequence[float]],
        grid_size:  Sequence[int],
        keep_shape: bool    = False,
    ) -> torch.Tensor:
    """Generates the uniform grid on the passed domain and grid size.
    
    -----
    `PendingDeprecationWarning`
    This function will be deprecated in the future.
    Use `custom_module.numerical.utils.space_grid` instead.
    
    -----
    ### Arguments
    @ `domain` (`Sequence[Sequence[float]]`)
        * The endpoints of the box-shaped domain are given.
    
    @ `grid_size` (`Sequence[int]`)
        * The number of the grids for each dimension are given.
    
    @ `keep_shape` (`bool`, default: `False`)
        * Determines whether to keep the shape of the domain.
        * If `True`, then the coordinates of the mesh points are aligned to the domain. 
        * If `False`, the coordinates are expanded in a 2-tensor of shape `(N, d)`, where `N` is the number of the mesh points and `d` is the dimension of the domain.
    """
    warnings.warn(
        '\n'.join([
            "This function will be deprecated in the future.",
            "Use `custom_module.numerical.utils.space_grid` instead.",
        ]),
        PendingDeprecationWarning
    )
    assert len(domain) == len(grid_size)
    grid = torch.meshgrid(
                [
                    torch.linspace(*domain[idx], grid_size[idx])
                    for idx in range(len(domain))
                ],
                indexing = 'ij'
            )
    grid = torch.hstack([g.reshape(-1, 1) for g in grid])
    if keep_shape:
        grid = grid.reshape(*grid_size, len(grid_size))
        assert grid.ndim == 1 + len(grid_size)
    else:
        assert grid.ndim == 2
    return grid


def generate_radius_graph(
        coords:             torch.Tensor,
        radius_range:       Sequence[float],
        radius_inclusion:   Sequence[bool]  = (True, True),
        max_neighbors:      int             = None,
        allow_loop:         bool            = False,
        p:                  float           = 2.0,
    ) -> torch.LongTensor:
    """Generates the radius graph on the passed point cloud.
    
    ### Description
    This function constructs the radius graph given a point cloud and the range of radius, the number of the maximum neighbors.
    
    ### Note
    The edge connectivity is constructed in accordance with the convention of the edge connectivity in PyTorch Geometric.
    In other words, if an output of this function is nominated by `edge_index`, then `edge_index` is a `torch.LongTensor` object of shape `(2, num_edges)`, with each edge connectivity `edge_index[:, i]` is a tuple of the source node and the target node.
    
    -----
    ### Arguments
    @ `coords` (`torch.Tensor`)
        * A point cloud in a 2-tensor of shape `(N, d)`, where `N` is the number of the points and `d` is the dimension of the ambient space.
    
    @ `radius_range` (`Sequence[float]`)
        * A pair `(r, s)` of real numbers with `r < s`.
        * Two points are connected whenever the distance `d` satisfies `r <(=) d <(=) s`.
        * Whether the marginal radii are allowed is determined by `radius_inclusion`.
    
    @ `radius_inclusion` (`Sequence[float]`, default: `(True, True)`)
        * A pair of two boolean variables.
        * The marginal radii are allowed if and only if the corresponding entry of `radius_inclusion` is `True`.
    
    @ `allow_loop` (`bool`, default: `False`)
        * The loops are allowed if and only if `allow_loop==True`.
    
    @ `p` (`float`, default: `2`)
        * The degree in the computation of the distances.
    """
    _cdist = torch.cdist(coords, coords, p = p)
    if max_neighbors is None or max_neighbors >= len(_cdist):
        max_neighbors = len(_cdist) - 1 # Connect with all the other nodes
    
    # Setting the mask
    _mask = (radius_range[0] <= _cdist) if radius_inclusion[0] else (radius_range[0] < _cdist)
    _mask *= (_cdist <= radius_range[1]) if radius_inclusion[1] else (_cdist < radius_range[1])
    
    # Masking in terms of `max_neighbors` (for a while, allow loops)
    _temp = torch.zeros_like(_cdist).type(torch.bool)
    knn: torch.LongTensor = torch.topk(_cdist, k = 1 + max_neighbors, largest = False)[-1]
    _r0 = torch.arange(len(knn)).repeat_interleave(1 + max_neighbors)
    _r1 = knn.flatten()
    _temp[_r0, _r1] = True
    _mask *= _temp
    
    # Masking in terms of `allow_loop`
    if not allow_loop:
        _mask = _mask & (_cdist != 0)
    
    # Set the source and target
    target, source = torch.where(_mask)
    return torch.stack([source, target])


def generate_k_nearest_neighbor_graph(
        coords:         torch.Tensor,
        max_neighbors:  int,
        allow_loop:     bool = False,
    ) -> torch.LongTensor:
    """Generates the k-nearest-neighbor graph on the passed point cloud.
    
    ### Description
    This function constructs the k-nearest-neighbor graph given a point cloud and the range of radius, the number of the maximum neighbors.
    
    ### Note
    The edge connectivity is constructed in accordance with the convention of the edge connectivity in PyTorch Geometric.
    In other words, if an output of this function is nominated by `edge_index`, then `edge_index` is a `torch.LongTensor` object of shape `(2, num_edges)`, with each edge connectivity `edge_index[:, i]` is a tuple of the source node and the target node.
    
    -----
    ### Arguments
    @ `coords` (`torch.Tensor`)
        * A point cloud in a 2-tensor of shape `(N, d)`, where `N` is the number of the points and `d` is the dimension of the ambient space.
    
    @ `max_neighbors` (`int`)
        * The number of the maximum neighbors for each node.
        
    @ `allow_loop` (`bool`)
        * The loops are allowed if and only if `allow_loop==True`.
        * Even if `allow_loop==True`, for each node, at most `max_neighbors` other nodes can be joined to each node.
    """
    knn: torch.LongTensor
    knn = torch.topk(torch.cdist(coords, coords), k = max_neighbors + 1, dim = -1, largest = False)[-1]
    if allow_loop:
        knn = knn.flatten()
        target = torch.arange(len(coords)).repeat_interleave(1 + max_neighbors)
    else:
        knn = knn[..., 1:].flatten()
        target = torch.arange(len(coords)).repeat_interleave(max_neighbors)
    return torch.stack([knn, target])


##################################################
##################################################


class GridGenerator():
    """## The base grid generator
    -----
    
    ### Description
    This class is a base class for other types of grid-generating classes.
    To this end, the initializer of this class contains various optional arguments, some of which may be omitted in the initialization and are initialized so as to construct a complete graph.
    
    This class provides the following features.
    
    1. Generation of grids
        Given two arguments `domain` and `grid_size` of the initializer, this class generates a uniform grid on `domain` of grid size `grid_size`, using the function `torch.meshgrid` with `ij` indexing.
    
    2. Sampling of subgraphs
        This class also provides some methods which are used to sample subgraphs.
        The types of the supported graphs are listed below.
        
        * Radius graph
            Given subgraph node indices or another point cloud with the range of radius for which nodes are joined, the method `radius_graph` constructs the corresponding edge connectivities.
            
        * k-nearest-neighbor graph
            Given subgraph node indices or another point cloud with the maximum number of the neighbors, the method `nearest_neighbor_graph` constructs the corresponding edge connectivities.
    """
    def __init__(
            self,
            
            # To generate a grid
            domain:             Sequence[Sequence[float]],
            grid_size:          Sequence[int],
            
            # To generate a radius graph
            radius_range:       Optional[Sequence[float]]   = _DEFAULT_RADIUS_RANGE,
            radius_inclusion:   Optional[Sequence[bool]]    = _DEFAULT_RADIUS_INCLUSION,
            
            # To generate a k-nearest-neighbor graph
            max_neighbors:      Optional[int]   = _DEFAULT_MAX_NEIGHBORS,
            
            # Allow loops
            allow_loop:         Optional[bool]  = _DEFAULT_ALLOW_LOOP,
        ) -> Self:
        self.__grid       = generate_grid(domain, grid_size, keep_shape = False)
        self.__grid_size  = grid_size
        
        self.__radius_range     = tuple(radius_range)
        self.__radius_inclusion = tuple(radius_inclusion)        

        self.__max_neighbors    = max_neighbors

        self.__allow_loop       = allow_loop
        
        return

    
    # From the arguments of the initializer
    @property
    def radius_range(self) -> tuple[float, float]:
        """The range of radius with respect to which the radius graph will be generated."""
        return self.__radius_range
    @property
    def radius_inclusion(self) -> tuple[bool, bool]:
        """The sequence which saves the inclusion of the marginal radii."""
        return self.__radius_inclusion
    @property
    def allow_loop(self) -> bool:
        """The boolean option `allow_loop`."""
        return self.__allow_loop
    @property
    def max_neighbors(self) -> int:
        """The maximum number of the neighbors."""
        return self.__max_neighbors
    
    # Grid information
    @property
    def grid_size(self) -> tuple[int]:
        """The size of the grid."""
        return self.__grid_size
    @property
    def grid(self) -> torch.Tensor:
        """The clone of the grid (coordinates)."""
        return self.__grid.clone()
    @property
    def num_nodes(self) -> int:
        """The number of the nodes in the grid."""
        return self.__grid.size(0)
    @property
    def dim_domain(self) -> int:
        """The dimension of the domain."""
        return self.__grid.size(1)
    
    # Node information
    @property
    def node_index(self) -> torch.LongTensor:
        """The tensor of the indices of the nodes in the grid."""
        return torch.arange(self.num_nodes)
    
    
    def construct_graph(
            self,
            subgrid_index:      Optional[torch.LongTensor]  = None,
            point_cloud:        Optional[torch.Tensor]      = None,
            radius_range:       Optional[Sequence[float]]   = None,
            radius_inclusion:   Optional[Sequence[bool]]    = None,
            max_neighbors:      Optional[int]               = None,
            allow_loop:         Optional[bool]              = None,
            p:                  float                       = 2.0,
        ) -> torch.LongTensor:
        """Generates the radius graph for the default (or passed) (sub)grid or a point cloud.
        
        -----
        ### Description
        This method constructs the radius graph for the passed subgraph or the default configuration.
        
        1. If either `subgrid_index` or `point_cloud` is given, the radius graph is computed for the given data. If both are given, `point_cloud` is ignored.
        2. As for the other arguments, see the docstring of `generate_radius_graph`.
        
        -----
        ### Note
        The usual radius graph can be constructed by letting `max_neighbors=inf`, and the usual k-nearest-neighbor graph can be constructed by letting `radius_range=(-c, inf)`, where `c` is any positive real number.
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
            coords              = coords,
            radius_range        = radius_range,
            radius_inclusion    = radius_inclusion,
            max_neighbors       = max_neighbors,
            allow_loop          = allow_loop,
            p                   = p,
        )


##################################################
##################################################
