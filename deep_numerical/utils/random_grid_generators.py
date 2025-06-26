from    typing              import  *
from    typing_extensions   import  Self

from    numpy.random    import  choice
import  torch

from    custom_modules.utils.graph  \
    import  GridGenerator, \
            _DEFAULT_RADIUS_RANGE, _DEFAULT_RADIUS_INCLUSION, _DEFAULT_MAX_NEIGHBORS, _DEFAULT_ALLOW_LOOP




##################################################
##################################################


__all__ = [
    "RandomGridGenerator",
    # "RandomMultiGridGenerator",
]


##################################################
##################################################


_MULTIGRID_SAMPLING_SUPPORTED_MODES = ("nested", "permutation")


##################################################
##################################################


class RandomGridGenerator(GridGenerator):
    """## The random grid generator
    
    -----
    ### Description
    This class generates a random grid by sampling from a uniform grid.
    
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

            # Grid
            domain:         Sequence[Sequence[float]],
            grid_size:      Sequence[int],
            sample_size:    int,
            
            # Graph
            radius:         Optional[float] = torch.inf,
            max_neighbors:  Optional[int]   = None,
            allow_loop:     Optional[bool]  = False,
        ) -> Self:
        super().__init__(
            domain              = domain,
            grid_size           = grid_size,
            radius_range        = (0, radius),
            radius_inclusion    = (True, True),
            max_neighbors       = max_neighbors,
            allow_loop          = allow_loop,
        )
        if sample_size <= 0 or sample_size > self.num_nodes:
            sample_size = self.num_nodes
        self.__sample_size: int = sample_size
        
        self.__subgrid_index:           torch.LongTensor    = None
        self.__subgraph_coordinates:    torch.Tensor        = None
        self.__subgraph_edge_index:     torch.LongTensor    = None
        
        return
    
    
    @property
    def sample_size(self) -> int:
        """The number of the nodes in the sampled subgraphs."""
        return self.__sample_size
    @property
    def subgrid_index(self) -> torch.LongTensor:
        """The grid indices of the sampled subgraph."""
        return self.__subgrid_index
    @property
    def subgraph_coordinates(self) -> torch.Tensor:
        """The coordinates of the sampled subgraph."""
        return self.__subgraph_coordinates
    @property
    def subgraph_edge_index(self) -> torch.LongTensor:
        """The edge connectivities of the sampled subgraph."""
        return self.__subgraph_edge_index
    
    
    def sample(self) -> None:
        """Sample a subgraph.
        
        @ `self.__subgrid_index`
            * The indices of the sampled mesh points.
            * Shape: `(sample_size,)`
            
        @ `self.__subgraph_coordinates`
            * The coordinates that each node corresponds to.
            * Shape: `(sample_size, dim_domain)`.
        
        @ `self.__subgraph_edge_index`
            * The `edge_index` attribute of the sampled subgraph.
            * Shape: `(2, num_edges_in_subgraph)`.
        """
        self.__subgrid_index = torch.LongTensor(choice(self.num_nodes, self.sample_size, replace = False))
        self.__subgraph_coordinates = self.grid[self.__subgrid_index]
        self.__subgraph_edge_index = self.construct_graph(self.__subgrid_index)
        
        return
    
    
    def idx_graph_to_grid(self, _range: Sequence) -> torch.LongTensor:
        """The method which converts a node index to coordinates (a grid value)."""
        return self.__subgrid_index[_range]
        

##################################################
##################################################
