import  warnings
from    typing              import  *
from    typing_extensions   import  Self, override

import  torch

try:
    from    torch_geometric.data    import  Data
    _HAS_TORCH_GEOMETRIC = True
except (ImportError, ModuleNotFoundError):
    _HAS_TORCH_GEOMETRIC = False

if not _HAS_TORCH_GEOMETRIC:
    warnings.warn(
        "Module 'torch_geometric' is not installed. 'RandomGraphGenerator' and 'RandomGridGenerator' will not be available.",
        UserWarning,
        stacklevel=2,
    )
    __all__: list[str] = []
else:
    from    .grid  import  space_grid

    __all__: list[str] = [
        "RandomGraphGenerator",
        "RandomGridGenerator",
    ]


    ##################################################
    ##################################################
    class RandomGraphGenerator():
        """The base class for random graph generators.
        
        ### Description
        This class provides the basic functionality for generating random graphs.
        It can be used as a base class for other random graph generators.
        The main purpose of this class is to provide a common interface for random graph generators.
        
        ### Note
        1. This class assumes that the attribute `pos` of the graph is nonempty, as all sampling methods are based on the `pos` attribute.
        2. The values of `edge_index` are inherited from the base graph. Using this property, the attributes `x` and `y` are also sampled from the base graph. However, `edge_attr` should be computed by users, using the attribute `edge_index` of the output subgraph.
        """
        def __init__(
                self,
                points: Optional[torch.Tensor]  = None,
                graph:  Optional[Data]          = None,
            ) -> None:
            """## The initializer of `RandomGraphGenerator`
            
            ## Description
            Initializes the random graph generator using either input point coordinates or a base PyG `Data` graph.
            
            ## Arguments
            `points` (`Optional[torch.Tensor]`, default: `None`): The tensor of node coordinates of shape `(num_nodes, dimension)`.
            `graph` (`Optional[Data]`, default: `None`): The base `torch_geometric.data.Data` object having attribute `pos`.
            
            ## Returns
            `None`: None.
            """
            self.__check_arguments(points, graph)
            if points is not None:
                graph = Data(pos=points)
            self.__num_nodes    = int(graph.pos.shape[0])
            self.__dimension    = int(graph.pos.shape[1])
            self.__base_graph   = graph
            return
        
        
        def __check_arguments(
                self,
                points: Optional[torch.Tensor],
                graph:  Optional[Data],
            ) -> None:
            if points is not None:
                if points.ndim != 2:
                    raise ValueError(f"The shape of 'points' should be (num_nodes, dimension), but got {points.shape}.")
            else:
                if graph.pos is None:
                    raise ValueError(f"The 'graph' should have the attribute 'pos'.")
                if graph.pos.ndim != 2:
                    raise ValueError(f"The shape of 'graph.pos' should be (num_nodes, dimension), but got {graph.pos.shape}.")
            return
        
        
        @property
        def num_nodes(self) -> int:
            """## Total number of nodes in the graph."""
            return self.__num_nodes

        @property
        def dimension(self) -> int:
            """## Spatial dimension of node coordinates."""
            return self.__dimension

        @property
        def base_graph(self) -> Data:
            """## Underlying base torch_geometric Data graph."""
            return self.__base_graph
        
        
        def _sample_node_indices(self, sample_size: int) -> torch.LongTensor:
            """Sample a subset of node indices from the graph."""
            from    numpy.random    import  choice
            sample_idx = choice(self.num_nodes, size=sample_size, replace=False)
            sample_idx.sort()
            sample_idx = torch.tensor(sample_idx, dtype=torch.long)
            return sample_idx
        
        
        def _sample_points(self, sample_size: int) -> torch.Tensor:
            """Sample a subset of nodes from the graph."""
            sample_idx = self._sample_node_indices(sample_size)
            return self.__base_graph.pos[sample_idx]        
        
        
        def sample_graph__knn(
                self,
                sample_size:    int,
                n_neighbors:    int,
                loop:           bool    = False,
            ) -> Data:
            """Sample a `k`-nn subgraph from the base graph.
            
            Arguments:
                `sample_size` (`int`): The number of nodes to sample from the base graph.
                `n_neighbors` (`int`): The number of neighbors to sample for each node.
                `loop` (`bool`, default: `False`): Whether to include self-loops in the graph.
            
            Returns:
                `Data`: A `torch_geometric.data.Data` object containing the sampled graph.
            """
            from    torch_geometric.nn  import  knn_graph
            node_index      = self._sample_node_indices(sample_size)
            points          = self.__base_graph.pos[node_index]
            sub_edge_index  = knn_graph(x=points, k=n_neighbors, loop=loop)
            
            sub_x:  Optional[torch.Tensor] = None
            sub_y:  Optional[torch.Tensor] = None
            if self.__base_graph.x is not None:
                sub_x = self.__base_graph.x[node_index]
            if self.__base_graph.y is not None:
                sub_y = self.__base_graph.y[node_index]
            return Data(x=sub_x, y=sub_y, pos=points, edge_index=sub_edge_index)
        
        
        def sample_graph__radius(
                self,
                sample_size:        int,
                radius:             float,
                loop:               bool    = False,
                max_num_neighbors:  int     = 64,
            ) -> Data:
            """## Sample a subgraph using radius-based connectivity
            
            ## Description
            Samples a subset of nodes uniformly from the base graph and constructs an edge connectivity based on a Euclidean radius threshold.
            
            ## Arguments
            `sample_size` (`int`): The number of nodes to sample from the base graph.
            `radius` (`float`): The distance cutoff for connecting edges between sampled nodes.
            `loop` (`bool`, default: `False`): Whether to include self-loops in the graph.
            `max_num_neighbors` (`int`, default: `64`): The maximum number of neighbors for each node in radius graph construction.
            
            ## Returns
            `Data`: A `torch_geometric.data.Data` object containing the sampled subgraph.
            """
            from    torch_geometric.nn  import  radius_graph
            node_index      = self._sample_node_indices(sample_size)
            points          = self.__base_graph.pos[node_index]
            sub_edge_index  = radius_graph(x=points, r=radius, loop=loop, max_num_neighbors=max_num_neighbors)
            
            sub_x:  Optional[torch.Tensor] = None
            sub_y:  Optional[torch.Tensor] = None
            if self.__base_graph.x is not None:
                sub_x = self.__base_graph.x[node_index]
            if self.__base_graph.y is not None:
                sub_y = self.__base_graph.y[node_index]
            return Data(x=sub_x, y=sub_y, pos=points, edge_index=sub_edge_index)


    ##################################################
    ##################################################
    class RandomGridGenerator(RandomGraphGenerator):
        """## Random Grid Generator
        
        ## Description
        A generator class that builds a regular Cartesian space grid on a multidimensional domain and wraps it as a base graph for random subgraph sampling.
        """
        def __init__(
                self,
                domain:         Sequence[Sequence[float]],
                num_grids:      Sequence[int],
                where_closed:   str = 'both',
            ) -> None:
            """The initializer of the class `RandomGridGenerator`
            
            Arguments:
                `domain` (`Sequence[Sequence[float]]`): The domain of the grid. Each element should be a sequence of length 2, where the first element is the lower bound and the second element is the upper bound.
                `num_grids` (`Sequence[int]`): The number of grids in each dimension. Each element should be a positive integer.
                `where_closed` (`str`, default: `'both'`): The type of the grid. It can be either `'both'`, `'left'`, `'right'`, or `'none'`. Default is `'both'`.
            """
            self.__check_arguments(domain, num_grids, where_closed)
            self.__dimension = len(domain)
            self.__num_grids = tuple(num_grids)
            min_values = tuple([x[0] for x in domain])
            max_values = tuple([x[1] for x in domain])
            points = space_grid(
                dimension       = self.__dimension,
                num_grids       = num_grids,
                max_values      = max_values,
                min_values      = min_values,
                where_closed    = where_closed,
            ).reshape(-1, self.__dimension)
            super().__init__(points=points)
            return
        
        
        @override
        def __check_arguments(
                self,
                domain:         Sequence[Sequence[float]],
                num_grids:      Sequence[int],
                where_closed:   str,
            ) -> None:
            if len(domain) != len(num_grids):
                raise ValueError(f"The length of 'domain' ({len(domain)}) and 'num_grids' ({len(num_grids)}) should be the same.")
            for idx, (x, n) in enumerate(zip(domain, num_grids)):
                if len(x) != 2:
                    raise ValueError(f"Each element of 'domain' should be a sequence of length 2, but got {len(x)} at index {idx}.")
                if x[0] >= x[1]:
                    raise ValueError(f"The first element of each element of 'domain' should be less than the second element, but got {x} at index {idx}.")
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(f"Each element of 'num_grids' should be a positive integer, but got {n} at index {idx}.")
            if where_closed not in ['both', 'left', 'right', 'none']:
                raise ValueError(f"'where_closed' should be either 'both', 'left', 'right', or 'none', but got {where_closed}.")
            return
        
        
        @property
        def num_grids(self) -> Sequence[int]:
            """## Number of grid intervals along each dimension."""
            return self.__num_grids


##################################################
##################################################
# End of file