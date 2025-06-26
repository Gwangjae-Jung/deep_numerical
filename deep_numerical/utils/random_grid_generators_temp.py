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
    "RandomMultiGridGenerator",
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
    def idx_graph_to_grid(self, _range: slice) -> torch.LongTensor:
        """The method which converts a node index to coordinates (a grid value)."""
        return self.__subgrid_index[_range]
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
        

##################################################
##################################################


class RandomMultiGridGenerator(GridGenerator):
    """## Multigrid generator with sampling
    -----
    ### Description
    
    
    ### Notes
    Here the usage of the prefixes and appendices in the name of the attributes are summarized.
    
    * The prefix `multigrid` is used to imply that the attribute is considered in the grid space.
    * The prefix `multigraph` is used to imply that the attribute is a tuple of information for each subgraph. Such attributes are processed so that they can be treated as information of a subgraph of the merged graph. For example, the node indices `4` in the subgraph of level 1 and 3, if they exist, are the same node.
    * The prefix `levelwise` is used to imply that the attribute is a tuple of information for each subgraph. Such attributes are processed so that they can be treated as information of a subgraph itself, which is in contrary to the prefix `multigraph`. For example, the node indices `4` in the subgraph of level 1 and 3, if they exist, need not be the same node.
    * The prefix `merged` is used to imply that the attribute is obtained by merging the subgraph information. The attributes with the prefix `merged` can be considered attributes of a single graph.
    * The prefix/appendix `ranges` is used to infer the range of the subgraph information in `merged` attributes.
        * Note that `ranges` will not be used as an appendix in the future; it will only be used as a prefix.
    """
    def __init__(
            self,
            
            # Grid
            domain:             Sequence[Sequence[float]],
            grid_size:          Sequence[int],
            sample_sizes:       Sequence[int],
            sampling_mode:      str = _MULTIGRID_SAMPLING_SUPPORTED_MODES[0],
            
            # Graph
            radii:              Optional[Sequence[float]]   = None,
            radii_geometric:    Optional[Sequence[float]]   = None,
            max_neighbors:      Optional[Sequence[int]]     = None,
            allow_loop:         Optional[bool]              = False,
        ) -> None:
        """## The initializer of the class `RandomMultiGridGenerator`
        -----
        ### Arguments
        
        [ Grid ]
        
        @ `domain` (`Sequence[Sequence[float]]`) and `grid_size` (`Sequence[int]`)
            * The input data of the function `torch.meshgrid`.
        
        @ `sample_sizes` (`Sequence[int]`)
            * The sequence of the number of the nodes for each level.
            * Should be given as a nonincreasing sequence, i.e., `sample_sizes = (m_1 > m_2 > ... > m_L)`, where `L` denotes the level.
        
        @ `sampling_mode` (`str`, default: `"permutation"`)
            Determines the mode of sampling.
            The supported modes are explained below.
        
        [ Graph ]
        
        @ `radii` (`Sequence[float]`, default: `None`) or `radii_geometric` (`Sequence[float]`, default: `None`)
            * `radii` is the sequence of radius to generate a graph for each level. When it is `None`, `radii_geometric` should be given.
            * `radii_geometric` is a sequence of `(radii[0], ratio)`, thereby `radii` will be set `(radii_geometric[0] * (ratio ** k) for k in range(len(sample_sizes)))`.
        
        @ `max_neighbors` (`Sequence[int]`)
        
        @ `allow_loop` (`bool`, default: `False`)
        
        -----
        ### Supported sampling modes
        * `"permutation"`: Uses `numpy.random.choice` as in the above paper.
        * `"nested"`: Constructs a nested subgraphs.
        """
        # Initialization
        super().__init__(domain = domain, grid_size = grid_size, allow_loop = allow_loop)
        self.__set_sample_sizes(sample_sizes)
        self.__set_sampling_mode(sampling_mode)
        self.__set_radii(radii, radii_geometric)
        self.__set_max_neighbors(max_neighbors)
        
        # Instantiate the storages for the sampling results
        ## Level-separated objects
        # `levelwise_grid_index==multigrid_index`
        self.__multigrid_index:             tuple[torch.LongTensor] = tuple()
        self.__multigraph_coordinates:      tuple[torch.Tensor]     = tuple()
        self.__multigraph_edge_index:       tuple[torch.LongTensor] = tuple()
        ## Merged objects (can be called by properties of the base class `GridGenerator`)
        self.__merged_grid_index:           torch.LongTensor    = None
        self.__merged_graph_coordinates:    torch.Tensor        = None
        self.__merged_graph_edge_index:     torch.LongTensor    = None
        ## Level-separated ranges
        self.__slices_node: tuple[slice] = tuple()
        self.__slices_edge: tuple[slice] = tuple()
        return
    
    
    def __set_sample_sizes(self, sample_sizes: Sequence[int]) -> None:
        if sample_sizes is None:
            raise ValueError(f"'sample_sizes' is not initialized.")
        if not isinstance(sample_sizes, Sequence):
            raise ValueError(f"'sample_sizes' should be given as a sequential data. ('sample_sizes': {sample_sizes})")
        self.__sample_sizes = sample_sizes
        return
    
    
    def __set_sampling_mode(self, sampling_mode: str) -> None:
        if sampling_mode is None:
            raise ValueError(f"'sampling_mode' is not initialized.")
        sampling_mode = sampling_mode.lower()
        if not sampling_mode in _MULTIGRID_SAMPLING_SUPPORTED_MODES:
            raise ValueError(
                f"The sampling mode {sampling_mode} is not supported.\n" \
                f"(Supported sampling modes: {_MULTIGRID_SAMPLING_SUPPORTED_MODES})"
            )
        self.__sampling_mode = sampling_mode
        return
    
    
    def __set_radii(self, radii: Optional[Sequence[float]], radii_geometric: Optional[Sequence[float]]) -> None:
        self.__radii: tuple[float] = None
        
        # Pass if the chosen type of graphs is the k-nearest-neighbor graph
        if radii is None and radii_geometric is None:
            return
        
        if radii is not None:
            if not radii[0] > 0:
                raise ValueError(
                        f"'radii[0]' should be a positive real number. (radii[0]: {radii[0]})"
                )
            for cnt in range(len(radii) - 1):
                if not radii[cnt] < radii[cnt + 1]:
                    raise ValueError(
                        f"'radii' should be an increasing sequence of positive real numbers. (radii: {radii})"
                    )
            self.__radii = tuple(radii)
        else:
            if not( radii_geometric[0] > 0 and radii_geometric[1] > 1):
                raise ValueError(
                        f"'radii_geometric' should be an increasing geometric sequence of positive real numbers. (radii[0]: {radii_geometric[0]}, ratio: {radii_geometric[1]})"
                )
            self.__radii = tuple([radii_geometric[0] * (radii_geometric[1] ** k) for k in range(self.num_levels)])
        
        # Check the size
        if len(self.sample_sizes) != len(self.__radii):
            raise ValueError(
                f"Invalid subgraph information. (len(sample_sizes): {len(self.sample_sizes)}, len(radii): {len(self.radii)})"
            )
        
        return
    
    
    def __set_max_neighbors(self, max_neighbors: Optional[Sequence[int]]) -> None:
        self.__max_neighbors: tuple[int] = None
        
        # Set the configurations for the k-nearest-neighbor graph
        if max_neighbors is not None:
            for cnt, k in enumerate(max_neighbors):
                if k <= 0:
                    raise ValueError(
                        f"'max_neighbors' encountered with a non-positive integer {k} at index {cnt}."
                    )
        self.__max_neighbors = tuple(max_neighbors)
        
        # Check the size
        if len(self.sample_sizes) != len(self.__max_neighbors):
            raise ValueError(
                f"Invalid subgraph information. (len(sample_sizes): {len(self.sample_sizes)}, len(max_neighbors): {len(self.__max_neighbors)})"
            )
        
        return
    
        
    @property
    def sample_sizes(self) -> Sequence[int]:
        """The number of the nodes in the sampled subgraphs."""
        return self.__sample_sizes
    @property
    def num_levels(self) -> int:
        """The level of the sampled multigraph, i.e., the sampled subgraphs."""
        return len(self.__sample_sizes)
    @property
    def total_sample_size(self) -> int:
        """The number of the nodes in the sampled multigraph, i.e., the sampled subgraphs."""
        return sum(self.__sample_sizes)
    @property
    def sampling_mode(self) -> str:
        """The sampling mode. (See the docstring of the module.)"""
        return self.__sampling_mode
    @property
    def radii(self) -> tuple[float]:
        """The tuple of the maximum radius for each subgraph."""
        return self.__radii
    @property
    def max_neighbors(self) -> tuple[float]:
        """The tuple of the maximally accepted number of neighbors for each subgraph."""
        return self.__max_neighbors
    
    @property
    def multigrid_index(self) -> tuple[torch.LongTensor]:
        """The grid indices of the sampled subgraphs."""
        return self.__multigrid_index
    @property
    def multigrid_coordinates(self) -> tuple[torch.Tensor]:
        """The coordinates of the sampled subgraphs. (Alias of `multigraph_coordinates`)"""
        return self.__multigraph_coordinates
    @property
    def multigraph_coordinates(self) -> tuple[torch.Tensor]:
        """The coordinates of the sampled subgraphs. (Alias of `multigrid_coordinates`)"""
        return self.__multigraph_coordinates
    @property
    def multigraph_edge_index(self) -> tuple[torch.LongTensor]:
        """The edge connectivities of the sampled subgraphs."""
        return self.__multigraph_edge_index
    @property
    def merged_grid_index(self) -> torch.LongTensor:
        """The grid indices of the sampled subgraphs."""
        return self.__merged_grid_index
    @property
    def merged_graph_coordinates(self) -> torch.Tensor:
        """The coordinates of the sampled subgraphs."""
        return self.__merged_graph_coordinates
    @property
    def merged_graph_edge_index(self) -> torch.Tensor:
        """The edge connectivities of the sampled subgraphs."""
        return self.__merged_graph_edge_index
    @property
    def levelwise_grid_index(self) -> tuple[torch.LongTensor]:
        """The grid indices of the sampled subgraphs."""
        return self.__multigrid_index
    @property
    def levelwise_coordinates(self) -> tuple[torch.Tensor]:
        """The coordinates of the sampled subgraphs."""
        return self.__multigraph_coordinates
    @property
    def levelwise_edge_index(self) -> tuple[torch.LongTensor]:
        """The edge connectivities of the sampled subgraphs."""
        if self.sampling_mode == "permutation":
            _ret = []
            _cumsum = 0
            for _num_nodes, edge_index in zip(self.sample_sizes, self.__multigraph_edge_index):
                _ret.append(edge_index - _cumsum)
                _cumsum += _num_nodes
            return tuple(_ret)
        elif self.sampling_mode== "nested":
            return self.__multigraph_edge_index
        else:
            raise ValueError(
                f"The sampling mode {self.sampling_mode} is not supported.\n"
                f"(Supported sampling modes: {_MULTIGRID_SAMPLING_SUPPORTED_MODES})"
            )
    
    @property
    def slices_node(self) -> tuple[slice]:
        """The tuple of the slices for the node-related attributes."""
        return self.__slices_node
    @property
    def slices_edge(self) -> tuple[slice]:
        """The tuple of the slices for the edge-related attributes."""
        return self.__slices_edge
    @property
    def masks_node(self) -> tuple[torch.LongTensor]:
        """The tuple of the masks for the node-related attributes."""
        return tuple([torch.arange(_sl.start, _sl.stop) for _sl in self.__slices_node])
    @property
    def masks_edge(self) -> tuple[torch.LongTensor]:
        """The tuple of the slices for the edge-related attributes."""
        return tuple([torch.arange(_sl.start, _sl.stop) for _sl in self.__slices_edge])
    
    
    def sample(self) -> None:
        """Do a sampling with a predetermined number of sample points.
        A sampling proceeds in the following order:
        
        1. Sampling hierarchical subgraphs.
        2. Merging the subgraphs to form a full graph.
        3. Saving the ranges of the subgraphs in the merged graph.
        
        -----
        ### Remark
        1. `multigrid_node_index` is attained by masking `self.node_index`. On the other hand, `multigrid_edge_index` is attained by considering the corresponding subgraph as a full graph.
        2. In contrast to the `sample()` method of `RandomGridGenerator`, this method does not return sampling information.
        """
        self.__sample_subgraphs()   # Step 1. Sample subgraphs
        self.__merge_subgraphs()    # Step 2. Merge multigrid information
        self.__slice_subgraphs()    # Step 3. Slice subgraphs
        return
    
    
    def __sample_subgraphs(self) -> None:
        """Sample subgraphs in a specified mode.
        
        In the supplementary code for the paper "Multipole Graph Neural Operator for Parametric Partial Differential Equations", hierarchical subgraphs are chosen which need not be nested.
        By passing specific mode of sampling, this method samples subgraphs, and set the following member variables.
        In what follows, all indices are zero-based.
        
        @ (`self.__multigrid_index`)
            `self.__multigrid_index` is a tuple whose `l`-th entry is the grid indices for the `l`-th subgraph.
            * If `self.sampling_mode=="permutation"`, then the subgraphs are sampled pairwise disjoint.
            * If `self.sampling_mode=="nested"`, then the subgraphs are sampled so that the nodes of the `l+1`-th subgraph are contained in the `l`-th subgraph.
        
        @ (`self.__multigraph_coordinates`)
            `self.__multigraph_coordinates` is a tuple whose `l`-th entry is the tensor of the coordinates of the nodes in the `l`-th subgraph.
            It satisfies `self.__multigraph_coordinates[l]==self.grid[self.__multigrid_index[l]]`.
        
        @ (`self.__multigraph_edge_index`)
            `self.__multigraph_coordinates` is a tuple whose `l`-th entry is the tensor of the edge connectivities in the `l`-th subgraph.
            The edge connectivity at level `l` is computed with the radius `r` satisfying `self.radii[l-1] <= r < self.radii[l]`, where `self.radii[-1]` refers to `0` in this docstring.
        """
        # Initialization
        self.__multigrid_index          = tuple()
        self.__multigraph_coordinates   = tuple()
        self.__multigraph_edge_index    = tuple()
        
        # Step 1. `self.__multigrid_index`
        if self.sampling_mode == "permutation":
            _temp = torch.LongTensor(choice(self.num_nodes, self.total_sample_size, replace = False))
            self.__multigrid_index = _temp.split(self.sample_sizes)
        elif self.sampling_mode == "nested":
            _temp = [torch.LongTensor(choice(self.num_nodes, self.sample_sizes[0], replace = False))]
            for _sample_size in self.sample_sizes[1:]:
                _temp.append(_temp[-1][:_sample_size])
            self.__multigrid_index = tuple(_temp)
        
        # Step 2. `self.__multigraph_coordinates`
        self.__multigraph_coordinates = tuple([self.grid[_r] for _r in self.__multigrid_index])
        
        # Step 3. `self.__multigrid_edge_index`
        _temp = []
        for _cnt, (_coords, _max_nbhd) in enumerate(
                zip(self.__multigraph_coordinates, self.__max_neighbors)
            ):
            _radius_range = [0., self.radii[0]] if _cnt == 0 else [self.radii[_cnt - 1], self.radii[_cnt]]
            _temp.append(
                self.construct_graph(
                    point_cloud         = _coords,
                    radius_range        = _radius_range,
                    radius_inclusion    = [True, False],    # (inclusive, exclusive)
                    allow_loop          = False,
                    max_neighbors       = _max_nbhd,
                )
            )
            # So far, `_temp[-1]` can only be treated as a full-graph information
            # Hence, correction of the `edge_index` attribute is required
            # Case 1. The vertex sets are pairwise disjoint --> Correction by translation
            if self.sampling_mode in ("permutation"):
                _temp[-1] = _temp[-1] + sum(self.sample_sizes[:_cnt])
            # Case 2. The vertex sets are nested --> No correction requried
        self.__multigraph_edge_index = tuple(_temp)
        
        return
    
    
    def __merge_subgraphs(self) -> None:
        """Merge subgraphs in a specified mode.
        
        In the supplementary code for the paper "Multipole Graph Neural Operator for Parametric Partial Differential Equations", hierarchical subgraphs are chosen which need not be nested.
        By passing specific mode of sampling, this method merges subgraphs, and set the following member variables.
        
        @ `self.__merged_grid_index`
        @ `self.__merged_graph_coordinates`
        @ `self.__merged_grid_edge_index`
        
        -----
        ### Cases
        
        Note that in any case `self.merged_grid == self.grid[self.merged_node_index]`.
        
        @ `self.sampling_mode == "permutation"'
            * `self.__merged_grid_index` is the union of all nodes of the subgraphs. `torch.cat()` is used to merge the nodes.
            * `self.__merged_graph_coordinates` is the union of all coordinates of the subgraphs. `torch.cat()` is used to merge the coordinates.
            * `self.__merged_graph_edge_index` is the union of all `edge_index` attributes. `torch.hstack()` is used to merge the edges.
            
        @ `self.sampling_mode == "nested"'
            * `self.__merged_grid_index` is the nodes in the finest subgraph.
            * `self.__merged_graph_coordinates` is the coordinates in the finest subgraph.
            * `self.__merged_graph_edge_index` is the union of all `edge_index` attributes. `torch.hstack()` is used to merge the edges, and note that no reduction process is done since edge connectivities are pairwise disjoint among levels.
        """
        if self.sampling_mode == "permutation":
            self.__merged_grid_index        = torch.cat(self.__multigrid_index)
            self.__merged_graph_coordinates = torch.cat(self.__multigraph_coordinates)
            self.__merged_graph_edge_index  = torch.hstack(self.__multigraph_edge_index)
        elif self.sampling_mode == "nested":
            self.__merged_grid_index        = self.__multigrid_index[0]
            self.__merged_graph_coordinates = self.__multigraph_coordinates[0]
            self.__merged_graph_edge_index  = torch.hstack(self.__multigraph_edge_index)
        return
    
    
    def __slice_subgraphs(self) -> None:
        """Determines the following member variables
        
        @ `self.__slices_node`
        @ `self.__slices_edge`
        """
        if self.sampling_mode == "permutation":
            __zero = torch.zeros((1,), dtype = torch.long)
            _temp_node = torch.LongTensor(self.sample_sizes)
            _temp_node = torch.hstack([__zero, torch.cumsum(_temp_node, dim = 0)])
            _temp_edge = torch.LongTensor([_edge_index.size(1) for _edge_index in self.__multigraph_edge_index])
            _temp_edge = torch.hstack([__zero, torch.cumsum(_temp_edge, dim = 0)])
            self.__slices_node = \
                tuple(
                    slice(_temp_node[idx].item(), _temp_node[idx + 1].item())
                    for idx in range(len(_temp_node) - 1)
                )
            self.__slices_edge = \
                tuple(
                    slice(_temp_edge[idx].item(), _temp_edge[idx + 1].item())
                    for idx in range(len(_temp_edge) - 1)
                )
        elif self.sampling_mode == "nested":
            __zero = torch.zeros((1,), dtype = torch.long)
            _temp_node = torch.LongTensor(self.sample_sizes)
            _temp_edge = torch.LongTensor([_edge_index.size(1) for _edge_index in self.multigraph_edge_index])
            _temp_edge = torch.hstack([__zero, torch.cumsum(_temp_edge, dim = 0)])
            self.__slices_node = \
                tuple(
                    slice(0, _num_nodes.item())
                    for _num_nodes in _temp_node
                )
            self.__slices_edge = \
                tuple(
                    slice(_temp_edge[idx].item(), _temp_edge[idx + 1].item())
                    for idx in range(len(_temp_edge) - 1)
                )
        return
    

##################################################
##################################################

