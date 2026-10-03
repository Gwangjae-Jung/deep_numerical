import warnings
from   typing            import Any, Callable, Optional, Sequence, Union
from   typing_extensions import override

import torch

try:
    from torch_geometric.data import Data
    _HAS_TORCH_GEOMETRIC: bool = True
except (ImportError, ModuleNotFoundError):
    _HAS_TORCH_GEOMETRIC: bool = False

if not _HAS_TORCH_GEOMETRIC:
    warnings.warn(
        "Module 'torch_geometric' is not installed. 'RandomGraphGenerator' and 'RandomGridGenerator' will not be available.",
        UserWarning,
        stacklevel=2,
    )
    __all__: list[str] = []
else:
    from .grid import space_grid

    __all__: list[str] = [
        "RandomGraphGenerator",
        "RandomGridGenerator",
    ]


    class RandomGraphGenerator():
        """The base class for random graph generators.

        ## Description
        This class provides the basic functionality for generating random graphs.
        It can be used as a base class for other random graph generators.
        The main purpose of this class is to provide a common interface for random graph generators.

        ## Arguments
        `points` (`Optional[torch.Tensor]`, default: `None`): The tensor of node coordinates of shape `(num_nodes, dimension)`.
        `graph` (`Optional[Data]`, default: `None`): The base `torch_geometric.data.Data` object having attribute `pos`.
        """

        def __init__(
            self,
            points: Optional[torch.Tensor] = None,
            graph:  Optional[Any]          = None,
        ) -> None:
            """The initializer of `RandomGraphGenerator`.

            ## Description
            Initializes the random graph generator using either input point coordinates or a base PyG `Data` graph.

            ## Arguments
            `points` (`Optional[torch.Tensor]`, default: `None`): The tensor of node coordinates of shape `(num_nodes, dimension)`.
            `graph` (`Optional[Any]`, default: `None`): The base `torch_geometric.data.Data` object having attribute `pos`.

            ## Returns
            `None`: None.
            """
            self.__check_arguments(points, graph)
            if points is not None:
                graph = Data(pos=points)
            self.__num_nodes:  int = int(graph.pos.shape[0])
            self.__dimension:  int = int(graph.pos.shape[1])
            self.__base_graph: Any = graph
            return None

        def __check_arguments(
            self,
            points: Optional[torch.Tensor],
            graph:  Optional[Any],
        ) -> None:
            if points is not None:
                if points.ndim != 2:
                    raise ValueError(f"The shape of 'points' should be (num_nodes, dimension), but got {points.shape}.")
            else:
                if graph.pos is None:
                    raise ValueError("The 'graph' should have the attribute 'pos'.")
                if graph.pos.ndim != 2:
                    raise ValueError(f"The shape of 'graph.pos' should be (num_nodes, dimension), but got {graph.pos.shape}.")
            return None

        @property
        def num_nodes(self) -> int:
            """Total number of nodes in the graph."""
            return self.__num_nodes

        @property
        def dimension(self) -> int:
            """Spatial dimension of node coordinates."""
            return self.__dimension

        @property
        def base_graph(self) -> Any:
            """Underlying base torch_geometric Data graph."""
            return self.__base_graph

        def sample_subgraph(
            self,
            num_nodes:   int,
            method:      str           = 'uniform',
            return_mask: bool          = False,
            generator:   Optional[Any] = None,
        ) -> Any:
            """Samples a subgraph of size `num_nodes`.

            ## Description
            Samples a subset of `num_nodes` from the base graph using the specified sampling method.

            ## Arguments
            `num_nodes` (`int`): The number of nodes in the sampled subgraph.
            `method` (`str`, default: `'uniform'`): The sampling method to use.
            `return_mask` (`bool`, default: `False`): Whether to also return the boolean selection mask.
            `generator` (`Optional[Any]`, default: `None`): PyTorch pseudo-random number generator.

            ## Returns
            `Any`: Sampled `Data` object, or tuple with selection mask.
            """
            if num_nodes > self.num_nodes:
                raise ValueError(f"The argument 'num_nodes' ({num_nodes}) should be less than or equal to the total number of nodes ({self.num_nodes}).")

            if method == 'uniform':
                perm: torch.Tensor = torch.randperm(self.num_nodes, generator=generator)
                selected_idx: torch.Tensor = perm[:num_nodes]
            else:
                raise ValueError(f"Unsupported sampling method: {method}")

            mask: torch.Tensor = torch.zeros(self.num_nodes, dtype=torch.bool)
            mask[selected_idx] = True

            sub_pos: torch.Tensor = self.base_graph.pos[selected_idx]
            sub_graph: Any        = Data(pos=sub_pos)

            if return_mask:
                return sub_graph, mask
            return sub_graph


    class RandomGridGenerator(RandomGraphGenerator):
        """Random grid generator based on spatial discretization.

        ## Description
        Generates random subgraphs sampled from a regular grid on a multi-dimensional box domain.

        ## Arguments
        `domain` (`Sequence[Sequence[float]]`): The domain bounds for each dimension.
        `num_grids` (`Sequence[int]`): The number of grid intervals in each dimension.
        `where_closed` (`str`, default: `'both'`): Boundary inclusion configuration (`'both'`, `'left'`, `'right'`, or `'none'`).
        """

        def __init__(
            self,
            domain:       Sequence[Sequence[float]],
            num_grids:    Sequence[int],
            where_closed: str = 'both',
        ) -> None:
            """The initializer of `RandomGridGenerator`.

            ## Description
            Initializes the regular grid graph generator on the given domain.

            ## Arguments
            `domain` (`Sequence[Sequence[float]]`): The domain bounding box.
            `num_grids` (`Sequence[int]`): Number of grids in each dimension.
            `where_closed` (`str`, default: `'both'`): Boundary closure configuration.

            ## Returns
            `None`: None.
            """
            self.__check_arguments(domain, num_grids, where_closed)
            self.__dimension: int                = len(domain)
            self.__num_grids: tuple[int, ...]     = tuple(num_grids)
            min_values:       tuple[float, ...]  = tuple([x[0] for x in domain])
            max_values:       tuple[float, ...]  = tuple([x[1] for x in domain])
            points: torch.Tensor = space_grid(
                dimension    = self.__dimension,
                num_grids    = num_grids,
                max_values   = max_values,
                min_values   = min_values,
                where_closed = where_closed,
            ).reshape(-1, self.__dimension)
            super().__init__(points=points)
            return None

        @override
        def __check_arguments(
            self,
            domain:       Sequence[Sequence[float]],
            num_grids:    Sequence[int],
            where_closed: str,
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
            return None

        @property
        def num_grids(self) -> tuple[int, ...]:
            """Number of grid intervals along each dimension."""
            return self.__num_grids


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()