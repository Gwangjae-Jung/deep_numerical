from    typing      import  Iterable, Optional, Sequence, TypeAlias, TypeVar, Union
from    typing_extensions   import  Self
from    pathlib     import  Path
from    numpy       import  ndarray
import  torch
from    torch_geometric.data    import  Data


##################################################
##################################################
__all__ = [
    "SUPPORTED_ARRAY_MODULES",
    "SUPPORTED_ARRAY_TYPES",
    "ArrayData",
    "PathLike",
    "Objects",
    "TORCH_DEFAULT_DTYPE",
    "TORCH_DEFAULT_DEVICE",
    "dtype_real_to_complex",
    "dtype_complex_to_real",
]


##################################################
##################################################
SUPPORTED_ARRAY_MODULES = ("np", "torch")
SUPPORTED_ARRAY_TYPES   = (ndarray, torch.Tensor)
        

##################################################
##################################################
_T = TypeVar("_T")
Objects:    TypeAlias   = Union[_T, Sequence[_T], Iterable[_T]]
"""The typealias for the available objects (`Any`, `Sequence`, and `Iterable`)."""


##################################################
##################################################
ArrayData:  TypeAlias   = Union[ndarray, torch.Tensor]
"""The typealias for the available tensors (`numpy.ndarray` and `torch.Tensor`)."""
PathLike:   TypeAlias   = Union[str, Path]
"""The typealias for the available path-like objects (`str` and `pathlib.Path`)."""


##################################################
##################################################
class MultiLevelData(Data):
    """The class for the multilevel graph data structure in `torch_geometric` (`torch_geometric.data.Data`).

    The class is used to store the multilevel graph data, including the node attributes, edge indices and attributes, and the multigrid information.
    """
    def __init__(
        self,
        x:              Optional[torch.Tensor] = None,
        edge_index:     Optional[torch.Tensor] = None,
        edge_attr:      Optional[torch.Tensor] = None,
        y:              Optional[Union[torch.Tensor, int, float]] = None,
        pos:            Optional[torch.Tensor] = None,
        time:           Optional[torch.Tensor] = None,
        
        intra_node_index:   Optional[Sequence[torch.Tensor]]    = None,
        intra_edge_index:   Optional[Sequence[torch.Tensor]]    = None,
        inter_edge_index:   Optional[Sequence[torch.Tensor]]    = None,
        
        **kwargs,
    ) -> Self:
        """The initializer of the class `MultiLevelData`.
        
        Arguments:
            `x` (`torch.Tensor`): The node attributes.
            `edge_index` (`torch.Tensor`): The edge indices.
            `edge_attr` (`torch.Tensor`): The edge attributes.
            `y` (`torch.Tensor`, `int`, or `float`): The graph-level attributes.
            `pos` (`torch.Tensor`): The node positions.
            `time` (`torch.Tensor`): The time attributes.
            
            `intra_node_index` (`Sequence[torch.Tensor]`): The sequence of the *subset* of the `node_index` attribute belonging to each level.
            `intra_edge_index` (`Sequence[torch.Tensor]`): The sequence of the *subset* of the `edge_index` attribute belonging to each level.
            `inter_edge_index` (`Sequence[torch.Tensor]`): The sequence of the *subset* of the `edge_index` attribute belonging to each pair of adjacent levels.        
        """
        # TODO: Merge edge-related attributes and save masks
        self.__check_multilevel_data(intra_node_index, intra_edge_index, inter_edge_index)
        super().__init__(
            x           = x,
            edge_index  = edge_index,
            edge_attr   = edge_attr,
            y           = y,
            pos         = pos,
            time        = time,
            **kwargs,
        )
        self.__level: int = len(intra_node_index)
        self.__intra_node_index:    tuple[torch.Tensor]     = tuple(intra_node_index)
        self.__intra_edge_index:    tuple[torch.Tensor]     = tuple(intra_edge_index)
        self.__inter_edge_index:    tuple[torch.Tensor]     = tuple(inter_edge_index)
        return
    
    
    def __check_multilevel_data(
        self,
        intra_node_index:   Optional[Sequence[torch.Tensor]]    = None,
        intra_edge_index:   Optional[Sequence[torch.Tensor]]    = None,
        inter_edge_index:   Optional[Sequence[torch.Tensor]]    = None,
    ) -> None:
        len1 = len(intra_node_index)
        len2 = len(intra_edge_index)
        len3 = len(inter_edge_index)
        if len1 != len2:
            raise ValueError(f"The length of `intra_node_index` ({len1}) and `intra_edge_index` ({len2}) must be the same.")
        if len1 != len3+1:
            raise ValueError(f"The length of `intra_node_index` ({len1}) should be greater than the length of `inter_edge_index` ({len3}) by 1.")
        return
    
    
    @property
    def level(self) -> int:
        """The property for the number of levels."""
        return self.__level
    @property
    def intra_node_index(self) -> tuple[torch.Tensor]:
        """The property for the sequence of the *subset* of the `node_index` attribute belonging to each level."""
        return self.__intra_node_index
    @property
    def intra_edge_index(self) -> tuple[torch.Tensor]:
        """The property for the sequence of the *subset* of the `edge_index` attribute belonging to each level."""
        return self.__intra_edge_index
    @property
    def inter_edge_index(self) -> tuple[torch.Tensor]:
        """The property for the sequence of the *subset* of the `edge_index` attribute belonging to each pair of adjacent levels."""
        return self.__inter_edge_index
    
    
    def data(self, level: int) -> Data:
        """The method for the data of the `level`-th level.
        
        Arguments:
            `level` (`int`): The level of the graph.
        """
        if level < 0 or level >= self.__level:
            raise ValueError(f"The level {level} is out of range.")
        return Data(
            x           = self.x[self.intra_node_index[level]],
            edge_index  = self.edge_index[self.intra_edge_index[level]],
            edge_attr   = self.edge_attr[self.intra_edge_index[level]],
            y           = self.y,
            pos         = self.pos,
            time        = self.time,
        )


Graph:  TypeAlias   = Data
"""The typealias for the graph data structure in `torch_geometric` (`torch_geometric.data.Data`)."""
MutiLevelGraph: TypeAlias   = MultiLevelData
"""The typealias for the multilevel graph data structure."""


##################################################
##################################################
TORCH_DEFAULT_DTYPE:    torch.dtype     = torch.float64
"""The default `torch` datatype for numerical opearations. It is set to `torch.float64` by default."""
TORCH_DEFAULT_DEVICE:   torch.device    = torch.device("cpu")
"""The default `torch` device for numerical opearations. It is set to `torch.device("cpu")` by default."""


def dtype_real_to_complex(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype is torch.float16:
        return torch.complex32
    elif real_dtype is torch.float32:
        return torch.complex64
    elif real_dtype is torch.float64:
        return torch.complex128
    else:
        raise TypeError(f"Input datatype: {real_dtype}")
    
    
def dtype_complex_to_real(complex_dtype: torch.dtype) -> torch.dtype:
    if complex_dtype is torch.complex32:
        return torch.float16
    elif complex_dtype is torch.complex64:
        return torch.complex32
    elif complex_dtype is torch.complex128:
        return torch.complex64
    else:
        raise TypeError(f"Input datatype: {complex_dtype}")


##################################################
##################################################
# End of file