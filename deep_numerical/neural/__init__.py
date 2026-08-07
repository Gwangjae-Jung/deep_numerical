from    typing      import  TYPE_CHECKING, Any, List, Set
import  importlib
import  torch
from    .utils      import  *
from    .utils      import  __all__     as  __all__utils


if TYPE_CHECKING:
    from    .utils      import  *
    from    .layer      import  *
    from    .network    import  *
    from    .operator   import  *
    from    .parameterized_op   import  *
    from    .collision_op       import  *


_SUBMODULES = {'utils', 'layer', 'network', 'operator', 'parameterized_op', 'collision_op'}
_CLASSES    = {'BaseModule'}
_UTILS      = Set(__all__utils)
__all__     = list(_SUBMODULES | _CLASSES | _UTILS)


##################################################
##################################################
class BaseModule(torch.nn.Module):
    """A custom base class for all deep learning architectures in this package.
    """
    def __init__(self) -> None:
        super().__init__()
        return None
    
    
    def count_parameters(self) -> int:
        """Count the number of the learnable parameters in the module.
        Note that each complex parameter is counted as two parameters.
        """
        cnt = 0
        for p in self.parameters():
            c = 2 if p.is_complex() else 1
            cnt += c * p.numel()
        return cnt
    
    
    def check_arguments(self, **kwargs) -> None:
        pass
    
    
    def __str__(self) -> str:
        msg: list[str] = []
        __half_line = '=' * 30
        _front = ''.join((__half_line, f'< {self.__class__.__name__} >', __half_line))
        _line  = '-' * len(_front)
        _back  = '=' * len(_front)
        
        msg.append(_front)
        msg.append(f"[ Subnetworks ]\n")
        for name, md in self.named_children():
            msg.append(f"* {name}")
            msg.append(str(md))
            msg.append('')
        
        msg.append(_line)
        msg.append(f"[ Parameters ]")
        named_params = self.named_parameters(recurse=False)
        for name, p in named_params:
            msg.append(f"( {name} )")
            msg.append(f"- Shape:       {list(p.shape)}")
            msg.append(f"- Data type:   {p.dtype}")
        
        msg.append(_line)
        msg.append(f"Number of parameters: {self.count_parameters()}")
        msg.append(_back)
        
        return '\n'.join(msg)


##################################################
##################################################
def __dir__() -> List[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', package=__name__)
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")


##################################################
##################################################
# End of file