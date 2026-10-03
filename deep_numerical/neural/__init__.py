import importlib
from   typing import TYPE_CHECKING, Any

import torch

from   .utils import *
from   .utils import __all__ as __all__utils


if TYPE_CHECKING:
    from .utils        import *
    from .layer        import *
    from .network      import *
    from .operator     import *
    from .collision_op import *


_SUBMODULES: set[str]  = {'utils', 'layer', 'network', 'operator', 'collision_op'}
_CLASSES:    set[str]  = {'BaseModule'}
_UTILS:      set[str]  = set(__all__utils)
__all__:     list[str] = list(_SUBMODULES | _CLASSES | _UTILS)


class BaseModule(torch.nn.Module):
    """Custom base class for all neural network modules and architectures in this package.

    ## Description
    Provides parameter counting, representation formatting, and validation hooks for deep learning models.
    """

    def __init__(self) -> None:
        """Initializes the `BaseModule`."""
        super().__init__()
        return None

    def count_parameters(self) -> int:
        """Counts the total number of learnable parameters in the module.

        ## Description
        Computes the number of parameters. Each complex parameter is counted as two real parameters.

        ## Returns
        `int`: Total number of learnable parameters.
        """
        cnt: int = 0
        for p in self.parameters():
            c: int = 2 if p.is_complex() else 1
            cnt += c * p.numel()
        return cnt

    def check_arguments(self, **kwargs: Any) -> None:
        """Validates layer and model arguments.

        ## Description
        Hook for subclasses to validate architectural parameters.

        ## Arguments
        `**kwargs` (`Any`): Keyword arguments to check.

        ## Returns
        `None`: Returns `None` if arguments are valid.
        """
        pass

    def __str__(self) -> str:
        """Returns a formatted summary string of subnetworks and parameters.

        ## Description
        Generates an ASCII overview table listing subnetworks, parameters, shapes, types, and parameter count.

        ## Returns
        `str`: Formatted overview string.
        """
        msg:         list[str] = []
        __half_line: str       = '=' * 30
        _front:      str       = ''.join((__half_line, f'< {self.__class__.__name__} >', __half_line))
        _line:       str       = '-' * len(_front)
        _back:       str       = '=' * len(_front)

        msg.append(_front)
        msg.append("[ Subnetworks ]\n")
        for name, md in self.named_children():
            msg.append(f"* {name}")
            msg.append(str(md))
            msg.append('')

        msg.append(_line)
        msg.append("[ Parameters ]")
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
def __dir__() -> list[str]:
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
def main() -> None:
    pass


if __name__ == '__main__':
    main()