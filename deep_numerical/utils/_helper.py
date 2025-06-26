import  warnings
from    typing      import  Callable
from    datetime    import  datetime


##################################################
##################################################
__all__: list[str] = [
    'EINSUM_STRING',
    'ones',
    'zeros',
    'repeat',
    'get_time_str',
    'warn_redundant_arguments',
]


##################################################
##################################################
EINSUM_STRING:  str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""The string of 26 uppercase alphabets (`ABC...XYZ`), which is used to define an Einstein summation command by slicing the string."""


ones:   Callable[[int], tuple[int]] = lambda k: tuple((1 for _ in range(k)))
"""Returns the tuple of `k` times of the integer `1`."""
zeros:  Callable[[int], tuple[int]] = lambda k: tuple((0 for _ in range(k)))
"""Returns the tuple of `k` times of the integer `0`."""
repeat: Callable[[object, int], tuple[object]] = lambda obj, k: tuple((obj for _ in range(k)))
"""Returns the tuple of `k` times of the object `obj`."""


def get_time_str() -> str:
    """Returns the datetime string of the form `{year}{month}{day}_{hour}{minute}{second}`."""
    current_time = datetime.now()
    return current_time.strftime("%Y%m%d_%H%M%S")


def warn_redundant_arguments(
        classtype:  type    = None,
        args:       tuple   = tuple(),
        kwargs:     dict    = dict(),
    ) -> None:
    """Warns the user about the redundant arguments given to the initializer of a class."""
    msg: list[str] = []
    if len(args) + len(kwargs) > 0:
        msg.append(f"({classtype.__name__}) The following unnecessary arguments are given to the initializer.")
        for cnt, _arg in enumerate(args):
            msg.append(f"* args[{cnt}] >>> {_arg}")
        for cnt, (k, v) in enumerate(kwargs.items()):
            msg.append(f"* kwargs[{cnt}] >>> {k}={v}")
        msg.append(f"Note that these arguments are ignored.")
        warnings.warn('\n'.join(msg), UserWarning)
    return


##################################################
##################################################
# End of file