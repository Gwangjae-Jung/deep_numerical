"""Utility functions and modules for deep_numerical.
"""
from deep_numerical.utils.dtype          import *
from deep_numerical.utils.grid           import *
from deep_numerical.utils.graph          import *
from deep_numerical.utils.metric         import *
from deep_numerical.utils.time_utils     import *
from deep_numerical.utils._random_grid   import *
from deep_numerical.utils._normalizers   import *
from deep_numerical.utils._uncategorized import *

from deep_numerical.utils.dtype          import __all__ as __all_dtype
from deep_numerical.utils.grid           import __all__ as __all_grid
from deep_numerical.utils.graph          import __all__ as __all_graph
from deep_numerical.utils.metric         import __all__ as __all_metric
from deep_numerical.utils.time_utils     import __all__ as __all_time
from deep_numerical.utils._normalizers   import __all__ as __all_norm
from deep_numerical.utils._uncategorized import __all__ as __all_uncat
from deep_numerical.utils._random_grid   import __all__ as __all_random_grid


__all__: list[str] = (
    list(__all_dtype)
    + list(__all_grid)
    + list(__all_graph)
    + list(__all_metric)
    + list(__all_time)
    + list(__all_norm)
    + list(__all_uncat)
    + list(__all_random_grid)
)


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()