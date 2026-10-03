"""Kernel modes for spectral Boltzmann collision operators.
"""
from .boltzmann_VHS import *
from .boltzmann_VHS import __all__ as __all__boltzmann_vhs


__all__: list[str] = list(__all__boltzmann_vhs)


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()