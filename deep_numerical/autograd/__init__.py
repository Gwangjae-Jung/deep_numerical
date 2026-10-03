"""Implementation of automatic differentiation (autograd) functionalities.

This submodule provides tools for computing derivatives, Jacobians, Hessians, and higher-order derivatives of functions using PyTorch's automatic differentiation capabilities.
In order to facilitate vectorized operations, it leverages `vmap` from `torch.func`.
"""
from deep_numerical.autograd.grad import compute_grad
from deep_numerical.autograd.vmap import derivatives, hessian, jacobian


__all__: list[str] = [
    "compute_grad",
    "derivatives",
    "hessian",
    "jacobian",
]


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()