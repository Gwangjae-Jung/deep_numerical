"""Implementation of automatic differentiation (autograd) functionalities.

This submodule provides tools for computing derivatives, Jacobians, Hessians, and higher-order derivatives of functions using PyTorch's automatic differentiation capabilities.
In order to facilitate vectorized operations, it leverages `vmap` from `torch.func`.
"""
from    deep_numerical.autograd.grad   import  compute_grad
from    deep_numerical.autograd.vmap   import  jacobian, hessian, derivatives