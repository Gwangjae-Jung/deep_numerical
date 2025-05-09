## Python implementation of several numerical methods for kinetic equations and deep-learning architectures

In this library, several numerical methods for solving kinetic equations (Fokker-Planck-Landau equation, Boltzmann equation) are implemented, as well as several deep learning architectures.

### Supported numerical methods

1. Discrete velocity method (DVM): To be constructed.
   1. Classical DVM
   2. Fast DVM
2. Spectral method
   1. Classical spectral method for the Boltzmann equation
      1. Only the solver for the elastic Boltzmann equation is implemented.
   2. Fast spectral method
      1. (Fokker-Planck-Landau equation) Only the solver for the elastic FPL equation is implemented.
      2. (Boltzmann equation) Reference should be given.

### Note

Use the submodule `pytorch`, an actively updated submodule, since it supports both numerical methods and deep-learning frameworks so the numerical methods provided by this library can be conducted using GPUs. For this reason, the submodule `numerical` is less actively updated, and it could be deprecated in future.

### References

TBD.
