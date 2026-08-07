## Python library for numerical methods for solving kinetic equations with neural network architectures

-----
This library provides a collection of the spectral methods for solving kinetic equations, such as the Fokker-Planck-Landau equation and the Boltzmann equation.
It also provides a collection of neural network architectures.

### A. Spectral methods
The spectral methods provided by this library can be found in the submodule `deep_numerical.numerical`, which includes the following methods.
1. Classical spectral method for the Boltzmann equation
    1. Only the solver for the elastic Boltzmann equation is implemented.
2. Fast spectral method
    1. (Fokker-Planck-Landau equation) [Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator](https://www.sciencedirect.com/science/article/pii/S0021999100966129)
    2. (Boltzmann equation) [A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels](https://epubs.siam.org/doi/10.1137/16M1096001)

### B. Neural network architectures
The neural network architectures provided by this library can be found in the submodules `deep_numerical.neural`.
1. `deep_numerical.neural.layer` contains several fundamental layers which are used to implement neural networks.
2. `deep_numerical.neural.network` contains, so far, the separable neural network, which is generally known as the [separable physics-informed neural network](https://neurips.cc/virtual/2022/59890).
3. `deep_numerical.neural.operator` contains several fundamental neural operators with layers from `deep_numerical.neural.layer`. These neural operators contains [Deep Operator Network](https://www.nature.com/articles/s42256-021-00302-5), [Multiple Input Operator Network](https://epubs.siam.org/doi/epdf/10.1137/22M1477751), [Graph Neural Operator](https://openreview.net/pdf?id=fg2ZFmXFO3), [Fourier Neural Operator](https://openreview.net/pdf?id=c8P9NQVtmnO), and [Galerkin Transformer](https://proceedings.neurips.cc/paper/2021/file/d0921d442ee91b896ad95059d13df618-Supplemental.pdf).
