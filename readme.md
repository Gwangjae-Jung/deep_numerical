<h1>Python library for numerical methods for solving kinetic equations with neural network architectures</h1>

<div>
This library provides a collection of spectral methods for solving kinetic equations, such as the Fokker-Planck-Landau equation and the Boltzmann equation.
It also provides a collection of neural network architectures.
</div>

<h2>Spectral methods</h2>
<div>
    The spectral methods provided by this library can be found in the submodule `deep_numerical.numerical`, which includes the following methods.
    <ul>
        <li>
            Classical spectral method for the Boltzmann equation
            <ul>
                <li>Only the solver for the elastic Boltzmann equation is implemented.</li>
            </ul>
        </li>
        <li>
            Fast spectral method
            <ul>
                <li>
                    (Fokker-Planck-Landau equation)
                    <a href="https://www.sciencedirect.com/science/article/pii/S0021999100966129">
                        Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator
                    </a>
                </li>
                <li>
                    (Boltzmann equation)
                    <a href="https://epubs.siam.org/doi/10.1137/16M1096001">
                        A Fast Spectral Method for the Boltzmann Collision Operator with General Collision Kernels
                    </a>
                </li>
            </ul>
        </li>
    </ul>
</div>

<h2>Neural network architectures</h2>
<div>
    The neural network architectures provided by this library can be found in the submodules `deep_numerical.neural`.
    <ul>
        <li>
            `deep_numerical.neural.layer` contains several fundamental layers, which are used to implement neural networks.
        </li>
        <li>
            `deep_numerical.neural.network` contains, so far, the separable neural network, which is generally known as the <a href="https://neurips.cc/virtual/2022/59890">separable physics-informed neural network</a>.
        </li>
        <li>
            `deep_numerical.neural.operator` contains several fundamental neural operators with layers from `deep_numerical.neural.layer`, including the following neural operators:
            <ul>
                <li>
                    <a href="https://www.nature.com/articles/s42256-021-00302-5">Deep Operator Network</a> and <a href="https://epubs.siam.org/doi/epdf/10.1137/22M1477751">Multiple Input Operator Network</a>.
                </li>
                <li>
                    <a href="https://openreview.net/pdf?id=fg2ZFmXFO3">Graph Neural Operator</a> and <a href="https://openreview.net/pdf?id=c8P9NQVtmnO">Fourier Neural Operator</a>, including several variants of the Fourier Neural Operator.
                </li>
                <li>
                    <a href="https://proceedings.neurips.cc/paper/2021/file/d0921d442ee91b896ad95059d13df618-Supplemental.pdf">Galerkin Transformer</a>.
                </li>
            </ul>
        </li>
    </ul>
</div>
