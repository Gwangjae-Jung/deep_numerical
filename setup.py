from    setuptools  import  setup, find_packages


setup(
    name            = "deep_numerical",
    version         = "2.5.3",
    description     = "A PyTorch-based library for deep learning and numerical methods for kinetic equations.",
    author          = "Gwangjae-Jung",
    packages        = find_packages(),
    python_requires = ">=3.10",
)