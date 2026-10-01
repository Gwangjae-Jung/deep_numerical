"""Models for operator learning

-----
### Description
This submodule provides some classes of neural operators which can be used to approximate continuous operators.

-----
### Features

* Deep Operator Network (DeepONet) and Multiple-Input Operator Network (MIONet)
* Graph Neural Operator (GNO)
* Fourier Neural Operator (FNO) and its variants (Factorized FNO, Separable FNO, Tensorized FNO, Radial FNO)
* Galerkin Transformer (GT)
"""
from    deep_numerical.neural.operator.fno \
    import  FourierNeuralOperator, FNO
from    deep_numerical.neural.operator.fno_factorized \
    import  FactorizedFourierNeuralOperator, FactorizedFNO
from    deep_numerical.neural.operator.fno_radial \
    import  RadialFourierNeuralOperator, RadialFNO
from    deep_numerical.neural.operator.fno_separable \
    import  SeparableFourierNeuralOperator, SeparableFNO
from    deep_numerical.neural.operator.hyper_sfno \
    import  HyperSFNO
from    deep_numerical.neural.operator.fno_tensorized \
    import  TensorizedFourierNeuralOperator, TensorizedFNO

from    deep_numerical.neural.operator.onet \
    import  DeepONet, DeepONetUnstructured, MIONet, MIONetUnstructured
from    deep_numerical.neural.operator.hyper_onet \
    import  HyperDeepONet, HyperMIONet, ParameterizedMIONet
from    deep_numerical.neural.operator.gt \
    import  GalerkinTransformer, GalerkinTransformerSelfAttention, GalerkinTransformerCrossAttention

__all__ = [
    'FourierNeuralOperator', 'FNO',
    'FactorizedFourierNeuralOperator', 'FactorizedFNO',
    'RadialFourierNeuralOperator', 'RadialFNO',
    'SeparableFourierNeuralOperator', 'SeparableFNO',
    'HyperSFNO',
    'TensorizedFourierNeuralOperator', 'TensorizedFNO',
    'DeepONet', 'DeepONetUnstructured', 'MIONet', 'MIONetUnstructured',
    'HyperDeepONet', 'HyperMIONet', 'ParameterizedMIONet',
    'GalerkinTransformer', 'GalerkinTransformerSelfAttention', 'GalerkinTransformerCrossAttention',
]

# Conditional imports for graph neural operators (requires torch_geometric)
try:
    from deep_numerical.neural.operator.gno \
        import GraphNeuralOperator, GraphKernelNetwork, GNO, GKN
    __all__.extend(['GraphNeuralOperator', 'GraphKernelNetwork', 'GNO', 'GKN'])
except (ImportError, ModuleNotFoundError, AttributeError):
    pass

try:
    from deep_numerical.neural.operator.mgno \
        import MultipoleGraphNeuralOperator, MGNO
    __all__.extend(['MultipoleGraphNeuralOperator', 'MGNO'])
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


##################################################
##################################################
# End of file