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
from    deep_numerical.neural.operator.fno_tensorized \
    import  TensorizedFourierNeuralOperator, TensorizedFNO

from    deep_numerical.neural.operator.onet \
    import  DeepONet, DeepONetUnstructured, MIONet, MIONetUnstructured
from    deep_numerical.neural.operator.gt \
    import  GalerkinTransformer, GalerkinTransformerSelfAttention, GalerkinTransformerCrossAttention
from    deep_numerical.neural.operator.gno \
    import  GraphNeuralOperator, GraphKernelNetwork


##################################################
##################################################
# End of file