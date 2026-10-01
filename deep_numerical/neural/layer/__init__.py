from    deep_numerical.neural.layer.general \
    import  MLP, HyperMLP, PatchEmbedding, Periodization1D
from    deep_numerical.neural.layer.attention \
    import  LinearSelfAttention, LinearCrossAttention, ModifiedMLP, HyperLinearSelfAttention
from    deep_numerical.neural.layer.integral_layers \
    import  IntegralLinearV1, IntegralLinear, IntegralConv2D, IntegralConv2D_
from    deep_numerical.neural.layer.fourier_layer \
    import  SpectralConv, FourierLayer
from    deep_numerical.neural.layer.fourier_layer_factorized \
    import  FactorizedSpectralConv, FactorizedFourierLayer
from    deep_numerical.neural.layer.fourier_layer_radial \
    import  RadialSpectralConv, RadialFourierLayer
from    deep_numerical.neural.layer.fourier_layer_separable \
    import  SeparableSpectralConv, SeparableFourierLayer
from    deep_numerical.neural.layer.fourier_layer_separable_h \
    import  HyperSeparableSpectralConv, HyperSeparableFourierLayer
from    deep_numerical.neural.layer.fourier_layer_tensorized \
    import  TensorizedSpectralConv, TensorizedFourierLayer
from    deep_numerical.neural.layer.kinetic_fourier_layer \
    import  FourierBoltzmannLayer
from    deep_numerical.neural.layer.galerkin_transformer \
    import  GalerkinTypeSelfAttention, GalerkinTypeCrossAttention, GalerkinTypeEncoderBlockSelfAttention, GalerkinTypeEncoderBlockCrossAttention

__all__ = [
    "MLP", "HyperMLP", "PatchEmbedding", "Periodization1D",
    "LinearSelfAttention", "LinearCrossAttention", "ModifiedMLP", "HyperLinearSelfAttention",
    "IntegralLinearV1", "IntegralLinear", "IntegralConv2D", "IntegralConv2D_",
    "SpectralConv", "FourierLayer",
    "FactorizedSpectralConv", "FactorizedFourierLayer",
    "RadialSpectralConv", "RadialFourierLayer",
    "SeparableSpectralConv", "SeparableFourierLayer",
    "HyperSeparableSpectralConv", "HyperSeparableFourierLayer",
    "TensorizedSpectralConv", "TensorizedFourierLayer",
    "FourierBoltzmannLayer",
    "GalerkinTypeSelfAttention", "GalerkinTypeCrossAttention",
    "GalerkinTypeEncoderBlockSelfAttention", "GalerkinTypeEncoderBlockCrossAttention",
]

# Conditional imports for graph neural network layers (requires torch_geometric)
try:
    from deep_numerical.neural.layer.graph_layer import GraphKernelLayer
    if GraphKernelLayer is not None:
        __all__.append("GraphKernelLayer")
except (ImportError, ModuleNotFoundError, AttributeError):
    pass

try:
    from deep_numerical.neural.layer.vector_self_attention import VectorSelfAttention
    if VectorSelfAttention is not None:
        __all__.append("VectorSelfAttention")
except (ImportError, ModuleNotFoundError, AttributeError):
    pass

try:
    from deep_numerical.neural.layer.multipole_graph_layer import MultipoleGraphKernelLayer
    if MultipoleGraphKernelLayer is not None:
        __all__.append("MultipoleGraphKernelLayer")
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


##################################################
##################################################
# End of file