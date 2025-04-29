from    typing      import  *
import  warnings

from    torch       import  nn, Tensor

from    ._dtype     import  Objects


##################################################
##################################################
__all__ = [
    "TORCH_ACTIVATION_DICT",
    "TORCH_INITIALIZER_DICT",
    "count_parameters",
    "get_activation",
    "initialize_weights",
    "warn_n_layers",
    "CheckShape",
]


##################################################
##################################################
TORCH_ACTIVATION_DICT: dict[str, nn.Module]= {
    "elu":          nn.ELU,
    "gelu":         nn.GELU,
    "identity":     nn.Identity,
    "leaky relu":   nn.LeakyReLU,
    "relu":         nn.ReLU,
    "silu":         nn.SiLU,
    "sigmoid":      nn.Sigmoid,
    "softmax":      nn.Softmax,
    "tanh":         nn.Tanh,
}
"""Supported activation functions in `torch`:
* elu
* gelu
* identity
* leaky relu
* relu
* silu
* sigmoid
* softmax
* tanh
"""

TORCH_INITIALIZER_DICT: dict[str, Callable[[Tensor, Any], Tensor]] = {
    "constant":         nn.init.constant_,
    "dirac":            nn.init.dirac_,
    "eye":              nn.init.eye_,
    "kaiming normal":   nn.init.kaiming_normal_,
    "kaiming uniform":  nn.init.kaiming_uniform_,
    "normal":           nn.init.normal_,
    "ones":             nn.init.ones_,
    "orthogonal":       nn.init.orthogonal_,
    "sparse":           nn.init.sparse_,
    "trunc normal":     nn.init.trunc_normal_,
    "uniform":          nn.init.uniform_,
    "xavier normal":    nn.init.xavier_normal_,
    "xavier uniform":   nn.init.xavier_uniform_,
    "zeros":            nn.init.zeros_,
}
        

##################################################
##################################################
def count_parameters(models: Objects[nn.Module], complex_as_two: bool=True) -> Union[int, list[int]]:
    if not isinstance(models, Sequence):
        models = [models]
    
    num_params: list[int] = []
    model: nn.Module
    for model in models:
        cnt = 0
        for p in model.parameters():
            cnt += p.numel() * (1 + (complex_as_two and p.is_complex()))
        num_params.append(cnt)
    
    if len(models) == 1:
        return num_params[0]
    else:
        return num_params


def get_activation(
        activation_name:    str,
        activation_kwargs:  dict[str, object] = {}
    ) -> nn.Module:
    return TORCH_ACTIVATION_DICT[activation_name](**activation_kwargs)


def initialize_weights(
        models:         Objects[nn.Module],
        init_name:      str,
        init_kwargs:    dict[str, object] = {},
    ) -> None:
    if not isinstance(models, Sequence):
        models = [models]
    
    try:
        initializer = TORCH_INITIALIZER_DICT[init_name]
    except:
        raise KeyError(
            f"The passed value {init_name} of 'init_name'is not in the list of supported initalization:\n{TORCH_INITIALIZER_DICT.keys()}"
        )
    model: nn.Module
    for model in models:
        for p in model.parameters():
            try:
                initializer(p, **init_kwargs)
            except:
                continue
    return


##################################################
##################################################
def warn_n_layers(n_layers: int) -> None:
    warnings.warn(f"The number of the hidden layers should be positive. ('n_layers': {n_layers})")
    return


class CheckShape(nn.Module):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.__msg = message
        return
    
    def forward(self, X: Tensor) -> Tensor:
        print(
                f"{self.__msg}\n\t*", 
                f"Shape: {X.shape} ({X.ndim}-dimensional)"
        )
        return X


##################################################
##################################################
# End of file