from    typing          import  TypeAlias, Any, Callable, Sequence, List, Literal, Union
import  torch
from    deep_numerical  import  Objects


__all__ = [
    "activations", "TORCH_ACTIVATION_DICT",
    "initializers", "TORCH_INITIALIZER_DICT",
    "get_activation", "initialize_weights", "count_parameters",
]


##################################################
##################################################
Activations: TypeAlias = Literal["elu", "gelu", "identity", "leaky relu", "relu", "silu", "sigmoid", "softmax", "tanh"]
Initializers: TypeAlias = Literal["constant", "dirac", "eye", "kaiming normal", "kaiming uniform", "normal", "ones", "orthogonal", "sparse", "trunc normal", "uniform", "xavier normal", "xavier uniform", "zeros"]


activations: dict[Activations, torch.nn.Module]= {
    "elu":          torch.nn.ELU,
    "gelu":         torch.nn.GELU,
    "identity":     torch.nn.Identity,
    "leaky relu":   torch.nn.LeakyReLU,
    "relu":         torch.nn.ReLU,
    "silu":         torch.nn.SiLU,
    "sigmoid":      torch.nn.Sigmoid,
    "softmax":      torch.nn.Softmax,
    "tanh":         torch.nn.Tanh,
}
initializers: dict[Initializers, Callable[[torch.Tensor, Any], torch.Tensor]] = {
    "constant":         torch.nn.init.constant_,
    "dirac":            torch.nn.init.dirac_,
    "eye":              torch.nn.init.eye_,
    "kaiming normal":   torch.nn.init.kaiming_normal_,
    "kaiming uniform":  torch.nn.init.kaiming_uniform_,
    "normal":           torch.nn.init.normal_,
    "ones":             torch.nn.init.ones_,
    "orthogonal":       torch.nn.init.orthogonal_,
    "sparse":           torch.nn.init.sparse_,
    "trunc normal":     torch.nn.init.trunc_normal_,
    "uniform":          torch.nn.init.uniform_,
    "xavier normal":    torch.nn.init.xavier_normal_,
    "xavier uniform":   torch.nn.init.xavier_uniform_,
    "zeros":            torch.nn.init.zeros_,
}
TORCH_ACTIVATION_DICT = activations
TORCH_INITIALIZER_DICT = initializers


##################################################
##################################################
def count_parameters(models: Objects[torch.nn.Module], complex_as_two: bool=True) -> Union[int, List[int]]:
    if not isinstance(models, Sequence):
        models = [models]
    num_params: List[int] = []
    model: torch.nn.Module
    for model in models:
        cnt = 0
        for p in model.parameters():
            cnt += p.numel() * (1 + (complex_as_two and p.is_complex()))
        num_params.append(cnt)
    if len(models)==1:  return num_params[0]
    else:               return num_params


def get_activation(
        activation_name:    Activations,
        activation_kwargs:  dict[str, object] = {},
    ) -> torch.nn.Module:
    return TORCH_ACTIVATION_DICT[activation_name](**activation_kwargs)


def initialize_weights(
        models:         Objects[torch.nn.Module],
        init_name:      Initializers = "xavier normal",
        init_kwargs:    dict[str, object] = {},
    ) -> None:
    """Initialize the weights in `models`.
    
    Arguments:
        `models` (`Union[nn.Module, Sequence[nn.Module]]`):
            A model or a sequence of models to be initialize.
        `init_name` (`Initializers`):
            The method to be used to initialize the model(s).
        `init_kwargs` (`dict[str, object]`, default: `{}`):
            Any further arguments for weight initialization.
    """
    if not isinstance(models, Sequence):
        models = [models]
    try:
        initializer = TORCH_INITIALIZER_DICT[init_name]
    except:
        raise KeyError(
            f"The passed value {init_name} of 'init_name'is not in the list of supported initalization:\n{TORCH_INITIALIZER_DICT.keys()}"
        )
    model: torch.nn.Module
    for model in models:
        for p in model.parameters():
            try:
                initializer(p, **init_kwargs)
            except:
                continue
    return


##################################################
##################################################
# End of file