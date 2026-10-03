from   collections.abc   import Collection
from   typing            import Any, Callable, List, Literal, Union
import warnings

import torch
from   typing_extensions import TypeAlias

from   deep_numerical    import Objects


__all__: list[str] = [
    "activations",
    "TORCH_ACTIVATION_DICT",
    "initializers",
    "TORCH_INITIALIZER_DICT",
    "get_activation",
    "initialize_weights",
    "count_parameters",
    "warn_redundant_arguments",
]


Activations:  TypeAlias = Literal["elu", "gelu", "identity", "leaky relu", "relu", "silu", "sigmoid", "softmax", "tanh"]
Initializers: TypeAlias = Literal["constant", "dirac", "eye", "kaiming normal", "kaiming uniform", "normal", "ones", "orthogonal", "sparse", "trunc normal", "uniform", "xavier normal", "xavier uniform", "zeros"]


activations: dict[Activations, Any] = {
    "elu":        torch.nn.ELU,
    "gelu":       torch.nn.GELU,
    "identity":   torch.nn.Identity,
    "leaky relu": torch.nn.LeakyReLU,
    "relu":       torch.nn.ReLU,
    "silu":       torch.nn.SiLU,
    "sigmoid":    torch.nn.Sigmoid,
    "softmax":    torch.nn.Softmax,
    "tanh":       torch.nn.Tanh,
}
initializers: dict[Initializers, Callable[..., torch.Tensor]] = {
    "constant":        torch.nn.init.constant_,
    "dirac":           torch.nn.init.dirac_,
    "eye":             torch.nn.init.eye_,
    "kaiming normal":  torch.nn.init.kaiming_normal_,
    "kaiming uniform": torch.nn.init.kaiming_uniform_,
    "normal":          torch.nn.init.normal_,
    "ones":            torch.nn.init.ones_,
    "orthogonal":      torch.nn.init.orthogonal_,
    "sparse":          torch.nn.init.sparse_,
    "trunc normal":    torch.nn.init.trunc_normal_,
    "uniform":         torch.nn.init.uniform_,
    "xavier normal":   torch.nn.init.xavier_normal_,
    "xavier uniform":  torch.nn.init.xavier_uniform_,
    "zeros":           torch.nn.init.zeros_,
}
TORCH_ACTIVATION_DICT:  dict[Activations, Any]                         = activations
TORCH_INITIALIZER_DICT: dict[Initializers, Callable[..., torch.Tensor]] = initializers


##################################################
def count_parameters(models: Objects[torch.nn.Module], complex_as_two: bool = True) -> Union[int, List[int]]:
    """Counts the number of learnable parameters in one or more PyTorch models.

    ## Description
    Counts the number of learnable parameters in one or more PyTorch models.
    Each complex parameter is counted as two real parameters if `complex_as_two` is `True`.

    ## Arguments
    `models` (`Objects[torch.nn.Module]`): A model or collection of models whose parameters are counted.
    `complex_as_two` (`bool`, default: `True`): If `True`, complex parameters are counted as two real parameters.

    ## Returns
    `Union[int, List[int]]`: The parameter count of the model, or a list of counts if multiple models are provided.
    """
    if not isinstance(models, Collection):
        models = [models]
    num_params: List[int] = []
    model: torch.nn.Module
    for model in models:
        cnt: int = 0
        for p in model.parameters():
            cnt += p.numel() * (1 + int(complex_as_two and p.is_complex()))
        num_params.append(cnt)
    if len(models) == 1:
        return num_params[0]
    else:
        return num_params


def get_activation(
        activation_name:   Activations,
        activation_kwargs: dict[str, object] = {},
    ) -> torch.nn.Module:
    """Retrieves and instantiates a PyTorch activation module.

    ## Description
    Retrieves and instantiates a PyTorch activation module corresponding to the specified activation name.

    ## Arguments
    `activation_name` (`Activations`): The name of the activation function (e.g., `'relu'`, `'silu'`, `'tanh'`).
    `activation_kwargs` (`dict[str, object]`, default: `{}`): Keyword arguments forwarded to the activation module initializer.

    ## Returns
    `torch.nn.Module`: The instantiated activation layer.
    """
    return TORCH_ACTIVATION_DICT[activation_name](**activation_kwargs)


def initialize_weights(
        models:      Objects[torch.nn.Module],
        init_name:   Initializers      = "xavier normal",
        init_kwargs: dict[str, object] = {},
    ) -> None:
    """Initializes weights in PyTorch model(s).

    ## Description
    Applies the specified initialization strategy to the parameters of one or multiple PyTorch models.

    ## Arguments
    `models` (`Objects[torch.nn.Module]`): A single model or collection of models to initialize.
    `init_name` (`Initializers`, default: `"xavier normal"`): The initialization method name.
    `init_kwargs` (`dict[str, object]`, default: `{}`): Additional keyword arguments passed to the initializer.

    ## Returns
    `None`: Initializes model parameters in-place.
    """
    if not isinstance(models, Collection):
        models = [models]
    try:
        initializer = TORCH_INITIALIZER_DICT[init_name]
    except KeyError:
        raise KeyError(
            f"The passed value {init_name} of 'init_name' is not in the list of supported initializations:\n{list(TORCH_INITIALIZER_DICT.keys())}"
        )
    model: torch.nn.Module
    for model in models:
        for p in model.parameters():
            try:
                initializer(p, **init_kwargs)
            except Exception:
                continue
    return


def warn_redundant_arguments(cls: type, kwargs: dict[str, Any]) -> None:
    """Warns if redundant keyword arguments are passed to a class initializer.

    ## Description
    Issues a `UserWarning` if extraneous keyword arguments are passed to a class initializer.

    ## Arguments
    `cls` (`type`): Target class whose initializer received extra kwargs.
    `kwargs` (`dict[str, Any]`): Extraneous keyword arguments dictionary.

    ## Returns
    `None`: Issues a warning if kwargs is non-empty.
    """
    if kwargs:
        warnings.warn(
            f"Redundant or unexpected arguments passed to '{cls.__name__}': {list(kwargs.keys())}",
            UserWarning,
            stacklevel=2,
        )
    return None


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()