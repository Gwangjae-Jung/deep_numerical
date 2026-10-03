from   typing import Optional, Sequence, Union
import torch


__all__: list[str] = ["absolute_error", "relative_error", "psnr"]


##################################################
def absolute_error(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    p:       Union[float, str]       = 2.0,
    dim:     Optional[Sequence[int]] = None,
    scale:   Optional[torch.Tensor]  = None,
) -> torch.Tensor:
    """Returns the instance-wise absolute error between `preds` and `targets`.

    ## Description
    Given sequences `preds` and `targets` of shape `(N, ...)`, returns the absolute error tensor of shape `(N,)`, where each entry is the norm of `preds - targets` of order `p`.

    ## Arguments
    `preds` (`torch.Tensor`): The predictions.
    `targets` (`torch.Tensor`): The targets.
    `p` (`Union[float, str]`, default: `2.0`): The order of the error norm. If string, must be `'inf'`, `'1'`, or `'2'`.
    `dim` (`Optional[Sequence[int]]`, default: `None`): The dimensions to compute the error over. If `None`, all dimensions except batch dimension are used.
    `scale` (`Optional[torch.Tensor]`, default: `None`): A scaling factor for the error.

    ## Returns
    `torch.Tensor`: The tensor of absolute errors.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    if dim is None:
        dim = tuple(range(1, preds.ndim))
    if isinstance(p, str):
        p = p.lower()
        if p == 'inf':
            p = torch.inf
    if scale is None:
        scale_tensor: torch.Tensor = torch.ones((preds.size(0),), device=preds.device)
    else:
        scale_tensor = scale
    diff:      torch.Tensor = torch.norm(preds - targets, dim=dim, p=p)
    scale_val: torch.Tensor = scale_tensor.reshape(diff.shape)
    abs_err:   torch.Tensor = scale_val * diff
    return abs_err


def relative_error(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    p:       Union[float, str]       = 2.0,
    dim:     Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Returns the instance-wise relative error between `preds` and `targets`.

    ## Description
    Given sequences `preds` and `targets` of shape `(N, ...)`, returns the relative error tensor of shape `(N,)`.

    ## Arguments
    `preds` (`torch.Tensor`): The predictions.
    `targets` (`torch.Tensor`): The targets.
    `p` (`Union[float, str]`, default: `2.0`): The order of the error norm. If string, must be `'inf'`, `'1'`, or `'2'`.
    `dim` (`Optional[Sequence[int]]`, default: `None`): The dimensions to compute the error over. If `None`, all dimensions except batch dimension are used.

    ## Returns
    `torch.Tensor`: The tensor of relative errors.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    if dim is None:
        dim = tuple(range(1, preds.ndim))
    if isinstance(p, str):
        p = p.lower()
        if p == 'inf':
            p = torch.inf
    numer: torch.Tensor = torch.norm(preds - targets, dim=dim, p=p)
    denom: torch.Tensor = torch.norm(targets, dim=dim, p=p)
    return numer / denom


def psnr(
    preds:         torch.Tensor,
    targets:       torch.Tensor,
    max_intensity: float = 1.0,
) -> torch.Tensor:
    """Returns the PSNR (peak signal-to-noise ratio) of `preds` to `targets`.

    ## Description
    Computes the peak signal-to-noise ratio (PSNR) in decibels between `preds` and `targets`.

    ## Arguments
    `preds` (`torch.Tensor`): The predictions.
    `targets` (`torch.Tensor`): The targets.
    `max_intensity` (`float`, default: `1.0`): The maximum intensity of the data.

    ## Returns
    `torch.Tensor`: The PSNR value tensor.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    ndim: int          = preds.ndim
    mse:  torch.Tensor = (preds - targets).pow(2).mean(tuple(range(1, ndim)))
    return 10 * ((max_intensity ** 2) / mse).log10()


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()