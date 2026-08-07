from    typing      import  Optional, Sequence, Union
import  torch


__all__ = ["absolute_error", "relative_error", "psnr"]


##################################################
##################################################
def absolute_error(
        preds:      torch.Tensor,
        targets:    torch.Tensor,
        p:          Union[float, str] = 2.0,
        dim:        Optional[Sequence[int]] = None,
        scale:      Optional[torch.Tensor]  = None,
    ) -> torch.Tensor:
    """Returns the instance-wise absolute error between the `preds` and `targets`.
    
    Given to sequences `preds` and `targets` of shape `(N, ...)` (where `N` is the number of the instances), this function returns the array `error` of shape `(N,)`, where `error[k]` is the absolute error of `preds[k]` of order `ord` from `targets[k]` for `k` in `range(N)`.
    
    Arguments:
        `preds` (`torch.Tensor`): The predictions.
        `targets` (`torch.Tensor`): The targets.
        `p` (`float` or `str`): The order of the error. If `p` is a string, it must be one of the following: 'inf', '1', '2'.
        `dim` (`Sequence[int]`, optional): The dimensions to compute the error over. If `None`, all dimensions are used.
        `scale` (`torch.Tensor`, optional): A scaling factor for the error. If `None`, no scaling is applied.
    
    Returns:
        `torch.Tensor`: The tensor of the absolute errors.
    
    ### Note
    To compute the error using the maximum function, pass `ord='inf'`.
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
        scale: torch.Tensor = torch.ones((preds.size(0),), device=preds.device)
    diff        = torch.norm(preds-targets, dim=dim, p=p)
    scale       = scale.reshape(diff.shape)
    abs_error   = scale * diff
    return abs_error


def relative_error(
        preds:      torch.Tensor,
        targets:    torch.Tensor,
        p:          Union[float, str] = 2.0,
        dim:        Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Returns the instance-wise relative error between the `preds` and `targets`.
    
    Given to sequences `preds` and `targets` of shape `(N, ...)` (where `N` is the number of the instances), this function returns the array `error` of shape `(N,)`, where `error[k]` is the relative error of `preds[k]` of order `ord` from `targets[k]` for `k` in `range(N)`.
    
    Arguments:
        `preds` (`torch.Tensor`): The predictions.
        `targets` (`torch.Tensor`): The targets.
        `p` (`float` or `str`): The order of the error. If `p` is a string, it must be one of the following: 'inf', '1', '2'.
        `dim` (`Sequence[int]`, optional): The dimensions to compute the error over. If `None`, all dimensions except for the batch dimension are used.
    
    Returns:
        `torch.Tensor`: The tensor of the relative errors of shape `(N,)`.
    
    ### Note
    To compute the error using the maximum function, pass `ord='inf'`.
    """
    if preds.shape != targets.shape:
        raise ValueError(f'The shapes of `preds` and `targets` must be equal, but got {preds.shape} and {targets.shape} instead.')
    if dim is None:
        dim = tuple(range(1, preds.ndim))
    if isinstance(p, str):
        p = p.lower()
        if p == 'inf':
            p = torch.inf
    numer = torch.norm(preds-targets, dim=dim, p=p)
    denom = torch.norm(targets, dim=dim, p=p)
    return numer/denom


##################################################
##################################################
def psnr(
        preds:          torch.Tensor,
        targets:        torch.Tensor,
        max_intensity:  float   = 1.0,
    ) -> torch.Tensor:
    """Returns the PSNR (peak signal-to-noise ratio) of `preds` to `targets`.
    
    Arguments:
        `preds` (`torch.Tensor`): The predictions.
        `targets` (`torch.Tensor`): The targets.
        `max_intensity` (`float`, default: `1.0): The maximum intensity of the data. In vision tasks with the images loaded as float (or double) tensors, images are usually normalized in `[0, 1]`, whence `max_intensity==1.0`.
    """
    assert preds.shape==targets.shape
    ndim = preds.ndim
    mse = (preds-targets).pow(2).mean(tuple(range(1, ndim)))
    return 10*((max_intensity**2)/mse).log10()


##################################################
##################################################
# End of file