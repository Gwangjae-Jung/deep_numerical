import  torch


##################################################
##################################################
__all__: list[str] = ['absolute_error', 'relative_error']


##################################################
##################################################
def absolute_error(
        pred:       torch.Tensor,
        target:     torch.Tensor,
        p:          float   = 2,
        is_batched: bool    = True
    ) -> torch.Tensor:
    assert pred.shape==target.shape
    ndim = pred.ndim
    if is_batched:
        norm_dim = tuple(range(1, ndim))
    else:
        norm_dim = tuple(range(ndim))
    return (pred-target).norm(p=p, dim=norm_dim)


def relative_error(
        pred:       torch.Tensor,
        target:     torch.Tensor,
        p:          float   = 2,
        is_batched: bool    = True
    ) -> torch.Tensor:
    assert pred.shape==target.shape
    ndim = pred.ndim
    if is_batched:
        norm_dim = tuple(range(1, ndim))
    else:
        norm_dim = tuple(range(ndim))
    return (pred-target).norm(p=p, dim=norm_dim) / target.norm(p=p, dim=norm_dim)


##################################################
##################################################
# End of file