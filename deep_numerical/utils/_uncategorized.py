from    typing      import  Sequence
import  torch

    
##################################################
##################################################
__all__:    list[int] = ['periodization']


##################################################
##################################################
def periodization(X: torch.Tensor, axes=Sequence[int]) -> torch.Tensor:
    sl          = [Ellipsis for _ in range(X.ndim)]
    pad_width   = [(0, 0)   for _ in range(X.ndim)]
    for ax in axes:
        sl[ax] = slice(0, -1)
        pad_width[ax] = (0, 1)
    return torch.nn.functional.pad(X[*sl], pad_width, mode="wrap")


##################################################
##################################################
# End of file