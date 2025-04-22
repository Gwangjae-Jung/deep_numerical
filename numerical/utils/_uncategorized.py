from    typing      import  Sequence
import  numpy       as      np

    
##################################################
##################################################
__all__:    list[int] = ['periodization']


##################################################
##################################################
def periodization(X: np.ndarray, axes=Sequence[int]) -> np.ndarray:
    sl          = [Ellipsis for _ in range(X.ndim)]
    pad_width   = [(0, 0)   for _ in range(X.ndim)]
    for ax in axes:
        sl[ax] = slice(0, -1)
        pad_width[ax] = (0, 1)
    return np.pad(X[*sl], pad_width, mode="wrap")


##################################################
##################################################
# End of file