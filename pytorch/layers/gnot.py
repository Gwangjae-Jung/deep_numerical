from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch       import  nn


##################################################
##################################################
__all__ = []


##################################################
##################################################


class HeterogeneousCrossAttention(nn.Module):
    """## Heterogeneous cross-attention"""
    def __init__(self) -> Self:
        super().__init__()
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X
    

##################################################
##################################################