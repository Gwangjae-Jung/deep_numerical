from    typing_extensions       import  Self

import  torch
from    torch       import  nn
from    torch.nn    import  functional  as  F




##################################################
##################################################


__all__ = [
    "TNet",
    "TNetTransform",
]


##################################################
##################################################


class TNet(nn.Module):
    """## The T-net in PointNet
    -----
    ### Description
    T-net aims to predict an affine transform matrix so as to align the input tensor to a canonical space before feature extraction.
    To be precise, given an input batch `X` of `B` point clouds (each of which consists of `N` points in the `C`-dimensional space; `X` is of shape `(B, C, N)`), T-net aims to return a 3-tensor `T` of shape `(B, in_channels, out_channels)` so that
    
    `torch.einsum("bin,bio->boj", [X, T])`
    
    is aligned in a canonical space.
    
    See https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf.
    """
    def __init__(
            self,
            num_channels:       int,
            hidden_dimensions:  list[int]   = [64, 128, 1024, 512, 256],
            maxpool_at:         int         = 3,
        ) -> Self:
        super().__init__()
        
        self.num_channels = num_channels
        DIMENSIONS = [
            num_channels,
            *hidden_dimensions,
            num_channels ** 2
        ]
        self.__MAXPOOL_AT = maxpool_at
        
        
        self.list_fc = nn.ModuleList([])
        self.list_bn = nn.ModuleList([])
        for idx in range(len(DIMENSIONS) - 1):
            _in_channels  = DIMENSIONS[idx]
            _out_channels = DIMENSIONS[idx + 1]
            self.list_fc.append(nn.Conv1d(_in_channels, _out_channels, kernel_size = 1))
            if idx != len(DIMENSIONS) - 2:
                self.list_bn.append(nn.BatchNorm1d(_out_channels))
        
        nn.init.zeros_(self.list_fc[-1].weight)
        self.list_fc[-1].bias = nn.Parameter(torch.eye(num_channels, dtype = torch.float).flatten())
        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input: `(B, C, N)`
        Output: `(B, C, C)`
        """
        for cnt in range(0, self.__MAXPOOL_AT):
            X = F.relu(self.list_bn[cnt](self.list_fc[cnt](X)))
        X = F.max_pool1d(X, kernel_size = X.shape[-1])
        for cnt in range(self.__MAXPOOL_AT, len(self.list_bn)):
            X = F.relu(self.list_bn[cnt](self.list_fc[cnt](X)))
        X = self.list_fc[-1](X)
        return X.reshape(-1, self.num_channels, self.num_channels)


class TNetTransform(nn.Module):
    """## The T-net transformer in PointNet
    """
    def __init__(
            self,
            num_channels:       int,
            hidden_dimensions:  list[int]   = [64, 128, 1024, 512, 256],
            maxpool_at:         int         = 3,
        ) -> Self:
        super().__init__()
        self.tnet = TNet(num_channels, hidden_dimensions, maxpool_at)
        return
    
    
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.tnet.forward(X)
        X = torch.einsum("bin,bij->bjn", [X, T])
        return (X, T)


##################################################
##################################################
