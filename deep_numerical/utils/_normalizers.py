from    typing_extensions   import  Self
import  torch


##################################################
##################################################
__all__ = [
    "GaussianNormalizer",
    "UnitGaussianNormalizer",
]


##################################################
##################################################
class UnitGaussianNormalizer():
    """Pointwise normalization of the input tensor `x`.
    
    Note
    1. The input tensor `x` is assumed to be aligned as `(batch, ..., channels)`.
    """
    def __init__(
            self,
            x:      torch.Tensor,
            eps:    float           = 1e-12,
        ) -> Self:
        self.__mean:    torch.Tensor    = torch.mean(x, 0, keepdim=True)
        self.__std:     torch.Tensor    = torch.std(x, 0, keepdim=True)
        self.__devie:   torch.device    = x.device
        self.__eps:     float           = eps
        return
    
    
    @property
    def mean(self) -> torch.Tensor:
        return self.__mean
    @property
    def std(self) -> torch.Tensor:
        return self.__std
    @property
    def device(self) -> torch.device:
        return self.__device
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor `x`."""
        # s = x.size()
        # x = x.view(s[0], -1)
        x = (x - self.__mean) / (self.__std + self.__eps)
        # x = x.view(s)
        return x
    def decode(self, x: torch.Tensor, sample_idx: slice=None) -> torch.Tensor:
        if sample_idx is None:
            std  = self.__std + self.__eps # n
            mean = self.__mean
        else:
            std = self.__std[sample_idx]+ self.__eps # batch * n
            mean = self.__mean[sample_idx]
        # s = x.size()
        # x = x.view(s[0], -1)
        x = (x * std) + mean
        # x = x.view(s)
        return x


    def to(self, device: torch.device) -> None:
        self.__mean = self.__mean.to(device)
        self.__std  = self.__std.to(device)
        return
    def cuda(self) -> None:
        self.__mean = self.__mean.cuda()
        self.__std  = self.__std.cuda()
        return
    def cpu(self) -> None:
        self.__mean = self.__mean.cpu()
        self.__std  = self.__std.cpu()
        return
    
    
    def __str__(self) -> str:
        return f"UnitGaussianNormalizer(mean.shape={tuple(self.__mean.shape)}, std.shape={tuple(self.__std.shape)})"


class GaussianNormalizer():
    """Instance-wise normalization of the input tensor `x`.
    
    Note
    1. The input tensor `x` is assumed to be aligned as `(batch, ..., channels)`.
    """
    def __init__(
            self,
            x:      torch.Tensor,
            eps:    float = 1e-12,
        ) -> Self:
        self.__ndim:        int     = x.ndim
        # self.__norm_config: dict    = {'dim': tuple(range(self.__ndim-1)), 'keepdim': True}
        self.__norm_config: dict    = {'dim': tuple(range(self.__ndim-1)), 'keepdim': False}
        self.__mean:    torch.Tensor    = torch.mean(x, **self.__norm_config)
        self.__std:     torch.Tensor    = torch.std( x, **self.__norm_config)
        self.__device:  torch.device    = x.device
        self.__eps:     float           = eps
        return
    
    
    @property
    def mean(self) -> torch.Tensor:
        return self.__mean
    @property
    def std(self) -> torch.Tensor:
        return self.__std
    @property
    def device(self) -> torch.device:
        return self.__device
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.__mean) / (self.__std + self.__eps)
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (self.__std + self.__eps)) + self.__mean


    def to(self, device: torch.device) -> None:
        self.__mean = self.__mean.to(device)
        self.__std  = self.__std.to(device)
        return
    def cuda(self) -> None:
        self.__mean = self.__mean.cuda()
        self.__std  = self.__std.cuda()
        return
    def cpu(self) -> None:
        self.__mean = self.__mean.cpu()
        self.__std  = self.__std.cpu()
        return
    
    
    def __str__(self) -> str:
        return f"GaussianNormalizer(mean={self.__mean.item():.4e}, std={self.__std.item():.4e})"


##################################################
##################################################
