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
    def __init__(
            self,
            x:      torch.Tensor,
            eps:    float           = 1e-12,
            device: torch.device    = torch.device('cpu')
        ) -> None:
        self.__mean = torch.mean(x, 0).to(device).view(-1)
        self.__std = torch.std(x, 0).to(device).view(-1)
        self.__eps = eps
        return
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.__mean) / (self.__std + self.__eps)
        x = x.view(s)
        return x
    def decode(self, x: torch.Tensor, sample_idx: slice = None) -> torch.Tensor:
        if sample_idx is None:
            std  = self.__std + self.__eps # n
            mean = self.__mean
        else:
            std = self.__std[sample_idx]+ self.__eps # batch * n
            mean = self.__mean[sample_idx]
        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
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


class GaussianNormalizer():
    """
    """
    def __init__(
                    self,
                    x:      torch.Tensor,
                    eps:    float = 1e-12,
                    device: torch.device = torch.device('cpu')
        ) -> torch.Tensor:
        self.__mean = torch.mean(x).to(device)
        self.__std = torch.std(x).to(device)
        self.__eps = eps
        return
    

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


##################################################
##################################################
