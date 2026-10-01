from    typing      import  Optional
import  torch


__all__:    list[str] = ["GaussianNormalizer", "UnitGaussianNormalizer",]


##################################################
##################################################
class UnitGaussianNormalizer():
    """## Pointwise Gaussian normalizer.

    ## Description
    Computes pointwise mean and standard deviation across batch elements (dimension 0)
    to normalize and denormalize input tensors.

    ## Arguments
    `x` (`torch.Tensor`): Input tensor of shape `(batch, ..., channels)` to compute statistics.
    `eps` (`float`, default: `1e-12`): Small epsilon value to avoid division by zero.
    """
    def __init__(
            self,
            x:      torch.Tensor,
            eps:    float           = 1e-12,
        ) -> None:
        """## Initializes the UnitGaussianNormalizer.

        ## Description
        Computes pointwise mean and standard deviation across dimension 0 of the input tensor.

        ## Arguments
        `x` (`torch.Tensor`): Reference tensor to compute mean and standard deviation.
        `eps` (`float`, default: `1e-12`): Numerical stability epsilon.

        ## Returns
        `None`.
        """
        self.__mean:    torch.Tensor    = torch.mean(x, 0, keepdim=True)
        self.__std:     torch.Tensor    = torch.std(x, 0, keepdim=True)
        self.__device:  torch.device    = x.device
        self.__eps:     float           = eps
        return None
    
    
    @property
    def mean(self) -> torch.Tensor:
        """## Returns the computed mean tensor."""
        return self.__mean

    @property
    def std(self) -> torch.Tensor:
        """## Returns the computed standard deviation tensor."""
        return self.__std

    @property
    def device(self) -> torch.device:
        """## Returns the device of the normalizer tensors."""
        return self.__device
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """## Normalizes the input tensor.

        ## Description
        Transforms `x` using pointwise z-score standardization: `(x - mean) / (std + eps)`.

        ## Arguments
        `x` (`torch.Tensor`): Input tensor to normalize.

        ## Returns
        `torch.Tensor`: Normalized tensor.
        """
        x = (x - self.__mean) / (self.__std + self.__eps)
        return x

    def decode(self, x: torch.Tensor, sample_idx: Optional[slice] = None) -> torch.Tensor:
        """## Inverts the normalization on the input tensor.

        ## Description
        Transforms normalized tensor `x` back to the original scale: `(x * std) + mean`.

        ## Arguments
        `x` (`torch.Tensor`): Normalized tensor to decode.
        `sample_idx` (`Optional[slice]`, default: `None`): Optional slice index for subset decoding.

        ## Returns
        `torch.Tensor`: Denormalized tensor.
        """
        if sample_idx is None:
            std  = self.__std + self.__eps # n
            mean = self.__mean
        else:
            std = self.__std[sample_idx]+ self.__eps # batch * n
            mean = self.__mean[sample_idx]
        x = (x * std) + mean
        return x


    def to(self, device: torch.device) -> None:
        """## Moves mean and standard deviation tensors to the specified device.

        ## Arguments
        `device` (`torch.device`): Target device.

        ## Returns
        `None`.
        """
        self.__mean = self.__mean.to(device)
        self.__std  = self.__std.to(device)
        return

    def cuda(self) -> None:
        """## Moves normalizer parameters to CUDA.

        ## Returns
        `None`.
        """
        self.__mean = self.__mean.cuda()
        self.__std  = self.__std.cuda()
        return

    def cpu(self) -> None:
        """## Moves normalizer parameters to CPU.

        ## Returns
        `None`.
        """
        self.__mean = self.__mean.cpu()
        self.__std  = self.__std.cpu()
        return
    
    
    def __str__(self) -> str:
        return f"UnitGaussianNormalizer(mean.shape={tuple(self.__mean.shape)}, std.shape={tuple(self.__std.shape)})"


class GaussianNormalizer():
    """## Instance-wise Gaussian normalizer.

    ## Description
    Computes global mean and standard deviation across all spatial and batch dimensions
    (leaving only channel dimensions if applicable) to normalize and denormalize input tensors.

    ## Arguments
    `x` (`torch.Tensor`): Input tensor of shape `(batch, ..., channels)`.
    `eps` (`float`, default: `1e-12`): Small epsilon value to avoid division by zero.
    """
    def __init__(
            self,
            x:      torch.Tensor,
            eps:    float = 1e-12,
        ) -> None:
        """## Initializes the GaussianNormalizer.

        ## Description
        Computes mean and standard deviation across all axes except the last dimension of `x`.

        ## Arguments
        `x` (`torch.Tensor`): Reference tensor to compute statistics.
        `eps` (`float`, default: `1e-12`): Numerical stability epsilon.

        ## Returns
        `None`.
        """
        self.__ndim:        int     = x.ndim
        self.__norm_config: dict    = {'dim': tuple(range(self.__ndim-1)), 'keepdim': False}
        self.__mean:    torch.Tensor    = torch.mean(x, **self.__norm_config)
        self.__std:     torch.Tensor    = torch.std( x, **self.__norm_config)
        self.__device:  torch.device    = x.device
        self.__eps:     float           = eps
        return
    
    
    @property
    def mean(self) -> torch.Tensor:
        """## Returns the computed mean tensor."""
        return self.__mean

    @property
    def std(self) -> torch.Tensor:
        """## Returns the computed standard deviation tensor."""
        return self.__std

    @property
    def device(self) -> torch.device:
        """## Returns the device of the normalizer tensors."""
        return self.__device
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """## Normalizes the input tensor.

        ## Description
        Transforms `x` using z-score standardization: `(x - mean) / (std + eps)`.

        ## Arguments
        `x` (`torch.Tensor`): Input tensor to normalize.

        ## Returns
        `torch.Tensor`: Normalized tensor.
        """
        return (x - self.__mean) / (self.__std + self.__eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """## Inverts the normalization on the input tensor.

        ## Description
        Transforms normalized tensor `x` back to original scale: `(x * (std + eps)) + mean`.

        ## Arguments
        `x` (`torch.Tensor`): Normalized tensor to decode.

        ## Returns
        `torch.Tensor`: Denormalized tensor.
        """
        return (x * (self.__std + self.__eps)) + self.__mean


    def to(self, device: torch.device) -> None:
        """## Moves mean and standard deviation tensors to the specified device.

        ## Arguments
        `device` (`torch.device`): Target device.

        ## Returns
        `None`.
        """
        self.__mean = self.__mean.to(device)
        self.__std  = self.__std.to(device)
        return

    def cuda(self) -> None:
        """## Moves normalizer parameters to CUDA.

        ## Returns
        `None`.
        """
        self.__mean = self.__mean.cuda()
        self.__std  = self.__std.cuda()
        return

    def cpu(self) -> None:
        """## Moves normalizer parameters to CPU.

        ## Returns
        `None`.
        """
        self.__mean = self.__mean.cpu()
        self.__std  = self.__std.cpu()
        return
    
    
    def __str__(self) -> str:
        return f"GaussianNormalizer(mean={self.__mean.item():.4e}, std={self.__std.item():.4e})"


##################################################
##################################################
# End of file
