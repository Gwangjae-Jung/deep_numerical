from    typing      import  Sequence, List, Tuple, Dict, Optional
from    math        import  prod
import  torch
from    torch       import  nn
from    deep_numerical          import  Objects, ones
from    deep_numerical.neural   import  get_activation, Activations


__all__ = ["MLP", "HyperMLP", "PatchEmbedding", "Periodization1D"]


##################################################
##################################################
class MLP(nn.Module):
    """## Multi-layer perceptron
    
    -----
    ### Description
    By passing the dimension of the input/output spaces and the hidden spaces, this class constructs a multi-layer perceptron.
    """
    def __init__(
            self,
            channels:           Sequence[int],
            bias:               bool    = True,
            activation_name:    Activations = "tanh",
            activation_kwargs:  Dict[str, object] = {},
            dtype:              Optional[torch.dtype] = None,
        ) -> None:
        """The initializer of the class `MLP`
        
        Arguments:
            `channels` (`Sequence[int]`): The number of the channels in each layer, from the input layer to the output layer.
            `bias` (`bool`, default: `True`): The `bias` argument of `torch.nn.Linear`.
            `activation_name` (`str`, default: `"tanh"`): The name of the activation function to be used.
            `activation_kwargs` (`Dict[str, object]`, default: `{}`): Further configurations for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the model.
        """
        super().__init__()
        self.__check_channels(channels)
        if dtype is None:   dtype = torch.get_default_dtype()
        
        # Save some member variables for representation
        self.__channels         = tuple(channels)
        self.__bias             = bias
        self.__activation_name  = activation_name
        self.__dtype            = dtype
        # Define the MLP
        self.net = nn.Sequential()
        for cnt in range(len(channels)-2):
            self.net.append(nn.Linear(channels[cnt], channels[cnt+1], bias=bias, dtype=dtype))
            self.net.append(get_activation(activation_name, activation_kwargs))
        self.net.append(nn.Linear(channels[-2], channels[-1], bias=bias, dtype=dtype))
        
        return None

    
    @property
    def channels(self) -> Tuple[int, ...]:
        """The widths in this network, from the input layer to the output layer."""
        return self.__channels
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net.forward(X)
    
    
    def __check_channels(self, channels: Sequence[int]) -> None:
        for cnt, dim in enumerate(channels):
            if not (type(dim) == int and dim >= 1):
                raise RuntimeError(f"The dimension of the layer {cnt} is set {dim}.")
        return None
    
    
    def __repr__(self) -> str:
        return f"MLP(layer={self.__channels}, bias={self.__bias}, activation={self.__activation_name})"




class HyperMLP(nn.Module):
    def __init__(
            self,
            channels:           Sequence[int],
            hyper_channels:     Sequence[int],
            bias:               bool                = True,
            activation_name:    str                 = "tanh",
            activation_kwargs:  Dict[str, object]   = {},
            dtype:              torch.dtype         = torch.float,
        ) -> None:
        """## The initializer of the class `HyperMLP`
        
        Arguments:
            `channels` (`Sequence[int]`): The number of the channels in each layer, from the input layer to the output layer.
                
            `hyper_channels` (`Sequence[int]`): The number of the channels in each layer, from the input layer to the pre-out layer.
            
                *Important* Users do not have to pass the output channels; the last linear layer will be automatically appended using `torch.nn.LazyLinear`.
            
            `bias` (`bool`, default: `True`): The `bias` argument of `torch.nn.Linear`.
            
            `activation_name` (`str`, default: "tanh"): The activation function which shall be used in each hidden layer.
            
            `activation_kwargs` (`Dict[str, object]`, defaule: `{}`): The keyword arguments for the activation function.
            
            `dtype` (`torch.dtype`, default: `torch.float`): The data type of the model.
        """
        super().__init__()
        self.__check_channels(channels)
        
        # Save some member variables for representation
        self.__channels             = tuple(channels)
        self.__bias                 = bias
        self.__activation_name      = activation_name
        self.__activation_kwargs    = activation_kwargs
        
        # Define variables for hypernetworks
        __shape_of_weights: List[Tuple[int, ...]]   = []
        __shape_of_biases:  List[Tuple[int, ...]]   = []
        for idx in range(len(channels)-1):
            ch_in, ch_out = channels[idx], channels[idx + 1]
            __shape_of_weights.append(tuple((ch_out, ch_in)))
            __shape_of_biases.append(tuple((ch_out,)))
        self.__shape_of_weights:    Tuple[Tuple[int, ...], ...] = tuple(__shape_of_weights)
        self.__shape_of_biases:     Tuple[Tuple[int, ...], ...] = tuple(__shape_of_biases)
        
        # Define the hypernetworks
        hypernet_weight = [
            MLP(
                (*hyper_channels, h*w),
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
                dtype               = dtype,
            )
            for (h, w) in self.__shape_of_weights
        ]
        hypernet_bias   = [] if not bias else [
            MLP(
                (*hyper_channels, h),
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
                dtype               = dtype,
            )
            for (h,) in self.__shape_of_biases
        ]
        self.hypernet_weight    = nn.ModuleList(hypernet_weight)
        self.hypernet_bias      = nn.ModuleList(hypernet_bias)
        
        return None

    
    @property
    def channels(self) -> Tuple[int, ...]:
        """The widths in this network, from the input layer to the output layer."""
        return self.__channels
    
    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """The forward pass of the class `HyperMLP`.
        
        Arguments:
            `X` (`torch.Tensor`):
                * The input tensor.
            `p` (`torch.Tensor`):
                * The hyperparameters.
        """
        weights = [net_w.forward(p) for net_w in self.hypernet_weight]
        biases  = [net_b.forward(p) for net_b in self.hypernet_bias] if self.__bias \
            else  [
                torch.zeros(p.size(0), 1, s, dtype=X.dtype)
                for s in self.__shape_of_biases
            ]
        for cnt, (w, b) in enumerate(zip(weights, biases)):
            w = torch.reshape(w, (-1, *self.__shape_of_weights[cnt]))
            b = torch.reshape(b, (-1, *ones(X.ndim-2), *self.__shape_of_biases[cnt]))
            X = torch.einsum('b...i, bji -> b...j', X, w) + b
            if cnt < len(weights)-1:
                X = get_activation(self.__activation_name, self.__activation_kwargs)(X)
        return X
        
    
    
    def __check_channels(self, channels: Sequence[int]) -> None:
        for cnt, dim in enumerate(channels):
            if not (type(dim) == int and dim >= 1):
                raise RuntimeError(f"The dimension of the layer {cnt} is set {dim}.")
        return None
    
    
    def __repr__(self) -> str:
        return f"HyperMLP(layer={self.__channels}, bias={self.__bias}, activation={self.__activation_name})"


class PatchEmbedding(nn.Module):
    """## Patch embedding
    
    -----
    ### Description
    This class aims to convert the input tensor into a sequence of patches.
    """
    def __init__(
            self,
            dim_domain:         int,
            in_channels:        int,
            patch_size:         Objects[int],
            dim_embed:          Optional[int] = None,
        ) -> None:
        """The initializer of the class `PatchEmbedding`
        
        Arguments:
            `dim_domain` (`int`): The dimension of the domain.
            `in_channels` (`int`): The number of the channels in the input tensor.
            `patch_size` (`Objects[int]`): The size of the patch in each dimension.
            `dim_embed` (`Optional[int]`, default: `None`): The number of the channels in the output tensor. If `None`, it is set to the `in_channels * prod(patch_size)`.
        """
        super().__init__()
        
        if dim_domain not in [1, 2, 3]:
            raise ValueError(f"The dimension of the domain should be 1, 2, or 3. ('dim_domain': {dim_domain})")
        if hasattr(patch_size, '__iter__') and len(patch_size) != dim_domain:
            raise ValueError(f"The length of the patch size should be equal to the dimension of the domain. ('dim_domain': {dim_domain}, 'patch_size': {patch_size})")
        
        if not hasattr(patch_size, '__iter__'):
            patch_size = [patch_size for _ in range(dim_domain)]
        
        self.__dim_domain:  int             = dim_domain
        self.__in_channels: int             = in_channels
        self.__patch_size:  Tuple[int, ...] = tuple(patch_size)
        self.__dim_embed:   int             = in_channels * prod(patch_size) if dim_embed is None else dim_embed
        
        self.patch_embed:   nn.Module = getattr(nn, f"Conv{dim_domain}d")(
            in_channels  = self.__in_channels,
            out_channels = self.__dim_embed,
            kernel_size  = self.__patch_size,
            stride       = self.__patch_size,
        )
        
        return None
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def in_channels(self) -> int:
        return self.__in_channels
    @property
    def patch_size(self) -> Tuple[int, ...]:
        return self.__patch_size
    @property
    def dim_embed(self) -> int:
        return self.__dim_embed
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.permute(self.__permute_1)
        X = self.patch_embed.forward(X)
        X = X.permute(self.__permute_2)
        return X
    
    
    @property
    def __permute_1(self) -> Tuple[int, ...]:
        return tuple((0, -1, *tuple(range(1, 1+self.dim_domain))))
    @property
    def __permute_2(self) -> Tuple[int, ...]:
        return tuple((0, *tuple(range(2, 2+self.dim_domain)), 1))


class Periodization1D(nn.Module):
    def __init__(self, max_freq: int, sup: float, inf: Optional[float]=None) -> None:
        super().__init__()
        if (not isinstance(max_freq, int)) or max_freq<=0:
            raise ValueError(f"'max_freq' should be a positive intege, but got {max_freq=}.")
        if inf is None:
            if sup<=0:  raise ValueError(f"If 'inf' is not given, 'sup' should be given as a positive real number, but got {(inf, sup)=}.")
            else:       inf = -sup
        else:
            if sup<=inf:    raise ValueError(f"The inequality [inf<sup] is not satisfied, as {(inf, sup)=}.")
        self.__period:      float   = sup-inf
        self.__max_freq:    int     = max_freq
        self.__inf:         float   = inf
        return None
    
    @property
    def in_channels(self) -> int:   return 1
    @property
    def out_channels(self) -> int:  return 2*self.__max_freq
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`): A 2-tensor of shape `(batch_size, 1)`.
        """
        X = (2*torch.pi)*(X-self.__inf)/self.__period
        Y = []
        kX = torch.zeros(X.shape, dtype=X.dtype, device=X.device)
        for _ in range(1, 1+self.__max_freq):
            kX = kX+X
            Y.append(kX.cos())
            Y.append(kX.sin())
        return torch.cat(Y, dim=1)
    
    
##################################################
##################################################
# End of file