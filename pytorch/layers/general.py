from    typing                  import  Sequence, Optional
from    typing_extensions       import  Self

from    math        import  prod

import  torch
from    torch       import  nn

from    ..utils     import  Objects, get_activation, ones


##################################################
##################################################
__all__ = [
    "MLP", "HyperMLP",
    "PatchEmbedding",
]


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
            activation_name:    str     = "tanh",
            activation_kwargs:  dict[str, object] = {},
        ) -> Self:
        """## The initializer of the class `MLP`
        
        -----
        ### Arguments
        * `channels` (`Sequence[int]`)
            * The number of the channels in each layer, from the input layer to the output layer.
        * `bias` (`bool`, default: `True`)
            * The `bias` argument of `torch.nn.Linear`.
        * `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        self.__check_channels(channels)
        
        # Save some member variables for representation
        self.__channels         = tuple(channels)
        self.__bias             = bias
        self.__activation_name  = activation_name
        
        # Define the MLP
        self.net = nn.Sequential()
        for cnt in range(len(channels)-2):
            self.net.append(nn.Linear(channels[cnt], channels[cnt + 1], bias = bias))
            self.net.append(get_activation(activation_name, activation_kwargs))
        self.net.append(nn.Linear(channels[-2], channels[-1], bias = bias))
        
        return

    
    @property
    def channels(self) -> tuple[int]:
        """The widths in this network, from the input layer to the output layer."""
        return self.__channels
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net.forward(X)
    
    
    def __check_channels(self, channels: Sequence[int]) -> None:
        for cnt, dim in enumerate(channels):
            if not (type(dim) == int and dim >= 1):
                raise RuntimeError(f"The dimension of the layer {cnt} is set {dim}.")
        return
    
    
    def __repr__(self) -> str:
        return f"MLP(layer={self.__channels}, bias={self.__bias}, activation={self.__activation_name})"


class HyperLinear(nn.Module):
    def __init__(
            self,
            in_channels:    int,
            out_channels:   int,
            bias:           bool    = True,
            hyper_channels_weight:  Optional[Sequence[int]] = None,
            hyper_channels_bias:    Optional[Sequence[int]] = None,
            activation_name:        str = "tanh",
            activation_kwargs:      dict[str, object] = {},
        ) -> Self:
        """## The initializer of the class `HyperMLP`
        
        -----
        ### Arguments
        * `in_channels` (`int`) and `out_channels` (`int`)
            * The number of the channels in the input layer and the output layer.
            
        * `hyper_channels_weight` (`Sequence[int]`) and ``hyper_channels_bias` (`Optional[Sequence[int]]`, default: `None`)
            * The number of the channels in each layer, from the input layer to the pre-out layer.
            * *Important* For each hypernet, the last linear layer will be appended in the initializer.
        
        * `bias` (`bool`, default: `True`)
            * The `bias` argument of `torch.nn.Linear`.
        
        * `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        
        # Save some member variables for representation
        self.__in_channels      = in_channels
        self.__out_channels     = out_channels
        self.__bias             = bias
        
        # Define variables for hypernetworks
        size_of_weight  = out_channels*in_channels
        size_of_bias    = out_channels
        self.__shape_of_weight = (-1, out_channels, in_channels)
        self.__shape_of_bias   = (-1, out_channels,)
        
        if hyper_channels_weight is None:
            raise ValueError(f"'hyper_channels_weight' should be set.")
        if hyper_channels_bias is None:
            hyper_channels_bias = [hyper_channels_weight[0]]
        # Define the hypernetworks
        self.hyper_weight = MLP(
            (*hyper_channels_weight, size_of_weight),
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        )
        self.hyper_bias:    Optional[MLP]   = MLP(
            (*hyper_channels_bias, size_of_bias),
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs,
        ) if bias else None
        
        return
    
    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        weight = self.hyper_weight.forward(p)
        bias  = self.hyper_bias.forward(p) if self.__bias else torch.zeros((p.size(0), self.__out_channels), dtype=X.dtype, device=X.device)
        weight = weight.reshape(self.__shape_of_weight)
        bias   = bias.reshape(-1, *ones(X.ndim-2), self.__shape_of_bias[1])
        X = torch.einsum('b...i, bji -> b...j', X, weight) + bias
        return X
    
    
    def __repr__(self) -> str:
        return f"HyperLinear(in_channels={self.__in_channels}, out_channels={self.__out_channels}, bias={self.__bias})"




class HyperMLP(nn.Module):
    def __init__(
            self,
            channels:           Sequence[int],
            hyper_channels:     Sequence[int],
            bias:               bool                = True,
            activation_name:    str                 = "tanh",
            activation_kwargs:  dict[str, object]   = {},
        ) -> Self:
        """## The initializer of the class `HyperMLP`
        
        -----
        ### Arguments
        * `channels` (`Sequence[int]`)
            * The number of the channels in each layer, from the input layer to the output layer.
            
        * `hyper_channels` (`Sequence[int]`)
            * The number of the channels in each layer, from the input layer to the pre-out layer.
            * *Important* For each hypernet, the last linear layer will be automatically appended using `torch.nn.LazyLinear`.
        
        * `bias` (`bool`, default: `True`)
            * The `bias` argument of `torch.nn.Linear`.
        
        * `activation_name` (`str`, default: "tanh") and `activation_kwargs` (`dict[str, object]`, defaule: `{}`)
            * The activation function which shall be used in each hidden layer.
        """
        super().__init__()
        self.__check_channels(channels)
        
        # Save some member variables for representation
        self.__channels         = tuple(channels)
        self.__bias             = bias
        self.__activation_name      = activation_name
        self.__activation_kwargs    = activation_kwargs
        
        # Define variables for hypernetworks
        __shape_of_weights:    list[tuple[int]]    = []
        __shape_of_biases:     list[tuple[int]]    = []
        for idx in range(len(channels)-1):
            ch_in, ch_out = channels[idx], channels[idx + 1]
            __shape_of_weights.append((ch_out, ch_in))
            __shape_of_biases.append((ch_out,))
        self.__shape_of_weights:    tuple[tuple[int]]    = tuple(__shape_of_weights)
        self.__shape_of_biases:     tuple[tuple[int]]    = tuple(__shape_of_biases)
        
        # Define the hypernetworks
        hypernet_weight = [
            MLP(
                (*hyper_channels, h*w),
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
            )
            for (h, w) in self.__shape_of_weights
        ]
        hypernet_bias   = [] if not bias else [
            MLP(
                (*hyper_channels, h),
                activation_name     = activation_name,
                activation_kwargs   = activation_kwargs,
            )
            for (h,) in self.__shape_of_biases
        ]
        self.hypernet_weight    = nn.ModuleList(hypernet_weight)
        self.hypernet_bias      = nn.ModuleList(hypernet_bias)
        
        return

    
    @property
    def channels(self) -> tuple[int]:
        """The widths in this network, from the input layer to the output layer."""
        return self.__channels
    
    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
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
        return
    
    
    def __repr__(self) -> str:
        return f"HyperMLP(layer={self.__channels}, bias={self.__bias}, activation={self.__activation_name})"


##################################################
##################################################
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
        ) -> Self:
        """## The initializer of the class `PatchEmbedding`
        
        -----
        ### Arguments
        * `dim_domain` (`int`)
            - The dimension of the domain.
        
        * `in_channels` (`int`)
            - The number of the channels in the input tensor.
        
        * `patch_size` (`Objects[int]`)
            - The size of the patch in each dimension.
            
        * `dim_embed` (`Optional[int]`, default: `None`)
            - The number of the channels in the output tensor.
            - If `None`, it is set to the `in_channels * prod(patch_size)`.
        """
        super().__init__()
        
        if dim_domain not in [1, 2, 3]:
            raise ValueError(f"The dimension of the domain should be 1, 2, or 3. ('dim_domain': {dim_domain})")
        if hasattr(patch_size, '__iter__') and len(patch_size) != dim_domain:
            raise ValueError(f"The length of the patch size should be equal to the dimension of the domain. ('dim_domain': {dim_domain}, 'patch_size': {patch_size})")
        
        if not hasattr(patch_size, '__iter__'):
            patch_size = [patch_size for _ in range(dim_domain)]
        
        self.__dim_domain:  int         = dim_domain
        self.__in_channels: int         = in_channels
        self.__patch_size:  tuple[int]  = tuple(patch_size)
        self.__dim_embed:   int         = in_channels * prod(patch_size) if dim_embed is None else dim_embed
        
        self.patch_embed:   nn.Module = getattr(nn, f"Conv{dim_domain}d")(
            in_channels  = self.__in_channels,
            out_channels = self.__dim_embed,
            kernel_size  = self.__patch_size,
            stride       = self.__patch_size,
        )
        
        return
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def in_channels(self) -> int:
        return self.__in_channels
    @property
    def patch_size(self) -> tuple[int]:
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
    def __permute_1(self) -> tuple[int]:
        return tuple((0, -1, *tuple(range(1, 1+self.dim_domain))))
    @property
    def __permute_2(self) -> tuple[int]:
        return tuple((0, *tuple(range(2, 2+self.dim_domain)), 1))


##################################################
##################################################
# End of file