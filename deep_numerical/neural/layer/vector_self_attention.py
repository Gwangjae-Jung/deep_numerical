import  warnings
from    typing      import  Optional

import  torch
from    torch       import  nn

try:
    from    torch_geometric.nn      import  MessagePassing
    _HAS_TORCH_GEOMETRIC = True
except (ImportError, ModuleNotFoundError):
    _HAS_TORCH_GEOMETRIC = False

if not _HAS_TORCH_GEOMETRIC:
    warnings.warn(
        "Module 'torch_geometric' is not installed. 'VectorSelfAttention' will not be available.",
        UserWarning,
        stacklevel=2,
    )
    __all__: list[str] = []
else:
    __all__ = ["VectorSelfAttention"]

    ##################################################
    ##################################################
    class VectorSelfAttention(MessagePassing):
        """## Vector self-attention
        
        -----
        ### Description
        This class aims to compute the vector-valued self-attention of given features from two distinct sources.
        The modifications from the original attention layer are listed below:
        
        ### Remark
        1. The softmax can be removed by passing `use_softmax=False` to the initializer.
        2. The layer normalization will not be used.
        """
        def __init__(
                self,
                
                channels:       int,
                n_heads:        int,
                pos_encoder:    nn.Module,
                
                use_softmax:    bool = True,
                use_linear:     bool = True,
                bias_linear:    bool = True,
                
                dtype:          Optional[torch.dtype] = None,
            ) -> None:
            """The initializer of `VectorSelfAttention`.
            
            Arguments:
                `channels` (`int`): The number of the hidden channels. `channels` is used to compute the einsum command.
                `n_heads` (`int`): The number of the heads in the attention layer.
                `pos_encoder` (`nn.Module`): The positional encoder. `pos_encoder` is used to compute the positional encoding.
                `use_softmax` (`bool`, default: `True`): If `True`, the softmax will be used in the attention layer. If `False`, the softmax will not be used in the attention layer.
                `use_linear` (`bool`, default: `True`): If `True`, the linear layer will be used as a skip connection. If `False`, the linear layer will not be used as a skip connection.
                `bias_linear` (`bool`, default: `True`): If `True`, the linear layer will use bias. If `False`, the linear layer will not use bias.
                `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the parameters. If `None`, the default data type of `torch` will be used.
            """
            # Check if the number of the hidden channels is divisible by the number of the heads
            # Initialization begins
            super().__init__(aggr='sum')
            
            # Save some variables for representation and computation
            self.__channels         = channels
            self.__n_heads          = n_heads
            self.__pos_encoder      = pos_encoder
            self.__use_softmax      = use_softmax
            self.__use_linear       = use_linear
            self.__EINSUM_COMMAND   = f"nc,chd->nhd"
            
            # Variables for attention
            _size   = (channels, n_heads, channels//n_heads)    # (C, H, C/H)
            _scale  = channels**2
            dtype   = dtype if dtype is not None else torch.get_default_dtype()
            # Feature maps
            self.sa_query   = nn.Parameter(torch.randn(size=_size, dtype=dtype) / _scale)
            self.sa_key     = nn.Parameter(torch.randn(size=_size, dtype=dtype) / _scale)
            self.sa_value   = nn.Parameter(torch.randn(size=_size, dtype=dtype) / _scale)
            # Internal linear layer as a skip connection
            self.linear     = nn.Linear(channels, channels, bias=bias_linear, dtype=dtype) if self.__use_linear else nn.Identity()
            
            return None
        
        
        @property
        def einsum_command(self) -> str:
            """## Einsum command string for multi-head projection."""
            return self.__EINSUM_COMMAND

        @property
        def channels(self) -> int:
            """## Total number of hidden channels."""
            return self.__channels

        @property
        def n_heads(self) -> int:
            """## Number of attention heads."""
            return self.__n_heads

        @property
        def use_softmax(self) -> bool:
            """## Whether softmax normalization is applied to attention weights."""
            return self.__use_softmax

        @property
        def use_linear(self) -> bool:
            """## Whether a linear skip connection is used."""
            return self.__use_linear
        
        
        def pos_encode(self, position: torch.Tensor) -> torch.Tensor:
            """
            Given a position tensor of shape `(N, k)`, this method computes the positional encoding of shape (N, H, C/H).
            Here,
            * `N` is the number of the nodes (points),
            * `k` is the dimension of the space in which the points exist,
            * `H` is the number of the heads,
            * `C` is the hidden dimension.
            """
            pos_enc: torch.Tensor
            pos_enc = self.__pos_encoder.forward(position)
            pos_enc = pos_enc.reshape(-1, self.n_heads, self.channels//self.n_heads)
            return pos_enc
        
        
        def forward(
                self,
                node_attr:  torch.Tensor,
                edge_index: torch.LongTensor,
                position:   torch.Tensor,
            ) -> torch.Tensor:
            """The forward method of `VectorSelfAttention`
            
            Arguments:
                `node_attr` (`torch.Tensor`):
                    * A `torch.Tensor` object of shape `(N, D)`, where `N` is the number of the points and `D` is the number of the input channels. Note that there is not dimension for the batch, which is in accordance with the operation of `torch_geometric`.
                `edge_index` (`torch.LongTensor`):
                    * A `torch.LongTensor` object of shape `(E, 2)`, where `E` is the number of the edges.
                `position` (`torch.Tensor`):
                    * A `torch.Tensor` object of shape `(N, k)`, where `k` is the dimension of the physical domain.
            
            Returns:
                `torch.Tensor`:
                    * A `torch.Tensor` object of shape `(N, D)`, where `N` is the number of the points and `D` is the number of the output channels.
            """
            # Map to the query/key/value spaces (resultant shape: `(N, hidden_channels)`)
            query   = torch.einsum(self.einsum_command, [node_attr, self.sa_query]).reshape(-1, self.channels)
            key     = torch.einsum(self.einsum_command, [node_attr, self.sa_key]).reshape(-1, self.channels)
            value   = torch.einsum(self.einsum_command, [node_attr, self.sa_value]).reshape(-1, self.channels)
            
            # Do message passing and compute the skip connection, if required
            prop    = self.propagate(
                edge_index,
                query = query, key = key, value = value,
                position = position,
            )
            if self.use_linear:
                skip = self.linear.forward(node_attr)
            else:
                skip = 0
                
            # Return the output
            return prop + skip
            

        def message(
                self,
                query_i:    torch.Tensor,
                key_j:      torch.Tensor,
                value_j:    torch.Tensor,
                position_i: torch.Tensor,
                position_j: torch.Tensor,
            ) -> torch.Tensor:
            """
            ### Note
            
            Here the shapes of the input tensors are summarized.
            * Query, key, value: `(N, C)`. Note that the features belonging to distinct heads are combined; otherwise, the message passing algorithm does not work.
            * Position: `(N, k)`
            
            With the above input tensors, the message passing algorithm returns a tensor of shape `(N, C)`.
            Here, `N` is the number of the nodes (points), `H` is the number of the heads, `C` is the number of the channels, and `k` is the dimension of the space in which the points exist.
            """
            qkv_shape = (-1, self.n_heads, self.channels//self.n_heads)
            query_i = query_i.reshape(qkv_shape)
            key_j   = key_j.reshape(qkv_shape)
            value_j = value_j.reshape(qkv_shape)
            
            vector_weights = query_i - key_j + self.pos_encode(position_i - position_j)
            if self.__use_softmax:
                vector_weights = torch.softmax(vector_weights, dim=-1)
            vectors = value_j + self.pos_encode(position_j)
            
            return (vector_weights * vectors).reshape(-1, self.channels)
        
        
        def __repr__(self) -> str:
            channels    = self.channels
            n_heads     = self.n_heads
            use_softmax = self.use_softmax
            use_linear  = self.use_linear
            return f"VectorSelfAttention({channels=}, {n_heads=}, {use_softmax=}, {use_linear=})"


##################################################
##################################################
# End of file
