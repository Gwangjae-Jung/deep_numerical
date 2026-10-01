from    typing              import  Sequence
from    typing_extensions   import  Self
import  torch
from    torch   import  nn
from    deep_numerical.neural   import  BaseModule
from    deep_numerical.neural.layer import  MLP, HyperMLP


__all__: list[str] = ['HyperDeepONet', 'HyperMIONet', 'ParameterizedMIONet']


##################################################
##################################################
def n_parameters_in_mlp(dimensions: Sequence[int]) -> int:
    """## Count MLP parameters
    
    ## Description
    Counts the total number of weights and biases in a multi-layer perceptron (MLP) with the specified layer dimensions.
    
    ## Arguments
    `dimensions` (`Sequence[int]`): The sequence of layer dimensions of the MLP.
    
    ## Returns
    `int`: The total number of parameters in the MLP.
    """
    length = len(dimensions)
    n_params = 0
    for idx in range(length-1):
        ch_in, ch_out = dimensions[idx], dimensions[idx+1]
        n_params += ch_in * ch_out  # Linear layer
        n_params += ch_out          # Bias
    return n_params
        

##################################################
##################################################
class HyperDeepONet(torch.nn.Module):
    """## Hypernetwork DeepONet
    
    ## Description
    A Deep Operator Network (DeepONet) variant where the trunk network weights and biases are generated dynamically by a hypernetwork branch.
    """
    def __init__(
            self,
            hypernet_prior: Sequence[torch.nn.Module],
            dim_trunk:      Sequence[int] = [1+2, 32, 32, 32, 32, 1],
        ) -> None:
        """## The initializer of `HyperDeepONet`
        
        ## Description
        Initializes the `HyperDeepONet` architecture with a prior hypernetwork module sequence and trunk layer dimensions.
        
        ## Arguments
        `hypernet_prior` (`Sequence[torch.nn.Module]`): Modules applied prior to the hypernetwork output linear projection.
        `dim_trunk` (`Sequence[int]`, default: `[3, 32, 32, 32, 32, 1]`): The sequence of channel dimensions for the trunk network.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
        self.__dim_trunk: Sequence[int] = dim_trunk
        self.hypernet = torch.nn.Sequential(
            *hypernet_prior,
            torch.nn.SiLU(),
            torch.nn.LazyLinear(n_parameters_in_mlp(dim_trunk)),
        )
        self.__splitter_weight: list[int] = []
        self.__splitter_bias:   list[int] = []
        self.__shapes_weight:   list[tuple[int, ...]] = []
        self.__shapes_bias:     list[tuple[int, ...]] = []
        
        _prev, _curr = 0, 0
        for idx in range(len(dim_trunk)-1):
            _prev = _curr
            ch_in, ch_out = dim_trunk[idx], dim_trunk[idx+1]
            _curr += ch_in*ch_out
            self.__splitter_weight.append(slice(_prev, _curr))
            _prev = _curr
            _curr += ch_out
            self.__splitter_bias.append(slice(_prev, _curr))
            self.__shapes_weight.append(tuple((ch_out, ch_in)))
            self.__shapes_bias.append(tuple((ch_out,)))
        
        return
    
    
    def forward(
            self,
            X:      torch.Tensor,
            query:  torch.Tensor,
        ) -> torch.Tensor:
        """## Forward propagation of `HyperDeepONet`
        
        ## Description
        Evaluates the hypernetwork on input branch features `X` to generate trunk parameters, then applies the synthesized trunk network to query points.
        
        ## Arguments
        `X` (`torch.Tensor`): The input tensor for the branch hypernetwork of shape `(batch_size, n_sensor_points)`.
        `query` (`torch.Tensor`): The input tensor for the target trunk network of shape `(n_query_points, dim_query)`.
        
        ## Returns
        `torch.Tensor`: The predicted operator evaluation tensor of shape `(batch_size, n_query_points, out_channels)`.
        """
        params = self.hypernet.forward(X)   # Shape: `(batch_size, n_params)`
        out = query
        w: torch.Tensor
        b: torch.Tensor
        for idx in range(len(self.__dim_trunk)-1):
            w = params[:, self.__splitter_weight[idx]]
            w = w.view(-1, *self.__shapes_weight[idx])
            b = params[:, self.__splitter_bias[idx]]
            b = b.view(-1, 1, *self.__shapes_bias[idx])
            out = torch.einsum('qd, bed -> bqe', out, w) + b
            if idx < len(self.__dim_trunk)-2:
                out = nn.functional.silu(out)
        return out


##################################################
##################################################
class HyperMIONet(BaseModule):
    """## Hypernetwork Multiple-Input Operator Network (HyperMIONet)
    
    ## Description
    A multiple-input operator network architecture with convolutional and hyper-MLP branches conditioned on parameters.
    """
    def __init__(
            self,
            dimension:  int,
        ) -> None:
        """## The initializer of `HyperMIONet`
        
        ## Description
        Initializes the `HyperMIONet` model components for the specified spatial dimension.
        
        ## Arguments
        `dimension` (`int`): The spatial dimension (e.g., 1, 2, or 3) for convolutional layers.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
        # Define the subnetworks
        self.branch1a = nn.ModuleList(
            [
                getattr(nn, f"Conv{dimension}d")(1, 8, 5, 2, 1),   # 33->16
                nn.SiLU(),
                getattr(nn, f"Conv{dimension}d")(8, 16, 5, 2, 1),  # 16->7
                nn.SiLU(),
                getattr(nn, f"Conv{dimension}d")(16, 32, 5, 2, 1),  # 7->3
                nn.SiLU(),
                nn.Flatten(),
            ]
        )
        self.branch1b = HyperMLP([32*9, 100, 64], [1, 50])
        self.branch2a = nn.ModuleList(
            [
                getattr(nn, f"Conv{dimension}d")(1, 8, 5, 2, 1),   # 33->16
                nn.SiLU(),
                getattr(nn, f"Conv{dimension}d")(8, 16, 5, 2, 1),  # 16->7
                nn.SiLU(),
                getattr(nn, f"Conv{dimension}d")(16, 32, 5, 2, 1),  # 7->3
                nn.SiLU(),
                nn.Flatten(),
            ]
        )
        self.branch2b = HyperMLP([32*9, 100, 64], [1, 50])
        self.trunk    = HyperMLP((dimension, 32, 64, 64), [1, 50])       
        
        return

    
    def forward(self, X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """## Forward propagation of `HyperMIONet`
        
        ## Description
        Computes the multiple-input operator evaluation given function data `X` and parameter `p`.
        
        ## Arguments
        `X` (`torch.Tensor`): Input function evaluation tensor.
        `p` (`torch.Tensor`): Condition parameter tensor.
        
        ## Returns
        `torch.Tensor`: The evaluated output tensor.
        """
        branch1 = X
        branch2 = X
        for md in self.branch1a:
            branch1 = md.forward(branch1)
        branch1 = self.branch1b.forward(branch1, p)
        for md in self.branch2a:
            branch2 = md.forward(branch2)
        branch2 = self.branch2b.forward(branch2, p)
        branch = branch1 * branch2
        trunk = self.trunk.forward(p)
        return branch @ trunk.T


class ParameterizedMIONet(BaseModule):
    """## Parameterized Multiple-Input Operator Network (ParameterizedMIONet)
    
    ## Description
    A parameterized MIONet architecture combining convolutional feature extraction, parameter encodings, and a trunk network.
    """
    def __init__(
            self,
            dimension:  int,
        ) -> None:
        """## The initializer of `ParameterizedMIONet`
        
        ## Description
        Initializes the subnetworks and convolutional branches of `ParameterizedMIONet` for the given dimension.
        
        ## Arguments
        `dimension` (`int`): The spatial dimension (e.g., 1, 2, or 3) for the convolutional branches.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
        # Define the subnetworks
        _conv_kwargs    = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self.branch1_f  = nn.Sequential(
            getattr(nn, f"Conv{dimension}d")(1, 4, **_conv_kwargs),
            nn.SiLU(),  # 33->16
            getattr(nn, f"Conv{dimension}d")(4, 8, **_conv_kwargs),
            nn.SiLU(),  # 16->8
            getattr(nn, f"Conv{dimension}d")(8, 16, **_conv_kwargs),
            nn.SiLU(),  # 8->4
            getattr(nn, f"MaxPool{dimension}d")(4),
            nn.SiLU(),  # 4->1
            nn.Flatten(),   # Output shape: `(batch_size, 16)`
        )
        self.branch1_p  = MLP([1, 16, 16])
        self.branch1    = MLP([32, 50, 50, 50])
        self.branch2_f = nn.Sequential(
            getattr(nn, f"Conv{dimension}d")(1, 4, **_conv_kwargs),
            nn.SiLU(),  # 33->16
            getattr(nn, f"Conv{dimension}d")(4, 8, **_conv_kwargs),
            nn.SiLU(),  # 16->8
            getattr(nn, f"Conv{dimension}d")(8, 16, **_conv_kwargs),
            nn.SiLU(),  # 8->4
            getattr(nn, f"MaxPool{dimension}d")(4),
            nn.SiLU(),  # 4->1
            nn.Flatten(),   # Output shape: `(batch_size, 16)`
        )
        self.branch2_p  = MLP([1, 16, 16])
        self.branch2    = MLP([32, 50, 50, 50])
        # self.trunk      = HyperMLP((dimension, 32, 64, 64, 64), [1, 50])       
        self.trunk      = MLP((dimension, 100, 100, 50))
        
        return

    
    def forward(
            self,
            X:      torch.Tensor,
            query:  torch.Tensor,
            params: torch.Tensor,
        ) -> torch.Tensor:
        """## Forward propagation of `ParameterizedMIONet`
        
        ## Description
        Evaluates the parameterized MIONet on input functions `X`, query locations `query`, and parameters `params`.
        
        ## Arguments
        `X` (`torch.Tensor`): Input function field tensor of shape `(batch_size, 1, *spatial_shape)`.
        `query` (`torch.Tensor`): Query coordinates tensor of shape `(n_queries, dimension)`.
        `params` (`torch.Tensor`): Parameter tensor of shape `(batch_size, 1)`.
        
        ## Returns
        `torch.Tensor`: The predicted values at the query locations of shape `(batch_size, n_queries)`.
        """
        branch1_f   = self.branch1_f.forward(X)
        branch1_p   = self.branch1_p.forward(params)
        branch1     = self.branch1.forward(torch.cat((branch1_f, branch1_p), dim=1))
        branch2_f   = self.branch2_f.forward(X)
        branch2_p   = self.branch2_p.forward(params)
        branch2     = self.branch2.forward(torch.cat((branch2_f, branch2_p), dim=1))
        branch = branch1 * branch2
        # trunk = self.trunk.forward(query, params)
        trunk = self.trunk.forward(query)
        return branch @ trunk.T
    

##################################################
##################################################
# End of file