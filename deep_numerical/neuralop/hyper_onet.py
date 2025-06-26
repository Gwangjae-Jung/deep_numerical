from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch   import  nn

from    ..utils         import  Objects, warn_redundant_arguments
from    ..layers        import  BaseModule, MLP, HyperMLP


##################################################
##################################################
__all__: list[str] = ['HyperDeepONet', 'HyperMIONet', 'ParameterizedMIONet']


##################################################
##################################################
def n_parameters_in_mlp(dimensions: Sequence[int]) -> int:
    """Counts the number of parameters in a multi-layer perceptron (MLP) with given dimensions."""
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
    def __init__(
            self,
            hypernet_prior: Sequence[torch.nn.Module],
            dim_trunk:      Sequence[int] = [1+2, 32, 32, 32, 32, 1],
        ) -> Self:
        super().__init__()
        self.__dim_trunk: Sequence[int] = dim_trunk
        self.hypernet = torch.nn.Sequential(
            *hypernet_prior,
            torch.nn.SiLU(),
            torch.nn.LazyLinear(n_parameters_in_mlp(dim_trunk)),
        )
        self.__splitter_weight: list[int] = []
        self.__splitter_bias:   list[int] = []
        self.__shapes_weight:   list[tuple[int]] = []
        self.__shapes_bias:     list[tuple[int]] = []
        
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
        """
        ### Arguments
        * `X`: The input tensor for the branch network (which is a hypernetwork) which is of shape `(batch_size, n_sensor_points)`.
        * `query`: The input tensor for the target network which is of shape `(n_query_points, dim_query)`.
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
    def __init__(
            self,
            dimension:  int,
        ) -> Self:
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
        branch1 = X
        branch2 = X
        for md in self.branch1:
            branch1 = md.forward(branch1, p)
        for md in self.branch2:
            branch2 = md.forward(branch2, p)
        branch = branch1 * branch2
        trunk = self.trunk.forward(p)
        return branch @ trunk.T


class ParameterizedMIONet(BaseModule):
    def __init__(
            self,
            dimension:  int,
        ) -> Self:
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