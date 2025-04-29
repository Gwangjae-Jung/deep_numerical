from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch   import  nn

from    ..utils         import  Objects, warn_redundant_arguments
from    ..layers        import  BaseModule, MLP


##################################################
##################################################
__all__: list[str] = ['HyperDeepONet', 'HyperMIONet']


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
class HyperMIONetSeparated(BaseModule):
    """## Multiple-Input Operator Network (MIONet) - Structured version
    """
    def __init__(self,) -> Self:
        super().__init__()
        
        self.num_inputs = self.num_branches = len(channels_branches)
        
        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name] * (self.num_branches + 1)
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs] * (self.num_branches + 1)

        # Define the subnetworks
        if branches is not None:
            self.branches = nn.ModuleList(branches)
        else:
            self.branches = nn.ModuleList(
                [
                    MLP(_ch_branch, True, _act_name, _act_kwargs)
                    for _ch_branch, _act_name, _act_kwargs in zip(
                        channels_branches,
                        activation_name[:-1],
                        activation_kwargs[:-1]
                    )
                ]
            )
        self.trunk  = \
            trunk if trunk is not None else \
            MLP(
                channels            = channels_trunk,
                activation_name     = activation_name[-1],
                activation_kwargs   = activation_kwargs[-1],
            )
        
        
        return

    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        r"""
        ### Arguments
        
        -----
        @ `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of multiple tensors of length `n + 1`, labelled `U_1, ..., U_n` and `Y` in order.
                * `U_i` for `i \in {1, ..., n}` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 2-tensor, where each row corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)

        -----
        ### Remark
        1. The forward operation is defined by `einsum` operation, whose specific definition is determined by `self.is_query_shared`.
            * `U_i.shape:               (batch_size,        n_sensor_points)`
            * `Y.shape:                 (n_query_points,    dim_query_space)`
            * `branch[i](U_i).shape:    (batch_size,        num_features)`
            * `trunk(Y).shape:          (n_query_points,    num_features)`
            * `output.shape:            (batch_size,        n_query_points)`
            * `forward:                 torch.einsum("bi,ti->bt", [prod(branch_1, ..., branch_n), trunk])`
        
        2. In contrast to the Deep Operator Network, the Multiple-Input Operator Network supports multiple inputs. Since this class assumes that each input instance is 1-dimensional, to input a multidimensional instance, one has to project the instance dimension by dimension. For example, if an input instance is a discretization of a continuous function into a 3-dimensional space, such instance should be given as 3 input instances, each of which is into a 1-dimensional space.
        """
        branch  = torch.stack(
            [
                self.branches[k].forward(inputs[k])
                for k in range(len(inputs)-1)
            ], dim=0
        ).type(torch.float)
        branch  = torch.prod(branch, dim=0)
        trunk   = self.trunk.forward(inputs[-1])
        inner_prod = branch @ trunk.T
        return inner_prod + self.bias


##################################################
##################################################
# End of file