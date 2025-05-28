from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch   import  nn

from    ..utils         import  Objects, warn_redundant_arguments
from    ..layers        import  BaseModule, MLP


##################################################
##################################################
__all__ = [
    "DeepONetStructured",
    "DeepONetUnstructured",
    "MIONetStructured",
    "MIONetUnstructured",
]


##################################################
##################################################
class DeepONetStructured(BaseModule):
    """## Deep Operator Network (DeepONet) - Structured version
    ### Approximation of a continuous operator of function spaces with a single input, using a branch net and a trunk net
    
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of certain types of function spaces.
    The universal approximation theorem holds for a continuous map `G: V -> C(K_2)`, where `V` is a compact subset of `C(K_1)` for some compact space `K_1` and `K_2` is a compact subset of a finite-dimensional Euclidean space.
    
    Reference: https://www.nature.com/articles/s42256-021-00302-5
    
    -----
    ### Remark
    1. It is assumed that the input and the output functions are scalar-valued.
    """
    def __init__(
            self,
            
            branch:             Optional[torch.nn.Module]   = None,
            trunk:              Optional[torch.nn.Module]   = None,
            channels_branch:    Optional[Sequence[int]]     = None,
            channels_trunk:     Optional[Sequence[int]]     = None,
            activation_name:    Objects[str]                = "relu",
            activation_kwargs:  Objects[dict[str, object]]  = {},
            bias:               bool                        = True,
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `DeepONet`
        
        -----
        ### Arguments
        * `channels_branch` (`Sequence[int]`)
            * The sequence of channels for the branch net, from the input to the output.
            * (Input channels) The number of the sensor points.
            * (Output channels) Should match with the number of the output channels of the trunk net.
        
        * `channels_trunk` (`Sequence[int]`)
            * The sequence of channels for the trunk net, from the input to the output.
            * (Input channels) The number of the coordinates of the query space.
            * (Output channels) Should match with the number of the output channels of the branch net.
        
        * `activation_name` (`Objects[str]`, default: `"relu"`) and `activation_kwargs` (`Objects[dict[str, object]]`, default: `{}`)
            * The activation functions.
            * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
            * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
        
        * `bias` (`bool`, default: `True`)
            * If `True`, then the bias is added to the output of the network.
            * If `False`, then the bias is not added to the output of the network.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name, activation_name]
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs, activation_kwargs]
        
        # Define the subnetworks
        self.branch = \
            branch if branch is not None else \
            MLP(
                channels            = channels_branch,
                activation_name     = activation_name[0],
                activation_kwargs   = activation_kwargs[0]
            )
        self.trunk  = \
            trunk if trunk is not None else \
            MLP(
                channels            = channels_trunk, 
                activation_name     = activation_name[1],
                activation_kwargs   = activation_kwargs[1]
            )
        self.bias   = nn.Parameter(torch.zeros(size=(1,), dtype=torch.float)) if bias else 0.0
        
        return
    
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        ### Arguments
        
        -----
        * `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of two tensors, labelled `U` and `Y` in order.
                * `U` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 2-tensor, where each row corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)
        
        -----
        ### Remark
        1. In the structured case, the given query points are used throughout all `U` instances.
            * `U.shape:         (batch_size,        n_sensor_points)`
            * `Y.shape:         (n_query_points,    dim_query_space)`
            * `branch(U).shape: (batch_size,        num_features)`
            * `trunk(Y).shape:  (n_query_points,    num_features)`
            * `output.shape:    (batch_size,        n_query_points)`
            * `forward:         torch.einsum("bi,ti->bt", [branch, trunk])`
        """ 
        branch  = self.branch.forward(inputs[0])
        trunk   = self.trunk.forward(inputs[1])
        inner_prod = branch @ trunk.T
        return inner_prod + self.bias




class DeepONetUnstructured(BaseModule):
    """## Deep Operator Network (DeepONet) - Unstructured version
    ### Approximation of a continuous operator of function spaces with a single input, using a branch net and a trunk net
    
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of certain types of function spaces.
    The universal approximation theorem holds for a continuous map `G: V -> C(K_2)`, where `V` is a compact subset of `C(K_1)` for some compact space `K_1` and `K_2` is a compact subset of a finite-dimensional Euclidean space.
    
    Reference: https://www.nature.com/articles/s42256-021-00302-5
    
    -----
    ### Remark
    1. It is assumed that the input and the output functions are scalar-valued.
    """
    def __init__(
            self,
            
            branch:             Optional[torch.nn.Module]   = None,
            trunk:              Optional[torch.nn.Module]   = None,
            channels_branch:    Optional[Sequence[int]]     = None,
            channels_trunk:     Optional[Sequence[int]]     = None,
            activation_name:    Objects[str]                = "relu",
            activation_kwargs:  Objects[dict[str, object]]  = {},
            bias:               bool                        = True,
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `DeepONet`
        
        -----
        ### Arguments
        * `channels_branch` (`Sequence[int]`)
            * The sequence of channels for the branch net, from the input to the output.
            * (Input layer) The number of the input channels is the number of the sensor points.
            * (Output layer) The number of the output channels must match the number of the output channels of the trunk net.
        
        * `channels_trunk` (`Sequence[int]`)
            * The sequence of channels for the trunk net, from the input to the output.
            * (Input layer) The number of the input channels is the number of the coordinates of the query space.
            * (Output layer) The number of the output channels must match the number of the output channels of the branch net.
        
        * `activation_name` (`Objects[str]`, default: `"relu"`) and `activation_kwargs` (`Objects[dict[str, object]]`, default: `{}`)
            * The activation functions.
            * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
            * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
            
        * `bias` (`bool`, default: `True`)
            * If `True`, then the bias is added to the output of the network.
            * If `False`, then the bias is not added to the output of the network.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name, activation_name]
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs, activation_kwargs]
        
        # Define the subnetworks
        self.branch = \
            branch if branch is not None else \
            MLP(
                channels            = channels_branch,
                activation_name     = activation_name[0],
                activation_kwargs   = activation_kwargs[0],
            )
        self.trunk  = \
            trunk if trunk is not None else \
            MLP(
                channels            = channels_trunk,
                activation_name     = activation_name[1],
                activation_kwargs   = activation_kwargs[1],
            )
        self.bias   = nn.Parameter(torch.zeros(size=(1,), dtype=torch.float)) if bias else 0.0
        
        return
    
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        ### Arguments
        
        -----
        * `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of two tensors, labelled `U` and `Y` in order.
                * `U` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 3-tensor, where each row of each 2-tensor in `Y` corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)
        
        -----
        ### Remark
        1. In the unstructured case, each `Y` instance corresponds to exactly one of the `U` instances; that is, each `Y[i]` is the tensor of the query points for `U[i]`. As each `Y` instance is a finite subset of the query space (single or not), `Y` should be given as a 3-tensor of shape `(batch_size, n_query_points, dim_query_space)`.
            * `U.shape:         (batch_size,        n_sensor_points)`
            * `Y.shape:         (batch_size,        n_query_points,     dim_query_space)`
            * `branch(U).shape: (batch_size,        num_features)`
            * `trunk(Y).shape:  (batch_size,        n_query_points,     num_features)`
            * `output.shape:    (batch_size,        n_query_points)`
            * `forward:         torch.einsum("bi,bfi->bf", [branch, trunk])`
        """ 
        branch  = self.branch.forward(inputs[0])
        trunk   = self.trunk.forward(inputs[1])
        inner_prod = torch.einsum("bi,bfi->bf", [branch, trunk])
        return inner_prod + self.bias


##################################################
##################################################
class MIONetStructured(BaseModule):
    """## Multiple-Input Operator Network (MIONet) - Structured version
    ### Approximation of a continuous operator of Banach spaces, using a branch neta and a trunk net
    
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of Banach spaces.
    The universal approximation theorem holds for a continuous map `G: K_1 x .. x K_n -> W`, where `K_i` is a compact subset of a Banach space `X_i` with a countable basis for `i in {1, ..., n}` and `W` is a Banach space.

    Reference: https://epubs.siam.org/doi/epdf/10.1137/22M1477751

    -----
    ### Remark
    1. For implementation convinience, the network is constructed in accordance with the part (c) of Corollary 2.6 of the paper.
    """
    def __init__(
            self,
            
            branches:           Optional[Sequence[torch.nn.Module]] = None,
            trunk:              Optional[torch.nn.Module]           = None,
            channels_branches:  Optional[Sequence[Sequence[int]]]   = None,
            channels_trunk:     Optional[Sequence[int]]             = None,
            activation_name:    Objects[str]                = "relu",
            activation_kwargs:  Objects[dict[str, object]]  = {},
            
            bias:               bool                        = True,
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `MIONet`
        
        -----
        ### Arguments
        * `channels_branches` (`Sequence[Sequence[int]]`)
            * The list of channels for the branch net, from the input to the output.
            * (Input layer) The number of the input channels is the number of the sensor points.
            * (Output layer) The number of the output channels must match the number of the output channels of the trunk net.
        
        * `channels_trunk` (`Sequence[int]`)
            * The list of channels for the trunk net, from the input to the output.
            * (Input layer) The number of the input channels is the number of the coordinates of the query space.
            * (Output layer) The number of the output channels must match the number of the output channels of the branch net.
        
        * `activation_name` (`Objects[str]`, default: `"relu"`) and `activation_kwargs` (`Objects[dict[str, object]]`, defaule: `{}`)
            * The activation functions.
            * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
            * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
        
        * `bias` (`bool`, default: `True`)
            * If `True`, then the bias is added to the output of the network.
            * If `False`, then the bias is not added to the output of the network.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
        
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
        self.bias   = nn.Parameter(torch.zeros(size=(1,), dtype=torch.float)) if bias else 0.0
        
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




class MIONetUnstructured(BaseModule):
    """## Multiple-Input Operator Network (MIONet) - Unstructured version
    ### Approximation of a continuous operator of Banach spaces, using a branch neta and a trunk net
    
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of Banach spaces.
    The universal approximation theorem holds for a continuous map `G: K_1 x .. x K_n -> W`, where `K_i` is a compact subset of a Banach space `X_i` with a countable basis for `i in {1, ..., n}` and `W` is a Banach space.

    Reference: https://epubs.siam.org/doi/epdf/10.1137/22M1477751

    -----
    ### Remark
    1. For implementation convinience, the network is constructed in accordance with the part (c) of Corollary 2.6 of the paper.
    """
    def __init__(
            self,
            
            branches:           Optional[Sequence[torch.nn.Module]] = None,
            trunk:              Optional[torch.nn.Module]           = None,
            channels_branches:  Optional[Sequence[Sequence[int]]]   = None,
            channels_trunk:     Optional[Sequence[int]]             = None,
            activation_name:    Objects[str]                = "relu",
            activation_kwargs:  Objects[dict[str, object]]  = {},
            bias:               bool                        = True,
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `MIONet`
        
        -----
        ### Arguments
        * `channels_branches` (`Sequence[Sequence[int]]`)
            * The list of channels for the branch net, from the input to the output.
            * (Input layer) The number of the input channels is the number of the sensor points.
            * (Output layer) The number of the output channels must match the number of the output channels of the trunk net.
        
        * `channels_trunk` (`Sequence[int]`)
            * The list of channels for the trunk net, from the input to the output.
            * (Input layer) The number of the input channels is the number of the coordinates of the query space.
            * (Output layer) The number of the output channels must match the number of the output channels of the branch net.
        
        * `activation_name` (`Objects[str]`, default: `"relu"`) and `activation_kwargs` (`Objects[dict[str, object]]`, defaule: `{}`)
            * The activation functions.
            * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
            * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
        
        * `bias` (`bool`, default: `True`)
            * If `True`, then the bias is added to the output of the network.
            * If `False`, then the bias is not added to the output of the network.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs=kwargs)
                
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
        self.bias   = nn.Parameter(torch.zeros(size=(1,), dtype=torch.float)) if bias else 0.0
        
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
                    * A 3-tensor, where each row of each 2-tensor in `Y` corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)

        -----
        ### Remark
        1. In the unstructured case, each `Y` instance corresponds to exactly one of the `U` instances; that is, each `Y[i]` is the tensor of the query points for `U[i]`. As each `Y` instance is a finite subset of the query space (single or not), `Y` should be given as a 3-tensor of shape `(batch_size, n_query_points, dim_query_space)`.
            * `U_i.shape:               (batch_size,        n_sensor_points)`
            * `Y.shape:                 (batch_size,        n_query_points,     dim_query_space)`
            * `branch[i](U_i).shape:    (batch_size,        num_features)`
            * `trunk(Y).shape:          (batch_size,        n_query_points,     num_features)`
            * `output.shape:            (batch_size,        n_query_points)`
            * `forward: torch.einsum("bi,bfi->bf", [(branch_1 * ... * branch_n), trunk])`
            
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
        inner_prod = torch.einsum("bi,bfi->bf", [branch, trunk])        
        return inner_prod + self.bias


##################################################
##################################################
# End of file