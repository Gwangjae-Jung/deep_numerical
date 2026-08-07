from    typing  import  Sequence, Dict, Optional
import  torch
from    torch   import  nn
from    deep_numerical  import  Objects
from    deep_numerical.neural   import  BaseModule
from    deep_numerical.neural.layer import  MLP


__all__: list[str] = [
    "DeepONet", "DeepONetUnstructured",
    "MIONet", "MIONetUnstructured",
]


##################################################
##################################################
class DeepONet(BaseModule):
    """## Deep Operator Network (DeepONet) - Structured version
    ### Approximation of a continuous operator of function spaces with a single input, using a branch net and a trunk net
    
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of certain types of function spaces.
    The universal approximation theorem holds for a continuous map `G: V -> C(K_2)`, where `V` is a compact subset of `C(K_1)` for some compact space `K_1` and `K_2` is a compact subset of a finite-dimensional Euclidean space.
    
    Reference: https://www.nature.com/articles/s42256-021-00302-5
    """
    def __init__(
            self,
            
            branch:             Optional[torch.nn.Module]   = None,
            trunk:              Optional[torch.nn.Module]   = None,
            channels_branch:    Optional[Sequence[int]]     = None,
            channels_trunk:     Optional[Sequence[int]]     = None,
            activation_name:    Objects[str]                = "relu",
            activation_kwargs:  Objects[Dict[str, object]]  = {},
            bias:               bool                        = True,
            dtype:              torch.dtype                 = torch.float,
        ) -> None:
        """## The initializer of the class `DeepONet`
        
        Arguments:
            `branch` (`Optional[torch.nn.Module]`, default: `None`):
                The branch network.
            `trunk` (`Optional[torch.nn.Module]`, default: `None`):
                The trunk network.
            `channels_branch` (`Sequence[int]`):
                The sequence of channels for the branch net, from the input to the output.
                    * (Input channels) The number of the sensor points.
                    * (Output channels) Should match with the number of the output channels of the trunk net.
            `channels_trunk` (`Sequence[int]`):
                The sequence of channels for the trunk net, from the input to the output.
                    * (Input channels) The number of the coordinates of the query space.
                    * (Output channels) Should match with the number of the output channels of the branch net.

            `activation_name` (`Objects[str]`, default: `"relu"`):
                The name of the activation functions to be used.
                    * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
                    * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
            
            `activation_kwargs` (`Objects[Dict[str, object]]`, default: `{}`):
                Further arguments for the activation functions.
            
            `bias` (`bool`, default: `True`):
                If `True`, then the bias is added to the output of the network. If `False`, then the bias is not added to the output of the network.
            
            `dtype` (`torch.dtype`, default: `torch.float`):
                The datatype to be used in this model.
        """
        super().__init__()
        
        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name, activation_name]
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs, activation_kwargs]
        
        # Define the subnetworks
        if branch is None:
            branch = nn.Sequential(
                MLP(
                    channels            = channels_branch,
                    activation_name     = activation_name[0],
                    activation_kwargs   = activation_kwargs[0],
                    dtype               = dtype,
                ),
                nn.Unflatten(-1, (channels_branch[-1], 1))
            )
        if trunk is None:
            trunk = nn.Sequential(
                MLP(
                    channels            = channels_trunk, 
                    activation_name     = activation_name[1],
                    activation_kwargs   = activation_kwargs[1],
                    dtype               = dtype,
                ),
                nn.Unflatten(-1, (channels_trunk[-1], 1))
            )
        self.branch = branch
        self.trunk  = trunk
        self.bias   = nn.Parameter(torch.zeros((1,), dtype=dtype), requires_grad=bias)
        
        return None
    
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            `inputs` (`Sequence[torch.Tensor]`):
                A sequence of two tensors, labelled `U` and `Y` in order. See Remark below.
        
        Returns:
            `torch.Tensor`:
                A tensor of shape `(batch_size, n_query_points, n_channels)`.
        
        ### Remark
        `U` is a 2-tensor of shape `(batch_size, n_sensor_points)`, where each row corresponds to the values of a function restricted to the sensor points. 
        Meanwhile, `Y` is a 2-tensor of shape `(n_query_points, dim_query_space)`, where each row corresponds to the coordinates of a query point.
        In this implementation, the query points are shared for all `U` instances.
        """ 
        branch: torch.Tensor = self.branch.forward(inputs[0])
        trunk:  torch.Tensor = self.trunk.forward(inputs[1])
        inner_prod = torch.einsum("bfc, qfc -> bqc", branch, trunk)
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
            activation_kwargs:  Objects[Dict[str, object]]  = {},
            bias:               bool                        = True,
            dtype:              torch.dtype                 = torch.float,
        ) -> None:
        """## The initializer of the class `DeepONetUnstructured`
        
        Arguments:
            `branch` (`Optional[torch.nn.Module]`, default: `None`):
                The branch network.
            `trunk` (`Optional[torch.nn.Module]`, default: `None`):
                The trunk network.
            `channels_branch` (`Sequence[int]`):
                The sequence of channels for the branch net, from the input to the output.
                    * (Input channels) The number of the sensor points.
                    * (Output channels) Should match with the number of the output channels of the trunk net.
            `channels_trunk` (`Sequence[int]`):
                The sequence of channels for the trunk net, from the input to the output.
                    * (Input channels) The number of the coordinates of the query space.
                    * (Output channels) Should match with the number of the output channels of the branch net.

            `activation_name` (`Objects[str]`, default: `"relu"`):
                The name of the activation functions to be used.
                    * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
                    * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
            
            `activation_kwargs` (`Objects[Dict[str, object]]`, default: `{}`):
                Further arguments for the activation functions.
            
            `bias` (`bool`, default: `True`):
                If `True`, then the bias is added to the output of the network. If `False`, then the bias is not added to the output of the network.
            
            `dtype` (`torch.dtype`, default: `torch.float`):
                The datatype to be used in this model.
        """
        super().__init__()
        
        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name, activation_name]
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs, activation_kwargs]
        
        # Define the subnetworks
        if branch is None:
            branch = nn.Sequential(
                MLP(
                    channels            = channels_branch,
                    activation_name     = activation_name[0],
                    activation_kwargs   = activation_kwargs[0],
                    dtype               = dtype,
                ),
                nn.Unflatten(-1, (channels_branch[-1], 1))
            )
        if trunk is None:
            trunk = nn.Sequential(
                MLP(
                    channels            = channels_trunk, 
                    activation_name     = activation_name[1],
                    activation_kwargs   = activation_kwargs[1],
                    dtype               = dtype,
                ),
                nn.Unflatten(-1, (channels_trunk[-1], 1))
            )
        self.branch = branch
        self.trunk  = trunk
        self.bias   = nn.Parameter(torch.zeros((1,), dtype=dtype), requires_grad=bias)
        
        return None
    
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            `inputs` (`Sequence[torch.Tensor]`):
                A sequence of two tensors, labelled `U` and `Y` in order. See Remark below.
        
        Returns:
            `torch.Tensor`:
                A tensor of shape `(batch_size, n_query_points, n_channels)`.
        
        -----
        ### Remark
        `U` is a 2-tensor of shape `(batch_size, n_sensor_points)`, where each row corresponds to the values of a function restricted to the sensor points. 
        Meanwhile, `Y` is a 2-tensor of shape `(batch_size, n_query_points, dim_query_space)`, from which it can be observed that the query points are *not* shared for all `U` instances.
        """
        branch: torch.Tensor = self.branch.forward(inputs[0])
        trunk:  torch.Tensor = self.trunk.forward(inputs[1])
        inner_prod = torch.einsum("bfc, bqfc -> bqc", [branch, trunk])
        return inner_prod + self.bias


##################################################
##################################################
class MIONet(BaseModule):
    """## Multiple-Input Operator Network (MIONet)
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
            activation_kwargs:  Objects[Dict[str, object]]  = {},
            bias:               bool                        = True,
            dtype:              torch.dtype                 = torch.float,
        ) -> None:
        """## The initializer of the class `MIONet`
        
        Arguments:
            `channels_branches` (`Sequence[Sequence[int]]`):
                The list of channels for the branch net, from the input to the output.
                    * (Input layer) The number of the input channels is the number of the sensor points.
                    * (Output layer) The number of the output channels must match the number of the output channels of the trunk net.
            `channels_trunk` (`Sequence[int]`):
                The list of channels for the trunk net, from the input to the output.
                    * (Input layer) The number of the input channels is the number of the coordinates of the query space.
                    * (Output layer) The number of the output channels must match the number of the output channels of the branch net.
            `activation_name` (`Objects[str]`, default: `"relu"`):
                The names of the activation functions to be used.
                    * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
                    * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
            `activation_kwargs` (`Objects[Dict[str, object]]`, default: `{}`):
                Further arguments for the activation functions.
            `bias` (`bool`, default: `True`):
                If `True`, then the bias is added to the output of the network. If `False`, then the bias is not added to the output of the network.
            `dtype` (`torch.dtype`, default: `torch.float`):
                The datatype to be used in this model.
        """
        super().__init__()
        
        num_branches: int
        if branches is not None:
            num_branches = len(branches)
        elif channels_branches is not None:
            num_branches = len(channels_branches)
        else:
            raise ValueError("Either `branches` or `channels_branches` must be provided.")
        
        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name] * (num_branches+1)
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs] * (num_branches+1)

        # Define the subnetworks
        if branches is None:
            branches = [
                nn.Sequential(
                    MLP(_ch_branch, True, _act_name, _act_kwargs, dtype=dtype),
                    nn.Unflatten(-1, (_ch_branch[-1], 1)),
                )
                for _ch_branch, _act_name, _act_kwargs in zip(
                    channels_branches,
                    activation_name[:-1],
                    activation_kwargs[:-1],
                )
            ]
        if trunk is None:
            trunk = nn.Sequential(
                MLP(
                    channels            = channels_trunk,
                    activation_name     = activation_name[-1],
                    activation_kwargs   = activation_kwargs[-1],
                    dtype               = dtype,
                ),
                nn.Unflatten(-1, (channels_trunk[-1], 1)),
            )
        self.branches   = nn.ModuleList(branches)
        self.trunk      = trunk
        self.bias       = nn.Parameter(torch.zeros((1,), dtype=dtype), requires_grad=bias)
        self.__einsum_cmd = ','.join(["bfc"]*num_branches + ["qfc"]) + " -> bqc"
        
        return None

    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            `inputs` (`Sequence[torch.Tensor]`):
                A sequence of tensors of length `n+1`, labelled `U_1, ..., U_n` and `Y` in order, where `n` is the number of the branch networks. See below remark.
        
        Returns:
            `torch.Tensor`:
                A tensor of shape `(batch_size, n_query_points, n_channels)`.

        -----
        ### Remark
        `U_1, ..., U_n` are 2-tensor of shape `(batch_size, n_sensor_points)`, where each row corresponds to the values of a function restricted to the sensor points. 
        Meanwhile, `Y` is a 2-tensor of shape `(n_query_points, dim_query_space)`.
        """
        branches = [self.branches[k].forward(inputs[k]) for k in range(len(inputs)-1)]
        trunk = self.trunk.forward(inputs[-1])
        inner_prod = torch.einsum(self.__einsum_cmd, *branches, trunk)
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
            activation_kwargs:  Objects[Dict[str, object]]  = {},
            bias:               bool                        = True,
            dtype:              torch.dtype                 = torch.float,
        ) -> None:
        """## The initializer of the class `MIONetUnstructured`
        
        Arguments:
            `channels_branches` (`Sequence[Sequence[int]]`):
                The list of channels for the branch net, from the input to the output.
                    * (Input layer) The number of the input channels is the number of the sensor points.
                    * (Output layer) The number of the output channels must match the number of the output channels of the trunk net.
            `channels_trunk` (`Sequence[int]`):
                The list of channels for the trunk net, from the input to the output.
                    * (Input layer) The number of the input channels is the number of the coordinates of the query space.
                    * (Output layer) The number of the output channels must match the number of the output channels of the branch net.
            `activation_name` (`Objects[str]`, default: `"relu"`):
                The names of the activation functions to be used.
                    * If `activation` is a string, then the same activation applies to both the branch net and the trunk net.
                    * If `activation` is a list of strings, then the branch net is activated by `activation[0]` and the trunk net is activated by `activation[1]`.
            `activation_kwargs` (`Objects[Dict[str, object]]`, default: `{}`):
                Further arguments for the activation functions.
            `bias` (`bool`, default: `True`):
                If `True`, then the bias is added to the output of the network. If `False`, then the bias is not added to the output of the network.
            `dtype` (`torch.dtype`, default: `torch.float`):
                The datatype to be used in this model.
        """
        super().__init__()
                
        num_branches: int
        if branches is not None:
            num_branches = len(branches)
        elif channels_branches is not None:
            num_branches = len(channels_branches)
        else:
            raise ValueError("Either `branches` or `channels_branches` must be provided.")

        # Save some information in lists
        if isinstance(activation_name, str):
            activation_name     = [activation_name] * (num_branches+1)
        if isinstance(activation_kwargs, dict):
            activation_kwargs   = [activation_kwargs] * (num_branches+1)

        # Define the subnetworks
        if branches is None:
            branches = [
                nn.Sequential(
                    MLP(_ch_branch, bias, _act_name, _act_kwargs, dtype=dtype),
                    nn.Unflatten(-1, (_ch_branch[-1], 1)),
                )
                for _ch_branch, _act_name, _act_kwargs in zip(
                    channels_branches,
                    activation_name[:-1],
                    activation_kwargs[:-1]
                )
            ]
        if trunk is None:
            trunk = nn.Sequential(
                MLP(
                    channels            = channels_trunk,
                    activation_name     = activation_name[-1],
                    activation_kwargs   = activation_kwargs[-1],
                    dtype               = dtype,
                ),
                nn.Unflatten(-1, (channels_trunk[-1], 1)),
            )
        self.branches   = nn.ModuleList(branches)
        self.trunk      = trunk
        self.bias       = nn.Parameter(torch.zeros((1,), dtype=dtype), requires_grad=bias)
        self.__einsum_cmd = ','.join(["bfc"]*num_branches + ["bqfc"]) + " -> bqc"
        
        return None

    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            `inputs` (`Sequence[torch.Tensor]`):
                A sequence of tensors of length `n+1`, labelled `U_1, ..., U_n` and `Y` in order, where `n` is the number of the branch networks. See below remark.
        
        Returns:
            `torch.Tensor`:
                A tensor of shape `(batch_size, n_query_points, n_channels)`.

        -----
        ### Remark
        `U_1, ..., U_n` are 2-tensor of shape `(batch_size, n_sensor_points)`, where each row corresponds to the values of a function restricted to the sensor points. 
        Meanwhile, `Y` is a 3-tensor of shape `(batch_size, n_query_points, dim_query_space)`, from which it can be observed that the query points are *not* shared for all `U` instances.
        """
        branches = [self.branches[k].forward(inputs[k]) for k in range(len(inputs)-1)]
        trunk = self.trunk.forward(inputs[-1])
        inner_prod = torch.einsum(self.__einsum_cmd, *branches, trunk)
        return inner_prod + self.bias


##################################################
##################################################
# End of file