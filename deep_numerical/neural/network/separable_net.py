from    typing                  import  Any, Callable, Sequence, List, Tuple, Optional
from    collections             import  deque
import  torch
from    deep_numerical          import  EINSUM_STRING
from    deep_numerical.autograd import  derivatives
from    deep_numerical.neural   import  BaseModule


__all__ = ['SeparableNet']


##################################################
##################################################
_BRANCH_IN_FEATURES:    int = 1
_BRANCH_NDIM:           int = 2


class SeparableNet(BaseModule):
    """## Separable neural network - Dimensionwise forward propagation for efficient training of neural networks
    
    -----
    ### Description
    `SeparableNet` is a generic implementation of a [separable neural network](https://neurips.cc/virtual/2022/59890), which is designed to handle multiple variables in a dimensionwise manner. This class allows for efficient training of neural networks by separating the forward propagation of each variable into distinct branches, which are then combined using `einsum`.
    
    -----
    ### Reference
    [1] Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park. 2023. Separable physics-informed neural networks. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NIPS '23). Curran Associates Inc., Red Hook, NY, USA, Article 1032, 23761–23788.
    """
    def __init__(
            self,
            networks:       Sequence[torch.nn.Module] = [],
            out_channels:   int = 1,
            deg_grad:       Sequence[int]   = [],
            bias:           bool            = True,
        ) -> None:
        """The initializer of the class `SeparableNet`.
        
        Arguments:
            `networks` (`Sequence[torch.nn.Module]`, default: `[]` (the empty list)): A sequence of neural network modules, where each module corresponds to a variable. If `[]`, default networks will be created for each variable using the `_default_network` function.
            `out_channels` (`int`, default: `1`): The number of the output channels of the network. Hence, the number of the output channels of each subnetwork given in `networks` should be divisible by `out_channels`.
            `deg_grad` (`Sequence[int]`, default: `[]` (the empty list)): A sequence of non-negative integers representing the degree of differentiation for each variable. If `[]`, no differentiation is applied.
            `bias` (`bool`, default: `True`): If `True`, includes a learnable bias term in the final output.
        """
        if len(networks)==0:
            raise ValueError("The 'networks' argument must contain at least one neural network module.")
        if len(networks)!=len(deg_grad):
            raise ValueError(f"The length of 'networks' ({len(networks)}) must match the length of 'deg_grad' ({len(deg_grad)}).")
        if any(_d<0 for _d in deg_grad):
            raise ValueError("All entries in deg_grad must be non-negative integers.")
        
        super().__init__()
        self.__n_vars:      int = len(deg_grad)
        self.__deg_grad:    Tuple[int, ...] = tuple(deg_grad)
        self.networks       = torch.nn.ModuleList(networks)
        self.bias           = torch.nn.Parameter(torch.randn((1,))) if bias else torch.zeros((1,))
        self.__out_channels = out_channels
        self.__einsum_cmd:      str = ""
        self.__base_einsum_cmd: str = ""
        self.__set_base_einsum_cmd()
        self.__diff_ops:    List[Callable[[torch.Tensor, Any], List[torch.Tensor]]] = \
            [derivatives(net, deg) for (net, deg) in zip(networks, deg_grad)]
        self.__branch_outputs: List[List[torch.Tensor]] = []
        return None
    
    
    @property
    def n_vars(self) -> int:
        return self.__n_vars
    @property
    def base_einsum_cmd(self) -> str:
        return self.__base_einsum_cmd
    @property
    def einsum_cmd(self) -> str:
        return self.__einsum_cmd
    @property
    def branch_outputs(self) -> List[List[torch.Tensor]]:
        """Returns the list of lists, where the internal list of index `idx` saves the output and the derivatives of the `idx`-th branch network with respect to its input variable."""
        return self.__branch_outputs
    
    
    def __set_base_einsum_cmd(self) -> None:
        _RHS = EINSUM_STRING[:self.__n_vars]
        _LHS = ','.join([f"{v}rc" for v in _RHS])
        self.__base_einsum_cmd = f"{_LHS}->{_RHS}c"
        return
    
    
    def forward(
            self,
            coordinates:            Sequence[torch.Tensor],
            einsum_command:         Optional[str]   = None,
            compute_derivatives:    Optional[bool]  = None,
        ) -> torch.Tensor:
        """
        Arguments:
            `coordinates` (`Sequence[torch.Tensor]`): A sequence of tensors, where the `i`-th entry of `coordinates` is a tensor of shape `(N_i, 1)`.
            `einsum_command` (`Optional[str]`, default: `None`): The command for the einstein summation. Generally, the einstein summation is conducted with `{i_1}rc, ..., {i_d}rc -> {i_1}...{i_d}c` for `d`-dimensional problems, which is the case where `einsum_command` is `None`. In order to compute the values on interface or boundary of a region, one may specify `einsum_command` in order to use the separable neural network as an ordinary neural network. Once `einsum_command` is set, this setting will be used until the following call of `forward()`.
            `compute_derivatives` (`Optional[bool]`, default: `None`): If `True`, computes the derivatives of the branch networks up to the degrees specified in `deg_grad` during the forward pass. If `False`, only computes the outputs of the branch networks without derivatives. If `None`, uses the module's training mode to decide.
        """
        self.__branch_outputs.clear()
        operands: List[torch.Tensor] = []
        if compute_derivatives is None:
            compute_derivatives = self.training
        if compute_derivatives: operands = self._forward_training(coordinates)
        else:                   operands = self._forward_inference(coordinates)
        self.__einsum_cmd = self.__base_einsum_cmd if einsum_command is None else einsum_command
        return torch.einsum(self.__einsum_cmd, *operands) + self.bias
    
    
    def _forward_training(self, coordinates: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        operands: List[torch.Tensor] = []
        for ops, coord in zip(self.__diff_ops, coordinates):
            if not (coord.ndim==_BRANCH_NDIM and coord.size(1)==_BRANCH_IN_FEATURES):
                raise ValueError(f"Each entry of 'coordinates' must have shape (N_i, 1). Found {coord.shape}.")
            res = coord.size(0)
            _outputs: deque[torch.Tensor] = deque(ops(coord))
            # NOTE: Since `net` is a function from $\mathbb{R}^$ into $\mathbb{R}^{r \cdot out_channels}$ and so are its derivatives.
            # NOTE: Thus, all tensors will be reshaped to `(N_i, rank)`.
            outputs: List[torch.Tensor] = []
            while _outputs:
                _out = _outputs.popleft()
                outputs.append(_out.reshape(res, -1, self.__out_channels))
            self.__branch_outputs.append(outputs)
            operands.append(outputs[0])
        return operands
    
    
    def _forward_inference(self, coordinates: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        operands: List[torch.Tensor] = []
        for idx, coord in enumerate(coordinates):
            if not (coord.ndim==_BRANCH_NDIM and coord.size(1)==_BRANCH_IN_FEATURES):
                raise ValueError(f"Each entry of 'coordinates' must have shape (N_i, 1). Found {coord.shape}.")
            out: torch.Tensor = self.networks[idx].forward(coord)
            out = out.reshape(coord.numel(), -1, self.__out_channels)
            operands.append(out)
        return operands
    
    
    def compute_gradient(self, deg_grad: Sequence[int]) -> torch.Tensor:
        if len(deg_grad)!=self.__n_vars:
            raise ValueError(f"Length of 'deg_grad' is {len(deg_grad)}, while 'n_vars' is {self.__n_vars}.")
        for idx, (d_in, d_max) in enumerate(zip(deg_grad, self.__deg_grad)):
            if d_in<0 or d_in>d_max:
                raise ValueError(f"Invalid differentiation degree for variable index {idx}: {d_in} (max: {d_max}).")
        operands: List[torch.Tensor] = [self.__branch_outputs[idx][d] for idx, d in enumerate(deg_grad)]
        return torch.einsum(self.__einsum_cmd, *operands)


##################################################
##################################################
# End of file