from    typing                  import  Sequence, Optional, Callable
from    typing_extensions       import  Self

import  torch
from    torch.func              import  vmap, jacrev

from    ..layers._base_module   import  BaseModule
from    ..layers                import  MLP


##################################################
##################################################
__all__: list[str] = ['SeparableNet']


##################################################
##################################################
class SeparableNet(BaseModule):
    def __init__(
            self,
            rank:           int,
            n_variables:    int,
            networks:       Optional[Sequence[torch.nn.Module]] = None,
        ) -> Self:
        """## Separable neural network - Dimensionwise forward propagation for efficient approximation
        
        -----
        ### Reference
        [1]: SPINN
        """
        if networks is None:
            networks = [
                torch.nn.Sequential(
                    torch.nn.Linear(1, 8),
                    torch.nn.Tanh(),
                    torch.nn.Linear(8, 32),
                    torch.nn.Tanh(),
                    torch.nn.Linear(32, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, rank),
                )
                for _ in range(n_variables)
            ]
        elif len(networks) != n_variables:
            raise ValueError(f"Number of networks ({len(networks)}) does not match number of variables ({n_variables}).")
        
        super().__init__()
        self.__rank         = rank
        self.__n_variables  = n_variables
        
        self.__network_inputs:  list[torch.Tensor] = []
        self.__network_outputs: list[torch.Tensor] = []
        self.__network_grads:   list[torch.Tensor] = []
        
        self.networks       = torch.nn.ModuleList(networks)
        self.functions:   list[Callable[[torch.Tensor], torch.Tensor]] = [
            lambda x1d: torch.squeeze(net.forward(torch.unsqueeze(x1d, 0)), 0)
            for net in networks
        ]
        
        return
    
    
    @property
    def num_variables(self) -> int:
        return self.__n_variables
    @property
    def rank(self) -> int:
        return self.__rank
    
    
    def forward(self, coordinates: list[torch.Tensor]) -> tuple[torch.Tensor]:
        """## Forward propagation of a separable neural network
        
        -----
        ### Arguments
        * `coordinates` (`Sequence[torch.Tensor]`)
            * A sequence of 1-dimensional tensors, where each entry is sampled from the corresponding axis of the input space.
            * Each entry (a 1-dimensional tensor) will be reshaped inside `forward` so that vectorized operations can be performed.
        """
        self.__compute_networks(coordinates)
        pred    = self.predict()
        diff_op = self.operator()
        return pred, diff_op
        
    
    def __compute_networks(self, coordinates: list[torch.Tensor]) -> None:
        # NOTE: Each tensor in `inputs` is a 1D tensor
        # Turn on gradients
        for _coord in coordinates:
            _coord.requires_grad_(True)
            
        # Conduct forward pass and compute Jacobian matrices
        outputs = [
            vmap(func)(x) for func, x in zip(self.functions, coordinates)
        ]
        grads   = [
            vmap(jacrev(func))(x) for func, x in zip(self.functions, coordinates)
        ]
        
        # Reshape outputs and gradients
        inputs  = [
            self.__reshape_output(_i, idx)
            for idx, _i in enumerate(coordinates)
        ]
        outputs = [
            self.__reshape_output(_o, idx)
            for idx, _o in enumerate(outputs)
        ]
        grads   = [
            self.__reshape_output(_g, idx)
            for idx, _g in enumerate(grads)
        ]
        
        self.__network_inputs  = inputs
        self.__network_outputs = outputs
        self.__network_grads   = grads
        return
    
    
    def __reshape_output(self, tensor: torch.Tensor, idx_coord: int) -> torch.Tensor:
        n_nets = len(self.networks)
        if tensor.ndim==1:
            n_points = tensor.size(0)
            newshape = [1]*n_nets
        elif tensor.ndim==2:
            n_points, rank = map(int, tensor.size())
            newshape = [1]*n_nets + [rank]
        else:
            raise ValueError(f"Reshaping tensors in 'SeparableNet' is allowed only for the tensors of dimension 1 or 2. ({tensor.ndim=})")
        newshape[idx_coord] = n_points
        return tensor.reshape(newshape)
    
    
    def predict(self) -> torch.Tensor:
        """The forward propagation of the class `SeparableNet`."""
        out = self.__network_outputs[0]
        for i in range(1, self.__n_variables):
            out = out * self.__network_outputs[i]
        return out.sum(dim=-1)
    
    
    def operator(self) -> torch.Tensor:
        r"""The differential operator of the PDE to be solved.
        
        -----
        ### Note
        1. The class `SeparableNet` will generally be used as the base class to construct a PINN, and the differential operator should be defined in the derived class. To this end, when defining `operator` in the derived class, it is recommended to begin with the following code, which loads the input tensors, output tensors, and the gradient tensors for each dimension:
        >>> inputs  = self.get_inputs()
        >>> outputs = self.get_outputs()
        >>> grads   = self.get_grads()
        
        -----
        ### Example
        1. The following code is an example of a 1-dimensional linear transport equation $u_t + v \cdot u_x = 0$, where the input tensor is aligned in $(t, x, v)$ convention:
        >>> inputs  = self.get_inputs()
        >>> outputs = self.get_outputs()
        >>> grads   = self.get_grads()
        >>> u_t = grads[0]
        >>> u_x = grads[1]
        >>> v   = inputs[2]
        >>> for i in range(self.__n_variables):
        >>>     if i!=0:
        >>>         u_t = u_t * outputs[i]
        >>> for i in range(self.__n_variables):
        >>>     if i!=1:
        >>>         u_x = u_x * outputs[i]
        >>> u_t = u_t.sum(dim=-1, keepdim=False)
        >>> u_x = u_x.sum(dim=-1, keepdim=False)
        >>> out = u_t + v * u_x
        """
        # NOTE: Below is an example of a differential operator
        pass
    
    
    def get_inputs(self) -> list[torch.Tensor]:
        """Get the inputs of the networks."""
        return self.__network_inputs
    def get_outputs(self) -> list[torch.Tensor]:
        """Get the outputs of the networks."""
        return self.__network_outputs
    def get_grads(self) -> list[torch.Tensor]:
        """Get the gradients of the networks."""
        return self.__network_grads


##################################################
##################################################
class HyperPINNinhomogeneous(BaseModule):
    def __init__(self, dim_domain: int, hidden_channels: int = 64) -> Self:
        super().__init__()
        input_dim = 1 + 2*dim_domain
        self.__dim_input = input_dim
        self.__hidden_channels = hidden_channels
        self.__dim_output = 1
        self.hypernets = torch.nn.ModuleList(
            [
                MLP(1, 32, 64, 256, (input_dim*hidden_channels)),
                MLP(1, 32, hidden_channels, hidden_channels**2)
            ]
        )



##################################################
##################################################
# End of file