from    typing                  import  Sequence, Optional, Callable
from    typing_extensions       import  Self

import  torch
from    torch.func              import  vmap, jacrev

from    ..layers._base_module   import  BaseModule


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
        
    
    def __compute_networks(self, inputs: list[torch.Tensor]) -> None:
        # NOTE: Each tensor in `inputs` is a 1D tensor
        # Turn on gradients
        for _input in inputs:
            _input.requires_grad_(True)
            
        # Conduct forward pass and compute Jacobian matrices
        outputs = [
            vmap(func)(x) for func, x in zip(self.functions, inputs)
        ]
        grads   = [
            vmap(jacrev(func))(x) for func, x in zip(self.functions, inputs)
        ]
        
        # Reshape outputs and gradients
        outputs = [
            self.__reshape_output(_o, idx)
            for idx, _o in enumerate(outputs)
        ]
        grads   = [
            self.__reshape_output(_g, idx)
            for idx, _g in enumerate(grads)
        ]
        
        self.__network_outputs = outputs
        self.__network_grads   = grads
        return 
    def __reshape_output(self, tensor: torch.Tensor, idx_coord: int) -> torch.Tensor:
        n_nets = len(self.networks)
        n_points, rank = map(int, tensor.size())
        newshape = [1]*n_nets + [rank]
        newshape[idx_coord] = n_points
        return tensor.reshape(newshape)
    
    
    def predict(self) -> torch.Tensor:
        """The forward propagation of the class `SeparableNet`."""
        out = self.__network_outputs[0]
        for i in range(1, self.__n_variables):
            out = out * self.__network_outputs[i]
        return out.sum(dim=-1)
    
    
    def operator(self) -> torch.Tensor:
        """The *experimental* differential operator of the class `SeparableNet`."""
        out = self.__network_grads[0]
        for i in range(1, self.__n_variables):
            out = out * self.__network_outputs[i]
        return out.sum(dim=-1)


##################################################
##################################################
# End of file