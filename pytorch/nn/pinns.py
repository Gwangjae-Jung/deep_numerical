from    typing                  import  Sequence
from    typing_extensions       import  Self

import  torch

from    ..layers._base_module   import  BaseModule


##################################################
##################################################
__all__: list[str] = ['SeparableNN']


##################################################
##################################################
class SeparableNN(BaseModule):
    def __init__(
            self,
            networks:   Sequence[torch.nn.Module],
        ) -> Self:
        """## Separable neural network - Dimensionwise forward propagation for efficient approximation
        
        -----
        ### Reference
        [1]: SPINN
        """
        super().__init__()
        self.__dimension = len(networks)
        self.networks   = torch.nn.ModuleList(networks)
        return
    
    
    def forward(self, coordinates: Sequence[torch.Tensor]) -> torch.Tensor:
        """## Forward propagation of a separable neural network
        
        -----
        ### Arguments
        * `coordinates` (`Sequence[torch.Tensor]`)
            * A sequence of 1-dimensional tensors, where each entry is sampled from the corresponding axis of the input space.
        """
        _outputs: list[torch.Tensor] = [
            md.forward(coord.flatten()[:, None]).reshape(*self.__newshape(coord.size(0), cnt), -1)
            for cnt, (md, coord) in enumerate(zip(self.networks, coordinates))
        ]
        out = _outputs[0]
        for idx in range(1, self.__dimension):
            out = out * _outputs[idx]
        return out.sum(dim=-1, keepdim=False)
    

    def __newshape(self, size: int, index: int) -> tuple[int]:
        ret = [1 for _ in range(self.__dimension)]
        ret[index] = size
        return tuple(ret)


##################################################
##################################################
# End of file