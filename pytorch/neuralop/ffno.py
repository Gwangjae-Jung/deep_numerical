from    typing              import  Sequence
from    typing_extensions   import  Self

import  torch
from    torch           import  nn

from    ..layers        import  MLP, FactorizedFourierLayer
from    ..utils         import  get_activation, warn_redundant_arguments


##################################################
##################################################
__all__: list[str] = [
    "FactorizedFourierNeuralOperator", "FFNO"
]


##################################################
##################################################
class FactorizedFourierNeuralOperator(nn.Module):
    """## Factorized Fourier Neural Operator (FFNO)
    
    -----
    ### Description
    The FFNO is a neural operator modelling integral operators with a skip connection, using the dimensionwise Fourier transform.
    
    Reference: https://openreview.net/forum?id=tmIiMPl4IPa
    """
    
    def __init__(
            self,
            n_modes:            Sequence[int],

            in_channels:        int,
            hidden_channels:    int,
            out_channels:       int,

            lift_layer:         Sequence[int]   = [256],
            n_layers:           int             = 4,
            project_layer:      Sequence[int]   = [256],

            activation_name:    str                 = "relu",
            activation_kwargs:  dict[str, object]   = {},
            
            **kwargs,
        ) -> Self:
        """## The initializer of the class `FFNO`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                * The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
                * Note that the length of `n_modes` is the dimension of the domain.
                    
            `in_channels` (`int`): The number of the input channels.
            `hidden_channels` (`int`): The number of the hidden channels.
            `out_channels` (`int`): The number of the output channels.
            
            `lift_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`): The number of hidden layers.
            `project_layer` (`Sequential[int]`, default: `(256)`): The numbers of channels inside the projection layer.
        """
        super().__init__()
        warn_redundant_arguments(type(self), kwargs)
        
        # Check the argument validity
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise RuntimeError(f"'k_max[{cnt}]' is chosen {item}, which is not positive.")
        
        # Save some member variables for representation
        self.__dim_domain = len(n_modes)
        
        # Define the subnetworks
        ## Lift
        self.network_lift   = MLP(
            [in_channels, *lift_layer, hidden_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
        ## Hidden layers
        self.network_hidden: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers <= 0:
            self.network_hidden.append(torch.nn.Identity())
        else:
            __fl_kwargs = {'n_modes': n_modes, 'in_channels': hidden_channels}
            for _ in range(n_layers):
                self.network_hidden.append(FactorizedFourierLayer(**__fl_kwargs))
        ## Projection
        self.network_projection = MLP(
            [hidden_channels, *project_layer, out_channels],
            activation_name     = activation_name,
            activation_kwargs   = activation_kwargs
        )
                        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`): The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        X = self.network_lift(X)
        X = self.network_hidden(X)
        X = self.network_projection(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain


##################################################
##################################################
FFNO = FactorizedFourierNeuralOperator


##################################################
##################################################
# End of file