from    typing      import  Sequence, Dict, Tuple, Optional
import  torch
from    deep_numerical.utils            import  type_as_real
from    deep_numerical.neural           import  BaseModule
from    deep_numerical.neural.utils     import  Activations
from    deep_numerical.neural.layer     import  MLP, FactorizedFourierLayer


__all__: list[str] = ["FactorizedFourierNeuralOperator", "FactorizedFNO", "FFNO"]


##################################################
##################################################
class FactorizedFourierNeuralOperator(BaseModule):
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
            bias:               bool            = True,

            activation_name:    Activations         = "relu",
            activation_kwargs:  Dict[str, object]   = {},
            
            dtype:              Optional[torch.dtype] = None,
        ) -> None:
        """## The initializer of the class `FFNO`
        
        Arguments:
            `n_modes` (`Sequence[int]`):
                The maximum degree of the Fourier modes to be preserved. To be precise, after performing the FFT, for the `i`-th Fourier transform, only the modes in `[-n_modes[i], +n_modes[i]]` will be preserved.
                Note that the length of `n_modes` is the dimension of the domain.
            `in_channels` (`int`):
                The number of the input channels.
            `hidden_channels` (`int`):
                The number of the hidden channels.
            `out_channels` (`int`):
                The number of the output channels.
            `bias` (`bool`, default: `True`):
                Whether to use bias terms in the subnetworks.
            `lift_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the lift layer.
            `n_layers` (`int`, default: `4`):
                The number of hidden layers.
            `project_layer` (`Sequence[int]`, default: `(256,)`):
                The numbers of channels inside the projection layer.
            `activation_name` (`str`, default: `"relu"`):
                The name of the activation function used in the MLPs.
            `activation_kwargs` (`Dict[str, object]`, default: `{}`):
                The keyword arguments for the activation function.
            `dtype` (`Optional[torch.dtype]`, default: `None`):
                The data type of the parameters.
                If `None`, the default data type of `torch` will be used.
        """
        super().__init__()
        
        # Save some member variables for representation
        self.__dim_domain   = len(n_modes)
        self.__n_modes      = tuple(n_modes)
        self.__dtype = dtype = type_as_real(dtype)
        
        # Define the subnetworks
        config_common   = {'bias': bias, 'activation_name': activation_name, 'activation_kwargs': activation_kwargs, 'dtype': dtype}
        config_fourier  = {'n_modes': n_modes, 'in_channels': hidden_channels}
        ## Lift
        self.network_lift = MLP([in_channels, *lift_layer, hidden_channels], **config_common)
        ## Hidden layers
        self.network_hidden: torch.nn.Sequential = torch.nn.Sequential()
        if n_layers<=0:
            import  warnings
            warnings.warn(f"The number of hidden layers is set to {n_layers}. Thus, the hidden layers are bypassed.", RuntimeWarning)
        for _ in range(n_layers):
            self.network_hidden.append(FactorizedFourierLayer(**config_common, **config_fourier))
        ## Projection
        self.network_projection = MLP([hidden_channels, *project_layer, out_channels], **config_common)
                        
        return None
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `X` (`torch.Tensor`): The input tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        
        Returns:
            `torch.Tensor`: The output tensor of shape `(B, s_1, ..., s_d, C)`, where `B` is the batch size, `s_i` is the size of the `i`-th dimension of the domain, and `C` is the number of channels.
        """
        X = self.network_lift.forward(X)
        X = self.network_hidden.forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    @property
    def dim_domain(self) -> int:
        return self.__dim_domain
    @property
    def n_modes(self) -> Tuple[int, ...]:
        return self.__n_modes
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype


FactorizedFNO = FactorizedFourierNeuralOperator
    

##################################################
##################################################
# End of file