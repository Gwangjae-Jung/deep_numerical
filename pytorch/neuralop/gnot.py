from    typing              import  *
from    typing_extensions   import  Self

import  torch
from    torch   import  nn

from    ..utils     import  warn_redundant_arguments
from    ..layers    import  \
        GalerkinTypeEncoderBlockSelfAttention       as  GTencoderSA, \
        GalerkinTypeEncoderBlockCrossAttention      as  GTencoderCA




##################################################
##################################################
__all__ = ["GeneralNeuralOperatorTransformer"]


##################################################
##################################################
class GeneralNeuralOperatorTransformer(nn.Module):
    """## General Neural Operator Transformer (GNOT)
    ### A neural operator for multiscale tasks with general and multiple inputs
    -----
    ### Description
    """
    def __init__(
            self,

            # Inputs
            num_inputs_function:    int,
            num_inputs_geometry:    int,
            num_inputs_parameter:   int,
            
            # Outputs
            out_channels:   int,
            
            # GNOT layer
            n_layers:   int,
            n_heads:    int,
            
            # Else
            **kwargs,
        ) -> Self:
        super().__init__()
        warn_redundant_arguments(type(self), kwargs = kwargs)
        
        # Save some member variables for representation
        
        
        # Variables to instantiate subnetworks
        self.__num_inputs_function  = num_inputs_function
        self.__num_inputs_geometry  = num_inputs_geometry
        self.__num_inputs_parameter = num_inputs_parameter
        
        # Set the subnetworks
        self.gate:              nn.Module   = nn.Identity()
        self.encoder_function : Optional[nn.ModuleList]  = nn.ModuleList([]) if self.is_function_input   else None
        self.encoder_geometry : Optional[nn.ModuleList]  = nn.ModuleList([]) if self.is_geometry_input   else None
        self.encoder_parameter: Optional[nn.ModuleList]  = nn.ModuleList([]) if self.is_parameter_input  else None
        
        return
    
    
    @property
    def num_inputs_function(self) -> int:
        return self.__num_inputs_function
    @property
    def num_inputs_geometry(self) -> int:
        return self.__num_inputs_geometry
    @property
    def num_inputs_parameter(self) -> int:
        return self.__num_inputs_parameter
    @property
    def is_function_input(self) -> bool:
        return self.__num_inputs_function > 0
    @property
    def is_geometry_input(self) -> bool:
        return self.__num_inputs_geometry > 0
    @property
    def is_parameter_input(self) -> bool:
        return self.__num_inputs_parameter > 0
    
    
    def __set_encoders(self) -> None:
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X

    
    def __repr__(self) -> str:
        return \
                f"GeneralNeuralOperatorTransformer(\n" \
                f")"


##################################################
##################################################
# End of file