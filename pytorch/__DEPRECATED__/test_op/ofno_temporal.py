from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers        import  LiftProject, FourierLayer
from    ..utils         import  *




##################################################
##################################################


class TempOFNO(nn.Module):
    """## Temporal ordinary Fourier Neural Operator (TempOFNO)
    ### Integral operator via discrete Fourier transform
    -----
    The FNO is a neural operator modelling integral operators with a skip connection, using the Fourier transform.
    
    Reference: https://arxiv.org/abs/2010.08895

    -----
    ### Model architecture
    
    As for the general architecture of the Fourier Neural Operator the docstring of the class `FNO`.
    To solve a time-dependent problem, the `TemporalFNO.forward()` method merges the channel dimension and the time dimension.
    """
    
    def __init__(
                    self,
                    n_modes:                ListData[int],
                    
                    in_channels:            int,
                    hidden_channels:        int,
                    out_channels:           int,
                    
                    n_input_times:          int,
                    
                    n_layers:               int,
                    
                    n_modes_as:             str = "count",
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256,
                    
                    **kwargs
        ) -> None:
        """## The initializer of the class `TemporalFNO`
        ----
        ### Arguments
        
        See the docstring of the class `FNO`.
        """
        
        super().__init__()
        
        for cnt, item in enumerate(n_modes, 1):
            if not (type(item) == int and item > 0):
                raise RuntimeError(f"'k_max[{cnt}]' is chosen {item}, which is not positive.")
        
        self.n_modes            = list(n_modes)
        self.dim_domain         = len(n_modes)
        
        self.in_channels        = in_channels
        self.hidden_channels    = hidden_channels
        self.out_channels       = out_channels
        
        self.n_input_times      = n_input_times
                
        self.network_lifting    = LiftProject(
                                                in_channels     = in_channels * n_input_times,
                                                hidden_channels = lifting_channels,
                                                out_channels    = hidden_channels,
                                                dim_domain      = self.dim_domain
                                    )
        self.network_projection = LiftProject(
                                                in_channels     = hidden_channels,
                                                hidden_channels = projection_channels,
                                                out_channels    = out_channels,
                                                dim_domain      = self.dim_domain
                                    )
        
        if n_layers <= 0:
            _network_fourier = None
        else:
            _network_fourier    = [
                                    FourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "gelu",
                                                    n_modes_as      = n_modes_as
                                                )
                                    for _ in range(n_layers - 1)
                                ] + [
                                    FourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "identity",
                                                    n_modes_as      = n_modes_as
                                                )
                                ]
        self.network_fourier    = nn.Sequential(*_network_fourier)
                        
        return;
    
    
    def forward(self, Xs: torch.Tensor) -> torch.Tensor:
        """
        Unlike in the ordinary FNO, `Xs` consists of the `X`-values at multiple temporal points.
        """
        Xs = self.network_lifting(Xs.flatten(start_dim = 1, end_dim = 2))    # (channel, time) --> (channel * time)
        Xs = self.network_fourier(Xs)
        Xs = self.network_projection(Xs)
        return Xs
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Ordinary Fourier neural operator ({self.dim_domain}D, temporal)"
        else:
            return f"OFNO ({self.dim_domain}D, temporal)"




##################################################
##################################################
