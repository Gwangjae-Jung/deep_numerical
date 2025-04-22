from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers        import  LiftProject, PDEiFourierLayer
from    ..utils         import  *




##################################################
##################################################


class HyperbolicFNO(nn.Module):
    """## Hyperbolic Fourier Neural Operator (HFNO)
    ### Integral operator via discrete Fourier transform
    -----
    The FNO is a neural operator modelling integral operators with a skip connection, using the Fourier transform.
    
    Reference: https://arxiv.org/abs/2010.08895

    -----
    ### Model architecture
    
    See the docstring of the class `FNO`.
    """
    
    def __init__(
                    self,
                    n_modes:                ListData[int],
                    
                    in_channels:            int,
                    hidden_channels:        int,
                    out_channels:           int,
                    
                    n_layers:               int,
                    
                    n_modes_as:             str = "count",
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256,
                    
                    lite:                   bool = False
        ) -> None:
        """## The initializer of the class `HyperbolicFNO`
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
                
        self.network_lifting    = LiftProject(
                                                in_channels     = in_channels,
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
                                    PDEiFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "gelu",
                                                    n_modes_as      = n_modes_as,
                                                    lite            = lite
                                                )
                                    for _ in range(n_layers - 1)
                                ] + [
                                    PDEiFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "identity",
                                                    n_modes_as      = n_modes_as,
                                                    lite            = lite
                                                )
                                ]
        self.network_fourier    = nn.Sequential(*_network_fourier)
                        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.network_lifting(X)
        X = self.network_fourier(X)
        X = self.network_projection(X)
        return X
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Hyperbolic Fourier neural operator ({self.dim_domain}D)"
        else:
            return f"HFNO ({self.dim_domain}D)"




##################################################
##################################################


class ParabolicFNO(nn.Module):
    """## Parabolic Fourier Neural Operator (PFNO)
    ### Integral operator via discrete Fourier transform
    -----
    The FNO is a neural operator modelling integral operators with a skip connection, using the Fourier transform.
    
    Reference: https://arxiv.org/abs/2010.08895

    -----
    ### Model architecture
    
    See the docstring of the class `FNO`.
    """
    
    def __init__(
                    self,
                    n_modes:                ListData[int],
                    
                    in_channels:            int,
                    hidden_channels:        int,
                    out_channels:           int,
                    
                    n_layers:               int,
                    
                    n_modes_as:             str = "count",
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256
        ) -> None:
        """## The initializer of the class `ParabolicFNO`
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
                
        self.network_lifting    = LiftProject(
                                                in_channels     = in_channels,
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
                                    PDEiFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "gelu",
                                                    n_modes_as      = n_modes_as
                                                )
                                    for _ in range(n_layers - 1)
                                ] + [
                                    PDEiFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "identity",
                                                    n_modes_as      = n_modes_as
                                                )
                                ]
        self.network_fourier    = nn.Sequential(*_network_fourier)
                        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.network_lifting(X)
        X = self.network_fourier(X)
        X = self.network_projection(X)
        return X
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Parabolic Fourier neural operator ({self.dim_domain}D)"
        else:
            return f"PFNO ({self.dim_domain}D)"




##################################################
##################################################
