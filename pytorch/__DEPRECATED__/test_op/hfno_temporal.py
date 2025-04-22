from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers        import  LiftProject
from    ..test_layers   import  OrbitalSpectralConv
from    ..utils         import  *




##################################################
##################################################


class TempHFNO(nn.Module):
    """## Temporal hyperbolic Fourier Neural Operator (TempHFNO)
    ### Integral operator via discrete Fourier transform
    -----
    The FNO is a neural operator modelling integral operators with a skip connection, using the Fourier transform.
    
    Reference: https://arxiv.org/abs/2010.08895

    -----
    ### Model architecture
    
    As for the general architecture of the Fourier Neural Operator the docstring of the class `FNO`.
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
                    
                    **kwargs
        ) -> None:
        """## The initializer of the class `TempHFNO`
        ----
        ### Arguments
        
        See the docstring of the class `FNO`.
        
        1. `time_step_size` (`float`)
            The size of the time step.
        
        2. `output_order` (`int`)
            The order of the Euler approximation.
            Currently, only 1 and 2 are supported.
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
        
        _network_spectral   = []
        if n_layers > 0:
            for _ in range(n_layers - 1):
                _network_spectral += [
                    OrbitalSpectralConv(
                        n_modes         = n_modes,
                        hidden_channels = hidden_channels,
                        n_modes_as      = n_modes_as
                    ),
                    getattr(nn, TORCH_ACTIVATION_DICT["gelu"])()
                ]
            _network_spectral += [
                    OrbitalSpectralConv(
                        n_modes         = n_modes,
                        hidden_channels = hidden_channels,
                        n_modes_as      = n_modes_as
                    )
                ]
        self.network_spectral   = nn.Sequential(*_network_spectral)
                        
        return;
    
    
    def forward(self, X_prev_curr: torch.Tensor) -> torch.Tensor:
        """Compute `X_next` from the Euler approximation.
        -----
        ### Propagation
        
        In a strictly reversible manner.
        
        -----
        ### Remark
        
        `X_prev_curr` is a tensor of the shape `(batch, channel, {prev time, curr time}, __domain__)`.
        So the tensor `X_prev` which corresponds to the previous time is given by `X_prec_curr[:, :, 0]`, and the tensor `X_curr` which corresponds to the current time is given by `X_prev_curr[:, :, 1]`.
        """
        
        # Lift the split tensors to the representation space
        X_prev = self.network_lifting.forward(X_prev_curr[:, :, 0])
        X_curr = self.network_lifting.forward(X_prev_curr[:, :, 1])
        Y_curr = X_curr
        
        # Transform in the representation space
        Y_curr = self.network_spectral.forward(Y_curr)
        
        # Project the updated current representation to the physical space
        return self.network_projection.forward((2 * X_curr - X_prev) + Y_curr)
    
    
    def reverse(self, X_curr_next: torch.Tensor) -> torch.Tensor:
        """Compute `X_prev` from the Euler approximation.
        -----
        ### Propagation
        
        In a strictly reversible manner.
        
        -----
        ### Remark
        
        `X_curr_next` is a tensor of the shape `(batch, channel, {curr time, next time}, __domain__)`.
        So the tensor `X_curr` which corresponds to the current time is given by `X_curr_next[:, :, 0]`, and the tensor `X_next` which corresponds to the next time is given by `X_curr_next[:, :, 1]`.
        """
        
        # Lift the split tensors to the representation space
        X_curr = self.network_lifting.forward(X_curr_next[:, :, 0])
        X_next = self.network_lifting.forward(X_curr_next[:, :, 1])
        Y_curr = X_curr
        
        # Transform in the representation space
        Y_curr = self.network_spectral.forward(Y_curr)
        
        # Project the updated current representation to the physical space
        return self.network_projection.forward((2 * X_curr - X_next) + Y_curr)
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Hyperbolic Fourier neural operator ({self.dim_domain}D, temporal)"
        else:
            return f"HFNO ({self.dim_domain}D, temporal)"




##################################################
##################################################
