from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers        import  LiftProject, PDEiTemporalFourierLayer
from    ..utils         import  *




##################################################
##################################################


class TemporalParabolicFNO(nn.Module):
    """## Temporal parabolic Fourier Neural Operator (TempPFNO)
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
                    
                    time_step_size:         float,
                    
                    n_layers:               int,
                    
                    n_modes_as:             str = "count",
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256,
                    
                    **kwargs
        ) -> None:
        """## The initializer of the class `TemporalParabolicFNO`
        ----
        ### Arguments
        
        See the docstring of the class `FNO`.
        
        1. `time_step_size` (`float`)
            The size of the time step.
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
        
        self.time_step_size     = time_step_size
        
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
                                    PDEiTemporalFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "gelu",
                                                    n_modes_as      = n_modes_as
                                                )
                                    for _ in range(n_layers - 1)
                                ] + [
                                    PDEiTemporalFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "identity",
                                                    n_modes_as      = n_modes_as
                                                )
                                ]
        self.network_fourier    = nn.ModuleList(_network_fourier)
                        
        return;
    
    
    def forward(self, X_prev_curr: torch.Tensor) -> torch.Tensor:
        """Compute `X_next` using the first order Euler approximation.
        
        -----
        ### Remark
        
        `X_prev_curr` is a tensor of the shape `(batch, channel, {prev time, curr time}, __domain__)`.
        So the tensor `X_prev` which corresponds to the previous time is given by `X_prec_curr[:, :, 0]`, and the tensor `X_curr` which corresponds to the current time is given by `X_prev_curr[:, :, 1]`.
        """
        
        X_prev = X_prev_curr[:, :, 0]           # (B, d_a, s_1, ..., s_d)
        X_curr = X_prev_curr[:, :, 1]           # (B, d_a, s_1, ..., s_d)
        
        Y_prev = self.network_lifting(X_prev)   # (B, d_v, s_1, ..., s_d)
        Y_curr = self.network_lifting(X_curr)   # (B, d_v, s_1, ..., s_d)
        
        for submodel in self.network_fourier:
            Y_curr = submodel(Y_prev, Y_curr)
        Y_curr = self.network_projection(Y_curr)
        
        return ( X_prev + (self.time_step_size * Y_curr) )
    
    
    def reverse(self, X_curr_next: torch.Tensor) -> torch.Tensor:
        """Compute `X_prev` using the first order Euler approximation.
        
        -----
        ### Remark
        
        `X_curr_next` is a tensor of the shape `(batch, channel, {prev time, curr time}, __domain__)`.
        So the tensor `X_curr` which corresponds to the current time is given by `X_curr_next[:, :, 0]`, and the tensor `X_next` which corresponds to the next time is given by `X_curr_next[:, :, 1]`.
        """
        
        X_curr = X_curr_next[:, :, 0]           # (B, C, s_1, ..., s_d)
        X_next = X_curr_next[:, :, 1]           # (B, C, s_1, ..., s_d)
        
        Y_curr = self.network_lifting(X_curr)   # (B, d_v, s_1, ..., s_d)
        Y_next = self.network_lifting(X_next)   # (B, d_v, s_1, ..., s_d)
        
        for submodel in self.network_fourier:
            Y_curr = submodel(Y_next, Y_curr)
        Y_curr = self.network_projection(Y_curr)
        
        return ( X_next - (self.time_step_size * Y_curr) )
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Parabolic Fourier neural operator ({self.dim_domain}D, temporal)"
        else:
            return f"PFNO ({self.dim_domain}D, temporal)"


##################################################
##################################################
