from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers        import  LiftProject, FourierLayer, TemporalFourierLayer
from    ..utils         import  *




##################################################
##################################################


class TemporalFNO(nn.Module):
    """## Temporal Fourier Neural Operator (TempFNO)
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
        """## The initializer of the class `ReversibleFNO`
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
            return f"Fourier neural operator ({self.dim_domain}D, temporal)"
        else:
            return f"FNO ({self.dim_domain}D, temporal)"




##################################################
##################################################


class ReversibleTemporalFNO(nn.Module):
    """## Reversible temporal Fourier Neural Operator (RevTempFNO)
    ### Integral operator via discrete Fourier transform
    -----
    The FNO is a neural operator modelling integral operators with a skip connection, using the Fourier transform.
    
    Reference: https://arxiv.org/abs/2010.08895

    -----
    ### Model architecture
    
    As for the general architecture of the Fourier Neural Operator the docstring of the class `FNO`.
    To solve a time-dependent problem, the `ReversibleTemporalFNO.forward()` method maps the input tensor `X_prev_curr` to the ambient space, and transforms `X_curr == X_prev_curr[:, :, 1]` via the frequency domain, and adds this with `X_prev == X_prev_curr[:, :, 0]`.
    Hence, RevTempFNO has a reversible structure.
    """
    
    def __init__(
                    self,
                    n_modes:                ListData[int],
                    
                    in_channels:            int,
                    hidden_channels:        int,
                    out_channels:           int,
                    
                    time_step_size:         float,
                    output_order:           int,
                    
                    n_layers:               int,
                    
                    n_modes_as:             str = "count",
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256,
                    
                    **kwargs
        ) -> None:
        """## The initializer of the class `ReversibleTemporalFNO`
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
        
        if output_order in [1, 2]:
            self.output_order   = output_order
        else:
            raise NotImplementedError(f"Currently, the model is implemented for the first and second derivatives. (Passed 'output_order': {output_order})")
        
        if n_layers <= 0:
            _network_fourier = None
        else:
            _network_fourier    = [
                                    TemporalFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "gelu",
                                                    n_modes_as      = n_modes_as
                                                )
                                    for _ in range(n_layers - 1)
                                ] + [
                                    TemporalFourierLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = "identity",
                                                    n_modes_as      = n_modes_as
                                                )
                                ]
        self.network_fourier    = nn.ModuleList(_network_fourier)
                        
        return;
    
    
    def forward(self, X_prev_curr: torch.Tensor) -> torch.Tensor:
        """Compute `X_next` from the Euler approximation.
        
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
        
        if self.output_order == 1:
            return ( X_prev + (self.time_step_size * Y_curr) )
        else:
            return ( (2 * X_curr - X_prev) + (self.time_step_size ** 2) * Y_curr )
    
    
    def reverse(self, X_curr_next: torch.Tensor) -> torch.Tensor:
        """Compute `X_next` from the Euler approximation.
        
        -----
        ### Remark
        
        `X_curr_next` is a tensor of the shape `(batch, channel, {curr time, next time}, __domain__)`.
        So the tensor `X_curr` which corresponds to the current time is given by `X_curr_next[:, :, 0]`, and the tensor `X_next` which corresponds to the next time is given by `X_curr_next[:, :, 1]`.
        """
        X_curr = X_curr_next[:, :, 0]           # (B, d_a, s_1, ..., s_d)
        X_next = X_curr_next[:, :, 1]           # (B, d_a, s_1, ..., s_d)
        
        Y_curr = self.network_lifting(X_curr)   # (B, d_v, s_1, ..., s_d)
        Y_next = self.network_lifting(Y_next)   # (B, d_v, s_1, ..., s_d)
        
        for submodel in self.network_fourier:
            Y_curr = submodel(Y_next, Y_curr)
        Y_curr = self.network_projection(Y_curr)
        
        if self.output_order == 1:
            return ( X_next - (self.time_step_size * Y_curr) )
        else:
            return ( (2 * X_curr - X_next) + (self.time_step_size ** 2) * Y_curr )
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Reversible Fourier neural operator ({self.dim_domain}D, temporal)"
        else:
            return f"RevFNO ({self.dim_domain}D, temporal)"




##################################################
##################################################
