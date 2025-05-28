# %%

from    typing              import  *

import  torch
from    torch               import  nn

from    .layers_hamiltonian_fno     import  *




##################################################
##################################################


class HamiltonianFNO(nn.Module):
    """## Hamiltonian Fourier Neural Operator
    -----
    
    ### Note
    
    This class is based on the usual `(B, C, ...)` convention.
    So it is required that the input is preprocessed to be aligned in the usual convention.
    
    -----
    
    ### Reference
    
    1. https://arxiv.org/abs/2010.08895
    2. ---

    -----
    ### Model architecture
    
    1. A tensor-type input data `X` of shape `(B, C, shape_of_domain)` is given, where\n
    
    2. (Lift)
    
    3. (Multiple passes of Fourier layers)
    
    4. (Projection)
        
    5. The projected tensor is returned as the result of the forward propagation.
    """
    
    def __init__(
                    self,
                    n_modes:                list[int] | tuple[int, ...],
                    
                    in_channels:            int,
                    hidden_channels:        int,
                    out_channels:           int,
                    
                    n_layers:               int,
                    
                    # is_temporal:            bool,
                    
                    lifting_channels:       int = 256,
                    projection_channels:    int = 256,
                    
                    activation:             object = nn.LeakyReLU(),
                    
                    **kwargs
        ) -> None:
        """## The initializer of the class `HamiltonianFNO`
        -----
    
        ### Note
        
        This class is based on the `(B, ..., C)` convention, rather than the usual `(B, C, ...)` convention.
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
        
        # self.is_temporal        = is_temporal
        
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
                                    HamiltonianLayer(
                                                    n_modes         = n_modes,
                                                    hidden_channels = hidden_channels,
                                                    activation      = activation,
                                                    # is_temporal     = is_temporal
                                                )
                                    for _ in range(n_layers)
                                ]
        self.network_fourier    = nn.ModuleList(_network_fourier)
                        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.network_lifting.forward(X)
        for cnt in range(len(self.network_fourier)):
            X = self.network_fourier[cnt].forward(X)
        X = self.network_projection.forward(X)
        return X
    
    
    def reverse(self, X: torch.Tensor) -> torch.Tensor:
        X = self.network_lifting.forward(X)
        for cnt in range(len(self.network_fourier)):
            X = self.network_fourier[cnt].reverse(X)
        X = self.network_projection.forward(X)
        return X
    
    
    def __name__(self, full: bool = True) -> str:
        if full:
            return f"Hamiltonian Fourier neural operator ({self.dim_domain}D)"
        else:
            return f"Hamiltonian FNO ({self.dim_domain}D)"




##################################################
##################################################

# %%
