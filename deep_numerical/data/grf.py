import  warnings
from    typing      import  *

import  numpy       as      np
import  torch

from    ..utils_main    import  Objects, ArrayData


__all__ = ['sample_grf']

def grf(
        grid_size:      Objects[int],
        domain_size:    Objects[float]  = [1.0, 1.0],
        length_scale:   Objects[float]  = 1.0,
    ) -> np.ndarray:
    """Sampling from a Gaussian random field.
    
    Main contribution by ChatGPT.
    """
    # Initialize
    if not isinstance(grid_size, Iterable):
        grid_size = (grid_size,)
    if not isinstance(domain_size, Iterable):
        domain_size = (domain_size,)
    
    # Create a grid of wave numbers
    k2 = np.meshgrid(
                *[
                    np.fft.fftfreq(_grid, _domain / _grid)
                    for _grid, _domain in zip(grid_size, domain_size)
                ], indexing = 'ij'
            )
    k2 = np.stack(k2, axis = -1)
    k2 = np.linalg.norm(k2, ord = 2, axis = -1) ** 2
    
    # Define the spectral density function (example: exponential decay)
    S_k = np.exp(-0.5 * k2 * (length_scale ** 2))  # Spectral density

    # Generate random complex numbers with the given spectral density
    random_phases = np.exp(2j * np.pi * np.random.rand(*grid_size))
    Z_k = np.sqrt(S_k) * random_phases

    # Perform the inverse FFT to get the GRF in spatial domain
    Z = np.fft.ifftn(Z_k).real
    return Z





def sample_grf(
        grid_size:      Objects[int],
        domain_size:    Objects[float]  = [1.0, 1.0],
        length_scale:   Objects[float]  = 1.0,
        sample_size:    int             = 1,
        array_type:     str             = "numpy",
    ) -> ArrayData:
    if not isinstance(length_scale, Iterable):
        length_scale = [length_scale for _ in range(sample_size)]
    
    Z = np.stack([grf(grid_size, domain_size, _len_sc) for _len_sc in length_scale])
    array_type = array_type.lower()
    if array_type in ("numpy", "np"):
        return Z
    elif array_type in ("torch"):
        return torch.from_numpy(Z).type(torch.float)
    else:
        warnings.warn(
            "'array_type' should be either 'numpy' or 'torch'.",
            UserWarning
        )
        return Z
    

if __name__ == "__main__":
    import  matplotlib.pyplot   as  plt
    
    DOMAIN_LENGTH   = 5.0
    NUM_GRIDS       = 256
    LENGTH_SCALE    = 0.1
    
    x = np.linspace(0, DOMAIN_LENGTH, NUM_GRIDS)
    y = grf(NUM_GRIDS, DOMAIN_LENGTH, LENGTH_SCALE)
    
    if x.ndim == 1:
        plt.plot(x, y, c = 'r')
        plt.title(f"GRF 1D (length scale: {LENGTH_SCALE:.2e})")
    elif x.ndim == 3:
        plt.title(f"GRF 2D (length scale: {LENGTH_SCALE:.2e})")
        plt.imshow(y)
        plt.xticks([x[:, 0].min(), x[:, 0].max()])
        plt.yticks([x[:, 1].min(), x[:, 1].max()])
        
    plt.show()
