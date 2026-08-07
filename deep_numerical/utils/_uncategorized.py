from    typing      import  Sequence
import  torch

    
##################################################
##################################################
__all__:    list[int] = [
    'isometric_augmentation_2D',
    'isometric_augmentation_3D',
    'periodization',
    'positional_encoding',
]


##################################################
##################################################
def isometric_augmentation_2D(data: torch.Tensor) -> torch.Tensor:
    """Conducts 8 isometries on the input `data`.
    
    Given a batch of 2D images of shape `(B, N, N, C)`, this function returns the tensor of shape `(8*B, N, N, C)`, where
    * the first half is obtained by rotating `data` along the dimensions `(1, 2)` by 0, 90, 180, and 270 degrees, and
    * the last half is obtained by permuting the dimensions `(1, 2)`.
    """
    data = torch.cat([data.rot90(k, dims=(1,2)) for k in range(4)], dim=0)
    data = torch.cat([data, data.flip((2,))], dim=0)
    return data
    

def isometric_augmentation_3D(data: torch.Tensor) -> torch.Tensor:
    """Conducts 48 isometries on the input `data`.
    
    Given a batch of 3D images of shape `(B, N, N, N, C)`, this function returns the tensor of shape `(48*B, N, N, N, C)`, where
    * the first one eights (from the batch index `0` to `6*B`) are obtained by permuting the axes,
    * and the remaining are obtained by flipping across each axes.
    """
    from    itertools       import  permutations
    dims = (1, 2, 3)
    perms = permutations(dims)
    data = torch.cat([data.permute(p) for p in perms], dim=0)
    for d in dims:
        data = torch.cat([data, data.flip(d)], dim=0)
    return data


def periodization(X: torch.Tensor, axes=Sequence[int]) -> torch.Tensor:
    sl          = [Ellipsis for _ in range(X.ndim)]
    pad_width   = [(0, 0)   for _ in range(X.ndim)]
    for ax in axes:
        sl[ax] = slice(0, -1)
        pad_width[ax] = (0, 1)
    return torch.nn.functional.pad(X[*sl], pad_width, mode="wrap")


def positional_encoding(
        shape:      Sequence[int],
        enc_type:   str,
        dtype:      torch.dtype     = torch.float,
        device:     torch.device    = torch.device('cpu'),
    ) -> torch.Tensor:
    """
    Generate a positional encoding tensor of the given shape.

    *Remark*:
        The shape of the tensor should be given in the form of `(batch_size, *space, num_channels)`. 
        
    Arguments:
        `shape` (`Sequence[int]`): The shape of the output tensor.
        `enc_type` (`str`): The type of positional encoding to generate.
            * Supported types include: `'cartesian'`, `'radial'`, `'sinusoidal'`.
        `dtype` (`torch.dtype`, default: `torch.float`): The data type of the output tensor. Defaults to torch.float.
        `device` (`torch.device`, default: `torch.device('cpu')`): The device on which to create the tensor. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: The generated positional encoding tensor.
    """
    from    deep_numerical.utils.grid  import  space_grid
    X_ndim = len(shape)
    dimension = X_ndim-2
    _grid = space_grid(dimension, shape[1:-1], 1, -1, 'none', dtype=dtype, device=device)
    if enc_type == 'cartesian':
        pos = _grid
    elif enc_type == 'radial':
        pos = _grid.norm(p=2, dim=-1, keepdim=True)
    elif enc_type == 'sinusoidal':
        pos = []
        for d in range(dimension):
            pos.append( torch.sin(torch.pi*_grid[..., d]) )
            pos.append( torch.cos(torch.pi*_grid[..., d]) )
        pos = torch.stack(pos, dim=-1)
    else:
        raise ValueError(f"Unsupported encoding type: {enc_type}")
    pos = pos[None, ...].repeat(shape[0], *(1 for _ in range(X_ndim-1)))
    return pos


##################################################
##################################################
# End of file