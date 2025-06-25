from    typing      import  Sequence
import  torch

    
##################################################
##################################################
__all__:    list[int] = ['periodization', 'positional_encoding']


##################################################
##################################################
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
    from    deep_numerical.utils._grid  import  space_grid
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