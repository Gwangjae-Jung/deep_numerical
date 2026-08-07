from    typing      import  Literal, Sequence, Optional
import  torch


__all__ = ['positional_encoding']


##################################################
##################################################
def positional_encoding(
        shape:      Sequence[int],
        enc_type:   Literal['cartesian', 'radial', 'sinusoidal'],
        dtype:      Optional[torch.dtype]   = None,
        device:     Optional[torch.device]  = None,
        reduce:     bool                    = False,
    ) -> torch.Tensor:
    """
    Generate a positional encoding tensor of the given shape.

    *Remark*:
        The shape of the tensor should be given in the form of `(batch_size, *space, num_channels)`. 
        
    Arguments:
        `shape` (`Sequence[int]`): The shape of the output tensor.
        `enc_type` (`str`): The type of positional encoding to generate.
            * Supported types include: `'cartesian'`, `'radial'`, `'sinusoidal'`.
        `dtype` (`Optional[torch.dtype]`, default: `None`): The data type of the output tensor. Defaults to `None`.
        `device` (`Optional[torch.device]`, default: `None`): The device on which to create the tensor. Defaults to `None`.
        `reduce` (`bool`, default: `False`): Whether to reduce the positional encoding tensor along the batch dimension. Note that `out[i]==out[j]` for all `i` and `j`, where `out` is the output tensor. Defaults to `False`.
        
    Returns:
        `torch.Tensor`: The generated positional encoding tensor.
    """
    from    deep_numerical.utils.grid  import  space_grid
    if dtype is None:   dtype = torch.get_default_dtype()
    if device is None:  device = torch.get_default_device()
    X_ndim = len(shape)
    dimension = X_ndim-2    # Remove the batch and channel dimensions
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
    
    if not reduce:
        pos = pos[None, ...].repeat(shape[0], *(1 for _ in range(X_ndim-1)))
    return pos


##################################################
##################################################
# End of file