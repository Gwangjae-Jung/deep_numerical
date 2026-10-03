from typing import Literal, Optional, Sequence
import torch

from   deep_numerical.utils.grid import space_grid


__all__: list[str] = ['positional_encoding']


##################################################
def positional_encoding(
        shape:    Sequence[int],
        enc_type: Literal['cartesian', 'radial', 'sinusoidal'],
        dtype:    Optional[torch.dtype]  = None,
        device:   Optional[torch.device] = None,
        reduce:   bool                   = False,
    ) -> torch.Tensor:
    """Generates a positional encoding tensor of the given shape.

    ## Description
    Constructs a coordinate-based positional encoding across spatial dimensions in cartesian, radial, or sinusoidal representation.
    The input shape should follow `(batch_size, *space, num_channels)`.

    ## Arguments
    `shape` (`Sequence[int]`): The shape of the output tensor `(batch_size, *space, num_channels)`.
    `enc_type` (`Literal['cartesian', 'radial', 'sinusoidal']`): The type of positional encoding.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Data type of the output tensor.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.
    `reduce` (`bool`, default: `False`): Whether to omit broadcasting along the batch dimension.

    ## Returns
    `torch.Tensor`: The generated positional encoding tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    X_ndim:    int = len(shape)
    dimension: int = X_ndim - 2  # Remove batch and channel dimensions
    _grid: torch.Tensor = space_grid(dimension, shape[1:-1], 1, -1, 'none', dtype=dtype, device=device)
    if enc_type == 'cartesian':
        pos = _grid
    elif enc_type == 'radial':
        pos = _grid.norm(p=2, dim=-1, keepdim=True)
    elif enc_type == 'sinusoidal':
        pos_list: list[torch.Tensor] = []
        for d in range(dimension):
            pos_list.append(torch.sin(torch.pi * _grid[..., d]))
            pos_list.append(torch.cos(torch.pi * _grid[..., d]))
        pos = torch.stack(pos_list, dim=-1)
    else:
        raise ValueError(f"Unsupported encoding type: {enc_type}")

    if not reduce:
        pos = pos[None, ...].repeat(shape[0], *(1 for _ in range(X_ndim - 1)))
    return pos


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()