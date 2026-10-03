from   typing import Sequence
import torch


__all__: list[str] = [
    'isometric_augmentation_2D',
    'isometric_augmentation_3D',
    'periodization',
    'positional_encoding',
]


##################################################
def isometric_augmentation_2D(data: torch.Tensor) -> torch.Tensor:
    """Conducts 8 isometries on the input `data`.

    ## Description
    Given a batch of 2D images of shape `(B, N, N, C)`, returns a tensor of shape `(8*B, N, N, C)` obtained by 4 rotations and reflections.

    ## Arguments
    `data` (`torch.Tensor`): Input 2D tensor of shape `(B, N, N, C)`.

    ## Returns
    `torch.Tensor`: Augmented tensor of shape `(8*B, N, N, C)`.
    """
    rotated:   torch.Tensor = torch.cat([data.rot90(k, dims=(1, 2)) for k in range(4)], dim=0)
    augmented: torch.Tensor = torch.cat([rotated, rotated.flip((2,))], dim=0)
    return augmented


def isometric_augmentation_3D(data: torch.Tensor) -> torch.Tensor:
    """Conducts 48 isometries on the input `data`.

    ## Description
    Given a batch of 3D images of shape `(B, N, N, N, C)`, returns a tensor of shape `(48*B, N, N, N, C)` obtained by permutations and axis flips.

    ## Arguments
    `data` (`torch.Tensor`): Input 3D tensor of shape `(B, N, N, N, C)`.

    ## Returns
    `torch.Tensor`: Augmented tensor of shape `(48*B, N, N, N, C)`.
    """
    from itertools import permutations
    dims: tuple[int, int, int] = (1, 2, 3)
    perms = permutations(dims)
    augmented: torch.Tensor = torch.cat([data.permute(p) for p in perms], dim=0)
    for d in dims:
        augmented = torch.cat([augmented, augmented.flip(d)], dim=0)
    return augmented


def periodization(X: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    """Applies periodic boundary padding along specified axes.

    ## Description
    Trims the last boundary element along the given `axes` and then applies circular wrapping (`mode="wrap"`) to enforce periodic boundary conditions on the tensor `X`.

    ## Arguments
    `X` (`torch.Tensor`): Input data tensor.
    `axes` (`Sequence[int]`): Sequence of dimension axes along which periodization is applied.

    ## Returns
    `torch.Tensor`: Periodized tensor.
    """
    sl:        list[object]         = [Ellipsis for _ in range(X.ndim)]
    pad_width: list[tuple[int, int]] = [(0, 0) for _ in range(X.ndim)]
    for ax in axes:
        sl[ax] = slice(0, -1)
        pad_width[ax] = (0, 1)
    return torch.nn.functional.pad(X[*sl], pad_width, mode="wrap")


def positional_encoding(
    shape:    Sequence[int],
    enc_type: str,
    dtype:    torch.dtype  = torch.float,
    device:   torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Generate a positional encoding tensor of the given shape.

    ## Description
    Generates a positional encoding tensor for the given grid shape using Cartesian, radial, or sinusoidal encoding.

    ## Arguments
    `shape` (`Sequence[int]`): The shape of the output tensor `(batch_size, *space, num_channels)`.
    `enc_type` (`str`): The type of positional encoding (`'cartesian'`, `'radial'`, or `'sinusoidal'`).
    `dtype` (`torch.dtype`, default: `torch.float`): The data type of the output tensor.
    `device` (`torch.device`, default: `torch.device('cpu')`): The device on which to create the tensor.

    ## Returns
    `torch.Tensor`: The generated positional encoding tensor.
    """
    from deep_numerical.utils.grid import space_grid

    x_ndim:    int = len(shape)
    dimension: int = x_ndim - 2
    _grid: torch.Tensor = space_grid(dimension, shape[1:-1], 1, -1, 'none', dtype=dtype, device=device)
    pos: torch.Tensor
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
    pos = pos[None, ...].repeat(shape[0], *(1 for _ in range(x_ndim - 1)))
    return pos


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()