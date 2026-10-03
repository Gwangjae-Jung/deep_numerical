import torch


__all__: list[str] = ["vhs"]


##################################################
def vhs(
    dimension:       int,
    resolution:      int,
    v_max:           float,
    v_where_closed:  str,
    exp_speed:       torch.Tensor,
    temporal_repeat: int,
) -> torch.Tensor:
    """VHS collision kernel grid.

    ## Description
    Computes a tensor of the Variable Hard Sphere (VHS) collision kernel evaluated over a discretized velocity grid.

    ## Arguments
    `dimension` (`int`): Dimension of the velocity domain.
    `resolution` (`int`): Number of grid points per velocity dimension.
    `v_max` (`float`): Truncation boundary of the velocity grid `[-v_max, v_max]^d`.
    `v_where_closed` (`str`): Endpoint boundary condition configuration (`'both'`, `'left'`, `'right'`, or `'none'`).
    `exp_speed` (`torch.Tensor`): Exponent `gamma` for the relative speed `|v|^gamma`.
    `temporal_repeat` (`int`): Repetition count along the temporal dimension.

    ## Returns
    `torch.Tensor`: The computed VHS kernel tensor.
    """
    from deep_numerical.utils import velocity_grid

    v_grid: torch.Tensor = velocity_grid(
        dimension,
        resolution,
        v_max,
        where_closed = v_where_closed,
        dtype        = exp_speed.dtype,
        device       = exp_speed.device,
    )[None, None, ...]
    exp_speed_reshaped: torch.Tensor = exp_speed.reshape(-1, *(1 for _ in range(2 + dimension)))
    kernel:             torch.Tensor = torch.norm(v_grid, 2, dim=-1, keepdim=True).pow(exp_speed_reshaped)
    return kernel.repeat(1, temporal_repeat, *(1 for _ in range(dimension + 1)))


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()