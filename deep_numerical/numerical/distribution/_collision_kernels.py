import  torch


##################################################
##################################################
__all__:    list[str]   = ["vhs"]


##################################################
##################################################
def vhs(
        dimension:          int,
        resolution:         int,
        v_max:              float,
        v_where_closed:     str,
        exp_speed:          torch.Tensor,
        temporal_repeat:    int,
    ) -> torch.Tensor:
    from    deep_numerical.utils    import  velocity_grid
    v_grid = velocity_grid(
        dimension, resolution, v_max,
        where_closed    = v_where_closed,
        dtype           = exp_speed.dtype,
        device          = exp_speed.device,
    )[None, None, ...]   # Generate new dimensions: batch, time
    exp_speed   = exp_speed.reshape(-1, *(1 for _ in range(2+dimension)))
    kernel: torch.Tensor = torch.norm(v_grid, 2, dim=-1, keepdim=True).pow(exp_speed)
    return kernel.repeat(1, temporal_repeat, *(1 for _ in range(dimension+1)))


##################################################
##################################################
# End of file