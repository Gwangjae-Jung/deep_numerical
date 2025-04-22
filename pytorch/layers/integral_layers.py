from    typing              import  *
from    typing_extensions   import  Self

from    math                import  prod

import  torch
from    torch.nn.functional import  grid_sample

from    ._base_module       import  BaseModule
from    ..utils             import  Objects, roots_linspace


##################################################
##################################################
__all__: list[int] = ['IntegralLinearV1', 'IntegralLinear', 'IntegralConv2D', 'IntegralConv2D_']


##################################################
##################################################
class IntegralLinearV1(BaseModule):
    """## Linear layer as a kernel integration.
    
    -----
    ### Description
    Usual linear layer can be regarded as numerical kernel integration at a fixed grid.
    This module sets trainable parameters as the modes of the kernel function, whence numerical kernel integration can be conducted on any discretization.
    """
    def __init__(
            self,
            
            n_modes_in:     int,
            n_modes_out:    int,
            
            default_channels_in:    Optional[int] = None,
            default_channels_out:   Optional[int] = None,
            
            sample_mode:    str = 'bicubic',
            
            bias:           bool = True,
        ) -> Self:
        super().__init__()
        self.__n_modes_in   = n_modes_in
        self.__n_modes_out  = n_modes_out
        self.__sample_mode  = sample_mode
        self.__bias = bias
        
        self.kernel_modes:  torch.nn.Parameter = \
            torch.nn.Parameter(torch.randn(size=(1, 1, n_modes_out, n_modes_in), dtype=torch.float))
        """
        The trainable parameter to approximate the kernel function, which is of shape `(1, 1, n_modes_out, n_modes_in)`.
        The first two dimensions are the batch, channel dimensions, respectively, and should not be squeezed.
        """
        self.bias_modes:    Union[torch.nn.Parameter, torch.Tensor] = \
            torch.nn.Parameter(torch.randn(size=(1, 1, n_modes_out, 1), dtype=torch.float)) if bias \
            else torch.zeros(size=(1, 1, n_modes_out, 1), dtype=torch.float)
        """
        The trainable parameter to approximate the bias function, which is of shape `(1, 1, n_modes_out, 1)`.
        The first two dimensions are the batch, channel dimensions, respectively, and should not be squeezed.
        The last dimension should also not be squeezed, too.
        """
        
        self.__default_channels_in:     Optional[int]   = None
        self.__default_channels_out:    Optional[int]   = None
        self.__default_grid:    Optional[torch.Tensor]  = None
        self.__default_weight:  Optional[torch.Tensor]  = None
        self.__grid_weight_set: bool = False
        self.__set_defualt_grid_weight(default_channels_in, default_channels_out)
        
        return
    
    
    @property
    def n_modes_in(self) -> int:
        return self.__n_modes_in
    @property
    def n_modes_out(self) -> int:
        return self.__n_modes_out
    @property
    def bias(self) -> bool:
        return self.__bias
    @property
    def default_channels_in(self) -> int:
        return self.__default_channels_in
    @property
    def default_channels_out(self) -> int:
        return self.__default_channels_out
    @property
    def grid_weight_set(self) -> bool:
        return self.__grid_weight_set
    
    
    def __set_defualt_grid_weight(
            self,
            default_channels_in:    Optional[int],
            default_channels_out:   Optional[int],
        ) -> None:
        if default_channels_in is None or default_channels_out is None:
            self.__default_channels_in  = None
            self.__default_channels_out = None
            self.__default_grid         = None
            self.__default_weight       = None
            self.__grid_weight_set      = False
        else:
            grid, weight = self.compute_grid_weight(default_channels_in, default_channels_out)
            self.__default_channels_in  = default_channels_in
            self.__default_channels_out = default_channels_out
            self.__default_grid         = grid
            self.__default_weight       = weight
            self.__grid_weight_set      = True
        return
    
    
    def compute_grid_weight(
            self,
            channels_in: int,
            channels_out: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ### Note
        This method returns the grid on `[-1, 1]^n`, which is not of the `'ij'`-indexing.
        The grid is indexed so that, *after expanding a new dimension at `dim=0`*,  it can be readily used as the input of `torch.nn.functional.grid_sample()`.
        
        * The input channel is aligned horizontally from left to right, the output channel is aligned vertically from top to bottom.
        
        -----
        ### Returns
        * `grid`: The grid on which the kernel function is interpolated.
        * `w_x`:  The weights to be used to conduct numerical integration along the input dimension (the $x$-axis).
        """
        grid_x, w_x = roots_linspace(channels_in,  -1, 1)
        grid_y, _   = roots_linspace(channels_out, -1, 1)
        grid    = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)
        return (grid, w_x)
        
    
    def forward(
            self,
            X:              torch.Tensor,
            channels_in:    Optional[int] = None,
            channels_out:   Optional[int] = None,
        ) -> torch.Tensor:
        """This forward propagation is different from the discrete linear layer.
        Note that the number of output channels should be specified.
        """
        grid:   torch.Tensor
        weight: torch.Tensor
        if self.__grid_weight_set and (channels_in is None or channels_out is None):
            grid    = self.__default_grid
            weight  = self.__default_weight
        else:
            grid, weight = self.compute_grid_weight(channels_in, channels_out)
        grid = grid[None, ...]
        
        # Shapes:
        # `kernel`: (out_channels, in_channels)
        # `bias`:   (out_channels,)
        kernel = grid_sample(self.kernel_modes, grid, mode=self.__sample_mode, padding_mode='border', align_corners=True)[0, 0]
        bias: torch.Tensor
        if self.__bias:
            bias = grid_sample(self.bias_modes, grid, mode=self.__sample_mode, padding_mode='border', align_corners=True)[0, 0, :, 0]
        else:
            bias = torch.zeros(size=(kernel.shape[0],))
        
        return torch.einsum('ij, ...j, j -> ...i', kernel, X, weight) + bias
        

##################################################
##################################################
class IntegralLinear(BaseModule):
    """## Linear layer as a kernel integration.
    
    -----
    ### Description
    Usual linear layer can be regarded as numerical kernel integration at a fixed grid.
    This module sets trainable parameters as the modes of the kernel function, whence numerical kernel integration can be conducted on any discretization.
    """
    def __init__(
            self,
            
            n_modes_in:     int,
            n_modes_out:    int,
            
            sample_mode:    str = 'bicubic',
            
            bias:           bool = True,
        ) -> Self:
        super().__init__()
        self.__n_modes_in   = n_modes_in
        self.__n_modes_out  = n_modes_out
        self.__sample_mode  = sample_mode
        self.__bias = bias
        
        self.kernel_modes:  torch.nn.Parameter = \
            torch.nn.Parameter(torch.randn(size=(1, 1, n_modes_out, n_modes_in), dtype=torch.float))
        """
        The trainable parameter to approximate the kernel function, which is of shape `(1, 1, n_modes_out, n_modes_in)`.
        The first two dimensions are the batch, channel dimensions, respectively, and should not be squeezed.
        """
        self.bias_modes:    Union[torch.nn.Parameter, torch.Tensor] = \
            torch.nn.Parameter(torch.randn(size=(1, 1, n_modes_out, 1), dtype=torch.float)) if bias \
            else torch.zeros(size=(1, 1, n_modes_out, 1), dtype=torch.float)
        """
        The trainable parameter to approximate the bias function, which is of shape `(1, 1, n_modes_out, 1)`.
        The first two dimensions are the batch, channel dimensions, respectively, and should not be squeezed.
        The last dimension should also not be squeezed, too.
        """
        
        return
    
    
    @property
    def n_modes_in(self) -> int:
        return self.__n_modes_in
    @property
    def n_modes_out(self) -> int:
        return self.__n_modes_out
    @property
    def bias(self) -> bool:
        return self.__bias
    
    
    def sample_filters(
            self,
            channels_in:    int,
            channels_out:   int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples the kernel and bias, together with quadrature weights.
        
        -----
        ### Returns
        * `filters`: The grid on which the kernel function is interpolated. (Shape: `(out_channels, in_channels)`)
        * `w_c_in`:  The weights to be used to conduct numerical integration along the input dimension (the $x$-axis). (Shape: `(in_channels,)`)
        * `bias`: The bias term. (Shape: `(out_channels,)`)
        """
        grid_c_in, w_c_in   = roots_linspace(channels_in,  -1, 1)
        grid_c_out, _       = roots_linspace(channels_out, -1, 1)
        grid_c = torch.stack(torch.meshgrid(grid_c_in, grid_c_out, indexing='xy'), dim=-1)
        filters = grid_sample(
            self.kernel_modes, grid_c[None, ...],
            mode            = self.__sample_mode,
            padding_mode    = 'border',
            align_corners   = True
        )[0, 0]
        bias = grid_sample(
            self.bias_modes, grid_c[None, ...],
            mode            = self.__sample_mode,
            padding_mode    = 'border',
            align_corners   = True
        )[0, 0, :, 0] if self.__bias else torch.zeros(size=(channels_out,))
        return (filters, w_c_in, bias)
        
    
    def forward(
            self,
            X:              torch.Tensor,
            channels_in:    int,
            channels_out:   int,
        ) -> torch.Tensor:
        """This forward propagation is different from the discrete linear layer.
        Note that the number of output channels should be specified.
        """
        # Shapes:
        # `kernel`: (out_channels, in_channels)
        # `weight`: (in_channels,)
        # `bias`:   (out_channels,)
        kernel, weight, bias = self.sample_filters(channels_in, channels_out)
        return torch.einsum('ij, ...j, j -> ...i', kernel, X, weight) + bias


##################################################
##################################################
class IntegralConv2D(BaseModule):
    def __init__(
            self,
            
            n_modes_channel_in:     int,
            n_modes_channel_out:    int,
            n_modes_axis1:          int,
            n_modes_axis2:          int,
            
            sample_mode:    str = 'bicubic',
            
            bias:   bool = True,
        ) -> Self:
        super().__init__()
        self.__n_modes_channel_in   = n_modes_channel_in
        self.__n_modes_channel_out  = n_modes_channel_out
        self.__n_modes_axis1        = n_modes_axis1
        self.__n_modes_axis2        = n_modes_axis2
        self.__sample_mode          = sample_mode
        self.__bias                 = bias
        
        self.filter_modes:    torch.nn.Parameter = \
            torch.nn.Parameter(
                torch.randn(
                    size    = (1, n_modes_axis1 * n_modes_axis2, n_modes_channel_out, n_modes_channel_in),
                    dtype   = torch.float
                )
            )
        self.bias_modes:      Union[torch.nn.Parameter, torch.Tensor] = \
            torch.nn.Parameter(
                torch.randn(
                    size    = (1, 1, n_modes_channel_out, 1),
                    dtype   = torch.float
                )
            ) if bias else torch.zeros(size=(1, 1, n_modes_channel_out, 1), dtype=torch.float)
        
        return
    
    
    @property
    def n_modes_channel_in(self) -> int:
        return self.__n_modes_channel_in
    @property
    def n_modes_channel_out(self) -> int:
        return self.__n_modes_channel_out
    @property
    def n_modes_axis1(self) -> int:
        return self.__n_modes_axis1
    @property
    def n_modes_axis2(self) -> int: 
        return self.__n_modes_axis2
    @property
    def sample_mode(self) -> str:
        return self.__sample_mode
    @property
    def bias(self) -> bool:
        return self.__bias
    
    
    def sample_filters(
            self,
            channels_in:    int,
            channels_out:   int,
            kernel_size:    Objects[int],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples the kernel and bias, together with quadrature weights.
        
        -----
        ### Returns
        * `filters`: The grid on which the kernel function is interpolated. (Shape: `(channels_out, channels_in, *kernel_size)`)
        * `w_c_in`:  The weights to be used to conduct numerical integration along the input dimension (the $x$-axis). (Shape: `(channels_in,)`)
        * `w_x`
        """
        # Step 1. Sample the convolutional weight
        ## Discretize the kernel function along the channel dimensions
        grid_c_in, w_c_in   = roots_linspace(channels_in,  -1, 1)
        grid_c_out, _       = roots_linspace(channels_out, -1, 1)
        grid_c  = torch.stack(torch.meshgrid(grid_c_in, grid_c_out, indexing='xy'), dim=-1)
        
        kernel_temp = grid_sample(
            self.filter_modes, grid_c[None, ...],
            mode            = self.__sample_mode,
            padding_mode    = 'border',
            align_corners   = True,
        ).reshape(self.__n_modes_axis1, self.__n_modes_axis2, channels_in, channels_out).permute(2,3,0,1)
        ## NOTE: Permutation: (c_out, c_in, x1, x2) -> (1, c_out*c_in, x1, x2)
        kernel_temp = kernel_temp.reshape(1, -1, *kernel_temp.shape[2:])
        
        ## Discritize the kernel function along the spatial dimensions
        kernel_size: tuple[int] = \
            tuple((kernel_size, kernel_size)) if isinstance(kernel_size, int) else tuple(kernel_size)
        grid_x1, w_x1 = roots_linspace(kernel_size[1], -1, 1)
        grid_x2, w_x2 = roots_linspace(kernel_size[0], -1, 1)
        grid_x  = torch.stack(torch.meshgrid(grid_x1, grid_x2, indexing='xy'), dim=-1)
        w_x     = w_x1[None, :] * w_x2[:, None]
        kernel  = grid_sample(kernel_temp, grid_x[None, ...], mode=self.__sample_mode, padding_mode='border', align_corners=True).reshape(channels_out, channels_in, *kernel_size)
        
        # Step 2. Sample the bias
        bias = grid_sample(
            self.bias_modes, grid_c[None, ...],
            mode            = self.__sample_mode,
            padding_mode    = 'border',
            align_corners   = True
        )[0, 0, :, 0] if self.__bias else torch.zeros(size=(channels_out,))
        return (kernel, w_c_in, w_x, bias)
    
    
    def forward(
            self,
            X:              torch.Tensor,
            channels_in:    Optional[int] = None,
            channels_out:   Optional[int] = None,
            kernel_size:    Optional[Objects[int]] = 1,
            stride:         Optional[Objects[int]] = 1,
            padding:        Optional[Objects[int]] = 0,
        ) -> torch.Tensor:
        filters, w_c_in, w_x, bias = self.sample_filters(channels_in, channels_out, kernel_size)
        filters = filters * w_c_in[None, :, None, None] * w_x[None, None, ...]
        return torch.nn.functional.conv2d(input=X, weight=filters, bias=bias, stride=stride, padding=padding)
    

##################################################
##################################################
# End of file





class IntegralConv2D_(torch.nn.Module):
    def __init__(
            self,
            
            n_modes_channel_in:     int,
            n_modes_channel_out:    int,
            n_modes_axis1:          int,
            n_modes_axis2:          int,
            
            sample_mode:    str = 'bicubic',
            
            default_channels_in:    Optional[int] = None,
            default_channels_out:   Optional[int] = None,
            
            bias:   bool = True,
        ) -> Self:
        super().__init__()
        self.__n_modes_channel_in   = n_modes_channel_in
        self.__n_modes_channel_out  = n_modes_channel_out
        self.__n_modes_axis1        = n_modes_axis1
        self.__n_modes_axis2        = n_modes_axis2
        self.__sample_mode          = sample_mode
        self.__bias                 = bias
        
        self.filter_modes:    torch.nn.Parameter = \
            torch.nn.Parameter(
                torch.randn(
                    size    = (1, n_modes_axis1 * n_modes_axis2, n_modes_channel_out, n_modes_channel_in),
                    dtype   = torch.float
                )
            )
        self.bias_modes:      Union[torch.nn.Parameter, torch.Tensor] = \
            torch.nn.Parameter(
                torch.randn(
                    size    = (1, 1, n_modes_channel_out, 1),
                    dtype   = torch.float
                )
            ) if bias else torch.zeros(size=(1, 1, n_modes_channel_out, 1), dtype=torch.float)
        
        self.__default_channels_in  = default_channels_in
        self.__default_channels_out = default_channels_out
        self.__default_kernel_ratio     = 8
        self.__default_padding_ratio    = 16
        self.__default_stride_ratio     = 32
        
        return
    
    
    @property
    def n_modes_channel_in(self) -> int:
        return self.__n_modes_channel_in
    @property
    def n_modes_channel_out(self) -> int:
        return self.__n_modes_channel_out
    @property
    def n_modes_axis1(self) -> int:
        return self.__n_modes_axis1
    @property
    def n_modes_axis2(self) -> int: 
        return self.__n_modes_axis2
    @property
    def sample_mode(self) -> str:
        return self.__sample_mode
    @property
    def bias(self) -> bool:
        return self.__bias
    
    
    def sample_filters(
            self,
            channels_in:    int,
            channels_out:   int,
            kernel_size:    Union[int, Sequence[int]],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples the kernel and bias, together with quadrature weights.
        
        -----
        ### Returns
        * `filters`: The grid on which the kernel function is interpolated. (Shape: `(channels_out, channels_in, *kernel_size)`)
        * `w_c_in`:  The weights to be used to conduct numerical integration along the input dimension (the $x$-axis). (Shape: `(channels_in,)`)
        * `w_x`
        """       
        # Step 1. Sample the convolutional weight
        ## Discretize the kernel function along the channel dimensions
        grid_c_in, w_c_in   = roots_linspace(channels_in,  -1, 1)
        grid_c_out, _       = roots_linspace(channels_out, -1, 1)
        grid_c  = torch.stack(torch.meshgrid(grid_c_in, grid_c_out, indexing='xy'), dim=-1)
        
        kernel_temp = grid_sample(
            self.filter_modes, grid_c[None, ...],
            mode            = self.__sample_mode,
            padding_mode    = 'border',
            align_corners   = True,
        ).reshape(self.__n_modes_axis1, self.__n_modes_axis2, channels_in, channels_out).permute(2,3,0,1)
        ## NOTE: Permutation: (c_out, c_in, x1, x2) -> (1, c_out*c_in, x1, x2)
        kernel_temp = kernel_temp.reshape(1, -1, *kernel_temp.shape[2:])
        
        ## Discritize the kernel function along the spatial dimensions
        kernel_size: tuple[int] = \
            tuple((kernel_size, kernel_size)) if isinstance(kernel_size, int) else tuple(kernel_size)      
        grid_x1, w_x1 = roots_linspace(kernel_size[1], -1, 1)
        grid_x2, w_x2 = roots_linspace(kernel_size[0], -1, 1)
        grid_x  = torch.stack(torch.meshgrid(grid_x1, grid_x2, indexing='xy'), dim=-1)
        w_x     = w_x1[None, :] * w_x2[:, None]
        kernel  = grid_sample(kernel_temp, grid_x[None, ...], mode=self.__sample_mode, padding_mode='border', align_corners=True).reshape(channels_out, channels_in, *kernel_size)
        
        # Step 2. Sample the bias
        bias = grid_sample(
            self.bias_modes, grid_c[None, ...],
            mode            = self.__sample_mode,
            padding_mode    = 'border',
            align_corners   = True
        )[0, 0, :, 0] if self.__bias else torch.zeros(size=(channels_out,))
        return (kernel, w_c_in, w_x, bias)
    
    
    def forward(
            self,
            X:              torch.Tensor,
            channels_in:    Optional[int] = None,
            channels_out:   Optional[int] = None,
        ) -> torch.Tensor:
        channels_in    = self.__default_channels_in  if channels_in  is None else channels_in
        channels_out   = self.__default_channels_out if channels_out is None else channels_out
        filters, w_c_in, w_x, bias = self.sample_filters(
            channels_in  = channels_in,
            channels_out = channels_out,
            kernel_size  = (X.shape[-1-i]//self.__default_kernel_ratio for i in range(2)),
        )
        filters = filters * w_c_in[None, :, None, None] * w_x[None, None, ...]
        return torch.nn.functional.conv2d(
            input=X, weight=filters, bias=bias,
            padding = (X.shape[-1-i]//self.__default_padding_ratio  for i in range(2)),
            stride  = (X.shape[-1-i]//self.__default_stride_ratio   for i in range(2)),
        )