"""A script for PointNet
-----
Remark

1. In forward propagation, the input tensor is assumed to have the shape `(batch, channel, num_points)`, as it is conventional in PyTorch.
"""
from   typing import List, Sequence, Tuple
import torch
from   torch import nn
import torch.nn.functional as F


__all__: list[str] = ["PointNetClassification", "PointNetSegmentation"]


class TNet(nn.Module):
    """T-net predicting canonical affine transformations for point clouds.
    
    ### Description
    T-net aims to predict an affine transform matrix so as to align the input tensor to a canonical space before feature extraction.
    To be precise, given an input batch `X` of `B` point clouds (each of which consists of `N` points in the `C`-dimensional space; `X` is of shape `(B, C, N)`), T-net aims to return a 3-tensor `T` of shape `(B, in_channels, out_channels)` so that
    
    `torch.einsum("bin,bio->boj", [X, T])`
    
    is aligned in a canonical space.
    
    See https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf.
    """
    def __init__(
            self,
            num_channels:       int,
            hidden_dimensions:  List[int]   = [64, 128, 1024, 512, 256],
            maxpool_at:         int         = 3,
        ) -> None:
        """The initializer of `TNet`
        
        ## Description
        Initializes the spatial transformation network with convolutional feature extraction layers and an affine transformation matrix predictor.
        
        ## Arguments
        `num_channels` (`int`): The number of input channels.
        `hidden_dimensions` (`List[int]`, default: `[64, 128, 1024, 512, 256]`): Intermediate hidden channel dimensions.
        `maxpool_at` (`int`, default: `3`): Index at which max pooling across points is performed.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
        
        self.num_channels = num_channels
        DIMENSIONS = [
            num_channels,
            *hidden_dimensions,
            num_channels ** 2
        ]
        self.__MAXPOOL_AT = maxpool_at
        
        
        self.list_fc = nn.ModuleList([])
        self.list_bn = nn.ModuleList([])
        for idx in range(len(DIMENSIONS) - 1):
            _in_channels  = DIMENSIONS[idx]
            _out_channels = DIMENSIONS[idx + 1]
            self.list_fc.append(nn.Conv1d(_in_channels, _out_channels, kernel_size = 1))
            if idx != len(DIMENSIONS) - 2:
                self.list_bn.append(nn.BatchNorm1d(_out_channels))
        
        nn.init.zeros_(self.list_fc[-1].weight)
        self.list_fc[-1].bias = nn.Parameter(torch.eye(num_channels, dtype = torch.float).flatten())
        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input: `(B, C, N)`
        Output: `(B, C, C)`
        """
        for cnt in range(0, self.__MAXPOOL_AT):
            X = F.relu(self.list_bn[cnt](self.list_fc[cnt](X)))
        X = F.max_pool1d(X, kernel_size = X.shape[-1])
        for cnt in range(self.__MAXPOOL_AT, len(self.list_bn)):
            X = F.relu(self.list_bn[cnt](self.list_fc[cnt](X)))
        X = self.list_fc[-1](X)
        return X.reshape(-1, self.num_channels, self.num_channels)


class TNetTransform(nn.Module):
    """T-net feature transformation layer for point clouds.
    """
    def __init__(
            self,
            num_channels:       int,
            hidden_dimensions:  List[int]   = [64, 128, 1024, 512, 256],
            maxpool_at:         int         = 3,
        ) -> None:
        """The initializer of `TNetTransform`
        
        ## Description
        Initializes the `TNetTransform` module wrapping a `TNet` instance for coordinate transformation.
        
        ## Arguments
        `num_channels` (`int`): The number of input channels.
        `hidden_dimensions` (`List[int]`, default: `[64, 128, 1024, 512, 256]`): Hidden dimensions of the internal `TNet`.
        `maxpool_at` (`int`, default: `3`): Max pooling layer index in `TNet`.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
        self.tnet = TNet(num_channels, hidden_dimensions, maxpool_at)
        return None
    
    
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of `TNetTransform`
        
        ## Description
        Computes the spatial transformation matrix `T` using `TNet` and applies it to the input point cloud `X`.
        
        ## Arguments
        `X` (`torch.Tensor`): Input point cloud tensor of shape `(batch_size, num_channels, num_points)`.
        
        ## Returns
        `Tuple[torch.Tensor, torch.Tensor]`: A tuple `(X_transformed, T)` containing the aligned point cloud and transformation matrix.
        """
        T = self.tnet.forward(X)
        X = torch.einsum("bin,bij->bjn", [X, T])
        return (X, T)


class PointNetBase(nn.Module):
    """Base model for PointNet
    """
    def __init__(
                    self,
                    in_channels:    int = 3,
                    base_channels:  int = 64,
                    
                    hidden_dimensions:  List[int]   = [64, 128, 1024, 512, 256],
                    maxpool_at:         int         = 3,
        ) -> None:
        """The initializer of `PointNetBase`
        
        ## Description
        Initializes the shared PointNet backbone consisting of input transform, local feature MLP, feature transform, and global feature MLP.
        
        ## Arguments
        `in_channels` (`int`, default: `3`): Number of input point feature channels (e.g., 3 for 3D coordinates).
        `base_channels` (`int`, default: `64`): Channel dimension after initial 1D convolution.
        `hidden_dimensions` (`List[int]`, default: `[64, 128, 1024, 512, 256]`): Hidden dimensions for the T-net modules.
        `maxpool_at` (`int`, default: `3`): Layer index at which max pooling is performed in the T-net.
        
        ## Returns
        `None`: None.
        """
        super().__init__()
           
        self.transform_input    = TNetTransform(in_channels, hidden_dimensions, maxpool_at)
        self.mlp_base           = nn.Conv1d(in_channels, base_channels, 1)
        self.bn_base            = nn.BatchNorm1d(base_channels)
        self.transorm_feature   = TNetTransform(base_channels, hidden_dimensions, maxpool_at)
        
        self.DIM_GLOBAL = [base_channels, 128, 1024]
        mlp_global = []
        for idx in range( len(self.DIM_GLOBAL) - 1 ):
            _in_channels    = self.DIM_GLOBAL[idx]
            _out_channels   = self.DIM_GLOBAL[idx + 1]
            mlp_global += [
                nn.Conv1d(_in_channels, _out_channels, 1),
                nn.BatchNorm1d(_out_channels),
                nn.ReLU(),
            ]
        mlp_global.pop(-1)  # Remove the activation of the last layer
        self.mlp_global = nn.Sequential(*mlp_global)
        
        return;
        
    
    def forward_base_global(self, X: torch.Tensor) -> Sequence[torch.Tensor]:
        """Computes base and global point cloud features.

        ## Description
        Transforms point clouds and extracts both local per-point base features and max-pooled global features.

        ## Arguments
        `X` (`torch.Tensor`): Input point cloud tensor of shape `(B, in_channels, N)`.

        ## Returns
        `Sequence[torch.Tensor]`: Tuple of `(X_base, X_global, T_input, T_feature)`.
        """
        X, T_input = self.transform_input(X)
        X = F.relu(self.bn_base(self.mlp_base(X)))
        X_base, T_feature = self.transorm_feature(X)
        X_global = self.mlp_global.forward(X_base)
        X_global = F.max_pool1d(X_global, kernel_size = X_global.size(-1))
        return (X_base, X_global, T_input, T_feature)


class PointNetClassification(PointNetBase):
    """PointNet architecture for point cloud classification.
    ### A neural network for point clouds
    -----
    ### Description
    See https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf.
    """
    def __init__(
                    self,
                    num_classes:    int,
                    in_channels:    int = 3,
                    base_channels:  int = 64,
                    
                    hidden_dimensions:  List[int]   = [64, 128, 1024, 512, 256],
                    maxpool_at:         int         = 3,
        ) -> None:
        """The initializer of `PointNetClassification`
        
        ## Description
        Initializes the PointNet architecture for point cloud classification with a classification MLP head.
        
        ## Arguments
        `num_classes` (`int`): The number of target classes.
        `in_channels` (`int`, default: `3`): Number of input point channels.
        `base_channels` (`int`, default: `64`): Base feature channel dimension.
        `hidden_dimensions` (`List[int]`, default: `[64, 128, 1024, 512, 256]`): Hidden dimensions for the T-net modules.
        `maxpool_at` (`int`, default: `3`): Max pooling index for T-net.
        
        ## Returns
        `None`: None.
        """
        super().__init__(
            in_channels         = in_channels,
            base_channels       = base_channels,
            hidden_dimensions   = hidden_dimensions,
            maxpool_at          = maxpool_at,
        )
        self.num_classes = num_classes
        
        # Subnetwork for classification
        DIM_CLS = [self.DIM_GLOBAL[-1], 256, num_classes]
        mlp_cls = []
        for idx in range(len(DIM_CLS) - 2):
            _in_channels    = DIM_CLS[idx]
            _out_channels   = DIM_CLS[idx + 1]
            mlp_cls += [
                nn.Linear(_in_channels, _out_channels),
                nn.BatchNorm1d(_out_channels),
                nn.ReLU()
            ]
        mlp_cls += [
                nn.Linear(DIM_CLS[-2], DIM_CLS[-1]),
                nn.Dropout(0.3),
                nn.BatchNorm1d(DIM_CLS[-1]),
                nn.ReLU()
            ]
        self.mlp_cls = nn.Sequential(*mlp_cls)
        
        return;
        
    
    def forward(self, X: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward pass for point cloud classification.

        ## Description
        Computes classification logits using extracted global point cloud representations.

        ## Arguments
        `X` (`torch.Tensor`): Input point cloud tensor of shape `(B, in_channels, N)`.

        ## Returns
        `Sequence[torch.Tensor]`: Tuple `(X_class, T_input, T_feature)`.
        """
        _, X_class, T_input, T_feature = self.forward_base_global(X)
        X_class = X_class.squeeze(-1)
        X_class = self.mlp_cls.forward(X_class)
        X_class = F.log_softmax(X_class, dim = -1)
        return (X_class, T_input, T_feature)
    
    
    def fit(self, X: torch.Tensor, y: torch.LongTensor) -> None:
        """Fits the classification model on training data.
        
        ## Description
        Runs a forward pass on training point clouds `X` and targets `y`.
        
        ## Arguments
        `X` (`torch.Tensor`): Input point cloud tensor of shape `(batch_size, in_channels, num_points)`.
        `y` (`torch.LongTensor`): Target class label tensor of shape `(batch_size,)`.
        
        ## Returns
        `None`: None.
        """
        pred, T_input, T_output = self.forward(X)
        
        return




class PointNetSegmentation(PointNetBase):
    """PointNet architecture for point cloud segmentation.
    ### A neural network for point clouds
    -----
    ### Description
    See https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf.
    """
    def __init__(
                    self,
                    num_classes:    int,
                    in_channels:    int = 3,
                    base_channels:  int = 64,
                    
                    hidden_dimensions:  List[int]   = [64, 128, 1024, 512, 256],
                    maxpool_at:         int         = 3,
        ) -> None:
        """The initializer of `PointNetSegmentation`
        
        ## Description
        Initializes the PointNet architecture for point cloud part/semantic segmentation.
        
        ## Arguments
        `num_classes` (`int`): The number of segmentation classes per point.
        `in_channels` (`int`, default: `3`): Number of input point channels.
        `base_channels` (`int`, default: `64`): Base feature channel dimension.
        `hidden_dimensions` (`List[int]`, default: `[64, 128, 1024, 512, 256]`): Hidden dimensions for the T-net modules.
        `maxpool_at` (`int`, default: `3`): Max pooling index for T-net.
        
        ## Returns
        `None`: None.
        """
        super().__init__(
            in_channels         = in_channels,
            base_channels       = base_channels,
            hidden_dimensions   = hidden_dimensions,
            maxpool_at          = maxpool_at,
        )
        self.num_classes = num_classes
        
        # Subnetwork for segmentation
        DIM_SEG = [base_channels + self.DIM_GLOBAL[-1], 512, 256, 128, num_classes]
        mlp_seg = []
        for idx in range(len(DIM_SEG) - 1):
            _in_channels    = DIM_SEG[idx]
            _out_channels   = DIM_SEG[idx + 1]
            mlp_seg += [
                nn.Conv1d(_in_channels, _out_channels, kernel_size = 1),
                nn.BatchNorm1d(_out_channels),
                nn.ReLU()
            ]
        mlp_seg.pop(-1)
        self.mlp_seg = nn.Sequential(*mlp_seg)
        
        return None
        
    
    def forward(self, X: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward pass for point cloud segmentation.

        ## Description
        Concatenates local and global feature representations to predict per-point segmentation classes.

        ## Arguments
        `X` (`torch.Tensor`): Input point cloud tensor of shape `(B, in_channels, N)`.

        ## Returns
        `Sequence[torch.Tensor]`: Tuple `(X_class, T_input, T_feature)` where `X_class` has shape `(B, num_classes, N)`.
        """
        X_base, X_global, T_input, T_feature = self.forward_base_global(X)
        X_base = torch.concat(
                [
                    X_base,
                    X_global.repeat(1, 1, X_base.size(-1))
                ],
                axis = 1
            )
        return (self.mlp_seg.forward(X_base), T_input, T_feature)


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()
