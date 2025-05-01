import  numpy                       as  np
import  torch
from    torch                       import  nn
from    torch.nn                    import  functional      as      F
from    torch_geometric.nn          import  MessagePassing, NNConv
from    custom_modules.torch.layers import  MLP

DenseNet = MLP

# multigraph1.py
class MKGN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, points, level, in_width=1, out_width=1):
        super(MKGN, self).__init__()
        self.depth = depth
        self.width = width
        self.level = level

        index = 0
        self.points = [0]
        for point in points:
            index = index + point
            self.points.append(index)
        print(level, self.points)

        self.points_total = np.sum(points)

        # in (P)
        self.fc_in = torch.nn.Linear(in_width, width)

        # Downward kernels
        self.conv_down_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_down_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_down_list = torch.nn.ModuleList(self.conv_down_list)

        # Isolevel kernels
        self.conv_list = []
        for l in range(level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=True, bias=False))
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        # Upward kernels
        self.conv_up_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_up_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_up_list = torch.nn.ModuleList(self.conv_up_list)

        # out (Q)
        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)


    def forward(self, data):
        """
        down: downward
        mid: isolevel
        up: upward
        """
        edge_index_down, edge_attr_down, range_down = \
            data.edge_index_down,   data.edge_attr_down,    data.edge_index_down_range
        edge_index_mid, edge_attr_mid, range_mid = \
            data.edge_index_mid,    data.edge_attr_mid,     data.edge_index_range
        edge_index_up, edge_attr_up, range_up = \
            data.edge_index_up,     data.edge_attr_up,      data.edge_index_up_range

        ##### Lift
        x = self.fc_in(data.x)

        ##### Hidden
        for t in range(self.depth):
            #downward
            for l in range(self.level-1):
                x = x + self.conv_down_list[l](
                            x,
                            edge_index_down[:,range_down[l,0]:range_down[l,1]],
                            edge_attr_down[range_down[l,0]:range_down[l,1],:]
                        )
                # `x` is modified only on `_range_down` by now, but...
                x = F.relu(x)
                # `x` is modified everywhere, due to the activation

            #upward
            for l in reversed(range(self.level)):
                # NOTE: No modification of `x` on `self.points[l]:self.points[l+1]` by now
                x[self.points[l]:self.points[l+1]] = \
                    self.conv_list[l](
                        x[self.points[l]:self.points[l+1]].clone(),
                        edge_index_mid[:,range_mid[l,0]:range_mid[l,1]]-self.points[l],
                        edge_attr_mid[range_mid[l,0]:range_mid[l,1],:]
                    )
                # `x` is modified only on `_range_iso`

                if l > 0:
                    x = x + self.conv_up_list[l-1](x, edge_index_up[:,range_up[l-1,0]:range_up[l-1,1]], edge_attr_up[range_up[l-1,0]:range_up[l-1,1],:])
                    x = F.relu(x)
                    # `x` is modified only on `_range_up`

        x = F.relu(self.fc_out1(x[:self.points[1]]))
        ##### Project
        x = self.fc_out2(x)
        return x




# multigraph2.py
class MGKN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width, s):
        super(MGKN, self).__init__()
        self.depth = depth
        self.width = width
        self.s = s
        self.level = int(np.log2(s)-1)

        # P
        self.fc1 = torch.nn.Linear(in_width, width)

        # Isolevel kernels
        # NOTE: Non-isolevel kernels are defined as methods
        self.conv_list = []
        for l in range(self.level + 1):
            ker_width_l = max( ker_width // (2**l), 16)
            kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_list.append(NNConv(width, width, kernel_l, aggr='mean'))
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        # Q
        self.fc2 = torch.nn.Linear(width, ker_width)
        self.fc3 = torch.nn.Linear(ker_width, 1)


    # K_{l,l+1}: Upward passage (Kernels can be found in `forward()`)
    def Upsample(self, x, channels, scale, s):
        """
        By default, `scale=2`.
        """
        x = x.transpose(0, 1).view(1,channels,s) # (K,width) to (1, width, s)
        x = F.upsample(x, scale_factor=scale, mode='nearest') # (1, width, s) to (1, width,  s*2)
        x = x.view(channels, -1).transpose(0, 1) # (1, width, s*2, s*2) to (K*4, width)
        return x

    # K_{l+1,l}: Downward passage (Kernels can be found in `forward()`)
    def Downsample(self, x, channels, scale, s):
        x = x.transpose(0, 1).view(1,channels,s) # (K,width) to (1, width,  s)
        x = F.avg_pool1d(x, kernel_size=scale)
        x = x.view(channels, -1).transpose(0, 1) # (1, width, s/2, s/2) to (K/4, width)
        return x

    def forward(self, data):
        X_list, _, edge_index_list, edge_attr_list = data
        level = len(X_list)
        x = X_list[0]
        x = self.fc1(x)
        phi = [None] * level # list of x, len=level, to save the results of the downward passes
        for k in range(self.depth):
            # downward
            # NOTE: There is no kernel in the downward passage
            for l in range(level):
                phi[l] = x
                if (l != level - 1):
                    # downsample
                    x = self.Downsample(x, channels=self.width, scale=2, s=self.s // (2 ** l) )

            # upward
            x = F.relu(x + self.conv_list[-1](phi[-1], edge_index_list[-1], edge_attr_list[-1]))
            for l in reversed(range(level)):
                if (l != 0):
                    # upsample
                    x = self.Upsample(x, channels=self.width, scale=2, s=self.s // (2 ** l))
                    # interactive neighbors
                    x = F.relu(x + self.conv_list[l](phi[l-1], edge_index_list[l], edge_attr_list[l]))
                else:
                    x = F.relu(x + self.conv_list[0](phi[0], edge_index_list[0], edge_attr_list[0]))

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x