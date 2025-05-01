import torch
import numpy as np
import scipy.io
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = np.load(self.file_path, allow_pickle = True)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()



# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# generate multi-level graph
class RandomMultiMeshGenerator(object):
    def __init__(self, real_space, mesh_size, level, sample_sizes):
        super(RandomMultiMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_sizes
        self.level = level

        assert len(sample_sizes) == level
        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        self.idx = []
        self.idx_all = None
        self.grid_sample = []
        self.grid_sample_all = None
        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []

    def sample(self):
        self.idx = []
        self.grid_sample = []

        perm = torch.randperm(self.n)
        index = 0
        for l in range(self.level):
            self.idx.append(perm[index: index + self.m[l]])
            self.grid_sample.append(self.grid[self.idx[l]])
            index = index + self.m[l]
        self.idx_all = perm[:index]
        self.grid_sample_all = self.grid[self.idx_all]
        return self.idx, self.idx_all

    def get_grid(self):
        grid_out = []
        for grid in self.grid_sample:
            grid_out.append(torch.tensor(grid, dtype=torch.float))
        return grid_out, torch.tensor(self.grid_sample_all, dtype=torch.float)

    def ball_connectivity(self, radius_inner, radius_inter):
        assert len(radius_inner) == self.level
        assert len(radius_inter) == self.level - 1

        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []
        edge_index_out = []
        edge_index_down_out = []
        edge_index_up_out = []

        index = 0
        for l in range(self.level):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l])
            edge_index = np.vstack(np.where(pwd <= radius_inner[l])) + index
            self.edge_index.append(edge_index)
            edge_index_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.n_edges_inner.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        index = 0
        for l in range(self.level - 1):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l], self.grid_sample[l + 1])
            edge_index = np.vstack(np.where(pwd <= radius_inter[l])) + index
            edge_index[1, :] = edge_index[1, :] + self.grid_sample[l].shape[0]
            self.edge_index_down.append(edge_index)
            edge_index_down_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.edge_index_up.append(edge_index[[1, 0], :])
            edge_index_up_out.append(torch.tensor(edge_index[[1, 0], :], dtype=torch.long))
            self.n_edges_inter.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        edge_index_out = torch.cat(edge_index_out, dim=1)
        edge_index_down_out = torch.cat(edge_index_down_out, dim=1)
        edge_index_up_out = torch.cat(edge_index_up_out, dim=1)

        return edge_index_out, edge_index_down_out, edge_index_up_out

    def get_edge_index_range(self):
        # in order to use graph network's data structure,
        # the edge index shall be stored as tensor instead of list
        # we concatenate the edge index list and label the range of each level

        edge_index_range = torch.zeros((self.level, 2), dtype=torch.long)
        edge_index_down_range = torch.zeros((self.level - 1, 2), dtype=torch.long)
        edge_index_up_range = torch.zeros((self.level - 1, 2), dtype=torch.long)

        n_edge_index = 0
        for l in range(self.level):
            edge_index_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index[l].shape[1]
            edge_index_range[l, 1] = n_edge_index

        n_edge_index = 0
        for l in range(self.level - 1):
            edge_index_down_range[l, 0] = n_edge_index
            edge_index_up_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index_down[l].shape[1]
            edge_index_down_range[l, 1] = n_edge_index
            edge_index_up_range[l, 1] = n_edge_index

        return edge_index_range, edge_index_down_range, edge_index_up_range

    def attributes(self, theta=None):
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []

        if theta is None:
            for l in range(self.level):
                edge_attr = self.grid_sample_all[self.edge_index[l].T].reshape((self.n_edges_inner[l], 2 * self.d))
                self.edge_attr.append(torch.tensor(edge_attr))

            for l in range(self.level - 1):
                edge_attr_down = self.grid_sample_all[self.edge_index_down[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_up = self.grid_sample_all[self.edge_index_up[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                self.edge_attr_down.append(torch.tensor(edge_attr_down))
                self.edge_attr_up.append(torch.tensor(edge_attr_up))
        else:
            theta = theta[self.idx_all]

            for l in range(self.level):
                edge_attr = np.zeros((self.n_edges_inner[l], 2 * self.d + 2))
                edge_attr[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index[l].T].reshape(
                    (self.n_edges_inner[l], 2 * self.d))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[l][0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[l][1]]
                self.edge_attr.append(torch.tensor(edge_attr, dtype=torch.float))

            for l in range(self.level - 1):
                edge_attr_down = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))
                edge_attr_up = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))

                edge_attr_down[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_down[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_down[:, 2 * self.d] = theta[self.edge_index_down[l][0]]
                edge_attr_down[:, 2 * self.d + 1] = theta[self.edge_index_down[l][1]]
                self.edge_attr_down.append(torch.tensor(edge_attr_down, dtype=torch.float))

                edge_attr_up[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_up[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_up[:, 2 * self.d] = theta[self.edge_index_up[l][0]]
                edge_attr_up[:, 2 * self.d + 1] = theta[self.edge_index_up[l][1]]
                self.edge_attr_up.append(torch.tensor(edge_attr_up, dtype=torch.float))

        edge_attr_out = torch.cat(self.edge_attr, dim=0)
        edge_attr_down_out = torch.cat(self.edge_attr_down, dim=0)
        edge_attr_up_out = torch.cat(self.edge_attr_up, dim=0)
        return edge_attr_out, edge_attr_down_out, edge_attr_up_out


# generate multi-level graph, with split and assemble
class RandomMultiMeshSplitter(object):
    def __init__(self, real_space, mesh_size, level, sample_sizes):
        super(RandomMultiMeshSplitter, self).__init__()

        self.d = len(real_space)
        self.ms = sample_sizes
        self.m = sample_sizes[0]
        self.level = level

        assert len(sample_sizes) == level
        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        self.splits = self.n // self.m  # number of sub-grid
        if self.splits * self.m < self.n:
            self.splits = self.splits + 1
        print('n:', self.n, ' m:', self.m, ' number of splits:', self.splits)

        self.perm = None
        self.idx = []
        self.idx_all = None
        self.grid_sample = []
        self.grid_sample_all = None
        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []

    def sample(self, new_sample=True, index0=0):
        self.idx = []
        self.grid_sample = []

        if (new_sample) or (self.perm is None):
            self.perm = torch.randperm(self.n)

        index = index0
        for l in range(self.level):
            index = index % self.n
            index_end = (index + self.ms[l]) % self.n

            if index < index_end:
                idx = self.perm[index: index_end]
            else:
                idx = torch.cat((self.perm[index:], self.perm[: index_end]), dim=0)
            self.idx.append(idx)
            self.grid_sample.append(self.grid[idx])
            index = index_end

        if index0 < index_end:
            idx_all = self.perm[index0: index_end]
        else:
            idx_all = torch.cat((self.perm[index0:], self.perm[: index_end]), dim=0)

        self.idx_all = idx_all
        self.grid_sample_all = self.grid[self.idx_all]
        return self.idx, self.idx_all

    def get_grid(self):
        grid_out = []
        for grid in self.grid_sample:
            grid_out.append(torch.tensor(grid, dtype=torch.float))
        return grid_out, torch.tensor(self.grid_sample_all, dtype=torch.float)

    def ball_connectivity(self, radius_inner, radius_inter):
        assert len(radius_inner) == self.level
        assert len(radius_inter) == self.level - 1

        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []
        edge_index_out = []
        edge_index_down_out = []
        edge_index_up_out = []

        index = 0
        for l in range(self.level):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l])
            edge_index = np.vstack(np.where(pwd <= radius_inner[l])) + index
            self.edge_index.append(edge_index)
            edge_index_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.n_edges_inner.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        index = 0
        for l in range(self.level - 1):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l], self.grid_sample[l + 1])
            edge_index = np.vstack(np.where(pwd <= radius_inter[l])) + index
            edge_index[1, :] = edge_index[1, :] + self.grid_sample[l].shape[0]
            self.edge_index_down.append(edge_index)
            edge_index_down_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.edge_index_up.append(edge_index[[1, 0], :])
            edge_index_up_out.append(torch.tensor(edge_index[[1, 0], :], dtype=torch.long))
            self.n_edges_inter.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        edge_index_out = torch.cat(edge_index_out, dim=1)
        edge_index_down_out = torch.cat(edge_index_down_out, dim=1)
        edge_index_up_out = torch.cat(edge_index_up_out, dim=1)

        return edge_index_out, edge_index_down_out, edge_index_up_out

    def get_edge_index_range(self):
        # in order to use graph network's data structure,
        # the edge index shall be stored as tensor instead of list
        # we concatenate the edge index list and label the range of each level

        edge_index_range = torch.zeros((self.level, 2), dtype=torch.long)
        edge_index_down_range = torch.zeros((self.level - 1, 2), dtype=torch.long)
        edge_index_up_range = torch.zeros((self.level - 1, 2), dtype=torch.long)

        n_edge_index = 0
        for l in range(self.level):
            edge_index_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index[l].shape[1]
            edge_index_range[l, 1] = n_edge_index

        n_edge_index = 0
        for l in range(self.level - 1):
            edge_index_down_range[l, 0] = n_edge_index
            edge_index_up_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index_down[l].shape[1]
            edge_index_down_range[l, 1] = n_edge_index
            edge_index_up_range[l, 1] = n_edge_index

        return edge_index_range, edge_index_down_range, edge_index_up_range

    def attributes(self, theta=None):
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []

        if theta is None:
            for l in range(self.level):
                edge_attr = self.grid_sample_all[self.edge_index[l].T].reshape((self.n_edges_inner[l], 2 * self.d))
                self.edge_attr.append(torch.tensor(edge_attr))

            for l in range(self.level - 1):
                edge_attr_down = self.grid_sample_all[self.edge_index_down[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_up = self.grid_sample_all[self.edge_index_up[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                self.edge_attr_down.append(torch.tensor(edge_attr_down))
                self.edge_attr_up.append(torch.tensor(edge_attr_up))
        else:
            theta = theta[self.idx_all]

            for l in range(self.level):
                edge_attr = np.zeros((self.n_edges_inner[l], 2 * self.d + 2))
                edge_attr[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index[l].T].reshape(
                    (self.n_edges_inner[l], 2 * self.d))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[l][0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[l][1]]
                self.edge_attr.append(torch.tensor(edge_attr, dtype=torch.float))

            for l in range(self.level - 1):
                edge_attr_down = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))
                edge_attr_up = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))

                edge_attr_down[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_down[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_down[:, 2 * self.d] = theta[self.edge_index_down[l][0]]
                edge_attr_down[:, 2 * self.d + 1] = theta[self.edge_index_down[l][1]]
                self.edge_attr_down.append(torch.tensor(edge_attr_down, dtype=torch.float))

                edge_attr_up[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_up[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_up[:, 2 * self.d] = theta[self.edge_index_up[l][0]]
                edge_attr_up[:, 2 * self.d + 1] = theta[self.edge_index_up[l][1]]
                self.edge_attr_up.append(torch.tensor(edge_attr_up, dtype=torch.float))

        edge_attr_out = torch.cat(self.edge_attr, dim=0)
        edge_attr_down_out = torch.cat(self.edge_attr_down, dim=0)
        edge_attr_up_out = torch.cat(self.edge_attr_up, dim=0)
        return edge_attr_out, edge_attr_down_out, edge_attr_up_out

    def splitter(self, radius_inner, radius_inter, theta_a, theta_all):
        # give a test mesh, generate a list of data
        data = []
        index = 0
        for i in range(self.splits):
            if i == 0:
                idx, idx_all = self.sample(new_sample=True, index0=index)
            else:
                idx, idx_all = self.sample(new_sample=False, index0=index)
            index = (index + self.m) % self.n

            grid, grid_all = self.get_grid()
            edge_index, edge_index_down, edge_index_up = self.ball_connectivity(radius_inner, radius_inter)
            edge_index_range, edge_index_down_range, edge_index_up_range = self.get_edge_index_range()
            edge_attr, edge_attr_down, edge_attr_up = self.attributes(theta=theta_a)
            x = torch.cat([grid_all, theta_all[idx_all, :]], dim=1)
            data.append(Data(x=x,
                             edge_index_mid=edge_index, edge_index_down=edge_index_down, edge_index_up=edge_index_up,
                             edge_index_range=edge_index_range, edge_index_down_range=edge_index_down_range,
                             edge_index_up_range=edge_index_up_range,
                             edge_attr_mid=edge_attr, edge_attr_down=edge_attr_down, edge_attr_up=edge_attr_up,
                             sample_idx=idx[0]))
        return data

    def assembler(self, out_list, sample_idx_list, is_cuda=False):
        assert len(out_list) == self.splits
        if is_cuda:
            pred = torch.zeros(self.n, ).cuda()
        else:
            pred = torch.zeros(self.n, )
        for i in range(self.splits):
            pred[sample_idx_list[i]] = out_list[i].reshape(-1)
        return pred


def multi_pole_grid1d(theta, theta_d, N, is_periodic=False):
    grid_list = []
    theta_list = []
    edge_index_list = []
    edge_index_list_cuda = []
    level = int(np.log2(s) - 1)
    print(level)
    for l in range(1, level+1):
        r_l = 2 ** (l - 1)
        s_l = s // r_l
        n_l = s_l
        print('level',s_l,r_l,n_l)
        xs = np.linspace(0.0, 1.0, s_l)
        grid_l = xs
        grid_l = torch.tensor(grid_l, dtype=torch.float)
        print(grid_l.shape)
        grid_list.append(grid_l)

        theta_l = theta[:,:,:theta_d].reshape(N, s, theta_d)
        theta_l = theta_l[:, ::r_l,  :]
        theta_l = theta_l.reshape(N, n_l, theta_d)
        theta_l = torch.tensor(theta_l, dtype=torch.float)
        print(theta_l.shape)
        theta_list.append(theta_l)

        # for the finest level, we construct the nearest neighbors (NN)
        if l==1:
            edge_index_nn = []
            for x_i in range(s_l):
                for x in (-1,1):
                    x_j = x_i + x

                    if is_periodic:
                        x_j = x_j % s_l

                    # if (xj, yj) is a valid node
                    if (x_j in range(s_l)):
                        edge_index_nn.append([x_i,x_j])
            edge_index_nn = torch.tensor(edge_index_nn, dtype=torch.long)
            edge_index_nn = edge_index_nn.transpose(0,1)
            edge_index_list.append(edge_index_nn)
            edge_index_list_cuda.append(edge_index_nn.cuda())
            print('edge', edge_index_nn.shape)

        # we then compute the interactive neighbors -- their parents are NN but they are not NearestNeighbor
        edge_index_inter = []
        for x_i in range(s_l):
            for x in range(-3,4):
                x_j = x_i + x
                # if (xj, yj) is a valid node
                if is_periodic:
                    x_j = x_j % s_l

                if (x_j in range(s_l)):
                    # if (xi, yi), (xj, yj) not NearestNeighbor
                    if abs(x)>=2:
                        # if their parents are NN
                        if abs(x_i//2 - x_j//2)%(s_l//2) <=1:
                            edge_index_inter.append([x_i,x_j])

        edge_index_inter = torch.tensor(edge_index_inter, dtype=torch.long)
        edge_index_inter = edge_index_inter.transpose(0,1)
        edge_index_list.append(edge_index_inter)
        edge_index_list_cuda.append(edge_index_inter.cuda())
        print('edge_inter', edge_index_inter.shape)

    print(len(grid_list),len(edge_index_list),len(theta_list))
    return grid_list, theta_list, edge_index_list, edge_index_list_cuda

def get_edge_attr(grid, theta, edge_index):
    n_edges = edge_index.shape[1]
    edge_attr = np.zeros((n_edges, 4))
    edge_attr[:, 0:2] = grid[edge_index.transpose(0,1)].reshape((n_edges, -1))
    edge_attr[:, 2] = theta[edge_index[0]]
    edge_attr[:, 3] = theta[edge_index[1]]
    return torch.tensor(edge_attr, dtype=torch.float)

def multi_pole_grid1d(theta, theta_d, s,  N, is_periodic=False):
    grid_list = []
    theta_list = []
    edge_index_list = []
    edge_index_list_cuda = []
    level = int(np.log2(s) - 1)
    print(level)
    for l in range(1, level+1):
        r_l = 2 ** (l - 1)
        s_l = s // r_l
        n_l = s_l
        print('level',s_l,r_l,n_l)
        xs = np.linspace(0.0, 1.0, s_l)
        grid_l = xs
        grid_l = torch.tensor(grid_l, dtype=torch.float)
        print(grid_l.shape)
        grid_list.append(grid_l)

        theta_l = theta[:,:,:theta_d].reshape(N, s, theta_d)
        theta_l = theta_l[:, ::r_l,  :]
        theta_l = theta_l.reshape(N, n_l, theta_d)
        theta_l = torch.tensor(theta_l, dtype=torch.float)
        print(theta_l.shape)
        theta_list.append(theta_l)

        # for the finest level, we construct the nearest neighbors (NN)
        if l==1:
            edge_index_nn = []
            for x_i in range(s_l):
                for x in (-1,1):
                    x_j = x_i + x

                    if is_periodic:
                        x_j = x_j % s_l

                    # if (xj, yj) is a valid node
                    if (x_j in range(s_l)):
                        edge_index_nn.append([x_i,x_j])
            edge_index_nn = torch.tensor(edge_index_nn, dtype=torch.long)
            edge_index_nn = edge_index_nn.transpose(0,1)
            edge_index_list.append(edge_index_nn)
            edge_index_list_cuda.append(edge_index_nn.cuda())
            print('edge', edge_index_nn.shape)

        # we then compute the interactive neighbors -- their parents are NN but they are not NearestNeighbor
        edge_index_inter = []
        for x_i in range(s_l):
            for x in range(-3,4):
                x_j = x_i + x
                # if (xj, yj) is a valid node
                if is_periodic:
                    x_j = x_j % s_l

                if (x_j in range(s_l)):
                    # if (xi, yi), (xj, yj) not NearestNeighbor
                    if abs(x)>=2:
                        # if their parents are NN
                        if abs(x_i//2 - x_j//2)%(s_l//2) <=1:
                            edge_index_inter.append([x_i,x_j])

        edge_index_inter = torch.tensor(edge_index_inter, dtype=torch.long)
        edge_index_inter = edge_index_inter.transpose(0,1)
        edge_index_list.append(edge_index_inter)
        edge_index_list_cuda.append(edge_index_inter.cuda())
        print('edge_inter', edge_index_inter.shape)

    print(len(grid_list),len(edge_index_list),len(theta_list))
    return grid_list, theta_list, edge_index_list, edge_index_list_cuda

def get_edge_attr(grid, theta, edge_index):
    n_edges = edge_index.shape[1]
    edge_attr = np.zeros((n_edges, 4))
    edge_attr[:, 0:2] = grid[edge_index.transpose(0,1)].reshape((n_edges, -1))
    edge_attr[:, 2] = theta[edge_index[0]]
    edge_attr[:, 3] = theta[edge_index[1]]
    return torch.tensor(edge_attr, dtype=torch.float)

