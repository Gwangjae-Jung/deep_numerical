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
            raise RuntimeError("씨발련아")

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






def multi_pole_grid1d(theta, theta_d, s,  N, is_periodic=False):
    """
    theta: function values (shape: (batch=N, resolution=s, channels=1))
    theta_d: dim_domain
    s: resolution
    N: number of the instances
    """
    print("=========================< Multipole grid 1D >=========================")
    grid_list:              list[torch.Tensor]      = []
    theta_list:             list[torch.Tensor]      = []
    edge_index_list:        list[torch.LongTensor]  = []
    edge_index_list_cuda:   list[torch.LongTensor]  = []
    level = int(np.log2(s) - 1)
    
    print(f"multi_pole_grid1d: {level}")
    
    for l in range(1, level+1):
        r_l = 2 ** (l - 1);     n_l = s_l = s // r_l
        
        grid_l = torch.linspace(0, 1, steps = s_l, dtype = torch.float)
        grid_list.append(grid_l)

        theta_l = theta[:,:,:theta_d].reshape(N, s, theta_d)
        theta_l = theta_l[:, ::r_l,  :]             # Downsampling, i.e., hierarchical subgraphs
        # theta_l = theta_l.reshape(N, n_l, theta_d)  # Useless
        theta_l = torch.tensor(theta_l, dtype=torch.float)
        theta_list.append(theta_l)

        # For the finest level, we construct the nearest neighbors (NN)
        # Internal points: two neighbors
        # Boundary points: one neighbor
        if l==1:
            edge_index_nn = []
            for x_i in range(s_l):
                for dx in (-1,1):   # Not `range(-1, 1 + 1)`
                    x_j = x_i + dx

                    if is_periodic:
                        x_j = x_j % s_l

                    # if (xj, yj) is a valid node
                    if (x_j in range(s_l)): # Required condition for non-periodic domain
                        edge_index_nn.append([x_i,x_j])
            edge_index_nn = torch.tensor(edge_index_nn, dtype=torch.long).transpose(0, 1)
            edge_index_list.append(edge_index_nn)
            edge_index_list_cuda.append(edge_index_nn.cuda())
        
        # Then compute the interactive neighbors
        # Their parents are NN, but they are not NN
        edge_index_inter = []
        for x_i in range(s_l):
            for dx in range(-3, 3 + 1):
                x_j = x_i + dx
                # if (xj, yj) is a valid node
                if is_periodic:
                    x_j = x_j % s_l

                if (x_j in range(s_l)): # Required condition for non-periodic domain
                    if abs(dx) > 1: # if xi and xj are not NearestNeighbor
                        if abs(x_i//2 - x_j//2)%(s_l//2) <= 1: # if their parents are NN
                            edge_index_inter.append([x_i,x_j])

        
        edge_index_inter = torch.tensor(edge_index_inter, dtype=torch.long).transpose(0, 1)
        edge_index_list.append(edge_index_inter)
        edge_index_list_cuda.append(edge_index_inter.cuda())

        if l == 3:
            print(f"Level {l}")
            print(f"* (r_l, n_l)=({r_l}, {n_l})")
            print(f"* grid_list:        {[_grid.shape       for _grid       in grid_list]}")
            print(f"* edge_index_list:  {[_edge_index.shape for _edge_index in edge_index_list]}")
            print(f"* theta_list:       {[_theta.shape      for _theta      in theta_list]}")
            print()

    print("=======================================================================")
    
    return grid_list, theta_list, edge_index_list, edge_index_list_cuda







def get_edge_attr(grid, theta, edge_index):
    n_edges = edge_index.shape[1]
    edge_attr = np.zeros((n_edges, 4))
    edge_attr[:, 0:2] = grid[edge_index.transpose(0,1)].reshape((n_edges, -1))
    edge_attr[:, 2] = theta[edge_index[0]]
    edge_attr[:, 3] = theta[edge_index[1]]
    return torch.tensor(edge_attr, dtype=torch.float)
