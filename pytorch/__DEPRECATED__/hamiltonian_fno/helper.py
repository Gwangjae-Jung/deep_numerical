import  torch




##################################################
##################################################


EINSUM_FULL_STRING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


TORCH_ACTIVATION_DICT= {
    "elu":          "ELU",
    "gelu":         "GELU",
    "identity":     "Identity",
    "leaky relu":   "LeakyReLU",
    "relu":         "ReLU",
    "silu":         "SiLU",
    "sigmoid":      "Sigmoid",
    "softmax":      "Softmax",
    "tanh":         "Tanh"
}

TORCH_INITIALIZER_DICT = {
    "constant":         "constant_",
    "dirac":            "dirac_",
    "eye":              "eye_",
    "kaiming normal":   "kaiming_normal_",
    "kaiming uniform":  "kaiming_uniform_",
    "normal":           "normal_",
    "ones":             "ones_",
    "orthogonal":       "orthogonal_",
    "sparse":           "sparse_",
    "trunc normal":     "trunc_normal_",
    "uniform":          "uniform_",
    "xavier normal":    "xavier_normal_",
    "xavier uniform":   "xavier_uniform_",
    "zeros":            "zeros_",
}


##################################################
##################################################


def inverse_permutation(
        perm:   list[int] | torch.Tensor
    ) -> torch.Tensor:
    
    if type(perm) != torch.Tensor:
        perm = torch.Tensor(perm).type(torch.long)
    if type(perm) == torch.Tensor and perm.ndim != 1:
        raise RuntimeError(f"The permutation tuple should be a 1-dimensional tensor, but got a tensor of shape {perm.shape}.")
        
    perm_inverse = torch.zeros_like(perm)
    perm_inverse[perm] = torch.arange(len(perm))
    return perm_inverse


def load_channels(
        channels_loaded:    list[int] | slice | torch.Tensor    = None,
        dim_domain:         int                                 = None,
    ) -> list[slice]:
    
    if channels_loaded == None:
        channels_loaded = slice(None)
    if dim_domain == None:
        raise RuntimeError(f"The dimension of domain is not given.")
        
    return [..., channels_loaded] + [slice(None) for _ in range(dim_domain)]
    

# def load_channels(
#         channels_loaded:    list[int] | slice | torch.Tensor    = None,
#         dim_domain:         int                                 = None,
#         is_temporal:        bool                                = None
#     ) -> list[slice]:
    
#     if channels_loaded == None:
#         channels_loaded = slice(None)
#     if dim_domain == None:
#         raise RuntimeError(f"The dimension of domain is not given.")
#     if is_temporal == None:
#         raise RuntimeError(f"Whether the problem is time-dependent is not given.")
    
#     _temporal = [slice(None)] if is_temporal else []
    
#     return [..., channels_loaded] + _temporal + [slice(None) for _ in range(dim_domain)]
    


##################################################
##################################################
