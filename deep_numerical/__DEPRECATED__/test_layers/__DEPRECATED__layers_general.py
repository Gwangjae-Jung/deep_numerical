from    typing      import  *

import  torch
from    torch       import  nn

from    ..utils     import  *




##################################################
##################################################


class MLP(nn.Module):
    """## Multi-layer perceptron
    -----
    By passing the dimension of the input/output spaces and the hidden spaces, this class constructs a multi-layer perceptron.
    """
    
    def __init__(
                    self,
                    channels:   ChannelsIHO,
                    activation: str = "tanh"
        ) -> None:
        """## The initializer of the class `MLP`
        -----
        ### Arguments
    
        1. `channels` (`ChannelsIHO`)
            The number of the channels in each layer, from the input layer to the output layer.
        
        2. `activation` (`str`, default: "tanh")
            The activation function which shall be used in each hidden layer.
        """
        
        super().__init__()
        self.__check_channels__(channels)
        
        _activation = getattr(nn, TORCH_ACTIVATION_DICT[activation])()
        
        _net = []
        for cnt in range(len(channels) - 2):
            _net += [
                        nn.Linear(channels[cnt], channels[cnt + 1]),
                        _activation
                    ]
            nn.init.xavier_normal_(_net[-2].weight)
            nn.init.zeros_(_net[-2].bias)
        _net.append(nn.Linear(channels[-2], channels[-1]))
        self.net = nn.Sequential(*_net)
        
        return;

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
    
    
    def __check_channels__(self, channels: ChannelsIHO) -> None:
        for cnt, dim in enumerate(channels):
            if not (type(dim) == int and dim >= 1):
                raise RuntimeError(f"The dimension of the layer {cnt} is set {dim}.")
        return;




##################################################
##################################################


class LiftProject(nn.Module):
    """## Lift and projection
    ### Affine transform with an activation acting on each point of the domain
    -----
    
    Given a batch of discretized functions, this class computes the activation of an affine transform of the input, where the affine transform is shared by all points of the domain.
    """
    def __init__(
                    self,
                    in_channels:        int,
                    hidden_channels:    int,
                    out_channels:       int,
                    dim_domain:         int,
                    activation:         str = "gelu"
        ) -> None:
        
        super().__init__()
        
        # Choose the convolutional layer to be used
        conv = getattr(nn, f"Conv{dim_domain}d", None)
        if conv is None:
            raise NotImplementedError(
                f"The PyTorch library provides the convolutional layers up to dimenion 3, but the domain is {dim_domain}-dimensional."
            )
        
        self.nn1 = conv(in_channels, hidden_channels, 1)
        self.nn2 = conv(hidden_channels, out_channels, 1)
        self.activation = getattr(nn, TORCH_ACTIVATION_DICT[activation])()
        
        return;
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.nn1(X)
        X = self.activation(X)
        X = self.nn2(X)
        return X




##################################################
##################################################
