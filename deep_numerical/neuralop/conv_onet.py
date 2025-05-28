from    typing          import  *

import  torch
from    torch           import  nn

from    ..layers._base_module   import  BaseModule
from    ..utils         import  *


##################################################
##################################################
class ConvDeepONetStructured(BaseModule):
    """## Convolutional Deep Operator Network (ConvDeepONet) - Structured version
    ### Approximation of a continuous operator of function spaces with a single input, using a branch net and a trunk net
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of certain types of function spaces.
    The universal approximation theorem holds for a continuous map `G: V -> C(K_2)`, where `V` is a compact subset of `C(K_1)` for some compact space `K_1` and `K_2` is a compact subset of a finite-dimensional Euclidean space.
    
    -----
    ### Remark
    1. It is assumed that the input and the output functions are scalar-valued.
    2. The input is assumed to be given in the `BCX` convention.
    """
    def __init__(
                    self,
                    branch: nn.Module,
                    trunk:  nn.Module,
        ) -> None:
        """## The initializer of the class `DeepONet`
        -----
        ### Arguments
        @ `branch` (`nn.Module`)
            * The branch subnetwork to encode the input functions.
            * (Input channels) Should be the number of the sensor points.
            * (Output channels) Should match with the number of the output channels of the branch net.
            
        @ `trunk` (`nn.Module`)
            * The trunk subnetwork to encode the query points.
            * (Output channels) Should match with the number of the output channels of the branch net.
        """
        super().__init__()
        self.branch = branch
        self.trunk  = trunk
        self.bias   = nn.Parameter(torch.zeros(size = (1,), dtype = torch.float))
        return
    
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        ### Arguments
        -----
        @ `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of two tensors, labelled `U` and `Y` in order.
                * `U` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 2-tensor, where each row corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)
        
        -----
        ### Remark
        1. In the structured case, the given query points are used throughout all `U` instances.
            * `U.shape:         (batch_size,        n_sensor_points)`
            * `Y.shape:         (n_query_points,    dim_query_space)`
            * `branch(U).shape: (batch_size,        num_features)`
            * `trunk(Y).shape:  (n_query_points,    num_features)`
            * `output.shape:    (batch_size,        n_query_points)`
            * `forward:         torch.einsum("bi,ti->bt", [branch, trunk])`
        """ 
        branch  = self.branch.forward(inputs[0])
        trunk   = self.trunk.forward(inputs[1])
        inner_prod = branch @ trunk.T
        return inner_prod + self.bias
    
    
    def __repr__(self) -> str:
        msg = \
            f"DeepONet(\n" \
            f"    structured,\n" \
            f"    branch={self.branch},\n" \
            f"    trunk ={self.trunk},\n" \
            f")"
        return msg




class ConvDeepONetUnstructured(BaseModule):
    """## Convolutional Deep Operator Network (ConvDeepONet) - Unstructured version
    ### Approximation of a continuous operator of function spaces with a single input, using a branch net and a trunk net
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of certain types of function spaces.
    The universal approximation theorem holds for a continuous map `G: V -> C(K_2)`, where `V` is a compact subset of `C(K_1)` for some compact space `K_1` and `K_2` is a compact subset of a finite-dimensional Euclidean space.
    
    Reference: https://www.nature.com/articles/s42256-021-00302-5
    
    -----
    ### Remark
    1. It is assumed that the input and the output functions are scalar-valued.
    """
    def __init__(
                    self,
                    branch: nn.Module,
                    trunk:  nn.Module,
        ) -> None:
        """## The initializer of the class `DeepONet`
        -----
        ### Arguments
        @ `branch` (`nn.Module`)
            * The branch subnetwork to encode the input functions.
            * (Input channels) Should be the number of the sensor points.
            * (Output channels) Should match with the number of the output channels of the branch net.
            
        @ `trunk` (`nn.Module`)
            * The trunk subnetwork to encode the query points.
            * (Output channels) Should match with the number of the output channels of the branch net.
        """
        super().__init__()
        self.branch = branch
        self.trunk  = trunk
        self.bias   = nn.Parameter(torch.zeros(size = (1,), dtype = torch.float))
        return
    
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        ### Arguments
        -----
        @ `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of two tensors, labelled `U` and `Y` in order.
                * `U` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 2-tensor, where each row corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)
        
        -----
        ### Remark
        1. In the unstructured case, each `Y` instance corresponds to exactly one of the `U` instances. As each `Y` instance is a finite subset of the query space (single or not), `Y` should be given as a 3-tensor of shape `(batch_size, n_query_points, dim_query_space)`.
            * `U.shape:         (batch_size,        n_sensor_points)`
            * `Y.shape:         (batch_size,        n_query_points,     dim_query_space)`
            * `branch(U).shape: (batch_size,        num_features)`
            * `trunk(Y).shape:  (batch_size,        n_query_points,     num_features)`
            * `output.shape:    (batch_size,        n_query_points)`
            * `forward:         torch.einsum("bi,bfi->bf", [branch, trunk])`
        """ 
        branch  = self.branch.forward(inputs[0])
        trunk   = self.trunk.forward(inputs[1])
        inner_prod = torch.einsum("bi,bfi->bf", [branch, trunk])
        return inner_prod + self.bias
    
    
    def __repr__(self) -> str:
        msg = \
            f"DeepONet(\n" \
            f"    unstructured,\n"\
            f"    branch={self.branch},\n" \
            f"    trunk ={self.trunk},\n" \
            f")"
        return msg


##################################################
##################################################
class ConvMIONetStructured(BaseModule):
    """## Convolutional Multiple-Input Operator Network (ConvMIONet) - Structured version
    ### Approximation of a continuous operator of Banach spaces, using a branch neta and a trunk net
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of Banach spaces.
    The universal approximation theorem holds for a continuous map `G: K_1 x .. x K_n -> W`, where `K_i` is a compact subset of a Banach space `X_i` with a countable basis for `i in {1, ..., n}` and `W` is a Banach space.

    Reference: https://epubs.siam.org/doi/epdf/10.1137/22M1477751

    -----
    ### Remark
    1. For implementation convinience, the network is constructed in accordance with the part (c) of Corollary 2.6 of the paper.
    """
    def __init__(
                    self,
                    branches:  Sequence[nn.Module],
                    trunk:     nn.Module,
        ) -> None:
        """## The initializer of the class `MIONet`
        -----
        ### Arguments
        @ `branches` (`Sequence[nn.Module]`)
            * The branch subnetworks to encode the input functions.
            * (Input channels) Should be the number of the sensor points.
            * (Output channels) Should match with the number of the output channels of the branch net.
            
        @ `trunk` (`nn.Module`)
            * The trunk subnetwork to encode the query points.
            * (Output channels) Should match with the number of the output channels of the branch net.
        """
        super().__init__()
        # Define the subnetworks
        self.branches = nn.ModuleList(*branches)
        self.trunk    = trunk
        self.bias     = nn.Parameter(torch.zeros(size = (1,), dtype = torch.float))
        
        return

    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        ### Arguments
        -----
        @ `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of multiple tensors of length `n + 1`, labelled `U_1, ..., U_n` and `Y` in order.
                * `U_i` for `i \in {1, ..., n}` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 2-tensor, where each row corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)

        -----
        ### Remark
        1. The forward operation is defined by `einsum` operation, whose specific definition is determined by `self.is_query_shared`.
            * `U_i.shape:               (batch_size,        n_sensor_points)`
            * `Y.shape:                 (n_query_points,    dim_query_space)`
            * `branch[i](U_i).shape:    (batch_size,        num_features)`
            * `trunk(Y).shape:          (n_query_points,    num_features)`
            * `output.shape:            (batch_size,        n_query_points)`
            * `forward:                 torch.einsum("bi,ti->bt", [(branch_1 * ... * branch_n), trunk])`
        
        2. In contrast to the Deep Operator Network, the Multiple-Input Operator Network supports multiple inputs. Since this class assumes that each input instance is 1-dimensional, to input a multidimensional instance, one has to project the instance dimension by dimension. For example, if an input instance is a discretization of a continuous function into a 3-dimensional space, such instance should be given as 3 input instances, each of which is into a 1-dimensional space.
        """
        branch  = torch.stack(
                        [
                            self.branches[k].forward(inputs[k])
                            for k in range(len(inputs) - 1)
                        ]
                    ).type(torch.float)
        branch  = torch.prod(branch, dim = 0)
        trunk   = self.trunk.forward(inputs[-1])
        inner_prod = branch @ trunk.T
        return inner_prod + self.bias
            
    
    def __repr__(self) -> str:
        msg = \
                f"MIONet(\n" \
                f"    structured,\n" \
                f"    branches:\n"
        for branch in self.branches:
            msg += \
                f"         * {branch},\n"
        msg +=  f"    turnk: {self.trunk},\n" \
                f")"
        return msg




class ConvMIONetUnstructured(BaseModule):
    """## Convolutional Multiple-Input Operator Network (ConvMIONet) - Unstructured version
    ### Approximation of a continuous operator of Banach spaces, using a branch neta and a trunk net
    -----
    ### Description
    This class constructs a network which approximates a continuous operator of Banach spaces.
    The universal approximation theorem holds for a continuous map `G: K_1 x .. x K_n -> W`, where `K_i` is a compact subset of a Banach space `X_i` with a countable basis for `i in {1, ..., n}` and `W` is a Banach space.

    Reference: https://epubs.siam.org/doi/epdf/10.1137/22M1477751

    -----
    ### Remark
    1. For implementation convinience, the network is constructed in accordance with the part (c) of Corollary 2.6 of the paper.
    """
    def __init__(
                    self,
                    branches:   Sequence[nn.Module],
                    trunk:      nn.Module,
        ) -> None:
        """## The initializer of the class `MIONet`
        -----
        ### Arguments
        @ `branches` (`Sequence[nn.Module]`)
            * The branch subnetworks to encode the input functions.
            * (Input channels) Should be the number of the sensor points.
            * (Output channels) Should match with the number of the output channels of the branch net.
            
        @ `trunk` (`nn.Module`)
            * The trunk subnetwork to encode the query points.
            * (Output channels) Should match with the number of the output channels of the branch net.
        """
        super().__init__()
        self.branches = nn.ModuleList(*branches)
        self.trunk    = trunk
        self.bias     = nn.Parameter(torch.zeros(size = (1,), dtype = torch.float))
        return;

    
    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        ### Arguments
        -----
        @ `inputs` (`Sequence[torch.Tensor]`)
            * A list or a tuple of multiple tensors of length `n + 1`, labelled `U_1, ..., U_n` and `Y` in order.
                * `U_i` for `i \in {1, ..., n}` (`torch.Tensor`)
                    * A 2-tensor of shape `(batch_size, n_sensor_points)`.
                    * Each row corresponds to the values of a function restricted to the sensor points.
                * `Y` (`torch.Tensor`)
                    * A 2-tensor, where each row corresponds to the coordinates of a query point.
                    * The shape and the contents of `Y` are up to whether the query points are shared for all `U` instances. (See the following remark.)

        -----
        ### Remark
        1. In the unstructured case, each `Y` instance corresponds to exactly one of the `U` instances. As each `Y` instance is a finite subset of the query space (single or not), `Y` should be given as a 3-tensor of shape `(batch_size, n_query_points, dim_query_space)`.
            * `U_i.shape:               (batch_size,        n_sensor_points)`
            * `Y.shape:                 (batch_size,        n_query_points,     dim_query_space)`
            * `branch[i](U_i).shape:    (batch_size,        num_features)`
            * `trunk(Y).shape:          (batch_size,        n_query_points,     num_features)`
            * `output.shape:            (batch_size,        n_query_points)`
            * `forward: torch.einsum("bi,bfi->bf", [(branch_1 * ... * branch_n), trunk])`
            
        2. In contrast to the Deep Operator Network, the Multiple-Input Operator Network supports multiple inputs. Since this class assumes that each input instance is 1-dimensional, to input a multidimensional instance, one has to project the instance dimension by dimension. For example, if an input instance is a discretization of a continuous function into a 3-dimensional space, such instance should be given as 3 input instances, each of which is into a 1-dimensional space.
        """
        branch  = torch.stack(
                        [
                            self.branches[k].forward(inputs[k])
                            for k in range(len(inputs) - 1)
                        ]
                    ).type(torch.float)
        branch  = torch.prod(branch, dim = 0)
        trunk   = self.trunk.forward(inputs[-1])
        inner_prod = torch.einsum("bi,bfi->bf", [branch, trunk])        
        return inner_prod + self.bias
            
    
    def __repr__(self) -> str:
        msg = \
                f"MIONet(\n" \
                f"    structured,\n" \
                f"    branches:\n"
        for branch in self.branches:
            msg += \
                f"         * {branch},\n"
        msg +=  f"    turnk: {self.trunk},\n" \
                f")"
        return msg


##################################################
##################################################
# End of file