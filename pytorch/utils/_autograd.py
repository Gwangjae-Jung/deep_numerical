from    typing      import  Callable, Any

import  torch
from    typing      import  *
from    torch.func  import  vmap, jacrev


##################################################
##################################################
__all__ = [
    "compute_grad",
    "compute_jacobian",
    "differential",
    "laplacian",
    "hessian",
]


##################################################
##################################################
def compute_grad(
        outputs:        torch.Tensor,
        inputs:         torch.Tensor,
        create_graph:   bool            = True,
        retain_graph:   Optional[bool]  = None,
    ) -> torch.Tensor:
    """## Autograd for computing gradients
    
    Arguments:
        `outputs` (`torch.Tensor`): A 1-dimensional tensor, which acts as an array of the values of a function of `inputs`. This function aims at computing the gradient of `outputs` at `inputs`.
        
        `inputs` (`torch.Tensor`): A tensor object at which the gradient of `outputs` shall be computed.

        `create_graph` (`bool`, default: `True`): See Appendix below.
        
        `retain_graph` (`bool`, default: `None`): See Appendix below. When this parameter is not initialized, it is initialized to `create_graph` by default.
    
    Returns:
        This function returns the gradient of `outputs` at `inputs`.
        
    -----
    ### Remark
    1. (Dimensionality)
        `outputs` is required to be a tensor of dimension 1.
    2. (Slicing)
        Since generates a new tensor, slicing `outputs` does not matter in back-propagation, while slicing `inputs` generates a tensor which is not connected with `outputs` in the computational graph of `outputs`.
    
    -----
    ### Examples

    Example 1.
    >>> x = torch.tensor([1, 2, 3], dtype = torch.float, requires_grad = True)
    >>> w = (x ** 2).sum()
    >>> compute_grad(outputs = w, inputs = x, create_graph = False)
    tensor([2., 4., 6.], grad_fn=<...>)

    Example 2.
    >>> x = torch.tensor([4, 0], dtype = torch.float, requires_grad = True)
    >>> u = 2 * x[0] + torch.exp(1 + x[1] ** 2 )
    >>> u_grad = compute_grad(u, x, create_graph = True)
    >>> u_grad
    tensor([2., 0.], grad_fn=<...>)
    >>> u_x = u_grad[0]
    >>> u_xy = compute_grad(u_x, x, create_graph = False)[1]
    >>> u_xy
    tensor([0.])
    >>> u_xy.requires_grad
    False

    Example 3.
    >>> To be added.

    -----
    ### Appendix. Some parameters of `torch.autograd.grad()`
    Here, `R` is the set of the real numbers.
    Let `U` be a nonempty open subset of `R^n` and `f: U --> R^k` be a map with a computational graph.

    1. (`outputs`, `inputs`, and `grad_outputs` (`torch.Tensor`))
    As the names indicate, `inputs` is a tensor of points in `U` and `outputs` is a tensor of values of `f` at each point in `inputs`.
    `torch.autograd.grad()` first computes the gradients (of `outputs` at the points listed in `inputs`), then does the vector-Jacobian multiplication.
    To compute the partial derivatives for each member of the input batch, `grad_outputs` has to be manually set `torch.oneslike(outputs)`, which is observed by the author.

    2-1. (`create_graph` (`bool`, default: `True`))
    This boolean parameter determines whether `torch.autograd.grad()` creates the computational graph for the derivative, which shall be generally used to compute derivatives of higher order.
        * If `True`, the computational graph for the derivative shall be constructed.
        * If `False`, the computational graph for the derivative is not constructed.

    2-2. (`retain_graph` (`bool`, default: `create_graph`)))
    This boolean parameter determines whether the computational graph for computing the derivative should be preserved.
        * When `True`, the computational graph is preserved.
        * When `False`, the computational graph is deleted.
    
    3. (`allow_unused` (`bool`))
    When `inputs` does not contribute in computing `outputs` (so that the true gradient is obviously the zero vector), then `inputs` is not contained in the computational graph of `outputs`, so `torch.autograd.grad()` cannot properly compute the gradient.
        * When `True`, then `torch.autograd.grad()` returns the tuple `(None,)`.
        * When `False`, then `torch.autograd.grad()` throws a runtime error.
    """
    # Initialize `retain_graph` if it is not initialized
    if retain_graph==None:
        retain_graph = create_graph
    
    # Compute the gradient
    return_value =  torch.autograd.grad(
        outputs         = outputs,
        inputs          = inputs,
        grad_outputs    = torch.ones_like(outputs),
        create_graph    = create_graph,
        retain_graph    = retain_graph,
        allow_unused    = True,
    )[0]
    
    # If no backward propagation is executed, set the gradient to the zero vector
    if (return_value == None):
        return_value = torch.zeros_like(inputs)
    
    return return_value


def compute_jacobian(
        func:   Callable[[torch.Tensor, Any], torch.Tensor],
        points: torch.Tensor,
        kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
    """## Computes the Jacobian of a differentiable function using `vmap`.
    
    Arguments:
        `func` (`Callable[[torch.Tensor, Any], torch.Tensor]`): A function that takes a `k`-dimensional tensor and returns a `d`-dimensional tensor. Specifically, `func` should be defined as a function which maps a tensor of shape `(k,)` to a tensor of shape `(d,)`. `torch.func.vmap` will be used to vectorize the function.
        `points` (`torch.Tensor`): A tensor of shape `(B, d)`.
        `kwargs` (`dict[str, Any]`, default: `{}`): Optional additional arguments to pass to the function.
    
    Returns:
        A tensor of shape `(B, d, k)` representing the Jacobian of `func` at the given points.
    """
    return vmap(jacrev(func))(points, **kwargs) # Returns (B, d, k)


##################################################
##################################################
# First -order operations
def differential(
        outputs:        torch.Tensor,
        inputs:         torch.Tensor,
        create_graph:   bool            = False
    ) -> torch.Tensor:
    """## Differential of vector-valued functions
    ### This function aims at computing the differential of a real-vector-valued function with a computational graph.
    -----
    ### Arguments
    @ `outputs` (`torch.Tensor`):
        * This is a `torch.Tensor` object of dimension 2, which is constructed depending on the entire elements of the parameter `inputs`.
        * This function aims at computing the gradient of `outputs` at `inputs`.

    @ `inputs` (`torch.Tensor`):
        * This is a `torch.Tensor` object of dimension 2, at which the gradient of `outputs` shall be computed.
        * It is further required that `inputs` consists of the input points in row-wise convention.
    
    3. `create_graph` (`bool`, default: `False`):
        * This parameter is sent as the parameter `create_graph` of the function `torch.autograd.grad()`.
        * As the derivatives of vector-valued functions are generally not differentiated again, this parameter is set `False` by default.
    
    -----
    ### Return
    This function returns the differential of `outputs` at `inputs` as a 3-dimensional tensor of 2-dimensional Jacobian matrices.
    
    -----
    ### Remark
    1. (Dimensionality)
        `outputs` and `inputs` are required to be tensors of dimension 2.
        And it is further required that `inputs` consists of the input points (at which `outputs` is evaluated) in the row-wise convention.
    
    -----
    ### Examples

    Example 1.
    >>> (Command)
    (Output)

    """
    assert (outputs.ndim == 2), \
        "Computation of the differentials is not supported when `outputs` is not 2-dimensional. " + \
        f"('outputs.ndim': {outputs.ndim} (greater than 2))"
    
    # Compute the partial derivatives to form a tensor of shape (num_components, num_points, num_variables)
    partial_derivatives = []
    for idx in range(outputs.shape[-1]):
        partial_derivatives.append(
            compute_grad(
                outputs[:, idx], inputs,
                create_graph = create_graph, retain_graph = True
            ).unsqueeze(0))
    # partial_derivatives = torch.tensor(partial_derivatives, dtype = torch.float, requires_grad = create_graph)
    partial_derivatives = torch.vstack(partial_derivatives).permute(1, 0, 2)
    
    # To align in (num_points, num_components, num_variables), permute the dimensions (0, 1, 2) to (1, 0, 2)
    return partial_derivatives


##################################################
##################################################
# Second-order operations
def laplacian(
        outputs:        torch.Tensor,
        inputs:         torch.Tensor,
        create_graph:   bool    = False,
    ) -> torch.Tensor:
    """## Laplacian of scalar-valued functions
    """
    dim_domain  = inputs.shape[-1]
    grad_once   = compute_grad(outputs, inputs, True)
    print(grad_once)
    print(grad_once.shape)
    grad_twice  = torch.stack(
                        [
                            compute_grad(grad_once[..., [d]], inputs, create_graph = create_graph, retain_graph = True)[..., d]
                            for d in range(dim_domain)
                        ], dim = -1
                    )
    print(grad_twice)
    print(grad_twice.shape)
    return grad_twice.sum(dim = -1, keepdim = True)


def hessian(
        outputs:        torch.Tensor,
        inputs:         torch.Tensor,
        create_graph:   bool    = False
    ) -> torch.Tensor:
    """## Hessian of vector-valued functions
    ### Computation of the Hessian of a real-vector-valued function with a computational graph.
    -----
    ### Arguments
    @ `outputs` (`torch.Tensor`):
        * This is a `torch.Tensor` object of dimension 1, which is constructed depending on the entire elements of the parameter `inputs`.
        * This function aims at computing the gradient of `outputs` at `inputs`.

    @ `inputs` (`torch.Tensor`):
        * This is a `torch.Tensor` object of dimension 2, at which the gradient of `outputs` shall be computed.
        * It is further required that `inputs` consists of the input points in row-wise convention.
    
    @ `create_graph` (`bool`, default: `False`):
        * This parameter is sent as the parameter `create_graph` of the function `torch.autograd.grad()`.
        * As the derivatives of vector-valued functions are generally not differentiated again, this parameter is set `False` by default.
    
    -----
    ### Return
    This function returns the Hessian of `outputs` at `inputs` as a 3-dimensional tensor of 2-dimensional Jacobian matrices.
    
    -----
    ### Remark
    1. (Dimensionality)
        `outputs` and `inputs` are required to be tensors of dimension 1 and 2, respectively.
        And it is further required that `inputs` consists of the input points (at which `outputs` is evaluated) in the row-wise convention.

    -----
    ### Examples
    
    Example 1.
    >>> (Command)
    (Output)

    """
    assert (outputs.ndim == 1), \
        "Computation of the differentials is not supported when `outputs` is not 1-dimensional. " + \
        f"('outputs.ndim': {outputs.ndim} (greater than 1))"
    
    
    # Compute the partial derivatives to form a tensor of shape (num_components, num_points, num_variables)
    gradients = compute_grad(outputs, inputs, create_graph = True, retain_graph = True)
    hessians = []
    for idx in range(gradients.shape[-1]):
        hessians.append(
            compute_grad(
                gradients[:, idx], inputs,
                create_graph = create_graph, retain_graph = True
            ).unsqueeze(0))
    hessians = torch.vstack(hessians).permute(1, 0, 2)
    
    return hessians


##################################################
##################################################
# End of file