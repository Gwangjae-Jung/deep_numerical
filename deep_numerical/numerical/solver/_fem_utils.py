from    typing          import  Callable, Optional, Sequence, Union
import  numpy           as      np
import  scipy.sparse    as      sp


##################################################
##################################################
__all__: list[str] = [
    'rectangular_mesh_2D',
    'element_areas_2D',
    'compute_P1_elementary_coeffs',
    'compute_P1_local_stiffness',
    'assemble_system',
    'dirichlet_condition',
]


##################################################
##################################################
# Generate mesh
def rectangular_mesh_2D(
        range_x1:       Sequence[float],
        range_x2:       Sequence[float],
        resolutions:    int | Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a triangulation on a rectagular domain."""
    for d, ab in enumerate((range_x1, range_x2)):
        if len(ab)!=2:
            raise ValueError(f"'x{d+1}' should be a sequence of two real numbers, but got {len(ab)} real numbers.")
        if ab[0]>=ab[1]:
            raise ValueError(f"At dimension {d}, we have a={ab[0]:.2f} and b={ab[1]:.2f}.")
    if isinstance(resolutions, int):
        resolutions = (resolutions, resolutions)
    elif len(resolutions)!=2:
        raise ValueError(f"'resolutions' should be a sequence of two integers, but got {len(resolutions)} integers.")
    
    # Generate points
    n1, n2 = resolutions
    grid_x1 = np.linspace(*range_x1, n1)
    grid_x2 = np.linspace(*range_x2, n2)
    points  = tuple(map(np.ravel, np.meshgrid(grid_x1, grid_x2, indexing='ij')))
    points  = np.stack(points, axis=1)
    
    # Compute all elementary indices
    idx1 = np.arange(n1)
    idx2 = np.arange(n2)
    base_indices = tuple(map(np.ravel, np.meshgrid(idx1, idx2, indexing='ij')))
    base_indices = np.ravel_multi_index(base_indices, (n1, n2)).reshape(n1, n2)
    element_indices_1 = np.stack(
        (
            base_indices[:-1, :-1],
            base_indices[1:, :-1],
            base_indices[1:, 1:],
        ),
        axis=-1,
    ).reshape(-1, 3)
    element_indices_2 = np.stack(
        (
            base_indices[:-1, :-1],
            base_indices[:-1, 1:],
            base_indices[1:, 1:],
        ),
        axis=-1,
    ).reshape(-1, 3)
    element_indices = np.concatenate((element_indices_1, element_indices_2), axis=0)
    
    # Find the boundary points
    is_boundary = (
        np.isclose(points[:, 0], range_x1[0]) | np.isclose(points[:, 0], range_x1[1]) | \
        np.isclose(points[:, 1], range_x2[0]) | np.isclose(points[:, 1], range_x2[1])
    )
    
    return points, element_indices, is_boundary


def element_areas_2D(points: np.ndarray, element_indices: np.ndarray) -> np.ndarray:
    """Computes the areas of the 2-dimensional elements.
    
    Arguments:
        `points` (`numpy.ndarray`): The collection of points of shape `(num_points, 2)`.
        `element_indices` (`numpy.ndarray`): The collection of the indices of the elements, which should be given as a `numpy.ndarray` object of shape `(num_elements, 3)`.
    
    Returns:
        `area` (`numpy.ndarrray`): The area of the elements of shape `(nun_elements,)`.
    """
    assert points.ndim==2 and points.shape[-1]==2
    assert element_indices.ndim==2 and element_indices.shape[-1]==3
    elems = points[element_indices]     # Shape: (num_points, 3, 2)
    return 0.5 * np.abs(
        np.linalg.det(
            np.stack(
                (elems[:, 0]-elems[:, 1], elems[:, 0]-elems[:, 2],),
                axis = 1,
            )
        )
    )


##################################################
# P1 basis
def compute_P1_elementary_coeffs(
        points:             np.ndarray,
        element_indices:    np.ndarray,
    ) -> np.ndarray:
    """Computes the coefficients of the elementary functions for the P1 basis functions.
    
    Arguments:
        `points` (`numpy.ndarray`): The collection of points of shape `(num_points, 2)`.
        `element_indices` (`numpy.ndarray`): The collection of the indices of the elements, which should be given as a `numpy.ndarray` object of shape `(num_elements, 3)`.
    
    Returns:
        `coeffs` (`numpy.ndarrray`): The array of the coefficients of the elementary functions. The shape of the returned array is `(num_elements, 3, 3)`. For the nonnegative integers `i<num_elements` and `j<3`, `coeffs[i, j]` is the 1-dimensional array `[a, b, c]`, where the `j`-th nodal basis at the `i`-th element maps `(x, y)` in the element to `a + b*x + c*y`.
    """
    element_points = points[element_indices]    # Shape: (num_points, 3, 2)
    arr_ones = np.ones((*element_points.shape[:-1], 1), dtype=points.dtype)
    element_points = np.concatenate((arr_ones, element_points), axis=-1)
    return np.linalg.inv(element_points).transpose((0, 2, 1))  # Transpose the array so that the coefficients are aligned in the last dimension


def compute_P1_local_stiffness(
        points:             np.ndarray,
        element_indices:    np.ndarray,
        
        weight:             Optional[Callable[[np.ndarray, object], np.ndarray]] = None,
        weight_kwargs:      dict[str, object] = {},
    ) -> np.ndarray:
    """Computes the local stiffness matrix using the Gaussian quadrature.
    
    Arguments:
        `points` (`numpy.ndarray`):
            * The array of the points in the domain.
            * Shape: `(num_points, dim_domain)`
        `element_indices` (`numpy.ndarray`):
            * The array of the points in the domain.
            * Shape: `(num_elements, num_nodes_in_an_element)`
        `weight` (`Optional[Callable[[numpy.ndarray, object], numpy.ndarray]]`, default: `None`):
            * The weight function.
        `weight_kwargs` (`dict[str, object]`, default: `{}`):
            * Further configurations for the weight function.
    
    Returns:
        `stiffness` (`numpy.ndarray`): The *local* stiffness matrix of the basis elements. The shape of the returned array is `(num_elements, 3, 3)`.
    """
    P1_coeffs   = compute_P1_elementary_coeffs(points, element_indices)[..., 1:]
    area        = element_areas_2D(points, element_indices)
    stiffness: np.ndarray = np.einsum("b, bni, bki -> bnk", area, P1_coeffs, P1_coeffs)
    return stiffness


##################################################
# Assembly
def assemble_system(
        points:             np.ndarray,
        element_indices:    np.ndarray,
        source:             Callable[[np.ndarray, object], np.ndarray],
        source_kwargs:      dict[str, object]={},
    ) -> tuple[sp.lil_matrix, np.ndarray]:
    """Assemble the global stiffness matrix and load vector.
    
    ### Note
    This function assembles the global stiffness matrix and load vector *without any initial and/or boundary condition*. Therefore, to conduct FEM, one should modify the global stiffness matrix and load vector.
    
    -----
    Arguments:
        `points` (`numpy.ndarray`): The points in the mesh.
        `element_indices` (`numpy.ndarray`): The array of the points in the domain. The shape of `element_indices` should be  `(num_elements, num_nodes_in_an_element)`.
        `source` (`Callable[[numpy.ndarray, object], numpy.ndarray]`): The source function of the Poisson equation.
        `source_kwargs` (`dict[str, object]`, default: `{}`): Further configurations for the source function.
    
    Returns:
        `stiffness` (`scipy.sparse.lil_matrix`): The global stiffness matrix of the basis elements. The shape of the returned array is `(num_points, num_points)`.
        `loads` (`numpy.ndarray`): The global load vector of the system. The shape of the returned array is `(num_points,)`.
    """
    local_stiffness = compute_P1_local_stiffness(points, element_indices)
    global_stiffness = _assemble_stiffness(points.shape[0], element_indices, local_stiffness)
    global_loads = _assemble_loads(points, element_indices, source, source_kwargs)
    return global_stiffness, global_loads


def _assemble_stiffness(
        num_points:         int,
        element_indices:    np.ndarray,
        local_stiffness:    Union[np.ndarray, Sequence[np.ndarray]],
    ) -> sp.lil_matrix:
    global_stiffness = sp.lil_matrix((num_points, num_points))
    for ord_elem, elem in enumerate(element_indices):
        for i_loc, i_global in enumerate(elem):
            for j_loc, j_global in enumerate(elem):
                global_stiffness[i_global, j_global] += local_stiffness[ord_elem, i_loc, j_loc]
    return global_stiffness


def _assemble_loads(
        points:             int,
        element_indices:    np.ndarray,
        source:             Callable[[np.ndarray], np.ndarray],
        source_kwargs:      dict[str, object] = {},
    ) -> np.ndarray:
    area = element_areas_2D(points, element_indices)
    element_centroids = points[element_indices].mean(axis=1)    # Shape: `(num_elements, 2)`
    element_loads = source(element_centroids, **source_kwargs).reshape(-1) * area / 3
    global_loads = np.zeros((points.shape[0],), dtype=points.dtype)
    for elem, load in zip(element_indices, element_loads):
        for i_global in elem:
            global_loads[i_global] = global_loads[i_global] + load
    return global_loads


##################################################
# Application of the Dirichlet, Neumann conditions
def dirichlet_condition(
        stiffness:          sp.lil_matrix,
        loads:              np.ndarray,
        points:             np.ndarray,
        arg_dirichlet:      Union[np.ndarray, Sequence[int]],
        condition:          Callable[[np.ndarray], np.ndarray],
        condition_kwargs:   dict[str, object] = {},
    ) -> tuple[sp.lil_matrix, np.ndarray]:
    boundary_values = condition(points[arg_dirichlet], **condition_kwargs)
    for idx, bd_val in zip(arg_dirichlet, boundary_values):
        stiffness.rows[idx] = [idx]
        stiffness.data[idx] = [1.0]
        loads[idx] = bd_val
    return stiffness, loads


def neumann_condition(
        stiffness:          sp.lil_matrix,
        loads:              np.ndarray,
        points:             np.ndarray,
        arg_neumann:        Union[np.ndarray, Sequence[int]],
        condition:          Callable[[np.ndarray, object], np.ndarray],
        condition_kwargs:   dict[str, object] = {},
    ) -> tuple[sp.lil_matrix, np.ndarray]:
    boundary_grads = condition(points[arg_neumann], **condition_kwargs)
    for idx, bd_grad in zip(arg_neumann, boundary_grads):
        stiffness.rows[idx] = ["all points adjacent to `idx`"]# [idx]
        stiffness.data[idx] = ["corresponding gradients"]#[1.0]
        loads[idx] = bd_grad
        pass
    return stiffness, loads


##################################################
##################################################
# End of file