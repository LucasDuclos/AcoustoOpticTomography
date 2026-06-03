"""
ReconTools.py

Unified reconstruction tools for AOT-BioMaps.
All operations (forward_projection, backward_projection, vector ops, potentials) implemented as standalone functions.
Works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

For GPU: Uses CUDA kernels from AOT_biomaps_kernels.cu when available.
For CPU: Uses NumPy operations.
"""

import os
import numpy as np
from scipy.signal.windows import hann
import warnings
from tqdm import trange

# Optional cupy imports for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from AOT_biomaps.AOT_Recon.ReconEnums import PotentialShapeType, PotentialType, PreconditionerType, StopCriterionType

# =============================================================================
# BASIC ARRAY OPERATIONS
# =============================================================================

def clamp_positive(SMatrix, x):
    """Clamp array values to be non-negative. Uses CUDA kernel when available."""   
    if check_gpu_available(SMatrix):
        sparse_mod = SMatrix.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                N = x_gpu.size
                threads = 256
                blocks = (N + threads - 1) // threads
                clamp_kernel = sparse_mod.get_function('clamp_positive_kernel')
                clamp_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                           args=[x_gpu.data.ptr, np.int32(N)])
                cp.cuda.Stream.null.synchronize()
                return x_gpu
            except Exception:
                pass
    
    xp = get_array_module(SMatrix)
    return xp.maximum(x, 0.0)

# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

def axpby(SMatrix, x, y, a, b):
    """Compute a*x + b*y. Uses CUDA kernel when available."""    
    if check_gpu_available(SMatrix):
        sparse_mod = SMatrix.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                y_gpu = cp.asarray(y) if not isinstance(y, cp.ndarray) else y
                N = x_gpu.size
                z_gpu = cp.empty(N, dtype=cp.float32)
                threads = 256
                blocks = (N + threads - 1) // threads
                axpby_kernel = sparse_mod.get_function('vector_axpby_kernel')
                axpby_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                           args=[z_gpu.data.ptr, x_gpu.data.ptr, y_gpu.data.ptr,
                                 np.float32(a), np.float32(b), np.int32(N)])
                cp.cuda.Stream.null.synchronize()
                return z_gpu
            except Exception:
                pass
    
    return a * x + b * y

def axpy(SMatrix, x, y, a):
    """Compute x + a*y. Uses CUDA kernel when available."""   
    if check_gpu_available(SMatrix):
        sparse_mod = SMatrix.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                y_gpu = cp.asarray(y) if not isinstance(y, cp.ndarray) else y
                N = x_gpu.size
                threads = 256
                blocks = (N + threads - 1) // threads
                axpy_kernel = sparse_mod.get_function('vector_axpy_kernel')
                # Note: vector_plus_axpy_kernel does r = r + alpha * z
                # We need to copy x first since it modifies in place
                r_gpu = x_gpu.copy()
                axpy_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                            args=[r_gpu.data.ptr, y_gpu.data.ptr, np.float32(a), np.int32(N)])
                cp.cuda.Stream.null.synchronize()
                return r_gpu
            except Exception:
                pass
    
    return x + a * y

def dot_product(SMatrix, x, y):
    """Compute dot product of two vectors."""
    xp = get_array_module(SMatrix)
    return xp.sum(x * y)

def vector_divide(SMatrix, x, y, epsilon=1e-12):
    """Element-wise division: x / y with protection against division by zero."""
    if check_gpu_available(SMatrix):
        if isinstance(x, np.ndarray):
            x = cp.asarray(x)
        if isinstance(y, np.ndarray):
            y = cp.asarray(y)
        xp = cp
    else:
        if isinstance(x, cp.ndarray):
            x = cp.asnumpy(x)
        if isinstance(y, cp.ndarray):
            y = cp.asnumpy(y)
        xp = np

    if check_gpu_available(SMatrix):
        sparse_mod = SMatrix.sparse_mod
        if sparse_mod is not None:
            try:
                N = x.size
                result_gpu = xp.empty(N, dtype=xp.float32)
                threads = 256
                blocks = (N + threads - 1) // threads
                invert_kernel = sparse_mod.get_function('invert_vector_kernel')
                invert_kernel(
                    grid=(blocks, 1),
                    block=(threads, 1, 1),
                    args=[result_gpu.data.ptr, y.data.ptr, np.float32(epsilon), np.int32(N)]
                )
                result_gpu = x * result_gpu
                cp.cuda.Stream.null.synchronize()
                return result_gpu
            except Exception:
                pass

    return x / (y + epsilon)

def apply_normalization(SMatrix, x, norm_factor):
    """Apply normalization to vector."""
    return x * norm_factor

# =============================================================================
# MATRIX-VECTOR OPERATIONS (delegated to SMatrix implementation)
# These use the CUDA kernels from SMatrix classes
# =============================================================================

def forward_projection(SMatrix, theta):
    """Forward projection: q = A * theta. Uses SMatrix.forward_projection() which calls CUDA kernels."""
    return SMatrix.forward_projection(theta)


def backward_projection(SMatrix, e):
    """Backprojection: c = A^T * e. Uses SMatrix.backward_projection() which calls CUDA kernels."""
    return SMatrix.backward_projection(e)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mse(SMatrix, lambda_true, lambda_pred):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.
    Equivalent to sklearn.metrics.mean_squared_error.
    """
    xp = get_array_module(SMatrix)

    lambda_true = xp.asarray(lambda_true)
    lambda_pred = xp.asarray(lambda_pred)
    return xp.mean((lambda_true - lambda_pred) ** 2)

# =============================================================================
# ALGORITHM FUNCTIONS
# =============================================================================

def calculate_step_size_reg(SMatrix, gamma, num_subsets, num_iters, show_logs):
    """
    Calculate step sizes tau and sigma for PDHG when alpha or sigma is "auto" in the regularized case.
    Args:
        - SMatrix: The system matrix, used to estimate the Lipschitz constant.
        - gamma: Regularization parameter
        - eta: Parameter for the step size calculation when alpha or sigma is "auto". Must be > 1 for convergence and < 2 for optimal convergence.
        - num_subsets: Number of subsets used in the algorithm (affects the effective Lipschitz constant)
        - num_iters: Number of iterations to use for the power method estimation of the Lipschitz constant
        - show_logs: If True, prints the estimated Lipschitz constant and chosen step sizes
    """
    L_estimate = estimate_operator_norm(SMatrix, num_iters=num_iters)
    L_grad = 8.0  # L_grad = 8.0 is the exact squared operator norm of the 2D finite difference matrix
    L_total = (L_estimate**2) + L_grad
    # Calculate Chambolle-Pock step sizes
    tau_val = float(0.99 / (np.sqrt(L_total) * gamma))
    sigma_val = float((0.99 * gamma / np.sqrt(L_total)) * num_subsets)
    
    if show_logs:
        print(f"Estimated Lipschitz: {L_estimate:.4f} | tau: {tau_val:.5e} | sigma: {sigma_val:.5e}")
        
    return tau_val, sigma_val
def calculate_step_size(SMatrix, eta, num_iters, show_logs):
    """
    Calculate the step size for the optimization algorithm if alpha is "auto".
    Args :
        SMatrix: The system matrix, used to estimate the Lipschitz constant.
        eta: Parameter for the step size calculation when alpha is "auto". Must be > 1 for convergence and < 2 for optimal convergence.
        num_iters: Number of iterations to use for the power method estimation of the Lipschitz constant
        show_logs: If True, prints the estimated Lipschitz constant and chosen alpha.
    """
    if eta is None:
        print("Warning: eta not set. Defaulting to 1.9.")
        eta = 1.9
    if not (1.0 < eta < 2.0):
        print(f"Warning: eta={eta} is outside (1.0, 2.0). Convergence might be suboptimal.")
    L_estimate = estimate_operator_norm(SMatrix, num_iters=num_iters)
    alpha = eta / L_estimate if L_estimate > 0 else 1.0
    if show_logs:
        print(f"Estimated Lipschitz constant: {L_estimate:.4f}, using step size alpha: {alpha:.5f}")
    return alpha

def estimate_operator_norm(SMatrix, num_iters: int = 15) -> float:
    """
    Estimate the spectral norm (largest singular value) of the forward operator A using power iteration.
     - SMatrix: The system matrix with forward_projection and backward_projection methods.
     - num_iters: Number of power iterations to perform (default 15).
    """
    xp = get_array_module(SMatrix)

    v = xp.random.rand(SMatrix.Z * SMatrix.X).astype(xp.float32)
    v /= xp.linalg.norm(v) + 1e-12

    eig = 0.0

    for _ in range(num_iters):
        Av = forward_projection(SMatrix, v)
        AtAv = backward_projection(SMatrix, Av)

        eig = xp.dot(v, AtAv)

        norm = xp.linalg.norm(AtAv)
        if norm > 1e-12:
            v = AtAv / norm

    return float(xp.sqrt(eig))

# =============================================================================
# POTENTIAL FUNCTIONS
# =============================================================================

def build_neighborhood_offsets(SMatrix, shape=PotentialShapeType.CROSS, radius=1):
    """
    Generates the spatial offsets (dz, dx) and associated weights for MRF gradients.
    It returns only the "half-neighborhood" to prevent computing identical edges twice 
    (since edge A-B is the same as B-A), which doubles GPU performance.
    
    Args:
        SMatrix: The system matrix.
        shape (PotentialShapeType): The shape of the neighborhood.
        radius (int): Maximum neighborhood distance.
        
    Returns:
        list of tuples: [(dz, dx, weight), ...]
    """
    xp = get_array_module(SMatrix)
    # 1. Vérification de la forme AVANT la boucle (Sécurité)
    if shape not in [PotentialShapeType.CROSS, PotentialShapeType.SQUARE, PotentialShapeType.CIRCLE]:
        raise ValueError(f"Unsupported neighborhood shape: {shape}, must be one of {list(PotentialShapeType)}.")

    offsets = []
    total_weight = 0.0
    
    # Iterate over the lower half of the 2D space (dz >= 0)
    for dz in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            # Skip the center pixel itself, and skip the left half of the same row (dz=0)
            # to strictly capture only half of the neighborhood edges.
            if dz == 0 and dx <= 0:
                continue
                
            dist_l2 = xp.sqrt(dz**2 + dx**2)
            dist_l1 = abs(dz) + abs(dx)
            dist_linf = max(abs(dz), abs(dx))
            
            # 2. Vérification géométrique de l'appartenance du pixel
            is_valid = False
            if shape == PotentialShapeType.CROSS:
                is_valid = (dist_l1 <= radius)
            elif shape == PotentialShapeType.SQUARE:
                is_valid = (dist_linf <= radius)
            elif shape == PotentialShapeType.CIRCLE:
                is_valid = (dist_l2 <= radius + 1e-5)

            if is_valid:
                # Standard isotropic weight is 1 / Euclidean distance
                weight = 1.0 / dist_l2
                offsets.append((dz, dx, weight))
                total_weight += weight * 2.0 # Multiply by 2 since we only compute half-edges
                
    # Normalize weights so the sum equals the standard 4-connectivity base (sum = 4.0).
    # This ensures your 'beta' parameter keeps the same scale regardless of the radius.
    normalization_factor = 4.0 / (total_weight + 1e-10)
    offsets = [(dz, dx, w * normalization_factor) for dz, dx, w in offsets]
    
    return offsets

def get_potential_function(potential_type, SMatrix, U, beta, shape, radius, delta=None, compute_grad=True, compute_hess=True, compute_energy=True):
    """
    Get potential function derivatives and energy dynamically.
    Returns (grad_U, hess_U, U_value). Elements are None if their compute flag is False.
    """
    xp = get_array_module(SMatrix)
    
    if potential_type == PotentialType.NONE:
        return (xp.zeros_like(U) if compute_grad else None, xp.zeros_like(U) if compute_hess else None, 0.0 if compute_energy else None)
    if potential_type == PotentialType.QUADRATIC:
        return quadratic_potential(SMatrix, U, beta, shape, radius, compute_grad, compute_hess, compute_energy)
    elif potential_type == PotentialType.HUBER:
        return huber_potential(SMatrix, U, beta, delta, shape, radius, compute_grad, compute_hess, compute_energy)
    elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
        return relative_difference_potential(SMatrix, U, beta, delta, shape, radius, compute_grad, compute_hess, compute_energy)
    elif potential_type == PotentialType.TOTAL_VARIATION:
        raise ValueError("Total Variation potential is not differentiable and thus not implemented in this framework. Consider using Huber potential with a small delta for an edge-preserving approximation.")
    else:
        raise ValueError(f"Unsupported potential type: {potential_type}")

def quadratic_potential(SMatrix, U, beta, shape="cross", radius=1, compute_grad=True, compute_hess=True, compute_energy=True):
    """
    True Spatial Quadratic Potential (Tikhonov / Markov Random Field).
    Penalizes the squared difference between neighboring pixels to smooth the image.
    Uses vectorized 2D array shifting for extreme GPU performance (no atomics).
    
    Returns:
        tuple: (grad_U, hess_U, U_value). Elements are None if compute flag is False.
    """
    xp = get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    U_img = U.reshape(Z, X)
    
    grad_img = xp.zeros_like(U_img) if compute_grad else None
    hess_img = xp.zeros_like(U_img) if compute_hess else None
    U_value = 0.0 if compute_energy else None
    
    # Get dynamic neighborhood offsets
    offsets = build_neighborhood_offsets(SMatrix, shape=shape, radius=radius)
    
    for dz, dx, weight in offsets:
        # Create dynamic slices for the center pixel and its neighbor
        slice_c_z = slice(None, -dz) if dz > 0 else slice(None)
        slice_n_z = slice(dz, None) if dz > 0 else slice(None)
        
        if dx > 0:
            slice_c_x = slice(None, -dx)
            slice_n_x = slice(dx, None)
        elif dx < 0:
            slice_c_x = slice(-dx, None)
            slice_n_x = slice(None, dx)
        else:
            slice_c_x = slice(None)
            slice_n_x = slice(None)

        # Calculate spatial difference: U_i - U_j
        diff = U_img[slice_c_z, slice_c_x] - U_img[slice_n_z, slice_n_x]
        
        # Accumulate Gradient
        if compute_grad:
            g = beta * weight * diff
            grad_img[slice_c_z, slice_c_x] += g
            grad_img[slice_n_z, slice_n_x] -= g
            
        # Accumulate Hessian (Constant for quadratic MRF)
        if compute_hess:
            h = beta * weight
            hess_img[slice_c_z, slice_c_x] += h
            hess_img[slice_n_z, slice_n_x] += h
            
        # Accumulate Energy
        if compute_energy:
            # e = 0.5 * beta * weight * (U_i - U_j)^2
            U_value += 0.5 * beta * weight * float(xp.sum(diff**2))

    return (grad_img.flatten() if compute_grad else None, 
            hess_img.flatten() if compute_hess else None, 
            U_value)

def huber_potential(SMatrix, U, beta, delta=0.01, shape="cross", radius=1, compute_grad=True, compute_hess=True, compute_energy=True):
    """
    True Spatial Huber Potential.
    Acts as a quadratic penalty for small differences (smoothing noise) 
    and a linear penalty for large differences (preserving edges).
    Uses vectorized 2D array shifting for extreme GPU performance.
    """
    xp = get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    U_img = U.reshape(Z, X)
    
    grad_img = xp.zeros_like(U_img) if compute_grad else None
    hess_img = xp.zeros_like(U_img) if compute_hess else None
    U_value = 0.0 if compute_energy else None

    # Get dynamic neighborhood offsets
    offsets = build_neighborhood_offsets(SMatrix, shape=shape, radius=radius)

    for dz, dx, weight in offsets:
        # Create dynamic slices
        slice_c_z = slice(None, -dz) if dz > 0 else slice(None)
        slice_n_z = slice(dz, None) if dz > 0 else slice(None)
        
        if dx > 0:
            slice_c_x = slice(None, -dx)
            slice_n_x = slice(dx, None)
        elif dx < 0:
            slice_c_x = slice(-dx, None)
            slice_n_x = slice(None, dx)
        else:
            slice_c_x = slice(None)
            slice_n_x = slice(None)

        # Calculate spatial difference
        diff = U_img[slice_c_z, slice_c_x] - U_img[slice_n_z, slice_n_x]
        
        # Huber Logic (split into quadratic and linear regions)
        abs_diff = xp.abs(diff)
        mask_quad = abs_diff <= delta
        mask_lin = ~mask_quad
        
        if compute_grad:
            g = xp.zeros_like(diff)
            g[mask_quad] = beta * weight * diff[mask_quad]
            g[mask_lin] = beta * weight * delta * xp.sign(diff[mask_lin])
            
            grad_img[slice_c_z, slice_c_x] += g
            grad_img[slice_n_z, slice_n_x] -= g
            
        if compute_hess:
            h = xp.zeros_like(diff)
            h[mask_quad] = beta * weight
            # Hessian in the linear region is technically 0
            
            hess_img[slice_c_z, slice_c_x] += h
            hess_img[slice_n_z, slice_n_x] += h
            
        if compute_energy:
            e = xp.zeros_like(diff)
            e[mask_quad] = 0.5 * beta * weight * diff[mask_quad]**2
            e[mask_lin] = beta * weight * delta * (abs_diff[mask_lin] - 0.5 * delta)
            
            U_value += float(xp.sum(e))

    return (grad_img.flatten() if compute_grad else None, 
            hess_img.flatten() if compute_hess else None, 
            U_value)

def relative_difference_potential(SMatrix, U, beta, delta=1.0, shape="cross", radius=1, compute_grad=True, compute_hess=True, compute_energy=True):
    """
    Relative Difference Prior (RDP).
    Designed specifically for emission tomography (PET/SPECT) and Poisson noise.
    Smooths low-contrast regions strongly while preserving high-contrast edges.
    """
    xp = get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    eps = 1e-8 # Safety constant to prevent division by zero (0/0)
    
    U_img = U.reshape(Z, X)
    grad_img = xp.zeros_like(U_img) if compute_grad else None
    hess_img = xp.zeros_like(U_img) if compute_hess else None
    U_value = 0.0 if compute_energy else None

    # Get dynamic neighborhood offsets
    offsets = build_neighborhood_offsets(SMatrix, shape=shape, radius=radius)

    for dz, dx, weight in offsets:
        # Create dynamic slices
        slice_c_z = slice(None, -dz) if dz > 0 else slice(None)
        slice_n_z = slice(dz, None) if dz > 0 else slice(None)
        
        if dx > 0:
            slice_c_x = slice(None, -dx)
            slice_n_x = slice(dx, None)
        elif dx < 0:
            slice_c_x = slice(-dx, None)
            slice_n_x = slice(None, dx)
        else:
            slice_c_x = slice(None)
            slice_n_x = slice(None)

        # Extract neighbors: u_i and u_j
        u_i = U_img[slice_c_z, slice_c_x]
        u_j = U_img[slice_n_z, slice_n_x]
        
        diff = u_i - u_j
        abs_diff = xp.abs(diff)
        sum_ij = u_i + u_j
        
        # Denominator: u_i + u_j + gamma * |u_i - u_j| + eps
        denom = sum_ij + delta * abs_diff + eps
        
        if compute_grad:
            # Gradient formulation for RDP
            sign_diff = xp.sign(diff)
            d_denom_dui = 1.0 + delta * sign_diff
            
            g = beta * weight * (2.0 * diff * denom - (diff**2) * d_denom_dui) / (denom**2)
            
            grad_img[slice_c_z, slice_c_x] += g
            grad_img[slice_n_z, slice_n_x] -= g
            
        if compute_hess:
            # We use an approximated constant Hessian for stability in OSL/De Pierro
            # Exact Hessian for RDP can easily go negative, causing algorithms to explode
            h = beta * weight * 2.0 / denom
            
            hess_img[slice_c_z, slice_c_x] += h
            hess_img[slice_n_z, slice_n_x] += h
            
        if compute_energy:
            # e = beta * weight * (u_i - u_j)^2 / denom
            e = beta * weight * (diff**2) / denom
            U_value += float(xp.sum(e))

    return (grad_img.flatten() if compute_grad else None, 
            hess_img.flatten() if compute_hess else None, 
            U_value)

# =============================================================================
# STOPPING CRITERIA
# =============================================================================

def check_stopping_criterion(SMatrix, current_lambda, prev_lambda, criterion_type, threshold, history = None, ground_truth = None):
    """
    Evaluates stopping criteria including MSE against ground truth.
    
    Args:
        current_lambda: Current reconstruction state.
        prev_lambda: State from previous iteration.
        iteration: Current iteration index.
        criterion_type: 'relative_change', 'cost_stagnation', or 'mse'.
        threshold: The epsilon value for the criterion.
        history: List of cost values.
        ground_truth: The reference image (required for 'mse').
    """
    xp = get_array_module(SMatrix)
    
    if criterion_type == StopCriterionType.MAX_ITERATIONS:
        return False, None  # This criterion is handled in the main loop, not here.
    elif criterion_type == StopCriterionType.RELATIVE_CHANGE:
        relative_change = xp.linalg.norm(current_lambda - prev_lambda) / (xp.linalg.norm(current_lambda) + 1e-10)
        return bool(relative_change < threshold), float(relative_change)
    elif criterion_type == StopCriterionType.COST_FUNCTION:
        if history is None or len(history) < 2:
            return False, history[-1]
        relative_diff = abs(history[-2] - history[-1]) / (abs(history[-2]) + 1e-10)
        return bool(relative_diff < threshold), float(relative_diff)
    elif criterion_type == StopCriterionType.MSE:
        if ground_truth is None:
            raise ValueError("Ground truth image required for MSE stopping criterion.")
        mse = mse(SMatrix, ground_truth, current_lambda)
        return bool(mse < threshold), float(mse)
    elif criterion_type == StopCriterionType.GRADIENT_NORM:
        grad_norm = xp.linalg.norm(current_lambda)
        return bool(grad_norm < threshold), float(grad_norm)
    else:
        raise ValueError(f"Unsupported stopping criterion type: {criterion_type}, must be one of {list(StopCriterionType)}.")
        
# =============================================================================
# GRADIENT AND DIVERGENCE OPERATIONS (for TV regularization)
# =============================================================================

def gradient_2d(SMatrix, x):
    """
    Compute the 2D spatial forward gradient of a flattened image vector.
    Gradients are calculated using forward differences.
    
    Args:
        SMatrix: SMatrix instance (for dimension and device abstraction)
        x: Flattened 1D image array of shape (Z*X,)
        
    Returns:
        tuple: (grad_x, grad_z) where each component is a flattened 1D array of shape (Z*X,)
    """
    xp = get_array_module(SMatrix)
    Z = SMatrix.Z
    X = SMatrix.X
    
    # Reshape into standard 2D space to apply spatial stencils safely
    x_img = x.reshape(Z, X)
    
    grad_x_img = xp.zeros_like(x_img)
    grad_z_img = xp.zeros_like(x_img)
    
    # Forward differences: ∂x / ∂x and ∂x / ∂z
    grad_x_img[:, :-1] = x_img[:, 1:] - x_img[:, :-1]
    grad_z_img[:-1, :] = x_img[1:, :] - x_img[:-1, :]
    
    return grad_x_img.flatten(), grad_z_img.flatten()

def divergence_2d(SMatrix, p_x, p_z):
    """
    Compute the 2D spatial divergence of a flattened dual vector field (p_x, p_z).
    Divergence is calculated using backward differences (Adjoint of forward gradient).
    
    Args:
        SMatrix: SMatrix instance
        p_x: Flattened 1D x-component array of shape (Z*X,)
        p_z: Flattened 1D z-component array of shape (Z*X,)
        
    Returns:
        div_p: Flattened 1D divergence array of shape (Z*X,)
    """
    xp = get_array_module(SMatrix)
    Z = SMatrix.Z
    X = SMatrix.X
    
    # Reshape dual vector fields back into their 2D layout
    p_x_img = p_x.reshape(Z, X)
    p_z_img = p_z.reshape(Z, X)
    
    div_x = xp.zeros_like(p_x_img)
    div_z = xp.zeros_like(p_z_img)
    
    div_x[:, 0] = -p_x_img[:, 0]
    div_x[:, 1:-1] = p_x_img[:, :-2] - p_x_img[:, 1:-1]
    div_x[:, -1] = p_x_img[:, -2]
    
    div_z[0, :] = -p_z_img[0, :]
    div_z[1:-1, :] = p_z_img[:-2, :] - p_z_img[1:-1, :]
    div_z[-1, :] = p_z_img[-2, :]
    
    return (div_x + div_z).flatten()

def proj_tv(SMatrix, p, radius=1.0):
    """
    Project a concatenated 1D dual vector field p = (p_x, p_z) onto the L∞ ball.
    The projection clip magnitudes to radius lambda_tv (default 1.0 for Chambolle-Pock).
    
    Args:
        SMatrix: SMatrix instance
        p: Flattened concatenated array of shape (2 * Z * X,)
        radius: Bound constraint radius of the L∞ ball (corresponds to weighted TV regularization strength)
        
    Returns:
        p_projected: Flattened projected array of shape (2 * Z * X,)
    ```"""
    xp = get_array_module(SMatrix)
    ZX = SMatrix.Z * SMatrix.X
    
    # Separate the stacked dual vector components cleanly
    p_x = p[:ZX]
    p_z = p[ZX:]
    
    # Compute conjoint spatial magnitude per pixel
    norm_p = xp.sqrt(p_x**2 + p_z**2 + 1e-12)
    mask = norm_p > radius
    
    # Apply threshold truncation mapping vector-wise
    p_x_proj = xp.where(mask, p_x * radius / norm_p, p_x)
    p_z_proj = xp.where(mask, p_z * radius / norm_p, p_z)
    
    return xp.concatenate([p_x_proj, p_z_proj])

# =============================================================================
# FILE I/O
# =============================================================================

def load_recon(hdr_path):
    """
    Load an Interfile (.hdr) and its binary file (.img) to reconstruct an image as Vinci does.

    Parameters:
    -----------
    - hdr_path : full path to the .hdr file

    Returns:
    --------
    - image : NumPy array containing the image
    - header : dictionary containing metadata from the .hdr file
    """
    header = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            if ':=' in line:
                key, value = line.split(':=', 1)
                key = key.strip().lower().replace('!', '')
                value = value.strip()
                header[key] = value

    data_file = header.get('name of data file')
    if data_file is None:
        raise ValueError(f"Cannot find data file associated with header file {hdr_path}")

    img_path = os.path.join(os.path.dirname(hdr_path), data_file)

    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 4) if f'matrix size [{i}]' in header]
    if shape and shape[-1] == 1:
        shape = shape[:-1]

    if not shape:
        raise ValueError("Cannot determine image shape from metadata.")

    data_type = header.get('number format', 'short float').lower()
    dtype_map = {
        'short float': np.float32,
        'float': np.float32,
        'int16': np.int16,
        'int32': np.int32,
        'uint16': np.uint16,
        'uint8': np.uint8
    }
    dtype = dtype_map.get(data_type)
    if dtype is None:
        raise ValueError(f"Unsupported data type: {data_type}")

    byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
    endianess = '<' if 'little' in byte_order else '>'

    img_size = os.path.getsize(img_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize

    if img_size != expected_size:
        raise ValueError(f"Image file size ({img_size} bytes) does not match expected size ({expected_size} bytes).")

    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)

    image = data.reshape(shape[::-1])

    rescale_slope = float(header.get('data rescale slope', 1))
    rescale_offset = float(header.get('data rescale offset', 0))
    image = image * rescale_slope + rescale_offset

    return image

# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def calculate_memory_requirement(SMatrix, y):
    """
    Calculate the memory required (in GB) for SMatrix and y.
    
    Args:
        SMatrix: Matrix object (SMatrix, SMatrix_CSR, SMatrix_SELL, SMatrix_DENSE, np.ndarray, cp.ndarray)
        y: Vector (float32)
    """
    total_bytes = 0

    try:
        matrix_size_gb = SMatrix.get_matrix_size()
        if isinstance(matrix_size_gb, dict) and 'error' in matrix_size_gb:
            raise ValueError(f"SMatrix allocation error: {matrix_size_gb['error']}")
        
        size_SMatrix = matrix_size_gb * (1024 ** 3)
        total_bytes += size_SMatrix
        print(f"SMatrix size: {matrix_size_gb:.3f} GB")
    except AttributeError:
        raise AttributeError("SMatrix must implement the get_matrix_size() method.")
    
    # Vector y
    if hasattr(y, 'nbytes'):
        size_y = y.nbytes
        total_bytes += size_y
        print(f"Vector y size: {size_y / (1024 ** 3):.3f} GB")
    else:
        raise ValueError("Vector y must be an array type exposing the .nbytes attribute.")
    
    return total_bytes / (1024 ** 3)

def check_gpu_memory(device_index, required_memory, show_logs=True):
    """Check if enough memory is available on the specified GPU."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available. Cannot check GPU memory.")
    
    free_memory = cp.cuda.runtime.memoryGetInfo()[0]
    free_memory_gb = free_memory / 1024**3
    
    if show_logs:
        print(f"Free memory on GPU {device_index}: {free_memory_gb:.2f} GB, Required memory: {required_memory:.2f} GB")
    
    return free_memory_gb >= required_memory

def check_gpu_available(SMatrix) -> bool:
    """Check if GPU operations are available."""
    if not isinstance(SMatrix.device, str) or "gpu" not in SMatrix.device:
        return False
    if not hasattr(SMatrix, 'sparse_mod'):
        return False
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy not available. Falling back to CPU.")
        SMatrix.device = 'cpu'
        return False
    return True

def get_array_module(SMatrix):
    """Get the appropriate array module based on device."""
    if check_gpu_available(SMatrix):
        return cp
    else:
        return np

# =============================================================================
# OLD CUDA/GPU FUNCTIONS (kept for backward compatibility with AnalyticRecon)
# These are CuPy-only functions used by analytic reconstruction methods
# =============================================================================

def fourierz_gpu(z, X):
    """Forward Fourier transform along z-axis (GPU)."""
    if CUPY_AVAILABLE:
        dz = float(z[1] - z[0])
        Nz = X.shape[0]
        return cp.fft.fftshift(
            cp.fft.fft(
                cp.fft.ifftshift(X, axes=0),
                axis=0
            ),
            axes=0
        ) * (Nz * dz)

def ifourierz_gpu(z, X):
    """Inverse Fourier transform along z-axis (GPU)."""
    if CUPY_AVAILABLE:
        dz = float(z[1] - z[0])
        Nz = X.shape[0]
        return cp.fft.ifftshift(
            cp.fft.ifft(
                cp.fft.fftshift(X, axes=0),
                axis=0
            ),
            axes=0
        ) * (1 / dz)

def ifourierx_gpu(F_fx_z, dx):
    """Inverse Fourier along X (axis=1), Matlab-compatible (GPU)."""
    if CUPY_AVAILABLE:
        return (
            cp.fft.ifftshift(
                cp.fft.ifft(
                    cp.fft.fftshift(F_fx_z, axes=1),
                    axis=1
                ),
                axes=1
            ) * (1.0 / dx)
        )

def EvalDelayLawOS_center(X_m, theta, DelayLAWS, ActiveLIST, c):
    """
    Return the rotation center C for each angle.
    """
    Nangle = DelayLAWS.shape[1]
    C = np.zeros((Nangle, 2))
    ct = DelayLAWS * c
    
    for i in range(Nangle):
        active_idx = np.where(ActiveLIST[:, i] == 1)[0]
        if len(active_idx) == 0:
            continue
        angle_i = np.round(theta[i], 5)
        u = np.array([np.sin(angle_i), np.cos(angle_i)])
        X0 = X_m - u[0] * ct[:, i]
        Z0 = 0 - u[1] * ct[:, i]
        if Z0[-1] - Z0[0] != 0:
            C[i, 0] = (Z0[-1]*X0[0] - Z0[0]*X0[-1]) / (Z0[-1] - Z0[0])
            C[i, 1] = 0
    return C

def rotate_theta_gpu(X, Z, Iin, theta, C):
    """GPU equivalent of RotateTheta.m"""
    if CUPY_AVAILABLE:
        X_rel = X - C[0]
        Z_rel = Z - C[1]
        c = cp.cos(theta)
        s = cp.sin(theta)
        Xout = c * X_rel + s * Z_rel
        Zout = -s * X_rel + c * Z_rel
        Xout += C[0]
        Zout += C[1]
        dx = X[0, 1] - X[0, 0]
        dz = Z[1, 0] - Z[0, 0]
        x0 = X[0, 0]
        z0 = Z[0, 0]
        ix = (Xout - x0) / dx
        iz = (Zout - z0) / dz
        coords = cp.stack([iz.ravel(), ix.ravel()])
        Iout = map_coordinates(
            Iin, coords, order=1, mode='constant', cval=0.0
        )
        return Iout.reshape(Iin.shape)

def filter_radon_gpu(fz, Fc):
    """Filter for filtered backprojection (GPU)."""
    if CUPY_AVAILABLE:
        FILTER = cp.abs(fz)
        FILTER = cp.where(cp.abs(fz) > Fc, 0, FILTER)
        FILTER *= cp.exp(-2 * cp.abs(fz / Fc)**10)
        return FILTER

def filter_radon(f, N, filter_type, Fc):
    """
    Implement filters for filtered backprojection (iRadon).
    Inspired by MATLAB FilterRadon function from Mamouna Bocoum.

    Parameters:
    -----------
    f : np.ndarray
        Frequency vector (e.g., f_t or f_z).
    N : int
        Filter size (length of f).
    filter_type : str
        Filter type: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'.
    Fc : float
        Cutoff frequency.

    Returns:
    -----------
    FILTER : np.ndarray
        Filter applied to frequencies.
    """
    FILTER = np.abs(f)

    if filter_type == 'ram-lak':
        pass
    elif filter_type == 'shepp-logan':
        with np.errstate(divide='ignore', invalid='ignore'):
            FILTER = FILTER * (np.sinc(2 * f / (2 * Fc)))
        FILTER[np.isnan(FILTER)] = 1.0
    elif filter_type == 'cosine':
        FILTER = FILTER * np.cos(2 * np.pi * f / (4 * Fc))
    elif filter_type == 'hamming':
        FILTER = FILTER * (0.54 + 0.46 * np.cos(2 * np.pi * f / Fc))
    elif filter_type == 'hann':
        FILTER = FILTER * (1 + np.cos(2 * np.pi * f / (4 * Fc))) / 2
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    FILTER[np.abs(f) > Fc] = 0
    FILTER = FILTER * np.exp(-2 * (np.abs(f) / Fc)**10)

    return FILTER

# =============================================================================
# PRECONDITIONERS
# =============================================================================

def _compute_diagonal_preconditioner(SMatrix):
    """
    Compute diagonal preconditioner: M = diag(A^T * 1)
    
    The diagonal preconditioner is computed as the sum of absolute values of each column
    of the system matrix A, which equals A^T * 1.
    
    This is commonly used in iterative reconstruction to normalize the sensitivity.
    
    Args:
        SMatrix: SMatrix instance (DENSE, CSR, or SELL) - must be allocated
        
    Returns:
        preconditioner: Diagonal vector (Z*X,) with A^T * 1 values, clamped to avoid zeros
        
    Compatible with: All SMatrix types (DENSE, CSR, SELL) and all devices (CPU, GPU)
    """
    xp = get_array_module(SMatrix)
    
    # Compute A^T * 1 (column sums)
    ones = xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32)
    
    if check_gpu_available(SMatrix):
        # Use GPU backprojection
        if hasattr(SMatrix, 'backward_projection'):
            preconditioner = SMatrix.backward_projection(ones)
        else:
            # Fallback: use CPU and convert
            preconditioner_cpu = SMatrix.backward_projection(cp.asnumpy(ones))
            preconditioner = cp.asarray(preconditioner_cpu)
    else:
        # Use CPU backprojection
        preconditioner = SMatrix.backward_projection(ones)
    
    # Ensure preconditioner is on correct device
    if check_gpu_available(SMatrix):
        preconditioner = cp.asarray(preconditioner)
    else:
        preconditioner = np.asarray(preconditioner)
    
    # Clamp to avoid division by zero
    preconditioner = xp.maximum(preconditioner, 1e-10)

    return preconditioner

def _apply_diagonal_preconditioner(U, preconditioner, SMatrix):
    """
    Apply diagonal preconditioner to a vector: U -> M^-1 * U
    
    Args:
        U: Vector to precondition (Z*X,)
        preconditioner: Diagonal preconditioner (Z*X,)
        SMatrix: SMatrix instance (for device information)
        
    Returns:
        Preconditioned vector on the same device as input
        
    Compatible with: All SMatrix types and all devices (CPU, GPU)
    """
    xp = get_array_module(SMatrix)
    
    # Ensure arrays are on the same device
    U = xp.asarray(U)
    preconditioner = xp.asarray(preconditioner)
    
    # Apply diagonal preconditioning: element-wise multiplication
    return U / preconditioner

def build_preconditioner(SMatrix, preconditioner_type):
    """
    Build preconditioner based on type.
    
    Args:
        SMatrix: SMatrix instance (must be allocated)
        preconditioner_type: PreconditionerType enum value
        
    Returns:
        preconditioner or None if NONE
        
    Compatible with: All SMatrix types and all devices (CPU, GPU)
    """    
    if preconditioner_type == PreconditionerType.NONE:
        return None
    elif preconditioner_type == PreconditionerType.DIAGONAL:
        return _compute_diagonal_preconditioner(SMatrix)
    else:
        raise ValueError(f"Unknown preconditioner type: {preconditioner_type}")

def apply_preconditioner(U, preconditioner, SMatrix):
    """
    Apply the specified preconditioner to vector U.
    
    Args:
        U: Vector to precondition (Z*X,)
        preconditioner: The preconditioner (Z*X,) or None if no preconditioning
        SMatrix: SMatrix instance (for device information)
        
    Returns:
        Preconditioned vector on the same device as input
    """
    if preconditioner is None:
        return U
    else:
        return _apply_diagonal_preconditioner(U, preconditioner, SMatrix)