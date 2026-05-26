"""
LS.py

Least Squares reconstruction algorithms.
Uses ReconTools functions for all matrix operations.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
"""

import numpy as np
from tqdm import trange

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def LS(
    SMatrix,
    y,
    numIterations=100,
    alpha=0.01,
    isSavingEachIteration=True,
    isCostFunction=False,
    withTumor=True,
    max_saves=5000,
    show_logs=True,
):
    """
    Least Squares reconstruction using Projected Gradient Descent (PGD).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        alpha: Step size parameter
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
        
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
        - reconstructed_image: Final or list of images (Z, X)
        - saved_indices: List of saved iteration indices (None if not saving)
        - cost_history: List of cost function values (None if not requested)
    """
    from AOT_biomaps.AOT_Recon.ReconTools import (
        projection, backprojection, axpby, minus_axpy, dot_product,
        clamp_positive, zeros
    )
    
    tumor_str = "WITH" if withTumor else "WITHOUT"
    
    # Get device from SMatrix
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type
    
    # Get dimensions
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X
    TN = SMatrix.N * SMatrix.T
    
    # Convert y to appropriate format
    if device == 'gpu' and CUPY_AVAILABLE:
        y_flat = cp.asarray(y.T.flatten().astype(np.float32))
        theta_flat = cp.full(ZX, 0.1, dtype=cp.float32)
        array_module = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        theta_flat = np.full(ZX, 0.1, dtype=np.float32)
        array_module = np
    
    # Setup save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = max(1, numIterations // max_saves)
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)
    
    saved_theta = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None
    
    description = f"AOT-BioMaps -- LS ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Compute gradient: g = A^T * (A * theta - y)
        q_flat = projection(SMatrix, theta_flat)
        r_flat = minus_axpy(SMatrix, y_flat, q_flat, -1.0)  # r = y - A*theta
        g_flat = backprojection(SMatrix, r_flat)  # g = A^T * r
        
        # Compute cost function if requested
        if isCostFunction:
            # LS cost: 0.5 * ||A*theta - y||^2
            cost = 0.5 * float(array_module.sum(r_flat**2))
            cost_history.append(cost)
        
        # Update: theta = theta - alpha * g
        theta_flat = minus_axpy(SMatrix, theta_flat, g_flat, alpha)
        
        # Clamp to non-negative
        theta_flat = clamp_positive(SMatrix, theta_flat)
        
        if isSavingEachIteration and it in save_indices:
            if device == 'gpu' and CUPY_AVAILABLE:
                saved_theta.append(cp.asnumpy(theta_flat.reshape(Z, X)))
            else:
                saved_theta.append(theta_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)
    
    if device == 'gpu' and CUPY_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(theta_flat.reshape(Z, X))
    else:
        final_result = theta_flat.reshape(Z, X)
    
    if isSavingEachIteration:
        return saved_theta, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history
