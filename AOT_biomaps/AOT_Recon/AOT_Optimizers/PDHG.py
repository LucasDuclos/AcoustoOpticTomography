"""
PDHG.py

Primal-Dual Hybrid Gradient (PDHG) algorithm for TV regularization.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
"""

from AOT_biomaps.AOT_Recon.ReconTools import (
    projection, backprojection, clamp_positive, calculate_memory_requirement, 
    check_gpu_memory, zeros, tv_potential, build_adjacency_indices
)
from AOT_biomaps.AOT_Recon.ReconEnums import NoiseType
from AOT_biomaps.Config import config

import numpy as np
from tqdm import trange

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def PDHG(
    SMatrix,
    y,
    numIterations=100,
    alpha=1.0,
    noise_type=NoiseType.POISSON,
    isSavingEachIteration=True,
    isCostFunction=False,
    withTumor=True,
    max_saves=5000,
    show_logs=True,
):
    """
    Primal-Dual Hybrid Gradient (PDHG) algorithm for TV regularization.
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        alpha: Regularization weight for TV
        noise_type: Type of noise (POISSON or GAUSSIAN)
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
        theta_flat = cp.zeros(ZX, dtype=cp.float32)
        x_flat = cp.zeros(ZX, dtype=cp.float32)
        y_flat_pos = cp.maximum(y_flat, 1e-10)
        array_module = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        theta_flat = np.zeros(ZX, dtype=np.float32)
        x_flat = np.zeros(ZX, dtype=np.float32)
        y_flat_pos = np.maximum(y_flat, 1e-10)
        array_module = np
    
    # PDHG parameters
    tau = 0.1
    sigma = 0.1
    
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
    
    description = f"AOT-BioMaps -- PDHG ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Primal update
        x_prev = x_flat.copy() if hasattr(x_flat, 'copy') else x_flat + 0
        
        # Compute gradient of data fidelity term
        if noise_type == NoiseType.POISSON:
            # For Poisson: gradient is A^T * (1 - y / (A*x + eps))
            Ax = projection(SMatrix, x_flat)
            grad_f = backprojection(SMatrix, 1.0 - y_flat / (Ax + 1e-10))
        else:  # GAUSSIAN
            # For Gaussian: gradient is A^T * (A*x - y)
            Ax = projection(SMatrix, x_flat)
            grad_f = backprojection(SMatrix, Ax - y_flat)
        
        # TV proximal operator (via subgradient)
        grad_U, _, _ = tv_potential(SMatrix, x_flat, alpha)
        
        # Primal update: x = prox_{tau * G}(x - tau * grad_f)
        x_flat = x_flat - tau * grad_f - tau * alpha * grad_U
        x_flat = clamp_positive(SMatrix, x_flat)
        
        # Dual update
        theta_flat = theta_flat + sigma * (projection(SMatrix, x_flat) - y_flat_pos)
        
        # Compute cost function if requested
        if isCostFunction:
            Ax = projection(SMatrix, x_flat)
            if noise_type == NoiseType.POISSON:
                # Poisson log-likelihood
                likelihood = array_module.sum(y_flat * array_module.log(Ax + 1e-10) - Ax)
                cost = float(-likelihood)
            else:
                # LS cost
                cost = 0.5 * float(array_module.sum((Ax - y_flat)**2))
            # Add TV
            _, _, tv_val = tv_potential(SMatrix, x_flat, alpha)
            cost += float(tv_val)
            cost_history.append(cost)
        
        if isSavingEachIteration and it in save_indices:
            if device == 'gpu' and CUPY_AVAILABLE:
                saved_theta.append(cp.asnumpy(x_flat.reshape(Z, X)))
            else:
                saved_theta.append(x_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)
    
    if device == 'gpu' and CUPY_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(x_flat.reshape(Z, X))
    else:
        final_result = x_flat.reshape(Z, X)
    
    if isSavingEachIteration:
        return saved_theta, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history
