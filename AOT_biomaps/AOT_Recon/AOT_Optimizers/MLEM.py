"""
MLEM.py

Maximum Likelihood Expectation Maximization (MLEM) reconstruction algorithm.
Uses ReconTools functions for all matrix operations.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def MLEM(
    SMatrix,
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    denominator_threshold: float = 1e-6,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    MLEM reconstruction algorithm.
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data (can be numpy array or cupy array)
        numIterations: Number of iterations
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        denominator_threshold: Threshold for denominator to avoid division by zero
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
        
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
        - reconstructed_image: Final or list of images (Z, X)
        - saved_indices: List of saved iteration indices (None if not saving)
        - cost_history: List of cost function values (None if not requested)
    """
    from AOT_biomaps.AOT_Recon.ReconTools import (
        projection, backprojection, vector_divide, apply_normalization
    )
    
    tumor_str = "WITH" if withTumor else "WITHOUT"
    
    # Get device from SMatrix
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type
    
    # Get matrix dimensions
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X
    
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
    
    description = f"AOT-BioMaps -- MLEM ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Forward projection
        q_flat = projection(SMatrix, theta_flat)
        
        # MLEM update: theta_new = theta * (A^T * (y / (A*theta + eps))) / (A^T * 1)
        # Compute denominator with threshold
        denominator = q_flat + denominator_threshold
        ratio = vector_divide(SMatrix, y_flat, denominator, epsilon=denominator_threshold)
        
        # Backprojection of ratio
        numerator = backprojection(SMatrix, ratio)
        
        # Backprojection of ones (sensitivity)
        sensitivity = backprojection(SMatrix, array_module.ones_like(q_flat))
        
        # MLEM update
        theta_flat = theta_flat * numerator / (sensitivity + denominator_threshold)
        
        # Compute cost function if requested (Poisson log-likelihood)
        if isCostFunction:
            # Cost = sum(y * log(q + eps) - q) where q = A*theta
            q_flat = projection(SMatrix, theta_flat)
            cost = float(array_module.sum(y_flat * array_module.log(q_flat + 1e-10) - q_flat))
            cost_history.append(-cost)  # Negative log-likelihood
        
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
