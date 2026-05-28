"""
DEPIERRO.py

DEPIERRO algorithm (De Pierro's optimization transfer for EM reconstruction).
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""

from AOT_biomaps.AOT_Recon.ReconTools import (
    projection, backprojection, build_adjacency_indices,
    quadratic_potential, huber_potential, relative_difference_potential,
    clamp_positive, calculate_memory_requirement, check_gpu_memory, zeros
)
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.Config import config

import numpy as np
from tqdm import trange

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def DEPIERRO(
    SMatrix,
    y,
    numIterations=100,
    beta=1.0,
    sigma=1.0,
    delta=0.01,
    potential_type=PotentialType.QUADRATIC,
    preconditioner_type=PreconditionerType.NONE,
    isSavingEachIteration=True,
    isCostFunction=False,
    withTumor=True,
    max_saves=5000,
    show_logs=True,
):
    """
    DEPIERRO reconstruction algorithm (De Pierro's optimization transfer for EM).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports potential functions:
    - QUADRATIC: p(u,v) = 0.5 * beta * (u-v)^2
    - HUBER: p(u,v,delta) = huber piecewise function
    - RELATIVE_DIFFERENCE: p(u,v,beta) = beta * (u-v)^2 / (u+v+beta*|u-v|)
    
    Supports preconditioning:
    - NONE: No preconditioning
    - DIAGONAL: Diagonal preconditioning using A^T * 1
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        beta: Regularization parameter (weight for potential)
        sigma: Additional parameter for DEPIERRO
        delta: Parameter for HUBER potential (threshold)
        potential_type: Type of potential function to use
        preconditioner_type: Type of preconditioner to use (default: NONE)
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
        theta_flat = cp.full(ZX, 0.1, dtype=cp.float32)
        array_module = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        theta_flat = np.full(ZX, 0.1, dtype=np.float32)
        array_module = np
    
    # Build adjacency for regularization
    adj_indices = build_adjacency_indices(SMatrix)
    
    # Select potential function
    def get_potential(U):
        """Get potential function based on potential_type."""
        if potential_type == PotentialType.QUADRATIC:
            return quadratic_potential(SMatrix, U, beta)
        elif potential_type == PotentialType.HUBER:
            return huber_potential(SMatrix, U, beta, delta)
        elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
            return relative_difference_potential(SMatrix, U, beta, 1.0)
        else:
            raise ValueError(f"DEPIERRO does not support potential type: {potential_type}. Use QUADRATIC, HUBER, or RELATIVE_DIFFERENCE.")
    
    # Compute preconditioner if requested
    preconditioner, preconditioner_inv = None, None
    if preconditioner_type != PreconditionerType.NONE:
        preconditioner, preconditioner_inv = build_preconditioner(SMatrix, preconditioner_type)
    
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
    
    description = f"AOT-BioMaps -- DEPIERRO ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Forward projection
        q_flat = projection(SMatrix, theta_flat)
        
        # Compute update factor
        ratio = y_flat / (q_flat + 1e-10)
        
        # Backprojection
        c_flat = backprojection(SMatrix, ratio)
        
        # Compute potential gradient and Hessian
        grad_U, hess_U, U_value = get_potential(theta_flat)
        
        # DEPIERRO update
        theta_flat = theta_flat * c_flat / (1 + sigma * hess_U)
        
        # Apply diagonal preconditioning if enabled
        if preconditioner_inv is not None:
            theta_flat = apply_diagonal_preconditioner(theta_flat, preconditioner_inv, SMatrix)
        
        # Clamp to non-negative
        theta_flat = clamp_positive(SMatrix, theta_flat)
        
        # Compute cost function if requested
        if isCostFunction:
            q_flat = projection(SMatrix, theta_flat)
            # Poisson log-likelihood + quadratic regularization
            likelihood = array_module.sum(y_flat * array_module.log(q_flat + 1e-10) - q_flat)
            cost = float(-likelihood + 0.5 * beta * array_module.sum(theta_flat**2))
            cost_history.append(cost)
        
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
