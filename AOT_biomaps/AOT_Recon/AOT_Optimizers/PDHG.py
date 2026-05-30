"""
PDHG.py

Primal-Dual Hybrid Gradient (PDHG) algorithm for regularized reconstruction.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports all potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE, TOTAL_VARIATION
"""

from AOT_biomaps.AOT_Recon.ReconTools import forward_projection, backward_projection, clamp_positive, huber_potential, quadratic_potential, relative_difference_potential, tv_potential, build_preconditioner, apply_diagonal_preconditioner
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType, NoiseType, PreconditionerType

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
    beta=1.0,
    delta=0.01,
    noise_type=NoiseType.POISSON,
    potential_type=PotentialType.TOTAL_VARIATION,
    preconditioner_type=PreconditionerType.NONE,
    isSavingEachIteration=True,
    isCostFunction=False,
    withTumor=True,
    max_saves=5000,
    show_logs=True,
):
    """
    Primal-Dual Hybrid Gradient (PDHG) algorithm for regularized reconstruction.
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports all potential functions:
    - QUADRATIC: p(u,v) = 0.5 * alpha * (u-v)^2
    - HUBER: p(u,v,delta) = huber piecewise function
    - RELATIVE_DIFFERENCE: p(u,v,beta) = alpha * (u-v)^2 / (u+v+beta*|u-v|)
    - TOTAL_VARIATION: p(u,v) = alpha * |u-v| (non-differentiable)
    
    Supports preconditioning:
    - NONE: No preconditioning
    - DIAGONAL: Diagonal preconditioning using A^T * 1
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        alpha: Regularization weight (primary parameter for all potentials)
        beta: Additional parameter for RELATIVE_DIFFERENCE potential
        delta: Parameter for HUBER potential (threshold)
        noise_type: Type of noise (POISSON or GAUSSIAN)
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
    
    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions (T={SMatrix.T}, N={SMatrix.N}).")
    
    # Convert y to appropriate format
    if device == 'gpu' and CUPY_AVAILABLE:
        y_flat = cp.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = cp.zeros(ZX, dtype=cp.float32)
        x_flat = cp.zeros(ZX, dtype=cp.float32)
        y_flat_pos = cp.maximum(y_flat, 1e-10)
        array_module = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = np.zeros(ZX, dtype=np.float32)
        x_flat = np.zeros(ZX, dtype=np.float32)
        y_flat_pos = np.maximum(y_flat, 1e-10)
        array_module = np
    
    # Select potential function
    def get_potential(U):
        """Get potential function based on potential_type."""
        if potential_type == PotentialType.QUADRATIC:
            return quadratic_potential(SMatrix, U, alpha)
        elif potential_type == PotentialType.HUBER:
            return huber_potential(SMatrix, U, alpha, delta)
        elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
            return relative_difference_potential(SMatrix, U, alpha, beta)
        elif potential_type == PotentialType.TOTAL_VARIATION:
            return tv_potential(SMatrix, U, alpha)
        else:
            raise ValueError(f"Unsupported potential type: {potential_type}")
    
    # Compute preconditioner if requested
    _, preconditioner_inv = None, None
    if preconditioner_type != PreconditionerType.NONE:
        _, preconditioner_inv = build_preconditioner(SMatrix, preconditioner_type)
    
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
    
    saved_lambda = []
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
            Ax = forward_projection(SMatrix, x_flat)
            grad_f = backward_projection(SMatrix, 1.0 - y_flat / (Ax + 1e-10))
        else:  # GAUSSIAN
            # For Gaussian: gradient is A^T * (A*x - y)
            Ax = forward_projection(SMatrix, x_flat)
            grad_f = backward_projection(SMatrix, Ax - y_flat)
        
        # Get potential gradient
        grad_U, _, _ = get_potential(x_flat)
        
        # Primal update: x = prox_{tau * G}(x - tau * grad_f)
        x_flat = x_flat - tau * grad_f - tau * grad_U
        
        # Apply diagonal preconditioning if enabled
        if preconditioner_inv is not None:
            x_flat = apply_diagonal_preconditioner(x_flat, preconditioner_inv, SMatrix)
        
        x_flat = clamp_positive(SMatrix, x_flat)
        
        # Dual update
        lambda_flat = lambda_flat + sigma * (forward_projection(SMatrix, x_flat) - y_flat_pos)
        
        # Compute cost function if requested
        if isCostFunction:
            Ax = forward_projection(SMatrix, x_flat)
            if noise_type == NoiseType.POISSON:
                # Poisson log-likelihood
                likelihood = array_module.sum(y_flat * array_module.log(Ax + 1e-10) - Ax)
                cost = float(-likelihood)
            else:
                # LS cost
                cost = 0.5 * float(array_module.sum((Ax - y_flat)**2))
            # Add potential function value
            _, _, potential_val = get_potential(x_flat)
            cost += float(potential_val)
            cost_history.append(cost)
        
        if isSavingEachIteration and it in save_indices:
            if device == 'gpu' and CUPY_AVAILABLE:
                saved_lambda.append(cp.asnumpy(lambda_flat.reshape(Z, X)))
            else:
                saved_lambda.append(lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)
    
    if device == 'gpu' and CUPY_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(lambda_flat.reshape(Z, X))
    else:
        final_result = lambda_flat.reshape(Z, X)
    
    if isSavingEachIteration:
        return saved_lambda, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history
