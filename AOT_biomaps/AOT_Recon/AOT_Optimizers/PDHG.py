"""
PDHG.py

Primal-Dual Hybrid Gradient (PDHG) algorithm for regularized reconstruction.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports all potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE, TOTAL_VARIATION
"""


import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import check_gpu_available, forward_projection, backward_projection, clamp_positive, build_preconditioner, get_potential_function
from AOT_biomaps.AOT_Recon.ReconEnums import NoiseType, PotentialType, PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def PDHG(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,
    delta: float = 0.01,
    tau: float = 0.1,
    sigma: float = 0.1,
    noise_type: NoiseType = NoiseType.POISSON,
    potential_type: PotentialType = PotentialType.TOTAL_VARIATION,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
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
        beta: Regularization weight (primary parameter for all potentials)
        delta: Additional parameter for RELATIVE_DIFFERENCE potential or for HUBER potential (threshold)
        tau: Step size parameter for primal update
        sigma: Step size parameter for dual update
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
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type.name
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions (T={SMatrix.T}, N={SMatrix.N}).")

    if check_gpu_available(SMatrix):
        y_flat = cp.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = cp.zeros(ZX, dtype=cp.float32)
        x_flat = cp.zeros(ZX, dtype=cp.float32)
        y_flat_pos = cp.maximum(y_flat, 1e-10)
        xp = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = np.zeros(ZX, dtype=np.float32)
        x_flat = np.zeros(ZX, dtype=np.float32)
        y_flat_pos = np.maximum(y_flat, 1e-10)
        xp = np

    # Compute preconditioner if requested
    preconditioner, preconditioner_inv = None, None
    if preconditioner_type != PreconditionerType.NONE:
        preconditioner, preconditioner_inv = build_preconditioner(SMatrix, preconditioner_type)
        # Normalize preconditioner by its mean (recommended)
        preconditioner = preconditioner / xp.mean(preconditioner)
        preconditioner_inv = 1.0 / preconditioner

    # PDHG parameters
    tau = 0.1
    sigma = 0.1

    # Adjust tau and sigma based on preconditioner
    if preconditioner_type != PreconditionerType.NONE:
        # Scale tau and sigma to account for preconditioning
        # This maintains the convergence condition: tau * sigma * ||A||^2 <= 1
        # Here, we use the mean of the preconditioner as a scaling factor
        mean_precond = xp.mean(preconditioner)
        tau = tau / mean_precond
        sigma = sigma * mean_precond

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
            Ax = forward_projection(SMatrix, x_flat)
            grad_f = backward_projection(SMatrix, 1.0 - y_flat / (Ax + 1e-10))
        else:  # GAUSSIAN
            Ax = forward_projection(SMatrix, x_flat)
            grad_f = backward_projection(SMatrix, Ax - y_flat)

        # Get potential gradient
        grad_U, _, _ = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta)

        # Primal update: x = x - tau * (grad_f + grad_U)
        x_flat = x_flat - tau * grad_f - tau * grad_U

        # Clamp to non-negative
        x_flat = clamp_positive(SMatrix, x_flat)

        # Dual update
        lambda_flat = lambda_flat + sigma * (forward_projection(SMatrix, x_flat) - y_flat_pos)

        # Compute cost function if requested
        if isCostFunction:
            Ax = forward_projection(SMatrix, x_flat)
            if noise_type == NoiseType.POISSON:
                likelihood = xp.sum(y_flat * xp.log(Ax + 1e-10) - Ax)
                cost = float(-likelihood)
            else:
                cost = 0.5 * float(xp.sum((Ax - y_flat)**2))
            _, _, potential_val = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta)
            cost += float(potential_val)
            cost_history.append(cost)

        if isSavingEachIteration and it in save_indices:
            if check_gpu_available(SMatrix):
                saved_lambda.append(cp.asnumpy(lambda_flat.reshape(Z, X)))
            else:
                saved_lambda.append(lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)

    if check_gpu_available(SMatrix):
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(x_flat.reshape(Z, X))  # Return x_flat (primal variable) instead of lambda_flat
    else:
        final_result = x_flat.reshape(Z, X)

    if isSavingEachIteration:
        return saved_lambda, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history
    