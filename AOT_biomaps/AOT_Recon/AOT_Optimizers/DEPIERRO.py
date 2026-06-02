"""
DEPIERRO.py

DEPIERRO algorithm (De Pierro's optimization transfer for EM reconstruction).
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""


import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import _get_array_module, apply_preconditioner, forward_projection, backward_projection, clamp_positive, get_potential_function, check_gpu_available
from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType, PotentialType, PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def DEPIERRO(
    SMatrix: Union[SMatrix_DENSE, SMatrix_CSR, SMatrix_SELL],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,
    delta: float = 1.5,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    DEPIERRO reconstruction algorithm (De Pierro's optimization transfer for EM).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports potential functions:
    - QUADRATIC: p(u,v) = 0.5 * beta * (u-v)^2
    - HUBER: p(u,v) = beta * (0.5 * (u-v)^2 if |u-v| <= delta else delta * (|u-v| - 0.5 * delta))
    - RELATIVE_DIFFERENCE: p(u,v) = beta * (u-v)^2 / (v + epsilon)
        
    Supports preconditioning:
    - DIAGONAL: Diagonal preconditioning using A^T * 1 (NECESSARY FOR CONVERGENCE)
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data (shape: (T, N))
        numIterations: Number of iterations
        beta: Regularization parameter (weight for potential)
        delta: Additional parameter for DEPIERRO
        potential_type: Type of potential function to use
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
    xp = _get_array_module(SMatrix)
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions (T={SMatrix.T}, N={SMatrix.N}).")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)

    sens_img = backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32) + 1e-10)

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

    description = f"AOT-BioMaps -- DEPIERRO ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # Forward projection
        q_flat = forward_projection(SMatrix, lambda_flat)

        # Compute potential and its Hessian
        _, hess_U, U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta)

        if isCostFunction:
            q_safe = xp.maximum(q_flat, 1e-10)
            data_fidelity = xp.sum(q_safe - y_flat * xp.log(q_safe))
            total_cost = float(data_fidelity + U_value)
            cost_history.append(total_cost)

        # Compute ratio: y / (A*λ + ε)
        ratio = y_flat / (q_flat + 1e-10)

        # Backprojection: A^T * (y / (A*λ + ε))
        c_flat = backward_projection(SMatrix, ratio) / sens_img

        hess_U = hess_U / sens_img

        # DEPIERRO update: λ_new = λ * (M^-1 * A^T * (y / (A*λ + ε))) / (M^-1 * (1 + δ * ∇²U(λ)))
        lambda_flat = lambda_flat * c_flat / (1 + delta * hess_U)

        # Clamp to non-negative
        lambda_flat = clamp_positive(SMatrix, lambda_flat)

        if isSavingEachIteration and it in save_indices:
            if check_gpu_available(SMatrix):
                saved_lambda.append(cp.asnumpy(lambda_flat.reshape(Z, X)))
            else:
                saved_lambda.append(lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)

    if check_gpu_available(SMatrix):
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(lambda_flat.reshape(Z, X))
    else:
        final_result = lambda_flat.reshape(Z, X)

    return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)
    