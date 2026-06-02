"""
MAPEM.py

MAP-EM algorithm (Maximum A Posteriori Expectation Maximization).
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
"""
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType, PotentialType, PreconditionerType
from AOT_biomaps.AOT_Recon.ReconTools import check_gpu_available, forward_projection, backward_projection, build_preconditioner, apply_preconditioner, clamp_positive, get_potential_function, cost_function
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def MAPEM(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,
    delta: float = 0.01,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    MAP-EM reconstruction algorithm (Maximum A Posteriori Expectation Maximization).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports preconditioning:
    - NONE: No preconditioning
    - DIAGONAL: Diagonal preconditioning using A^T * 1
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        beta: Regularization weight
        delta: Parameter for Huber potential or relative difference potential
        potential_type: Type of potential function (QUADRATIC, HUBER, RELATIVE_DIFFERENCE, TOTAL_VARIATION)
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
        lambda_flat = cp.full(ZX, 0.1, dtype=cp.float32)
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = np.full(ZX, 0.1, dtype=np.float32)

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

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    description = f"AOT-BioMaps -- MAPEM ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # Forward projection
        q_flat = forward_projection(SMatrix, lambda_flat)

        # Compute ratio: y / (A*λ + ε)
        ratio = y_flat / (q_flat + 1e-10)

        # Backward projection: A^T * (y / (A*λ + ε))
        c_flat = backward_projection(SMatrix, ratio)

        # Apply preconditioner to c_flat: M^-1 * A^T * (y / (A*λ + ε))
        if preconditioner_inv is not None:
            c_flat = apply_preconditioner(c_flat, preconditioner_inv, SMatrix)

        # Compute potential Hessian
        _, hess_U, _ = get_potential_function(lambda_flat, potential_type, SMatrix, beta=beta, delta=delta)

        # Apply preconditioner to Hessian term: M^-1 * (1 + ∇²U(λ))
        if preconditioner_inv is not None:
            hess_U = apply_preconditioner(hess_U, preconditioner_inv, SMatrix)

        # MAP-EM update: λ_new = λ * (M^-1 * A^T * (y / (A*λ + ε))) / (M^-1 * (1 + ∇²U(λ)))
        lambda_flat = lambda_flat * c_flat / (1 + hess_U)

        # Clamp to non-negative
        lambda_flat = clamp_positive(SMatrix, lambda_flat)

        # Compute cost function if requested
        if isCostFunction:
            cost_history.append(cost_function(SMatrix, lambda_flat, y_flat, optimizer=OptimizerType.MAPEM, beta=beta, delta=delta))

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

    if isSavingEachIteration:
        return saved_lambda, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history