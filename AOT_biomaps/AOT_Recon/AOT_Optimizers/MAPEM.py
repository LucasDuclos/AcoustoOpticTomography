"""
MAPEM.py

MAP-EM algorithm (Maximum A Posteriori Expectation Maximization / OSL).
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports spatial potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType, PotentialType, PotentialShapeType
from AOT_biomaps.AOT_Recon.ReconTools import _get_array_module, check_gpu_available, forward_projection, backward_projection, clamp_positive, get_potential_function
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
    delta: float = 1.0,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    potential_shape: PotentialShapeType = PotentialShapeType.CROSS,
    potential_radius: int = 2,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    MAP-EM reconstruction algorithm (One-Step Late).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        beta: Regularization weight
        delta: Threshold parameter for Huber or Relative Difference potentials
        potential_type: Type of potential function
        potential_shape: Neighborhood shape (PotentialShapeType enum)
        potential_radius: Neighborhood radius in pixels
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
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
    
    # Pre-compute sensitivity image (A^T * 1)
    sens_img = backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32))
    sens_img = xp.maximum(sens_img, 1e-10)

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

    description = f"AOT-BioMaps -- MAPEM ({matrix_type}) with {potential_type.name} potential (shape: {potential_shape.name}, radius: {potential_radius}) β={beta} & δ={delta} ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # 1. Forward projection
        q_flat = forward_projection(SMatrix, lambda_flat)

        # 2. Compute potential Gradient dynamically (Hessian is not used in OSL/MAP-EM)
        grad_U, _, U_value = get_potential_function(
            potential_type, SMatrix, lambda_flat, 
            beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
            compute_grad=True, compute_hess=False, compute_energy=isCostFunction
        )

        # 3. Track cost function (Negative Log-Likelihood + Penalty)
        if isCostFunction:
            q_safe = xp.maximum(q_flat, 1e-10)
            nllh = xp.sum(q_safe - y_flat * xp.log(q_safe))
            cost_history.append(float(nllh + U_value))

        # 4. Compute EM ratio: y / (A*λ + ε)
        ratio = y_flat / xp.maximum(q_flat, 1e-10)

        # 5. Backward projection: A^T * (y / Ax)
        c_flat = backward_projection(SMatrix, ratio)
        
        # 6. MAP-EM (OSL) Update Rule
        # Denominator = (A^T 1) + ∇U(λ)
        denominator = xp.maximum(sens_img + grad_U, 1e-10)
        
        lambda_flat = lambda_flat * c_flat / denominator

        # 7. Clamp to non-negative
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