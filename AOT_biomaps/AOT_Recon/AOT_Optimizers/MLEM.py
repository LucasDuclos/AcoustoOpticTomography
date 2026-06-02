"""
MLEM.py

Maximum Likelihood Expectation Maximization (MLEM) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import (
    check_gpu_available, forward_projection, backward_projection, 
    _get_array_module, cost_function
)
from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

def MLEM(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    denominator_threshold: float = 1e-10,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    MLEM reconstruction algorithm.
    Optimized for GPU/CPU with direct array operations.
    
    Args:
        SMatrix: System matrix interface.
        y: Measurement data vector.
        numIterations: Total number of iterations.
        denominator_threshold: Small epsilon to avoid division by zero.
        isSavingEachIteration: Toggle to store intermediate states.
        isCostFunction: Toggle to track the log-likelihood history.
        withTumor: Flag for description.
        max_saves: Max number of history frames to keep in memory.
        show_logs: Display progress bar.
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type.name
    xp = _get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)

    # 1. Pre-calculate Sensitivity (A^T * 1) - The native preconditioner of EM
    sens_img = backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32)+1e-10)
    sens_img = xp.maximum(sens_img, denominator_threshold)

    save_indices = list(range(0, numIterations, max(1, numIterations // max_saves)))
    if save_indices[-1] != numIterations - 1:
        save_indices.append(numIterations - 1)

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    description = f"AOT-BioMaps -- MLEM ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # 2. Forward Projection
        q_flat = forward_projection(SMatrix, lambda_flat)

        # 3. MLEM Update: lambda = lambda * (A^T * (y / Ax)) / (A^T * 1)
        # Note: We use xp.maximum for numerical stability
        ratio = y_flat / xp.maximum(q_flat, denominator_threshold)
        correction = backward_projection(SMatrix, ratio) / sens_img
        lambda_flat = lambda_flat * correction

        # 4. Tracking cost (Negative Log-Likelihood)
        if isCostFunction:
            q_safe = xp.maximum(q_flat, 1e-10)
            llh = xp.sum(q_safe - y_flat * xp.log(q_safe))
            cost_history.append(float(llh))

        if isSavingEachIteration and it in save_indices:
            saved_lambda.append(lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)

    final_result = lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X)

    return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)