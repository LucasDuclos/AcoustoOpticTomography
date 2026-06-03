"""
MLEM.py

Maximum Likelihood Expectation Maximization (MLEM) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, forward_projection, backward_projection, check_stopping_criterion
from AOT_biomaps.AOT_Recon.ReconEnums import StopCriterionType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def MLEM(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    denominator_threshold: float = 1e-10,
    stop_criterion: StopCriterionType = StopCriterionType.MAX_ITERATIONS,
    stop_threshold: float = 100.0,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
    show_criterion: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    MLEM reconstruction algorithm. Optimized for GPU/CPU with direct array operations.

    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

    Supports potential functions:
        - NONE: No regularization

    Supports stopping criteria:
        - MAX_ITERATIONS: Stop after a fixed number of iterations
        - RELATIVE_CHANGE: Stop when relative change in lambda is below threshold
        - COST_FUNCTION: Stop when cost function value changes by less than threshold
        - MSE: Stop when mean squared error with respect to ground truth is below threshold (requires ground truth) ONLY FOR SIMULATED DATA 
        - GRADIENT_NORM: Stop when norm of the update step is below threshold
    
    Args:
        SMatrix: System matrix interface.
        y: Measurement data vector.
        numIterations: Total number of iterations.
        denominator_threshold: Small epsilon to avoid division by zero.
        stop_criterion: Criterion for stopping the iterations
        stop_threshold: Threshold value for the stopping criterion (for MAX_iterations, this is ignored)
        isSavingEachIteration: Toggle to store intermediate states.
        isCostFunction: Toggle to track the log-likelihood history.
        withTumor: Flag for description.
        max_saves: Max number of history frames to keep in memory.
        show_logs: Display progress bar.
        show_criterion: If True, shows stopping criterion evolution in progress bar
    
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
        - reconstructed_image: Final or list of images (Z, X)
        - saved_indices: List of saved iteration indices (None if not saving)
        - cost_history: List of cost function values (None if not requested)
    """
    xp = get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)

    # Pre-calculate Sensitivity (A^T * 1) - The native preconditioner of EM
    sens_img = xp.maximum(backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32)+1e-10), denominator_threshold)

    # Setup save indices
    save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    description = f"AOT-BioMaps -- MLEM ({SMatrix.matrix_type.name}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        prev_lambda = lambda_flat.copy()
        q_flat = forward_projection(SMatrix, lambda_flat)

        # MLEM Update: lambda = lambda * (A^T * (y / Ax)) / (A^T * 1)
        lambda_flat = lambda_flat * backward_projection(SMatrix, y_flat / xp.maximum(q_flat, denominator_threshold)) / sens_img

        # Track cost function (Negative Log-Likelihood)
        if isCostFunction:
            cost_history.append(float(xp.sum(xp.maximum(q_flat, 1e-10) - y_flat * xp.log(xp.maximum(q_flat, 1e-10)))))

        # Stopping Criterion
        if stop_criterion != StopCriterionType.MAX_ITERATIONS:
            ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
            isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, cost_history, ground_truth)
            if show_logs and show_criterion:
                iterator.set_postfix_str(f"{stop_criterion.name}: {val:.2e}")
            if isStop:
                if show_logs: print(f"\n[Stopping] Criterion {stop_criterion.name} reached at iteration {it}.")
                break

        if isSavingEachIteration and it in save_indices:
            saved_lambda.append(lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)

    final_result = lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X)
    return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)
    