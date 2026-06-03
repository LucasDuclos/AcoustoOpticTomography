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

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, forward_projection, backward_projection, clamp_positive, get_potential_function, check_stopping_criterion
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType, PotentialShapeType, StopCriterionType
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
    potential_shape: PotentialShapeType = PotentialShapeType.CROSS,
    potential_radius: int = 2,
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
    DEPIERRO reconstruction algorithm (De Pierro's optimization transfer for EM).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports potential functions:
    - QUADRATIC: 0.5 * β * (u-v)^2
    - HUBER: β * (0.5 * (u-v)^2 if |u-v| <= δ else δ * (|u-v| - 0.5 * δ))
    - RELATIVE_DIFFERENCE: β * (u-v)^2 / (v + ε)

    Supports stopping criteria:
    - MAX_ITERATIONS: Stop after a fixed number of iterations
    - RELATIVE_CHANGE: Stop when relative change in lambda is below threshold
    - COST_FUNCTION: Stop when cost function value changes by less than threshold
    - MSE: Stop when mean squared error with respect to ground truth is below threshold (requires ground truth) ONLY FOR SIMULATED DATA 
    - GRADIENT_NORM: Stop when norm of the update step is below threshold

    Supports preconditioning:
    - DIAGONAL: Diagonal preconditioning using A^T * 1 (NECESSARY FOR CONVERGENCE)
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data (shape: (T, N))
        numIterations: Number of iterations
        beta: Regularization parameter (weight for potential)
        delta: Additional parameter for DEPIERRO
        potential_type: Type of potential function to use
        potential_shape: Neighborhood shape (PotentialShapeType enum)
        potential_radius: Neighborhood radius in pixels
        stop_criterion: Criterion for stopping the iterations (StopCriterionType enum)
        stop_threshold: Threshold value for the stopping criterion (for MAX_iterations, this is ignored)
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
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

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)

    # Pre-compute sensitivity image (A^T * 1)
    sens_img = xp.maximum(backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32)), 1e-10)

    # Setup save indices
    save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    description = f"AOT-BioMaps -- DEPIERRO ({SMatrix.matrix_type.name})  with {potential_type.name} potential (shape: {potential_shape.name}, radius: {potential_radius}) β={beta} & δ={delta} ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        prev_lambda = lambda_flat.copy()
        q_flat = forward_projection(SMatrix, lambda_flat)

        # Compute potential Gradient and Hessian dynamically
        grad_U, hess_U, U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=True, compute_hess=True, compute_energy=isCostFunction)

        # Track cost function (Negative Log-Likelihood + Penalty)
        if isCostFunction:
            cost_history.append(float(xp.sum(xp.maximum(q_flat, 1e-10) - y_flat * xp.log(xp.maximum(q_flat, 1e-10))) + U_value))

        # De Pierro Update: λ = λ + (λ * (A^T * ((y - Ax) / Ax) - grad_U)) / (A^T * 1 + λ * hess_U)
        lambda_flat = clamp_positive(SMatrix, lambda_flat + lambda_flat * (backward_projection(SMatrix, (y_flat - q_flat) / xp.maximum(q_flat, 1e-10)) - grad_U) / xp.maximum(sens_img + lambda_flat * hess_U, 1e-10))

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
    