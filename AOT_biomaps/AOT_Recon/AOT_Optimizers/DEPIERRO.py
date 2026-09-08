"""
DEPIERRO.py

DEPIERRO algorithm (De Pierro's optimization transfer for EM reconstruction).
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""
import contextlib
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, get_device_context, forward_projection, backward_projection, get_potential_function, check_stopping_criterion
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

# =====================================================================
# CuPy Kernels definition for fusion of operations (Zero-Allocation)
# =====================================================================
if CUPY_AVAILABLE:
    # Kernel for: (y - q) / max(q, eps)
    depierro_ratio_kernel = cp.ElementwiseKernel(
        'float32 y, float32 q, float32 eps',
        'float32 out',
        '''
        float q_safe = q < eps ? eps : q;
        out = (y - q) / q_safe;
        ''',
        'depierro_ratio_kernel'
    )

    # Kernel for the complete update of De Pierro with positive clamping
    depierro_update_kernel = cp.ElementwiseKernel(
        'float32 lam, float32 backproj, float32 grad, float32 hess, float32 sens, float32 eps',
        'float32 lam_out',
        '''
        float denominator = sens + lam * hess;
        if (denominator < eps) denominator = eps;
        
        float step = lam * (backproj - grad) / denominator;
        float new_val = lam + step;
        
        lam_out = new_val > 0.0f ? new_val : 0.0f;
        ''',
        'depierro_update_kernel'
    )

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
    stop_window_size: int = 5,
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
        - NONE: No regularization
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
        delta: Huber threshold (only used if potential_type is HUBER)
        potential_type: Type of potential function to use
        potential_shape: Neighborhood shape (PotentialShapeType enum)
        potential_radius: Neighborhood radius in pixels
        stop_criterion: Criterion for stopping the iterations (StopCriterionType enum)
        stop_threshold: Threshold value for the stopping criterion (for MAX_iterations, this is ignored)
        stop_window_size: Window size (used to avoid early stop due to oscillations)
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
    is_gpu = (xp.__name__ == 'cupy')

    with get_device_context(SMatrix):
        Z, X = SMatrix.Z, SMatrix.X
        ZX = Z * X

        y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
        lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)
        ratio_buffer = xp.empty_like(y_flat)

        # Pre-compute sensitivity image (A^T * 1)
        sens_img = backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32))
        xp.maximum(sens_img, 1e-10, out=sens_img)

        # Setup save indices
        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()
        
        saved_lambda = []
        saved_indices_list = []
        cost_history = [] if isCostFunction else None
        window_history = []

        description = f"[AOT-biomaps] DEPIERRO ({SMatrix.matrix_type.name})  with {potential_type.name} potential (shape: {potential_shape.name}, radius: {potential_radius}) β={beta} & δ={delta} ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        for it in iterator:
            prev_lambda = lambda_flat.copy() if stop_criterion != StopCriterionType.MAX_ITERATIONS else None
            q_flat = forward_projection(SMatrix, lambda_flat)
            
            if is_gpu:
                depierro_ratio_kernel(y_flat, q_flat, 1e-10, ratio_buffer)
            else:
                xp.maximum(q_flat, 1e-10, out=q_flat)
                xp.subtract(y_flat, q_flat, out=ratio_buffer)
                xp.divide(ratio_buffer, q_flat, out=ratio_buffer)

            backproj_ratio = backward_projection(SMatrix, ratio_buffer)

            # Compute potential Gradient and Hessian dynamically
            grad_U, hess_U, U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=True, compute_hess=True, compute_energy=isCostFunction, use_surrogate_hessian=True)

            # Track cost function (Negative Log-Likelihood + Penalty)
            if isCostFunction:
                q_safe = xp.maximum(q_flat, 1e-10)
                cost_history.append(float(xp.sum(q_safe - y_flat * xp.log(q_safe)) + U_value))

            # De Pierro Update: λ = λ + (λ * (A^T * ((y - Ax) / Ax) - grad_U)) / (A^T * 1 + λ * hess_U)
            if is_gpu:
                depierro_update_kernel(lambda_flat, backproj_ratio, grad_U, hess_U, sens_img, 1e-10, lambda_flat)
            else:
                # Fallback CPU
                denominator = sens_img + lambda_flat * hess_U
                xp.maximum(denominator, 1e-10, out=denominator)
                
                update_step = backproj_ratio - grad_U
                update_step *= lambda_flat
                update_step /= denominator
                
                lambda_flat += update_step
                xp.maximum(lambda_flat, 0.0, out=lambda_flat)

            # Stopping Criterion
            if stop_criterion != StopCriterionType.MAX_ITERATIONS:
                if SMatrix.experiment.OpticImage is None:
                    ground_truth = None
                else:
                    ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
                isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, window_size=stop_window_size, history=cost_history, ground_truth=ground_truth, gradient=None, window_history=window_history)
                if show_logs and show_criterion:
                    iterator.set_postfix_str(f"{stop_criterion.name}: {val:.2e}")
                if isStop:
                    if show_logs: print(f"\n[AOT-biomaps] Stopping Criterion {stop_criterion.name} reached at iteration {it}.")
                    cost_history.pop() if isCostFunction else None
                    break
            
            if isSavingEachIteration and it in save_indices:
                saved_lambda.append(lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X).copy())
                saved_indices_list.append(it)

        final_result = lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X)
        return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)
        