"""
PGC.py

Penalized Gauss-Newton Conjugate Gradient (PGC) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports spatial potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""
import contextlib
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, get_device_context, forward_projection, backward_projection, get_potential_function, check_stopping_criterion, calculate_step_size_PGC
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType, PotentialShapeType, StopCriterionType
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.NoPreconditioner import NoPreconditioner
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
    # Fused Kernel for PGC Update & Clamp Positive (Zero-Allocation)
    pgc_update_kernel = cp.ElementwiseKernel(
        'float32 lam_in, float32 d, float32 alpha',
        'float32 lam_out',
        '''
        float new_val = lam_in + alpha * d;
        lam_out = new_val > 0.0f ? new_val : 0.0f;
        ''',
        'pgc_update_kernel'
    )

def PGC(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: Union[float, str] = "auto",     
    beta: float = 1.0,       
    delta: float = 0.01,   
    eta: Optional[float] = None, 
    numIterations_stepCalculation: int = 20, 
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
    Penalized Gauss-Newton Conjugate Gradient (PGC) reconstruction algorithm.
    Uses a Gauss-Newton approximation to the Hessian and a conjugate gradient update for efficient optimization.

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
        - NONE: No preconditioning (equivalent to standard Gauss-Newton)
    
    Args:
        SMatrix: System matrix object.
        y: Measurement data vector.
        numIterations: Total number of iterations.
        alpha: Step size parameter (float or 'auto' for power method estimation of Lipschitz constant)
        beta: Regularization weight parameter.
        delta: Threshold for non-quadratic potentials.
        eta: Parameter for Lipschitz estimation.
        numIterations_stepCalculation: Number of iterations for power method when alpha is "auto"
        potential_type: Type of MRF spatial potential to use.
        potential_shape: Neighborhood shape (PotentialShapeType enum).
        potential_radius: Neighborhood radius in pixels.
        stop_criterion: Criterion for stopping the iterations (StopCriterionType enum)
        stop_threshold: Threshold value for the stopping criterion (for MAX_iterations, this is ignored)
        stop_window_size: Window size (used to avoid early stop due to oscillations)
        isSavingEachIteration: If True, stores intermediate reconstructions.
        isCostFunction: If True, tracks the cost function history.
        withTumor: Flag for description.
        max_saves: Limit on stored iterations.
        show_logs: Displays tqdm progress bar.
        show_criterion: If True, shows stopping criterion evolution in progress bar
    """
    xp = get_array_module(SMatrix)
    is_gpu = (xp.__name__ == 'cupy')

    with get_device_context(SMatrix):
        Z, X = SMatrix.Z, SMatrix.X
        ZX = Z * X

        if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
            raise ValueError(f"[AOT-biomaps] Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

        y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
        lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)
        
        # Conjugate Gradient vectors: r (residual), d (direction)
        r = xp.zeros_like(lambda_flat)
        d = xp.zeros_like(lambda_flat)
        residual_buffer = xp.empty_like(y_flat)
        prev_r_dot = 0.0

        preconditioner = NoPreconditioner(SMatrix=SMatrix)  # No preconditioning for PGC
        alpha = calculate_step_size_PGC(SMatrix, preconditioner, eta, numIterations_stepCalculation, show_logs) if alpha == "auto" else alpha

        # Setup save indices
        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

        saved_lambda = []
        saved_indices_list = []
        cost_history = [] if isCostFunction else None
        window_history = []

        description = f"[AOT-biomaps] PGC ({SMatrix.matrix_type.name}) with {potential_type.name} (shape: {potential_shape.name}, r: {potential_radius}) β={beta} ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        for it in iterator:
            prev_lambda = lambda_flat.copy() if stop_criterion != StopCriterionType.MAX_ITERATIONS else None

            q_flat = forward_projection(SMatrix, lambda_flat)
            xp.subtract(q_flat, y_flat, out=residual_buffer)

            # Compute potential gradient dynamically (Hessian is not used in PGC)
            grad_U, _, U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=True, compute_hess=False, compute_energy=isCostFunction)

            if isCostFunction:
                cost_history.append(0.5 * float(xp.vdot(residual_buffer, residual_buffer)) + U_value)

            grad_fidelity = backward_projection(SMatrix, residual_buffer)
            xp.add(grad_fidelity, grad_U, out=r)
            xp.negative(r, out=r)
            r_dot = float(xp.vdot(r, r))

            # Update conjugate direction: d = r + (r_dot / prev_r_dot) * d
            if it == 0:
                xp.copyto(d, r)
            else:
                beta_cg = r_dot / (prev_r_dot + 1e-10)
                # In-place equivalence of: d = r + beta_cg * d
                d *= beta_cg
                d += r

            if is_gpu:
                pgc_update_kernel(lambda_flat, d, float(alpha), lambda_flat)
            else:
                lambda_flat += alpha * d
                np.maximum(lambda_flat, 0.0, out=lambda_flat)
            
            prev_r_dot = r_dot

            # Stopping Criterion
            if stop_criterion != StopCriterionType.MAX_ITERATIONS:
                if SMatrix.experiment.OpticImage is None:
                    ground_truth = None
                else:
                    ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
                gradient_for_stop = r if stop_criterion == StopCriterionType.GRADIENT_NORM else None
                isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, window_size=stop_window_size, history=cost_history, ground_truth=ground_truth, gradient=gradient_for_stop, window_history=window_history)
                if show_logs and show_criterion:
                    iterator.set_postfix_str(f"{stop_criterion.name}: {val:.2e}")
                if isStop:
                    if show_logs: print(f"\n[AOT-biomaps] Stopping criterion {stop_criterion.name} reached at iteration {it}.")
                    cost_history.pop() if isCostFunction else None
                    break
                
            if isSavingEachIteration and it in save_indices:
                saved_lambda.append(lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X).copy())
                saved_indices_list.append(it)

        final_result = lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X)
        return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)