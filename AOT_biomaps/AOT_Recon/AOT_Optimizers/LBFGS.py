"""
LBFGS.py

Unconstrained L-BFGS optimization algorithm using variable transformation (lambda = w^2)
to enforce non-negativity natively without clipping artifacts.
Uses unified SMatrix interface and ReconTools functions.

Supports spatial potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, forward_projection, backward_projection, get_potential_function, check_stopping_criterion
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

def LBFGS(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,
    delta: float = 0.01,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    potential_shape: PotentialShapeType = PotentialShapeType.CROSS,
    potential_radius: int = 1,
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
    Limited Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm with variable transformation (lambda = w^2) to enforce non-negativity.
    
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
        - NONE : No preconditioning  (preconditionning is directly integrated into the L-BFGS two-loop recursion)

    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement vector (shape: T x N)
        numIterations: Maximum number of iterations
        beta: Regularization strength for the potential function
        delta: Threshold parameter for Huber potential
        potential_type: Type of potential function (QUADRATIC, HUBER, RELATIVE_DIFFERENCE)
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

    """    
    xp = get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)
    w_flat = xp.sqrt(lambda_flat) # lambda = w^2 guarantees non-negativity natively.

    # L-BFGS Memory initialization
    m = 10 
    s_history = []
    y_history = []
    rho_history = []

    save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    description = f"AOT-BioMaps -- LBFGS ({SMatrix.matrix_type.name}) with {potential_type.name} (w^2 transform) β={beta} ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    # --- Initial computations ---
    q_flat = forward_projection(SMatrix, lambda_flat)
    residual = q_flat - y_flat
    
    # Compute potential Gradient dynamically (Hessian is not used in L-BFGS)
    grad_U_lambda, _, U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=True, compute_hess=False, compute_energy=True)
    
    # Chain rule: gradient w.r.t w (d_f / d_w = d_f / d_lambda * 2w)
    grad_w = backward_projection(SMatrix, residual) + grad_U_lambda * (2.0 * w_flat)
    
    current_cost = 0.5 * float(xp.sum(residual**2)) + (float(U_value) if U_value is not None else 0.0)

    for it in iterator:
        prev_lambda = lambda_flat.copy()
        if isCostFunction:
            cost_history.append(current_cost)

        # --- L-BFGS TWO-LOOP RECURSION (Operating on 'w') ---
        q = grad_w.copy()
        alphas = []
        
        for s, y_hist, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alphas.append(rho * xp.sum(s * q))
            q = q - alphas[-1] * y_hist
            
        alphas.reverse()

        gamma_k = xp.sum(s_history[-1] * y_history[-1]) / (xp.sum(y_history[-1] * y_history[-1]) + 1e-10) if len(s_history) > 0 else 1.0
            
        d_w = gamma_k * q # Initial scaling of the gradient
        
        for s, y_hist, rho, alpha_i in zip(s_history, y_history, rho_history, alphas):
            d_w = d_w + s * (alpha_i - rho * xp.sum(y_hist * d_w))

        # Search Direction
        d_w = -d_w

        # --- PURE BACKTRACKING LINE SEARCH (Unconstrained) ---
        c1 = 1e-4
        step = 1.0
        max_ls_iter = 20
        ls_success = False

        for _ in range(max_ls_iter):
            # 1. Update w
            w_probe = w_flat + step * d_w
            
            # 2. Automatically get non-negative lambda (NO CLAMP!)
            lambda_probe = w_probe ** 2
            
            # 3. Evaluate new cost
            res_probe = forward_projection(SMatrix, lambda_probe) - y_flat
            _, _, U_probe = get_potential_function(potential_type, SMatrix, lambda_probe, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=False, compute_hess=False, compute_energy=True)
            
            probe_cost = 0.5 * float(xp.sum(res_probe**2)) + float(U_probe)

            # Armijo Condition
            if probe_cost <= current_cost + c1 * step * float(xp.sum(grad_w * d_w)):
                w_new = w_probe
                lambda_new = lambda_probe
                new_cost = probe_cost
                res_new = res_probe
                break
                
            step *= 0.5

        # --- UPDATE GRADIENTS & L-BFGS HISTORY ---
        s_k = w_new - w_flat
        
        # New gradients w.r.t lambda
        grad_f_new = backward_projection(SMatrix, res_new)
        grad_U_new, _, _ = get_potential_function(
            potential_type, SMatrix, lambda_new, 
            beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
            compute_grad=True, compute_hess=False, compute_energy=False
        )
        grad_lambda_new = grad_f_new + grad_U_new
        
        # Chain rule: gradient w.r.t w
        grad_w_new = grad_lambda_new * (2.0 * w_new)
        
        y_k = grad_w_new - grad_w
        
        curvature = float(xp.sum(y_k * s_k))
        if curvature > 1e-10:
            if len(s_history) >= m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_k)
            y_history.append(y_k)
            rho_history.append(1.0 / curvature)

        # Shift variables
        w_flat = w_new
        lambda_flat = lambda_new
        grad_w = grad_w_new
        current_cost = new_cost
        
        # Stopping Criterion
        if stop_criterion != StopCriterionType.MAX_ITERATIONS:
            ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
            isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, cost_history, ground_truth)
            if show_logs and show_criterion:
                iterator.set_postfix_str(f"{stop_criterion.name}: {val:.2e}")
            if isStop:
                if show_logs: print(f"\n[Stopping] Criterion {stop_criterion.name} reached at iteration {it}.")
                break

        # Save states
        if isSavingEachIteration and it in save_indices:
            saved_lambda.append(lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)

    final_result = lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X)
    return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)