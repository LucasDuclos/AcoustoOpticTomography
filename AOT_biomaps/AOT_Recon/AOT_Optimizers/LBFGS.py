"""
LBFGS.py

Unconstrained L-BFGS optimization algorithm using variable transformation (lambda = w^2)
to enforce non-negativity natively without clipping artifacts.
Uses unified SMatrix interface and ReconTools functions.

Supports spatial potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
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
    
    # Kernel for the polynomial evaluation in the line search: r(step) = r_current + step * q1 + step^2 * q2
    line_search_kernel = cp.ElementwiseKernel(
        'float32 res, float32 q1, float32 q2, float32 step',
        'float32 res_out',
        'res_out = res + step * q1 + (step * step) * q2',
        'lbfgs_ls_kernel'
    )

    # Kernel for generating probe variables in-place: w_probe = w + step * d_w; lam_probe = w_probe^2
    lbfgs_probe_kernel = cp.ElementwiseKernel(
        'float32 w, float32 d_w, float32 step',
        'float32 w_probe, float32 lam_probe',
        '''
        w_probe = w + step * d_w;
        lam_probe = w_probe * w_probe;
        ''',
        'lbfgs_probe_kernel'
    )

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
    stop_window_size: int = 5,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
    show_criterion: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    Limited Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm with variable transformation (lambda = w^2) to enforce non-negativity.
    """    
    xp = get_array_module(SMatrix)
    is_gpu = (xp.__name__ == 'cupy')

    with get_device_context(SMatrix):
        Z, X = SMatrix.Z, SMatrix.X
        ZX = Z * X

        if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
            raise ValueError(f"[AOT-biomaps] Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

        y_flat = xp.asarray(y.T.flatten(), dtype=xp.float32)
        lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)
        w_flat = xp.sqrt(lambda_flat)

        # L-BFGS Memory initialization
        m = 10 
        s_history = []
        y_history = []
        rho_history = []

        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

        saved_lambda = []
        saved_indices_list = []
        cost_history = [] if isCostFunction else None
        window_history = []

        description = f"[AOT-biomaps] LBFGS ({SMatrix.matrix_type.name}) with {potential_type.name} (w^2) β={beta} ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        # --- Initial computations ---
        q_flat = forward_projection(SMatrix, lambda_flat)
        residual = q_flat - y_flat
        
        grad_U_lambda, _, U_value = get_potential_function(
            potential_type, SMatrix, lambda_flat, beta=beta, delta=delta, 
            shape=potential_shape, radius=potential_radius, 
            compute_grad=True, compute_hess=False, compute_energy=True, use_surrogate_hessian=False
        )
        
        # Chain rule: gradient w.r.t w
        grad_w = (backward_projection(SMatrix, residual) + grad_U_lambda) * (2.0 * w_flat)
        current_cost = 0.5 * float(xp.vdot(residual, residual)) + (float(U_value) if U_value is not None else 0.0)

        for it in iterator:
            prev_lambda = lambda_flat.copy()
            if isCostFunction:
                cost_history.append(current_cost)

            # --- L-BFGS TWO-LOOP RECURSION ---
            q = grad_w.copy()
            alphas = []
            
            # Backward pass
            for s, y_hist, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
                alpha = rho * float(xp.vdot(s, q))
                alphas.append(alpha)
                q -= alpha * y_hist 
                
            alphas.reverse()

            if len(s_history) > 0:
                gamma_k = float(xp.vdot(s_history[-1], y_history[-1])) / (float(xp.vdot(y_history[-1], y_history[-1])) + 1e-10)
            else:
                gamma_k = 1.0 / (float(xp.linalg.norm(q)) + 1e-5)
                    
            d_w = gamma_k * q
            
            # Forward pass
            for s, y_hist, rho, alpha_i in zip(s_history, y_history, rho_history, alphas):
                beta_i = rho * float(xp.vdot(y_hist, d_w))
                d_w += s * (alpha_i - beta_i) 

            d_w = -d_w

            dir_deriv = float(xp.vdot(grad_w, d_w))
            if dir_deriv >= 0.0:
                if show_logs and it > 0: print(f"\n[AOT-biomaps] Not a descent direction at {it}. Resetting L-BFGS.")
                s_history.clear()
                y_history.clear()
                rho_history.clear()
                d_w = -grad_w / (float(xp.linalg.norm(grad_w)) + 1e-8)
                dir_deriv = float(xp.vdot(grad_w, d_w))

            # --- OPTIMIZED BACKTRACKING LINE SEARCH ---
            c1 = 1e-4
            step = 1.0
            max_ls_iter = 20
            
            delta1 = xp.float32(2.0) * w_flat * d_w
            delta2 = d_w * d_w
            q1 = forward_projection(SMatrix, delta1)
            q2 = forward_projection(SMatrix, delta2)
            
            w_new = w_flat
            lambda_new = lambda_flat
            new_cost = current_cost
            res_new = residual.copy()

            res_probe_buffer = xp.empty_like(residual)
            w_probe_buffer = xp.empty_like(w_flat)
            lambda_probe_buffer = xp.empty_like(lambda_flat)

            q1_safe = q1.astype(xp.float32, copy=False) if is_gpu else q1
            q2_safe = q2.astype(xp.float32, copy=False) if is_gpu else q2

            for _ in range(max_ls_iter):
                if is_gpu:
                    lbfgs_probe_kernel(w_flat, d_w, float(step), w_probe_buffer, lambda_probe_buffer)
                else:
                    np.multiply(d_w, step, out=w_probe_buffer)
                    np.add(w_flat, w_probe_buffer, out=w_probe_buffer)
                    np.multiply(w_probe_buffer, w_probe_buffer, out=lambda_probe_buffer)
                
                if is_gpu:
                    line_search_kernel(residual, q1_safe, q2_safe, float(step), res_probe_buffer)
                else:
                    np.multiply(q1_safe, step, out=res_probe_buffer)
                    res_probe_buffer += residual
                    res_probe_buffer += (step * step) * q2_safe
                
                _, _, U_probe = get_potential_function(
                    potential_type, SMatrix, lambda_probe_buffer, beta=beta, delta=delta, 
                    shape=potential_shape, radius=potential_radius, 
                    compute_grad=False, compute_hess=False, compute_energy=True, use_surrogate_hessian=False
                )
                
                probe_cost = 0.5 * float(xp.vdot(res_probe_buffer, res_probe_buffer)) + float(U_probe)

                if probe_cost <= current_cost + c1 * step * dir_deriv:
                    w_new = w_probe_buffer.copy()
                    lambda_new = lambda_probe_buffer.copy()
                    new_cost = probe_cost
                    res_new = res_probe_buffer.copy()
                    break
                        
                step *= 0.5
            else:
                if show_logs: print(f"\n[AOT-biomaps] Stop Criterion reached (Float32) at iteration {it}. Stopping early.")
                break 

            # --- UPDATE GRADIENTS & L-BFGS HISTORY ---
            s_k = w_new - w_flat
            
            grad_f_new = backward_projection(SMatrix, res_new)
            grad_U_new, _, _ = get_potential_function(
                potential_type, SMatrix, lambda_new, 
                beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
                compute_grad=True, compute_hess=False, compute_energy=False, use_surrogate_hessian=False
            )
            
            grad_w_new = (grad_f_new + grad_U_new) * (2.0 * w_new)
            y_k = grad_w_new - grad_w
            
            curvature = float(xp.vdot(y_k, s_k))
            
            if curvature > 1e-10:
                if len(s_history) >= m:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)
                s_history.append(s_k)
                y_history.append(y_k)
                rho_history.append(1.0 / curvature)
            else:
                s_history.clear()
                y_history.clear()
                rho_history.clear()

            # Shift variables
            w_flat = w_new
            lambda_flat = lambda_new
            grad_w = grad_w_new
            residual = res_new
            current_cost = new_cost
            
            # Stopping Criterion
            if stop_criterion != StopCriterionType.MAX_ITERATIONS:
                if SMatrix.experiment.OpticImage is None:
                    ground_truth = None
                else:
                    ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
                gradient_for_stop = grad_w_new if stop_criterion == StopCriterionType.GRADIENT_NORM else None
                isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, window_size=stop_window_size, history=cost_history, ground_truth=ground_truth, gradient=gradient_for_stop, window_history=window_history)
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