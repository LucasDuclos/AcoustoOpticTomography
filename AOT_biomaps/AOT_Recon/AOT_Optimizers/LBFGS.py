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

from AOT_biomaps.AOT_Recon.ReconTools import (
    apply_preconditioner, check_gpu_available, forward_projection, 
    backward_projection, build_preconditioner, get_potential_function, _get_array_module
)
from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType, PotentialType, PreconditionerType, PotentialShapeType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

_NON_DIFFERENTIABLE_POTENTIALS = {PotentialType.TOTAL_VARIATION}


def LBFGS(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,
    delta: float = 0.01,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    potential_shape: PotentialShapeType = PotentialShapeType.CROSS,
    potential_radius: int = 1,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    
    if potential_type in _NON_DIFFERENTIABLE_POTENTIALS:
        raise ValueError(f"LBFGS cannot handle non-differentiable potentials like {potential_type.name}. Use PDHG instead.")

    tumor_str = "WITH" if withTumor else "WITHOUT"
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type.name
    xp = _get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    
    # ---------------------------------------------------------
    # VARIABLE TRANSFORMATION: Optimize 'w' instead of 'lambda'
    # lambda = w^2 guarantees non-negativity natively.
    # ---------------------------------------------------------
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)
    w_flat = xp.sqrt(lambda_flat) 

    # L-BFGS Memory initialization
    m = 10 
    s_history = []
    y_history = []
    rho_history = []

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

    description = f"AOT-BioMaps -- LBFGS ({matrix_type}) with {potential_type.name} (w^2 transform) β={beta} ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    # --- Initial computations ---
    q_flat = forward_projection(SMatrix, lambda_flat)
    residual = q_flat - y_flat
    grad_f_lambda = backward_projection(SMatrix, residual)
    
    grad_U_lambda, _, U_value = get_potential_function(
        potential_type, SMatrix, lambda_flat, 
        beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
        compute_grad=True, compute_hess=False, compute_energy=True
    )
    
    # Standard gradient w.r.t lambda
    grad_lambda = grad_f_lambda + grad_U_lambda
    
    # Chain rule: gradient w.r.t w (d_f / d_w = d_f / d_lambda * 2w)
    grad_w = grad_lambda * (2.0 * w_flat)
    
    current_cost = 0.5 * float(xp.sum(residual**2)) + (float(U_value) if U_value is not None else 0.0)

    for it in iterator:
        if isCostFunction:
            cost_history.append(current_cost)

        # --- L-BFGS TWO-LOOP RECURSION (Operating on 'w') ---
        q = grad_w.copy()
        alphas = []
        
        for s, y_hist, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alpha_i = rho * xp.sum(s * q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_hist
            
        alphas.reverse()

        if len(s_history) > 0:
            gamma_k = xp.sum(s_history[-1] * y_history[-1]) / (xp.sum(y_history[-1] * y_history[-1]) + 1e-10)
        else:
            gamma_k = 1.0
            
        d_w = gamma_k * q
        
        for s, y_hist, rho, alpha_i in zip(s_history, y_history, rho_history, alphas):
            beta_i = rho * xp.sum(y_hist * d_w)
            d_w = d_w + s * (alpha_i - beta_i)

        # Search Direction
        d_w = -d_w

        # --- PURE BACKTRACKING LINE SEARCH (Unconstrained) ---
        c1 = 1e-4
        step = 1.0
        max_ls_iter = 20
        ls_success = False

        for ls_iter in range(max_ls_iter):
            # 1. Update w
            w_probe = w_flat + step * d_w
            
            # 2. Automatically get non-negative lambda (NO CLAMP!)
            lambda_probe = w_probe ** 2
            
            # 3. Evaluate new cost
            q_probe = forward_projection(SMatrix, lambda_probe)
            res_probe = q_probe - y_flat
            _, _, U_probe = get_potential_function(
                potential_type, SMatrix, lambda_probe, 
                beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
                compute_grad=False, compute_hess=False, compute_energy=True
            )
            
            probe_cost = 0.5 * float(xp.sum(res_probe**2)) + float(U_probe)

            # Armijo Condition
            if probe_cost <= current_cost + c1 * step * float(xp.sum(grad_w * d_w)):
                w_new = w_probe
                lambda_new = lambda_probe
                new_cost = probe_cost
                q_new = q_probe
                res_new = res_probe
                ls_success = True
                break
                
            step *= 0.5

        if not ls_success:
            if len(s_history) > 0:
                s_history.clear()
                y_history.clear()
                rho_history.clear()
                continue
            else:
                if show_logs:
                    print(f"\n[L-BFGS] Convergence atteinte à l'itération {it}.")
                break

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

        # Save states
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