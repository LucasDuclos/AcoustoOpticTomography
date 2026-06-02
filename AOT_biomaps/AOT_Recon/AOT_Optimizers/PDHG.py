"""
PDHG.py

Primal-Dual Hybrid Gradient (PDHG) algorithm for regularized reconstruction.
Strictly implemented via the mathematically stable Chambolle-Pock framework, supporting both 
deterministic and Stochastic updates (SPDHG / Subsets) for accelerated reconstruction.

Tailored for Acousto-Optic Tomography (AOT) matrix pipelines where N represents acoustic emissions
and T represents temporal propagation frames.
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import (
    check_gpu_available, forward_projection, backward_projection, clamp_positive, 
    build_preconditioner, apply_preconditioner, _get_array_module,
    gradient_2d, divergence_2d, proj_tv, estimate_operator_norm
)
from AOT_biomaps.AOT_Recon.ReconEnums import NoiseType, PotentialType, PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def PDHG(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,         # TV regularization weight parameter (lambda)
    gamma: float = 1.0,        # Scaling factor balancing the tau/sigma step ratio
    theta: float = 1.0,        # Extrapolation parameter for primal variable (relaxation step)
    tau: Union[float, str] = "auto",
    sigma: Union[float, str] = "auto",
    num_subsets: int = 1,      # Number of subsets for stochastic acoustic emissions acceleration
    reshuffle_period: int = 10,
    noise_type: NoiseType = NoiseType.GAUSSIAN,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves=5000,
    show_logs=True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    Primal-Dual Hybrid Gradient (PDHG) function - Implemented via Chambolle-Pock Algorithm.
    
    Uses ReconTools functions for all matrix operations, making it compatible with
    any SMatrix type (CSR, SELL, DENSE) and device (CPU, GPU).
    Supports both deterministic and stochastic updates (SPDHG) for accelerated convergence.
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        beta: TV regularization weight parameter (lambda)
        delta: Kept for backward compatibility with main call signatures
        gamma: Balancing step-size parameter (scales tau down and sigma up)
        theta: Extrapolation parameter for primal variable (relaxation step)
        tau: Step size parameter for primal update ('auto' for operator norm adaptation)
        sigma: Step size parameter for dual update ('auto' for operator norm adaptation)
        num_subsets: Number of subsets for stochastic updates (SPDHG). If 1, runs deterministic PDHG.
        reshuffle_period: Number of iterations after which subsets are reshuffled
        noise_type: Type of noise (POISSON or GAUSSIAN)
        potential_type: Kept for signature compatibility, operates strictly as TOTAL_VARIATION
        preconditioner_type: Type of preconditioner to use (default: NONE)
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
        
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type.name
    xp = _get_array_module(SMatrix)
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X
    
    num_emissions = SMatrix.N    
    num_time_frames = SMatrix.T  
    NT = num_emissions * num_time_frames

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    q = xp.zeros(NT, dtype=xp.float32)
    p_x = xp.zeros(ZX, dtype=xp.float32)
    p_z = xp.zeros(ZX, dtype=xp.float32)

    if noise_type == NoiseType.POISSON:
        y_flat = xp.maximum(y_flat, 0.0)  
    
    y_data = y_flat.copy()
    x_flat = xp.zeros(ZX, dtype=xp.float32)
    x_bar = xp.zeros(ZX, dtype=xp.float32)

    emission_indices = np.random.permutation(num_emissions)
    subset_slices = np.array_split(emission_indices, num_subsets)
    p_i = 1.0 / num_subsets 

    # --- TRUE DIAGONAL PRECONDITIONING (Independent of Beta) ---
    if preconditioner_type != PreconditionerType.NONE:
        ones_x = xp.ones(ZX, dtype=xp.float32)
        A_ones = forward_projection(SMatrix, ones_x) + 1e-8  
        
        ones_q = xp.ones(NT, dtype=xp.float32)
        At_ones = backward_projection(SMatrix, ones_q) + 1e-8

        rho = 0.99
        
        sigma_q = gamma * rho / A_ones
        sigma_p = gamma * rho / 2.0  
        
        tau_vec = (rho * p_i / gamma) / (At_ones + 4.0) 
        
        if show_logs:
            print(f"Using Diagonal Preconditioning vectors | Subsets: {num_subsets}.")
    else:
        if str(tau).lower() == "auto" or str(sigma).lower() == "auto":
            L_A = estimate_operator_norm(SMatrix, num_iters=20)
            L_grad = 8.0  
            L_total = (L_A**2) + L_grad  # Pure operator norm without beta scale

            tau_val = 0.99 / (np.sqrt(L_total) * gamma)
            sigma_val = (0.99 * gamma / np.sqrt(L_total)) * float(num_subsets)
            
            tau_vec = tau_val
            sigma_q = sigma_val
            sigma_p = sigma_val
            
            if show_logs:
                print(f"Using spectral norm estimation | Subsets: {num_subsets} | L_A: {L_A:.4f} | Selected scalar tau: {tau_val:.5f}, sigma: {sigma_val:.5f}.")

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

    algo_name = f"SPDHG (Chambolle-Pock) ({num_subsets} subsets)" if num_subsets > 1 else "PDHG (Chambolle-Pock)"
    description = f"AOT-BioMaps -- {algo_name} ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        x_prev = x_flat.copy() if hasattr(x_flat, 'copy') else x_flat + 0
        
        Ax_bar = forward_projection(SMatrix, x_bar)
        
        if noise_type == NoiseType.POISSON:
            Ax_bar = xp.maximum(Ax_bar, 1e-8)

        # --- DUAL UPDATE 1: Data Fidelity Proximal Resolution ---
        if num_subsets == 1:            
            if noise_type == NoiseType.POISSON:
                q_tilde = q + sigma_q * Ax_bar
                inner_term = (1.0 - q_tilde)**2 + 4.0 * sigma_q * y_data
                q = 0.5 * (1.0 + q_tilde - xp.sqrt(xp.maximum(inner_term, 0.0)))
            else:  
                q = (q + sigma_q * (Ax_bar - y_data)) / (1.0 + sigma_q)
            
            AT_q = backward_projection(SMatrix, q)
        else:
            if it > 0 and it % reshuffle_period == 0:
                emission_indices = np.random.permutation(num_emissions)
                subset_slices = np.array_split(emission_indices, num_subsets)
                
            active_subset_idx = it % num_subsets
            active_emissions = subset_slices[active_subset_idx]
            
            if check_gpu_available(SMatrix):
                subset_mask = cp.zeros(NT, dtype=cp.bool_)
                for emis in active_emissions:
                    subset_mask[emis * num_time_frames : (emis + 1) * num_time_frames] = True
            else:
                subset_mask = np.zeros(NT, dtype=np.bool_)
                for emis in active_emissions:
                    subset_mask[emis * num_time_frames : (emis + 1) * num_time_frames] = True
            
            if isinstance(sigma_q, float):
                sig_q_sub = sigma_q
            else:
                sig_q_sub = sigma_q[subset_mask]

            if noise_type == NoiseType.POISSON:
                q_sub = q[subset_mask]
                Ax_sub = Ax_bar[subset_mask]
                y_sub = y_data[subset_mask]
                
                q_sub_tilde = q_sub + sig_q_sub * Ax_sub
                inner_term = (1.0 - q_sub_tilde)**2 + 4.0 * sig_q_sub * y_sub
                q[subset_mask] = 0.5 * (1.0 + q_sub_tilde - xp.sqrt(xp.maximum(inner_term, 0.0)))
            else:  
                q[subset_mask] = (q[subset_mask] + sig_q_sub * (Ax_bar[subset_mask] - y_data[subset_mask])) / (1.0 + sig_q_sub)

            AT_q = backward_projection(SMatrix, q)

        # --- DUAL UPDATE 2: Total Variation Regularization ---
        grad_x, grad_z = gradient_2d(SMatrix, x_bar)
        p_x += sigma_p * grad_x
        p_z += sigma_p * grad_z

        p_stacked = xp.concatenate([p_x, p_z])
        # PURE CP: Beta is exclusively acting as the projection radius limit
        p_projected = proj_tv(SMatrix, p_stacked, radius=beta)
        p_x = p_projected[:ZX]
        p_z = p_projected[ZX:]

        # --- PRIMAL UPDATE: Element-wise execution using tau vector ---
        div_p = divergence_2d(SMatrix, p_x, p_z)

        # Primal update execution (Negative div_p maps exact adjointness)
        x_flat = x_flat - tau_vec * AT_q - tau_vec * div_p
        x_flat = clamp_positive(SMatrix, x_flat)

        # --- PRIMAL EXTRAPOLATION ---
        x_bar = x_flat + theta * (x_flat - x_prev)

        # Compute cost function metrics if requested
        if isCostFunction:
            Ax = forward_projection(SMatrix, x_flat)
            if noise_type == NoiseType.POISSON:
                Ax_clamped = xp.maximum(Ax, 1e-10)
                likelihood = xp.sum(y_data * xp.log(Ax_clamped) - Ax_clamped)
                cost = float(-likelihood)
            else:  
                cost = 0.5 * float(xp.sum((Ax - y_data)**2))
                
            gx_eval, gz_eval = gradient_2d(SMatrix, x_flat)
            tv_val = float(xp.sum(xp.sqrt(gx_eval**2 + gz_eval**2 + 1e-12)))
            cost += beta * tv_val
            cost_history.append(cost)

        if isSavingEachIteration and it in save_indices:
            if check_gpu_available(SMatrix):
                saved_lambda.append(cp.asnumpy(x_flat.reshape(Z, X)))  
            else:
                saved_lambda.append(x_flat.reshape(Z, X).copy())     
            saved_indices_list.append(it)

    if check_gpu_available(SMatrix):
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(x_flat.reshape(Z, X)) 
    else:
        final_result = x_flat.reshape(Z, X)  

    if isSavingEachIteration:
        return saved_lambda, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history