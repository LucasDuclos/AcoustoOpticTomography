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

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, forward_projection, backward_projection, clamp_positive, check_stopping_criterion, calculate_step_size_reg, gradient_2d, divergence_2d, proj_tv
from AOT_biomaps.AOT_Recon.ReconEnums import NoiseType, PreconditionerType, StopCriterionType
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
    beta: float = 1.0,         
    gamma: float = 1.0,        
    theta: float = 1.0,        
    tau: Union[float, str] = "auto",
    sigma: Union[float, str] = "auto",
    eta: Optional[float] = None,
    numIterations_stepCalculation: int = 20,
    num_subsets: int = 1,     
    reshuffle_period: int = 10,
    noise_type: NoiseType = NoiseType.GAUSSIAN,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
    stop_criterion: StopCriterionType = StopCriterionType.MAX_ITERATIONS,
    stop_threshold: float = 100.0,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves=5000,
    show_logs=True,
    show_criterion: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    Primal-Dual Hybrid Gradient (PDHG) function - Implemented via Chambolle-Pock Algorithm.
    
    Uses ReconTools functions for all matrix operations, making it compatible with
    any SMatrix type (CSR, SELL, DENSE) and device (CPU, GPU).
    Supports both deterministic and stochastic updates (SPDHG) for accelerated convergence.
    
    Supports potential functions:
        - NONE: No regularization
        - TOTAL_VARIATION: β*||∇u||_1 (Isotropic Total Variation regularization (implemented via dual variable projection))

    Supports stopping criteria:
        - MAX_ITERATIONS: Stop after a fixed number of iterations
        - RELATIVE_CHANGE: Stop when relative change in lambda is below threshold
        - COST_FUNCTION: Stop when cost function value changes by less than threshold
        - MSE: Stop when mean squared error with respect to ground truth is below threshold (requires ground truth) ONLY FOR SIMULATED DATA 
        - GRADIENT_NORM: Stop when norm of the update step is below threshold
    
    Supports preconditioning:
        - DIAGONAL: Diagonal preconditioning using A^T * 1
        - NONE: No preconditioning (equivalent to standard gradient descent, but may not converge for ill-conditioned problems)

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
        eta: Kept for backward compatibility with main call signatures
        num_subsets: Number of subsets for stochastic updates (SPDHG). If 1, runs deterministic PDHG.
        reshuffle_period: Number of iterations after which subsets are reshuffled
        noise_type: Type of noise (POISSON or GAUSSIAN)
        potential_type: Kept for signature compatibility, operates strictly as TOTAL_VARIATION
        preconditioner_type: Type of preconditioner to use (default: NONE)
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
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X
    NT = SMatrix.N * SMatrix.T

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    q = xp.zeros(NT, dtype=xp.float32)
    p_x = xp.zeros(ZX, dtype=xp.float32)
    p_z = xp.zeros(ZX, dtype=xp.float32)

    if noise_type == NoiseType.POISSON:
        y_flat = xp.maximum(y_flat, 0.0)  
    
    y_data = y_flat.copy()
    lambda_flat = xp.zeros(ZX, dtype=xp.float32)
    x_bar = xp.zeros(ZX, dtype=xp.float32)

    emission_indices = np.random.permutation(NT)
    subset_slices = np.array_split(emission_indices, num_subsets)
    p_i = 1.0 / num_subsets 

    # --- TRUE DIAGONAL PRECONDITIONING (Independent of Beta) ---
    if preconditioner_type != PreconditionerType.NONE:
        sigma_q = gamma * 0.99 / xp.maximum(forward_projection(SMatrix, xp.ones(ZX, dtype=xp.float32)), 1e-10)
        sigma_p = gamma * 0.99 / 2.0  
        tau_vec = (0.99 * p_i / gamma) / (xp.maximum(backward_projection(SMatrix, xp.ones(NT, dtype=xp.float32)), 1e-10) + 4.0) 
    else:
        tau_val, sigma_val = calculate_step_size_reg(SMatrix, gamma, num_subsets, numIterations_stepCalculation, show_logs) if tau == "auto" or sigma == "auto" else (tau, sigma)
        sigma_q = sigma_val
        sigma_p = sigma_val
        tau_vec = tau_val

    save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    algo_name = f"SPDHG (Chambolle-Pock) ({num_subsets} subsets)" if num_subsets > 1 else "PDHG (Chambolle-Pock)"
    description = f"AOT-BioMaps --- {algo_name} ({SMatrix.matrix_type.name}) --- Diagonal preconditioning : {'YES' if preconditioner_type != PreconditionerType.NONE else 'NO'} --- {'WITH' if withTumor else 'WITHOUT'} TUMOR --- {SMatrix.device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        prev_lambda = lambda_flat.copy() if hasattr(lambda_flat, 'copy') else lambda_flat + 0
        
        Ax_bar = xp.maximum(forward_projection(SMatrix, x_bar), 1e-8) if noise_type == NoiseType.POISSON else forward_projection(SMatrix, x_bar)
            
        # --- DUAL UPDATE 1: Data Fidelity Proximal Resolution --- For Poisson noise: q = (q + sigma * A * x_bar - sqrt((1 - (q + sigma * A * x_bar))^2 + 4 * sigma * y)) / 2  --- For Gaussian noise: q = (q + sigma * (A * x_bar - y)) / (1 + sigma)
        if num_subsets == 1:  
            q = 0.5 * (1.0 + (q + sigma_q * Ax_bar) - xp.sqrt(xp.maximum(((1.0 - (q + sigma_q * Ax_bar))**2 + 4.0 * sigma_q * y_data), 0.0))) if noise_type == NoiseType.POISSON else (q + sigma_q * (Ax_bar - y_data)) / (1.0 + sigma_q)           
        else:
            subset_mask = xp.zeros(NT, dtype=xp.bool_)
            subset_slices = np.array_split(np.random.permutation(SMatrix.N), num_subsets) if it % reshuffle_period == 0 else subset_slices

            for emis in subset_slices[it % num_subsets]:
                subset_mask[emis * SMatrix.T : (emis + 1) * SMatrix.T] = True

            sig_q_sub = sigma_q if isinstance(sigma_q, float) else sigma_q[subset_mask]
            q[subset_mask] = 0.5 * (1.0 + (q[subset_mask] + sig_q_sub * Ax_bar[subset_mask]) - xp.sqrt(xp.maximum(((1.0 - (q[subset_mask] + sig_q_sub * Ax_bar[subset_mask]))**2 + 4.0 * sig_q_sub * y_data[subset_mask]), 0.0))) if noise_type == NoiseType.POISSON else (q[subset_mask] + sig_q_sub * (Ax_bar[subset_mask] - y_data[subset_mask])) / (1.0 + sig_q_sub)


        # --- DUAL UPDATE 2: Total Variation Regularization ---
        grad_x, grad_z = gradient_2d(SMatrix, x_bar)
        p_projected = proj_tv(SMatrix, xp.concatenate([p_x + sigma_p * grad_x, p_z + sigma_p * grad_z]), radius=beta) # PURE CP: Beta is exclusively acting as the projection radius limit
        p_x, p_z = p_projected[:ZX], p_projected[ZX:]
        # --- PRIMAL UPDATE: Element-wise execution using tau vector --- (Negative div_p maps exact adjointness)
        lambda_flat = clamp_positive(SMatrix, (lambda_flat - tau_vec * backward_projection(SMatrix, q) - tau_vec * divergence_2d(SMatrix, p_x, p_z)))

        # --- EXTRAPOLATION ---
        x_bar = lambda_flat + theta * (lambda_flat - prev_lambda)

        # Compute cost function metrics if requested
        if isCostFunction:
            Ax = forward_projection(SMatrix, lambda_flat)
            gx_eval, gz_eval = gradient_2d(SMatrix, lambda_flat)
            cost_history.append(float(-(xp.sum(y_data * xp.log(xp.maximum(Ax, 1e-10)) - xp.maximum(Ax, 1e-10)))) + beta * float(xp.sum(xp.sqrt(gx_eval**2 + gz_eval**2 + 1e-12))) if noise_type == NoiseType.POISSON else 0.5 * float(xp.sum((Ax - y_data)**2)) + beta * float(xp.sum(xp.sqrt(gx_eval**2 + gz_eval**2 + 1e-10))))

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