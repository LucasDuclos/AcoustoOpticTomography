"""
PDHG.py

Primal-Dual Hybrid Gradient (PDHG) algorithm for regularized reconstruction.
Strictly implemented via the mathematically stable Chambolle-Pock framework, supporting both 
deterministic and Stochastic updates (SPDHG / Subsets) for accelerated reconstruction.

Tailored for Acousto-Optic Tomography (AOT) matrix pipelines where N represents acoustic emissions
and T represents temporal propagation frames.
Note : Algorithm handles both real and complex data (4-phase quadrature representation) seamlessly, 
with dynamic kernel selection based on data type.

=============================================================================
MATHEMATICAL FRAMEWORK & OPTIMIZATION THEORY:
=============================================================================

1. Saddle-Point Optimization Problem (Chambolle-Pock Framework):
   We seek to solve a nonsmooth convex optimization problem involving a data fidelity term 
   and a total variation (TV) regularization prior subject to non-negativity:
   
       minimize_{λ >= 0}  { 0.5 * ||A * λ - y||_2^2 + β * ||grad λ||_1 }

   Using convex duality, this is cast as a saddle-point problem:
   
       min_{λ >= 0} max_{q, p}  { <A * λ, q> - 0.5 * ||q||_2^2 - <q, y> + <grad λ, p> - σ_p * δ_{|p| <= β}(p) }

   where:
   - A is the forward system matrix (SMatrix).
   - q is the dual variable associated with the data fidelity term.
   - p = (p_x, p_z) is the dual variable associated with the spatial gradient operator (∇λ).
   - β scales the dual feasible set (L_∞ norm ball for isotropic TV).

2. Dual Update Steps:
   - Data Fidelity Dual Variable (Proximal operator of the conjugate data term):
     
       q_{k+1} = (q_k + σ_q * (A * bar{λ}_k - y)) / (1 + σ_q)
     
   - Regularization Dual Variable (Gradient ascent followed by point-wise projection onto the L_∞ ball):
     
       p_{k+1} = P_{|p| <= β} (p_k + σ_p * grad bar{λ}_k)

3. Primal Update & Extrapolation (Nesterov-like Overrelaxation):
   - Primal variable update via backward projection and divergence operator, combined with 
     an orthogonal projection onto the non-negativity orthant R_+^{Z * X}:
     
       λ_{k+1} = max(0, λ_k - τ * (A^H q_{k+1} + div p_{k+1}))
     
   - Extrapolation step for the next iteration:
     
       λ_bar_{k+1} = λ_{k+1} + θ * (λ_{k+1} - λ_k)

4. Step Size Rules & Convergence Bounds:
   To ensure convergence, the primal step size τ and dual step sizes σ_q, σ_p 
   satisfy the operator norm condition based on the maximum singular value of A and the gradient operator:
   
       τ * σ * ||K||^2 <= 1
   
   - For diagonal preconditioning, vector-valued step sizes are computed following Ehrhardt et al. (2019) 
     using absolute row and column sums of A and the discrete finite-difference stencil weights.
=============================================================================
"""
import contextlib
import warnings
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, get_device_context, forward_projection, backward_projection, check_stopping_criterion, estimate_lipschitz_constant, gradient_2d, divergence_2d, proj_tv
from AOT_biomaps.AOT_Recon.ReconEnums import StopCriterionType
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.PreconditionerEnums import PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.NoPreconditioner import NoPreconditioner
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

if CUPY_AVAILABLE:
    pdhg_gaussian_kernel__REAL = cp.ElementwiseKernel(
        'float32 q_in, float32 Alambda_bar, float32 y, float32 sigma, bool mask',
        'float32 q_out',
        '''
        if (mask) {
            q_out = (q_in + sigma * (Alambda_bar - y)) / (1.0f + sigma);
        } else {
            q_out = q_in;
        }
        ''',
        'pdhg_gaussian_kernel__REAL'
    )

    pdhg_gaussian_kernel__COMPLEX = cp.ElementwiseKernel(
        'complex64 q_in, complex64 Alambda_bar, complex64 y, float32 sigma, bool mask',
        'complex64 q_out',
        '''
        if (mask) {
            q_out = (q_in + sigma * (Alambda_bar - y)) / (1.0f + sigma);
        } else {
            q_out = q_in;
        }
        ''',
        'pdhg_gaussian_kernel__COMPLEX'
    )

    pdhg_primal_kernel = cp.ElementwiseKernel(
        'float32 lam_in, float32 backproj, float32 div, float32 tau, float32 theta',
        'float32 lam_out, float32 x_bar_out',
        '''
        float lam_new = lam_in - tau * backproj - tau * div;
        lam_new = lam_new > 0.0f ? lam_new : 0.0f; // Projection R+
        lam_out = lam_new;
        x_bar_out = lam_new + theta * (lam_new - lam_in); // Extrapolation
        ''',
        'pdhg_primal_kernel'
    )

def calculate_step_size_PDHG(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    preconditioner_type: PreconditionerType,
    num_subsets: int,
    num_iters: int,
    show_logs: bool
) -> Tuple[Union[float, np.ndarray, cp.ndarray], Union[float, np.ndarray, cp.ndarray], float]:
    """
    Compute primal and dual step sizes for the PDHG algorithm.
    
    Mathematical framework:
    - Standard PDHG: Scalar steps based on the operator spectral radius (Lipschitz constant L).
      L = lambda_max(A^H A)
    - Diagonal Preconditioner: Vector-valued steps based on Ehrhardt et al. (2019, Theorem 2),
      leveraging absolute row/col sums to handle complex systems safely:
      τ_j = ρ / (p_i * (col_sums_j + ||∇||^2))
      σ_i = ρ / row_sums_i

    Args:
        SMatrix: System matrix wrapper instance.
        preconditioner_type: PreconditionerType.NONE or PreconditionerType.DIAGONAL.
        num_subsets: Number of subsets for stochastic updates (SPDHG).
        num_iters: Number of power iterations for spectral radius estimation.
        show_logs: If True, prints diagnostic information.

    Returns:
        tuple: (tau_vec, sigma_q_vec, sigma_p_val)
    """
    xp = get_array_module(SMatrix)
    is_gpu = (xp.__name__ == 'cupy')

    with get_device_context(SMatrix):
        rho = 0.99  # Safety convergence factor strictly less than 1

        # ==========================================================
        # 1. STANDARD PDHG (Scalar Steps)
        # ==========================================================
        if preconditioner_type == PreconditionerType.NONE:
            L_data = estimate_lipschitz_constant(
                SMatrix,
                preconditioner=NoPreconditioner(SMatrix=SMatrix),
                num_iters=num_iters
            )
            L_grad = 8.0  # ||∇||^2 <= 8 for 2D finite differences
            L_total = num_subsets * L_data + L_grad

            tau = float(rho / np.sqrt(L_total))
            sigma_q = float(rho * num_subsets / np.sqrt(L_total))
            sigma_p = float(rho / np.sqrt(L_total))

            if show_logs:
                print(f"[AOT-biomaps] L_data={L_data:.3e} | L_total={L_total:.3e} | tau={tau:.3e} | sigma_q={sigma_q:.3e}")

            return xp.asarray(tau, dtype=xp.float32), xp.asarray(sigma_q, dtype=xp.float32), float(sigma_p)

        # ==========================================================
        # 2. DIAGONAL PRECONDITIONER (Ehrhardt et al. 2019)
        # ==========================================================
        if show_logs:
            print("[AOT-biomaps] Computing Ehrhardt diagonal step sizes...")

        row_sums, col_sums = SMatrix.compute_absolute_row_col_sums()

        row_sums = xp.asarray(row_sums, dtype=xp.float32)
        col_sums = xp.asarray(col_sums, dtype=xp.float32)

        # Robust stabilization clipping
        eps_row = max(float(xp.median(row_sums)) * 1e-6, 1e-12)
        eps_col = max(float(xp.median(col_sums)) * 1e-6, 1e-12)

        row_sums = xp.maximum(row_sums, eps_row)
        col_sums = xp.maximum(col_sums, eps_col)

        # Spatial gradient row sum contribution: ||∇||^T 1
        # Stencil weights for 2D forward differences: corners=2, edges=3, center=4
        grad_col_sum = xp.full((SMatrix.Z, SMatrix.X), 4, dtype=xp.float32)
        grad_col_sum[0, :] -= 1
        grad_col_sum[-1, :] -= 1
        grad_col_sum[:, 0] -= 1
        grad_col_sum[:, -1] -= 1
        grad_col_sum = grad_col_sum.ravel()

        # Step size computation (Theorem 2)
        p_i = 1.0 / num_subsets
        tau_vec = rho * p_i / (col_sums + grad_col_sum)
        sigma_q_vec = rho / row_sums
        sigma_p = rho / 2.0  # Dual step size associated with the TV operator

        if show_logs:
            print(f"[AOT-biomaps] tau median={float(xp.median(tau_vec)):.3e} | min={float(xp.min(tau_vec)):.3e} | max={float(xp.max(tau_vec)):.3e}")
            print(f"[AOT-biomaps] sigma_q median={float(xp.median(sigma_q_vec)):.3e} | min={float(xp.min(sigma_q_vec)):.3e} | max={float(xp.max(sigma_q_vec)):.3e}")

        return tau_vec, sigma_q_vec, float(sigma_p)

def PDHG(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,            
    theta: float = 1.0,        
    tau: Union[float, str] = "auto",
    sigma: Union[float, str] = "auto",
    numIterations_stepCalculation: int = 20,
    num_subsets: int = 1,     
    reshuffle_period: int = 10,
    preconditioner_type: PreconditionerType = PreconditionerType.DIAGONAL,
    stop_criterion: StopCriterionType = StopCriterionType.MAX_ITERATIONS,
    stop_threshold: float = 100.0,
    stop_window_size: int = 5,
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
        theta: Extrapolation parameter (Chambolle-Pock)
        tau: Step size parameter for primal update ('auto' for operator norm adaptation)
        sigma: Step size parameter for dual update ('auto' for operator norm adaptation)
        numIterations_stepCalculation: Number of iterations for operator norm estimation (if tau or sigma is 'auto')
        num_subsets: Number of subsets for stochastic updates (SPDHG). If 1, runs deterministic PDHG.
        reshuffle_period: Number of iterations after which subsets are reshuffled
        preconditioner_type: Preconditioner type (see PreconditionerEnums for available types)
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
    device_context = cp.cuda.Device(SMatrix.gpu_index) if is_gpu else contextlib.nullcontext()

    with device_context:
        Z, X = SMatrix.Z, SMatrix.X
        ZX = Z * X
        NT = SMatrix.N * SMatrix.T

        if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
            raise ValueError(f"[AOT-biomaps] Shape mismatch: y {y.shape} vs SMatrix (T={SMatrix.T}, N={SMatrix.N})")

        # Adapt data dtype based on whether the matrix is complex (4-phase quadrature) or real
        data_dtype = xp.complex64 if SMatrix.isComplexSMatrix else xp.float32

        # Normalize data y
        y_max = float(np.max(np.abs(y))) if SMatrix.isComplexSMatrix else float(np.max(y))
        if y_max > 0:
            y_norm = y / y_max 

        y_flat = xp.asarray(y_norm.T.flatten().astype(data_dtype))

        lambda_flat = xp.zeros(ZX, dtype=xp.float32)
        lambda_bar = lambda_flat.copy() 
        q = xp.zeros(NT, dtype=data_dtype)
        subset_mask = xp.ones(NT, dtype=xp.bool_)
        
        p_x = xp.zeros(ZX, dtype=xp.float32)
        p_z = xp.zeros(ZX, dtype=xp.float32)

        emission_indices = np.random.permutation(SMatrix.N)
        subset_slices = np.array_split(emission_indices, num_subsets)

        # Step sizes configuration
        if tau == "auto" or sigma == "auto":
            tau_res, sigma_q_res, sigma_p = calculate_step_size_PDHG(SMatrix, preconditioner_type, num_subsets, numIterations_stepCalculation, show_logs)
            tau_vec = xp.asarray(tau_res, dtype=xp.float32) if isinstance(tau_res, (np.ndarray, cp.ndarray)) else xp.full(ZX, tau_res, dtype=xp.float32)
            sigma_q = xp.asarray(sigma_q_res, dtype=xp.float32) if isinstance(sigma_q_res, (np.ndarray, cp.ndarray)) else xp.full(NT, sigma_q_res, dtype=xp.float32)
        else:
            sigma_q = xp.full(NT, sigma, dtype=xp.float32)
            sigma_p = sigma
            tau_vec = xp.full(ZX, tau, dtype=xp.float32)

        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()
        saved_lambda, saved_indices_list = [], []
        cost_history = [] if isCostFunction else None
        window_history = []

        algo_name = f"SPDHG (Chambolle-Pock) ({num_subsets} subsets)" if num_subsets > 1 else "PDHG (Chambolle-Pock)"
        prec_str = preconditioner_type.name.replace("_", " ")
        cplx_str = "COMPLEX (4-phases quadrature) " if SMatrix.isComplexSMatrix else "REAL "
        description = f"[AOT-biomaps] {cplx_str}{algo_name} --- ({SMatrix.matrix_type.name}) --- {prec_str} --- {'WITH' if withTumor else 'WITHOUT'} TUMOR --- DEVICE: {SMatrix.device.upper()}"
        
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        # ==========================================================
        # MAIN OPTIMIZATION LOOP
        # ==========================================================
        for it in iterator:
            if num_subsets > 1:
                if it % reshuffle_period == 0:
                    subset_slices = np.array_split(np.random.permutation(SMatrix.N), num_subsets)
                subset_mask.fill(False)
                for emis in subset_slices[it % num_subsets]:
                    subset_mask[emis * SMatrix.T : (emis + 1) * SMatrix.T] = True

            prev_lambda = lambda_flat.copy()

            Alambda_bar = forward_projection(SMatrix, lambda_bar)

            # --- DUAL UPDATE (Data Fidelity Proximal Operator) --- 
            if is_gpu:
                pdhg_gaussian_kernel__COMPLEX(q, Alambda_bar, y_flat, sigma_q, subset_mask, q) if SMatrix.isComplexSMatrix else pdhg_gaussian_kernel__REAL(q, Alambda_bar, y_flat, sigma_q, subset_mask, q)
            else:
                q[subset_mask] = (q[subset_mask] + sigma_q[subset_mask] * (Alambda_bar[subset_mask] - y_flat[subset_mask])) / (1.0 + sigma_q[subset_mask])

            # --- DUAL UPDATE (Regularization Total Variation) --- 
            grad_x, grad_z = gradient_2d(SMatrix, lambda_bar)
            p_x += sigma_p * grad_x
            p_z += sigma_p * grad_z

            p_projected = proj_tv(SMatrix, xp.concatenate([p_x, p_z]), radius=beta)
            p_x, p_z = p_projected[:ZX], p_projected[ZX:]

            # --- PRIMAL UPDATE & EXTRAPOLATION ---
            backproj_q = xp.ascontiguousarray(xp.real(backward_projection(SMatrix, q)), dtype=xp.float32) if SMatrix.isComplexSMatrix else xp.ascontiguousarray(backward_projection(SMatrix, q), dtype=xp.float32)

            div_p = divergence_2d(SMatrix, p_x, p_z)

            if is_gpu:
                pdhg_primal_kernel(lambda_flat.astype(xp.float32, copy=False), backproj_q.astype(xp.float32, copy=False), div_p.astype(xp.float32, copy=False), tau_vec.astype(xp.float32, copy=False), float(theta), lambda_flat, lambda_bar)
            else:
                lambda_flat = lambda_flat - tau_vec * backproj_q - tau_vec * div_p
                np.maximum(lambda_flat, 0.0, out=lambda_flat)
                lambda_bar = lambda_flat + theta * (lambda_flat - prev_lambda)

            if isCostFunction:
                Ax = forward_projection(SMatrix, lambda_flat)
                gx_eval, gz_eval = gradient_2d(SMatrix, lambda_flat)
                tv_penalty = beta * float(xp.sum(xp.sqrt(gx_eval**2 + gz_eval**2 + 1e-12)))
                fidelity = 0.5 * float(xp.vdot(Ax - y_flat, Ax - y_flat).real)
                cost_history.append(fidelity + tv_penalty)

            if stop_criterion != StopCriterionType.MAX_ITERATIONS:
                if SMatrix.experiment.OpticImage is None:
                    ground_truth = None
                else:
                    ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
                gradient = backproj_q + div_p if stop_criterion == StopCriterionType.GRADIENT_NORM else None
                isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, window_size=stop_window_size, history=cost_history, ground_truth=ground_truth, gradient=gradient, window_history=window_history)
                if show_logs and show_criterion:
                    iterator.set_postfix_str(f"{stop_criterion.name}: {val:.2e}")
                if isStop:
                    if show_logs: 
                        print(f"\n[AOT-biomaps] Stopping criterion {stop_criterion.name} reached at iteration {it}.")
                    cost_history.pop() if isCostFunction else None
                    break

            if isSavingEachIteration and it in save_indices:
                lambda_snapshot = lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X).copy()
                saved_lambda.append(lambda_snapshot * y_max / SMatrix.normalization_factor)
                saved_indices_list.append(it)

        final_result = lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X)
        final_result *= y_max / SMatrix.normalization_factor
        
        return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)