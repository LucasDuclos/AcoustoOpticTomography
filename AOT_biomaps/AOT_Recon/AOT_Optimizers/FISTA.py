"""
FISTA.py

Fast Iterative Shrinkage-Thresholding Algorithm (Accelerated PGD).
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

=============================================================================
MATHEMATICAL FRAMEWORK & OPTIMIZATION THEORY:
=============================================================================

1. Objective Function (Regularized Least Squares):
   We seek to reconstruct an optimal spatial distribution (image) represented by the real, 
   non-negative vector λ in ℝ^{Z * X} from acoustic-optic measurements y in ℂ^T (or ℝ^T). 
   The optimization problem incorporates a data fidelity term and a spatial regularization prior 
   subject to a non-negativity constraint:
   
       minimize   F(λ) = f(λ) + β * U(λ) = 0.5 * ||A * λ - y||_2^2 + β * U(λ)
       subject to λ >= 0

   where:
   - A is the forward system matrix (SMatrix) modeling light propagation and interaction.
   - y represents the acquired experimental data (e.g., 4-phase quadrature representation).
   - U(λ) is a spatial potential function enforcing smoothness or edge-preservation 
     (e.g., Quadratic, Huber, or Relative Difference).
   - β is the regularization hyperparameter balancing data-fit and prior strength.

2. Gradient Derivation of the Objective:
   The total gradient of the objective function evaluated at an extrapolation point λ_bar 
   combines the data fidelity gradient and the regularization prior gradient:
   
       ∇F(λ_bar}) = P^{-1} A^H (A * λ_bar - y) + ∇U(λ_bar)

   where P^{-1} is an optional preconditioner (e.g., diagonal preconditioning via diag(A^H * A)).

3. Lipschitz Constant & Adaptive Step Size:
   To ensure global convergence and stability of the accelerated scheme, the step size alpha 
   must be bounded by the inverse of the total Lipschitz constant L_total of the gradient operator:
   
       L_total = L_data + L_prior
   
   - L_data is estimated dynamically using the power iteration method (`estimate_lipschitz_constant`).
   - L_prior is bounded analytically using Gershgorin circle theorems on the Hessian of the potential 
     (e.g., L_prior = 8 * beta for quadratic/Huber neighborhood structures).
   
       α = 1 / (L_data + L_prior)

4. Nesterov Acceleration (Momentum Extrapolation):
   Unlike standard gradient descent, FISTA maintains an extrapolation point λ_bar computed 
   via a momentum coefficient derived from a sequence of scalar weights t_k:
   
       t_{k+1} = (1 + sqrt(1 + 4 * t_k^2)) / 2
       momentum = (t_k - 1) / t_{k+1}
       λ_bar_k = λ_k + momentum * (λ_k - λ_{k-1})

5. Proximal / Projected Gradient Step:
   The update combines the accelerated gradient step with an orthogonal projection onto the 
   non-negative orthant R_+^{Z * X}:
   
       λ_{k+1} = max(0, λ_bar_k - α * ∇F(λ_bar_k))
=============================================================================
""" 
import contextlib
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, get_device_context, forward_projection, backward_projection, get_potential_function, check_stopping_criterion, estimate_lipschitz_constant
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType, PotentialShapeType, StopCriterionType
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.PreconditionerEnums import PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.NoPreconditioner import NoPreconditioner
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.DiagPreconditioner import DiagPreconditioner
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
    fista_update_kernel = cp.ElementwiseKernel(
        'float32 prev_lambda, float32 z_in, float32 grad, float32 alpha, float32 momentum',
        'float32 x_new, float32 z_out',
        '''
        // Gradient descent step: z_in - alpha * grad(F(z_in))
        double step = (double)z_in - (double)alpha * (double)grad;
        
        // Non-negativity projection: x_{k+1} = max(0, step)
        double x_val = step > 0.0 ? step : 0.0;
        x_new = (float)x_val;
        
        // Nesterov Extrapolation: z_{k+1} = x_{k+1} + momentum * (x_{k+1} - x_k)
        z_out = (float)(x_val + (double)momentum * (x_val - (double)prev_lambda));
        ''',
        'fista_update_kernel'
    )

def calculate_step_size_FISTA(SMatrix, preconditioner, potential_type, beta, delta, num_iters, show_logs):
    """
    Calculate the optimal gradient descent step size α for the FISTA algorithm.
    
    Mathematical framework:
    To guarantee convergence, the step size must satisfy α ≤ 1 / L_total, 
    where L_total is the Lipschitz constant of the objective function's gradient.
    L_total = L_data + L_prior
    
    - L_data = λ_max(P⁻¹ Aᴴ A) (estimated via Power Iteration).
    - L_prior is derived from the maximum eigenvalue of the regularization Hessian (Gerschgorin bounds).
    """
    L_data = estimate_lipschitz_constant(SMatrix, preconditioner=preconditioner, num_iters=num_iters)
    
    # Bounding the Hessian of the regularization term
    if potential_type == PotentialType.QUADRATIC:
        # L_prior = 8β (Exact for 2D cross-neighborhood quadratic penalty)
        L_prior = 8.0 * beta
    elif potential_type == PotentialType.HUBER:
        # Keep 8.0 * beta. Mathematically inaccurate for the quadratic region (the tip, where L=8β/δ),
        # but pragmatically perfect for the linear TV-like region where 99% of the pixels actually reside.
        L_prior = 8.0 * beta 
    elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
        # Keep the division by gamma here. Because gamma (0.05) acts as a macroscopic baseline,
        # RD strictly requires this damping factor to prevent the step size from exploding.
        # L_prior = 8 * (2β / γ)
        gamma = 0.05  # Must strictly match the 'gamma' defined in the RDP kernel
        L_prior = 8.0 * (2.0 * beta / gamma)
    else:
        L_prior = 0.0

    # α = 1 / (L_data + L_prior)
    alpha = 1.0 / (L_data + L_prior)

    if show_logs:
        print(f"[AOT-biomaps] Using step size α = {alpha:.3e} (L_data={L_data:.2e}, L_prior={L_prior:.2e})")

    return alpha


def FISTA(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: Union[float, str] = "auto",     
    beta: float = 1.0,      
    delta: float = 0.01,     
    numIterations_stepCalculation: int = 20,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    potential_shape: PotentialShapeType = PotentialShapeType.CROSS,
    potential_radius: int = 2,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
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
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for least squares reconstruction 
    with potential regularization and Nesterov acceleration.
    
    Uses ReconTools functions for all matrix operations, ensuring compatibility with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

    Supports potential functions U(x):
        - NONE: No regularization
        - QUADRATIC: ½ β (u-v)²
        - HUBER: β * (½ (u-v)² if |u-v| ≤ δ else δ * (|u-v| - ½ δ))
        - RELATIVE_DIFFERENCE: β * (u-v)² / (u + v + γ + δ|u-v|)

    Supports stopping criteria:
        - MAX_ITERATIONS: Stop after a fixed number of iterations
        - RELATIVE_CHANGE: Stop when relative change in lambda is below threshold
        - COST_FUNCTION: Stop when objective function value changes by less than threshold
        - MSE: Stop when MSE with respect to ground truth is below threshold (Simulations only)
        - GRADIENT_NORM: Stop when norm of the update step is below threshold
    
    Supports preconditioning P⁻¹:
        - DIAGONAL: Jacobi preconditioner using diag(AᴴA) + damping (RECOMMENDED)
        - NONE: Standard gradient descent
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data (shape: (T, N))
        numIterations: Number of optimization steps
        alpha: Step size α for gradient update (float or "auto" for Lipschitz estimation)
        beta: Regularization parameter β (weight of the prior)
        delta: Threshold δ (used for Huber linearity boundary or RDP denominator scaling)
        numIterations_stepCalculation: Iterations for Power Iteration estimating L_data
        potential_type: Type of potential function (PotentialType enum)
        potential_shape: Neighborhood geometry (PotentialShapeType enum)
        potential_radius: Neighborhood radius in pixels
        preconditioner_type: Preconditioner type (see PreconditionerEnums for available types)
        stop_criterion: Metric evaluated for early stopping (StopCriterionType enum)
        stop_threshold: Trigger value for the stopping criterion
        stop_window_size: Sliding window size to prevent premature stopping due to oscillations
        isSavingEachIteration: If True, saves intermediate lambda states
        isCostFunction: If True, computes objective function history ½||Ax-y||² + βU(x)
        withTumor: Boolean flag for logging description
        max_saves: Memory cap for intermediate saves
        show_logs: If True, displays progress bar
        show_criterion: If True, displays stopping criterion metric in progress bar
    
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
        - reconstructed_image: Final numpy array or list of arrays (Z, X)
        - saved_indices: List of iterations where state was saved
        - cost_history: Evolution of the objective function (if requested)
    """
    xp = get_array_module(SMatrix)
    is_gpu = (xp.__name__ == 'cupy')

    with get_device_context(SMatrix):
        Z, X = SMatrix.Z, SMatrix.X
        ZX = Z * X

        if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
            raise ValueError(f"[AOT-biomaps] Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

        # Adapt data dtype based on whether the matrix is complex (4-phase quadrature) or real
        data_dtype = xp.complex64 if SMatrix.isComplexSMatrix else xp.float32

        # Normalize data y
        y_max = float(np.max(np.abs(y))) if SMatrix.isComplexSMatrix else float(np.max(y))
        if y_max > 0:
            y_norm = y / y_max 

        y_flat = xp.asarray(y_norm.T.flatten().astype(data_dtype)) # can be complex or real

        # Scale β and δ strictly according to amplitude physics
        beta_phys = beta
        delta_phys = delta

        if potential_type == PotentialType.QUADRATIC:
            # Quadratic gradient scales naturally with image amplitude. No y_max division.
            beta = beta_phys / (SMatrix.normalization_factor)
        elif potential_type == PotentialType.HUBER:
            # Huber linear gradient is sign(x), invariant to amplitude. Division by y_max is required.
            beta = beta_phys / (SMatrix.normalization_factor * y_max)
            delta = delta_phys * (SMatrix.normalization_factor / y_max)
        elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
            # RDP incorporates relative normalization.
            beta = beta_phys / (SMatrix.normalization_factor * y_max)

        # Preconditioner Initialization
        if preconditioner_type == PreconditionerType.DIAGONAL:
            preconditioner = DiagPreconditioner(SMatrix=SMatrix)
            preconditioner.build() # Computes diag(AᴴA) and applies Tikhonov damping
        else:
            preconditioner = NoPreconditioner(SMatrix=SMatrix)
        
        # Variables initialization
        lambda_flat = xp.zeros(ZX, dtype=xp.float32)        # λ: Actual image (always real)
        lambda_bar = xp.zeros(ZX, dtype=xp.float32)         # λ_bar : Extrapolation point (always real)
        t = 1.0                                             # t_k: Momentum tracker
        
        residual_buffer = xp.empty_like(y_flat)
        alpha_val = calculate_step_size_FISTA(SMatrix, preconditioner, potential_type, beta, delta, numIterations_stepCalculation, show_logs) if alpha == "auto" else alpha
        alpha_vec = xp.full(ZX, alpha_val, dtype=xp.float32)


        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()
        saved_lambda = []
        saved_indices_list = []
        cost_history = [] if isCostFunction else None
        window_history = []

        prec_str = preconditioner.get_name()
        cplx_str = "COMPLEX (4-phases quadrature) " if SMatrix.isComplexSMatrix else "REAL "
        delta_str = f" / δ={delta:.2e}" if potential_type == PotentialType.HUBER else ""
        description = f"[AOT-biomaps] {cplx_str}FISTA --- {prec_str} ({SMatrix.matrix_type.name}) with {potential_type.name} β={beta:.2e}{delta_str} --- {'WITH' if withTumor else 'WITHOUT'} TUMOR --- DEVICE: {SMatrix.device.upper()}"
        
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        # ==========================================================
        # MAIN OPTIMIZATION LOOP
        # ==========================================================
        for it in iterator:
            prev_lambda = lambda_flat.copy()

            Alambda_bar = forward_projection(SMatrix, lambda_bar)   
            xp.subtract(Alambda_bar, y_flat, out=residual_buffer) # residual = (A * λ_bar) - y

            # --- PRIOR GRADIENT --- ∇U(λ_bar)
            grad_U, _, _ = get_potential_function(potential_type, SMatrix, lambda_bar, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=True, compute_hess=False, compute_energy=isCostFunction, use_surrogate_hessian=False)

            # --- DATA FIDELITY GRADIENT WITH PRECONDITIONING --- Apply inverse preconditioner ONLY to the data fidelity term to protect L_prior stability : ∇f_prec = P⁻¹ ∇f_raw where ∇f_raw = Aᴴ(A * λ_bar - y)
            grad_fidelity_prec = xp.ascontiguousarray(preconditioner.apply_inverse(xp.real(backward_projection(SMatrix, residual_buffer))), dtype=xp.float32) # if preconditioner is none, this is just the identity operation. Always real-valued.

            # --- TOTAL GRADIENT --- 
            # ∇F(λ_bar) = P⁻¹ ∇f(λ_bar) + ∇U(λ_bar)
            total_grad = grad_fidelity_prec + grad_U

            # --- NESTEROV MOMENTUM ---
            # t_{k+1} = (1 + √(1 + 4 t_k²)) / 2
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_next

            # --- FISTA UPDATE (GRADIENT DESCENT + PROJECTION + EXTRAPOLATION) ---
            if is_gpu:
                fista_update_kernel(prev_lambda, lambda_bar, total_grad.astype(xp.float32, copy=False), alpha_vec, float(momentum), lambda_flat, lambda_bar)
            else:
                lambda_flat = lambda_bar - alpha_vec * total_grad # x_{k+1} = max(0, λ_bar - α ∇F(λ_bar))
                np.maximum(lambda_flat, 0.0, out=lambda_flat)
                lambda_bar = lambda_flat + float(momentum) * (lambda_flat - prev_lambda) # λ_bar_{k+1} = λ_{k+1} + momentum * (λ_{k+1} - λ_k)
                
            t = t_next

            if isCostFunction:
                Ax = forward_projection(SMatrix, lambda_flat)
                _, _, U_x = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta, shape=potential_shape, radius=potential_radius, compute_grad=False, compute_hess=False, compute_energy=True, use_surrogate_hessian=False)
                fidelity = 0.5 * float(xp.vdot(Ax - y_flat, Ax - y_flat).real)
                # F(x) = ½ ||A * λ - y||² + U(x)
                cost_history.append(fidelity + U_x)

            if stop_criterion != StopCriterionType.MAX_ITERATIONS:
                if SMatrix.experiment.OpticImage is None:
                    ground_truth = None
                else:
                    ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
                gradient_for_stop = total_grad if stop_criterion == StopCriterionType.GRADIENT_NORM else None
                isStop, val = check_stopping_criterion(SMatrix, lambda_flat, prev_lambda, stop_criterion, stop_threshold, window_size=stop_window_size, history=cost_history, ground_truth=ground_truth, gradient=gradient_for_stop, window_history=window_history)
                if show_logs and show_criterion:
                    iterator.set_postfix_str(f"{stop_criterion.name}: {val:.2e}")
                if isStop:
                    if show_logs: print(f"\n[AOT-biomaps] Stopping Criterion {stop_criterion.name} reached at iteration {it}.")
                    cost_history.pop() if isCostFunction else None
                    break
                
            if isSavingEachIteration and it in save_indices:
                lambda_snapshot = lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X).copy()
                saved_lambda.append(lambda_snapshot * y_max / SMatrix.normalization_factor)
                saved_indices_list.append(it)

        # Reshape and rescale the final result to true physical amplitude
        final_result = lambda_flat.reshape(Z, X).get() if is_gpu else lambda_flat.reshape(Z, X)
        final_result *= y_max / SMatrix.normalization_factor
        
        return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)