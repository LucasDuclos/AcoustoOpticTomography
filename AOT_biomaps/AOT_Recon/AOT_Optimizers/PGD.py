"""
PGD.py

Projected Gradient Descent (PGD) algorithm for un-regularized Least Squares reconstruction.
This is the simplest and most basic optimization algorithm for solving the inverse problem in acousto-optic tomography.
Uses ReconTools functions for all matrix operations.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

=============================================================================
MATHEMATICAL FORMULATION & OPTIMIZATION THEORY:
=============================================================================

1. Objective Function (Least Squares Data Fidelity):
   We seek to reconstruct an optimal spatial distribution (image) represented by the real, 
   non-negative vector lambda in ℝ^{Z * X} from acoustic-optic measurements y in ℂ^T (or ℝ^T). 
   The optimization problem is formulated as the minimization of a smooth, convex quadratic 
   cost function subject to a physical non-negativity constraint:
   
       minimize   f(λ) = 0.5 * ||A * λ - y||_2^2
       subject to λ >= 0

   where:
   - A is the forward system matrix (SMatrix) modeling light propagation and acousto-optic interaction.
   - y represents the acquired or demodulated experimental signal (e.g., 4-phase quadrature representation).
   - || . ||_2 denotes the standard Euclidean norm (or complex Hermitian inner product norm).

2. Gradient Derivation:
   The Fréchet derivative (gradient) of the data-fidelity term with respect to the image lambda 
   is derived via the adjoint operator A^H (implemented through backward projection):
   
       ∇f(λ) = A^H (A * λ - y)

3. Lipschitz Constant Estimation & Adaptive Step Size:
   The convergence rate of gradient descent is dictated by the maximum eigenvalue of the Hessian matrix 
   (i.e., the Lipschitz constant L of the gradient operator ∇f). 
   Since L corresponds to the spectral radius of the normal operator A^H A:
   
       L = ||A^H A||_2 = max_eigenvalue(A^H A)
   
   It is efficiently estimated using the power iteration method (`estimate_lipschitz_constant`). 
   The optimal descent step size (learning rate) alpha is then set as the inverse of the Lipschitz constant:
   
       α = 1 / L

4. Preconditioning:
   Acousto-optic inverse problems frequently suffer from severe ill-conditioning due to the diffuse 
   nature of optical transport and limited view configurations. To rescale the energy across spatial frequencies 
   and accelerate convergence, a preconditioning operator P is introduced.
   
   - NONE: Standard gradient descent (P = I).
   - DIAGONAL: Uses a diagonal approximation of the Fisher information matrix / normal operator, 
     typically derived from the back-projection of uniform weights (P ~ diag(A^H * 1)), incorporating 
     Tikhonov damping for numerical stability:
     
       ∇f_prec = P^{-1} * ∇f(λ)

5. Projection Operator (Non-Negativity Constraint):
   To respect the physical reality of optical absorption or concentration parameters (which cannot be negative), 
   an orthogonal projection onto the non-negative orthant ℝ_+^{Z * X} is applied at each iteration:
   
       P_{>= 0}(x) = max(0, x)

6. Iterative Update Scheme (PGD Iteration):
   Combining the gradient descent step, the preconditioning, and the projection yields the core PGD update rule:
   
       λ_{k+1} = max(0, λ_k - α * P^{-1} * A^H (A * λ_k - y))
=============================================================================
"""
import contextlib
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, get_device_context, forward_projection, backward_projection, check_stopping_criterion, estimate_lipschitz_constant
from AOT_biomaps.AOT_Recon.ReconEnums import StopCriterionType
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
    # Fused Kernel : Update PGD + Clamp (Zero-Allocation)
    pgd_update_kernel = cp.ElementwiseKernel(
    'float32 lam_in, float32 grad, float32 alpha',
    'float32 lam_out',
    '''
    float new_val = lam_in - alpha * grad; 
    lam_out = (new_val < 0.0f) ? 0.0f : new_val;
    ''',
    'pgd_update_kernel'
)
    
def calculate_step_size_PGD(SMatrix, preconditioner, num_iters, show_logs):
    """
    Compute the gradient descent step size in the PGD algorithm.
    Args:
        - SMatrix : System matrix.
        - preconditioner : Preconditioner object implementing apply_inverse().
        - eta : Relaxation parameter (typically 1.8-1.99).
        - num_iters : Number of power iterations.
        - show_logs : Print information.
    """

    L = estimate_lipschitz_constant(SMatrix, preconditioner=preconditioner, num_iters=num_iters)

    alpha = 1.0 / L if L > 0 else 1.0

    if show_logs:
        print(f"[AOT-biomaps] Using step size alpha = {alpha:.3e}")

    return alpha


def PGD(
    SMatrix : Union[SMatrix_DENSE, SMatrix_CSR, SMatrix_SELL],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: Union[str, float] = "auto",
    numIterations_stepCalculation: int = 20,
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
    Least Squares reconstruction using Projected Gradient Descent (PGD).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports potential functions:
        - NONE: No regularization

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
        alpha: Step size parameter (float or 'auto' for power method estimation of Lipschitz constant)
        numIterations_stepCalculation: Number of iterations for power method when alpha is "auto"
        preconditioner_type: Preconditioner type (see PreconditionerEnums for available types)
        stop_criterion: Criterion for stopping the iterations
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
        Z = SMatrix.Z
        X = SMatrix.X
        ZX = Z * X

        if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
            raise ValueError(f"[AOT-biomaps] Shape of y {y.shape} does not match SMatrix dimensions (T={SMatrix.T}, N={SMatrix.N}).")

        data_dtype = xp.complex64 if SMatrix.isComplexSMatrix else xp.float32

        y_max = float(np.max(np.abs(y))) if SMatrix.isComplexSMatrix else float(np.max(y))
        if y_max > 0:
            y_norm = y / y_max 

        y_flat = xp.asarray(y_norm.T.flatten().astype(data_dtype))

        # Preconditioner Initialization
        if preconditioner_type == PreconditionerType.DIAGONAL:
            preconditioner = DiagPreconditioner(SMatrix=SMatrix)
            preconditioner.build() # Computes diag(AᴴA) and applies Tikhonov damping
        else:
            preconditioner = NoPreconditioner(SMatrix=SMatrix)

        lambda_flat = xp.zeros(ZX, dtype=xp.float32)        # λ: Actual image (always real)
        residual_buffer = xp.empty_like(y_flat)

        # Setup save indices
        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

        saved_lambda = []
        saved_indices_list = []
        cost_history = [] if isCostFunction else None
        window_history = []

        alpha = calculate_step_size_PGD(SMatrix, preconditioner, numIterations_stepCalculation, show_logs) if alpha == "auto" else alpha

        prec_str = preconditioner.get_name()
        cplx_str = "COMPLEX (4-phases quadrature) " if SMatrix.isComplexSMatrix else "REAL "
        description = f"[AOT-biomaps] {cplx_str} PGD 4-phases ({SMatrix.matrix_type.name}) --- {prec_str} --- {'WITH' if withTumor else 'WITHOUT'} TUMOR --- DEVICE: {SMatrix.device.upper()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        for it in iterator:
            prev_lambda = lambda_flat.copy()
            
            Alambda_bar = forward_projection(SMatrix, lambda_flat)
            xp.subtract(Alambda_bar, y_flat, out=residual_buffer) # residual = (A * λ_bar) - y
    
            # --- DATA FIDELITY GRADIENT WITH PRECONDITIONING --- Apply inverse preconditioner ONLY to the data fidelity term to protect L_prior stability : ∇f_prec = P⁻¹ ∇f_raw where ∇f_raw = Aᴴ(A * λ_bar - y)
            grad_fidelity_prec = xp.ascontiguousarray(preconditioner.apply_inverse(xp.real(backward_projection(SMatrix, residual_buffer))), dtype=xp.float32)

            # Update: λ = P_{>= 0}(λ - α * ∇f_prec)
            if is_gpu:
                pgd_update_kernel(lambda_flat, grad_fidelity_prec, float(alpha), lambda_flat)
            else:
                grad_fidelity_prec *= float(alpha)
                lambda_flat -= grad_fidelity_prec
                np.maximum(lambda_flat, 0.0, out=lambda_flat)

            if isCostFunction:
                # F(x) = 0.5 * ||A * λ - y||²
                cost_history.append(0.5 * float(xp.vdot(residual_buffer, residual_buffer).real)) if SMatrix.isComplexSMatrix else cost_history.append(0.5 * float(xp.vdot(residual_buffer, residual_buffer)))

            # Stopping Criterion
            if stop_criterion != StopCriterionType.MAX_ITERATIONS:
                if SMatrix.experiment.OpticImage is None:
                    ground_truth = None
                else:
                    ground_truth = SMatrix.experiment.OpticImage.phantom if withTumor else SMatrix.experiment.OpticImage.laser.intensity
                gradient_for_stop = grad_fidelity_prec if stop_criterion == StopCriterionType.GRADIENT_NORM else None
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