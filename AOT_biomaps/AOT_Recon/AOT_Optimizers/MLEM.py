"""
MLEM.py

Maximum Likelihood Expectation Maximization (MLEM) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
Note : Algorithm handles both real and complex data (4-phase quadrature representation) seamlessly, with dynamic kernel selection based on data type.
"""
import contextlib
import warnings
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import get_array_module, get_device_context, forward_projection, backward_projection, check_stopping_criterion
from AOT_biomaps.AOT_Recon.ReconEnums import StopCriterionType
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

    # ---------------------------------------------------------
    # RATIO KERNELS : REAL (float32) and COMPLEX (complex64)
    # ---------------------------------------------------------
    # Kernel for: y / max(q, eps)
    mlem_ratio_kernel__REAL = cp.ElementwiseKernel(
        'float32 y, float32 q, float32 eps',
        'float32 out',
        '''
        float q_safe = q < eps ? eps : q;
        out = y / q_safe;
        ''',
        'mlem_ratio_kernel__REAL'
    )

    mlem_ratio_kernel__COMPLEX = cp.ElementwiseKernel(
        'complex64 y, complex64 q, float32 eps',
        'complex64 out',
        '''
        // abs() is supported natively in CuPy for complex numbers
        float mag = abs(q);
        if (mag < eps) {
            out = y / eps; // Implicit conversion of eps to complex for safe division
        } else {
            out = y / q;
        }
        ''',
        'mlem_ratio_kernel__COMPLEX'
    )

    # ---------------------------------------------------------
    # UPDATE KERNEL : ALWAYS REAL
    # (The reconstructed optical contrast lambda is strictly real and positive)
    # ---------------------------------------------------------
    # Kernel for MLEM Update: lambda * backproj / sens
    mlem_update_kernel = cp.ElementwiseKernel(
        'float32 lam, float32 backproj, float32 sens',
        'float32 lam_out',
        '''
        // Sens is already clamped with eps outside the loop
        float new_val = lam * backproj / sens;
        lam_out = new_val > 0.0f ? new_val : 0.0f; // Clamp to ensure strict positivity
        ''',
        'mlem_update_kernel'
    )

def MLEM(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    denominator_threshold: float = 1e-10,
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
    MLEM reconstruction algorithm. Optimized for GPU/CPU with direct array operations.

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
    
    Args:
        SMatrix: System matrix interface.
        y: Measurement data vector.
        numIterations: Total number of iterations.
        denominator_threshold: Small epsilon to avoid division by zero.
        stop_criterion: Criterion for stopping the iterations
        stop_threshold: Threshold value for the stopping criterion (for MAX_iterations, this is ignored)
        stop_window_size: Window size (used to avoid early stop due to oscillations)
        isSavingEachIteration: Toggle to store intermediate states.
        isCostFunction: Toggle to track the log-likelihood history.
        withTumor: Flag for description.
        max_saves: Max number of history frames to keep in memory.
        show_logs: Display progress bar.
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

        if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
            raise ValueError(f"[AOT-biomaps] Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

        # Dynamic type definition (Data = Real or Complex)
        data_dtype = xp.complex64 if SMatrix.isComplexSMatrix else xp.float32
        
        if SMatrix.isComplexSMatrix:
            print("[AOT-biomaps] Warning: MLEM relies on Poisson statistics mathematically defined for real positive values. Applying heuristic magnitude-based updates for complex data.")

        y_flat = xp.asarray(y.T.flatten().astype(data_dtype))
        lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)
        ratio_buffer = xp.empty_like(y_flat)

        # Pre-calculate Sensitivity (A^T * 1) - The native preconditioner of EM
        ones_vec = xp.ones(SMatrix.N * SMatrix.T, dtype=data_dtype)
        sens_img_raw = backward_projection(SMatrix, ones_vec)
        
        # If the system is complex, we use the absolute value of the sensitivity for stability
        if SMatrix.isComplexSMatrix:
            sens_img = xp.abs(sens_img_raw).astype(xp.float32)
        else:
            sens_img = sens_img_raw.astype(xp.float32)
            
        xp.maximum(sens_img, 1e-10, out=sens_img)

        # Setup save indices
        save_indices = np.unique(np.append(np.arange(0, numIterations, max(1, numIterations // max_saves)), numIterations - 1)).tolist()

        saved_lambda = []
        saved_indices_list = []
        cost_history = [] if isCostFunction else None
        window_history = []

        # Progress bar configuration
        if SMatrix.isComplexSMatrix:
            description = f"[AOT-biomaps] 4-phases quadrature MLEM ({SMatrix.matrix_type.name}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
        else:
            description = f"[AOT-biomaps] MLEM ({SMatrix.matrix_type.name}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- {SMatrix.device.upper()}"
            
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        for it in iterator:
            prev_lambda = lambda_flat.copy() if stop_criterion != StopCriterionType.MAX_ITERATIONS else None
            
            # Forward projection (Complex or Real depending on SMatrix)
            q_flat = forward_projection(SMatrix, lambda_flat)

            # =========================================================
            # RATIO CALCULATION (y / q)
            # =========================================================
            if is_gpu:
                if SMatrix.isComplexSMatrix:
                    mlem_ratio_kernel__COMPLEX(y_flat, q_flat, denominator_threshold, ratio_buffer)
                else:
                    mlem_ratio_kernel__REAL(y_flat, q_flat, denominator_threshold, ratio_buffer)
            else:
                if SMatrix.isComplexSMatrix:
                    # Complex safe division fallback for CPU
                    mask = np.abs(q_flat) < denominator_threshold
                    q_safe = q_flat.copy()
                    q_safe[mask] = denominator_threshold
                    np.divide(y_flat, q_safe, out=ratio_buffer)
                else:
                    np.maximum(q_flat, denominator_threshold, out=q_flat)
                    np.divide(y_flat, q_flat, out=ratio_buffer)

            # =========================================================
            # BACKPROJECTION AND IMAGE UPDATE
            # =========================================================
            backproj_ratio_raw = backward_projection(SMatrix, ratio_buffer)
            
            # Crucial extraction of the real part for the image update
            if SMatrix.isComplexSMatrix:
                backproj_ratio = xp.real(backproj_ratio_raw).astype(xp.float32)
            else:
                backproj_ratio = backproj_ratio_raw

            # MLEM Update: lambda = lambda * (A^T * (y / Ax)) / (A^T * 1)
            if is_gpu:
                mlem_update_kernel(lambda_flat, backproj_ratio, sens_img, lambda_flat)
            else:
                lambda_flat *= backproj_ratio
                lambda_flat /= sens_img
                np.maximum(lambda_flat, 0.0, out=lambda_flat)

            # Track cost function (Negative Log-Likelihood)
            if isCostFunction:
                if SMatrix.isComplexSMatrix:
                    # Heuristic absolute log-likelihood for complex representations
                    q_mag = xp.maximum(xp.abs(q_flat), 1e-10)
                    y_mag = xp.abs(y_flat)
                    cost_history.append(float(xp.sum(q_mag - y_mag * xp.log(q_mag))))
                else:
                    q_safe = xp.maximum(q_flat, 1e-10)
                    cost_history.append(float(xp.sum(q_safe - y_flat * xp.log(q_safe))))

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