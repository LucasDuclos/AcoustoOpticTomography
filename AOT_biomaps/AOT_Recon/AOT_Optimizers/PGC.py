"""
PGC.py

Penalized Gauss-Newton Conjugate Gradient (PGC) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports spatial potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import (
    check_gpu_available, forward_projection, backward_projection, 
    clamp_positive, get_potential_function, _get_array_module, estimate_operator_norm
)
from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType, PotentialType, PotentialShapeType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def PGC(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: Union[float, str] = "auto",     
    beta: float = 1.0,       
    delta: float = 0.01,    
    eta: Optional[float] = None, 
    potential_type: PotentialType = PotentialType.QUADRATIC,
    potential_shape: PotentialShapeType = PotentialShapeType.CROSS,
    potential_radius: int = 2,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    Penalized Gauss-Newton Conjugate Gradient (PGC) reconstruction algorithm.
    Uses the Polak-Ribière conjugate direction method to accelerate convergence.
    
    Args:
        SMatrix: System matrix object.
        y: Measurement data vector.
        numIterations: Total number of iterations.
        alpha: Step size parameter (float or 'auto' for power method estimation of Lipschitz constant)
        beta: Regularization weight parameter.
        delta: Threshold for non-quadratic potentials.
        eta: Parameter for Lipschitz estimation.
        potential_type: Type of MRF spatial potential to use.
        potential_shape: Neighborhood shape (PotentialShapeType enum).
        potential_radius: Neighborhood radius in pixels.
        isSavingEachIteration: If True, stores intermediate reconstructions.
        isCostFunction: If True, tracks the cost function history.
        withTumor: Flag for description.
        max_saves: Limit on stored iterations.
        show_logs: Displays tqdm progress bar.
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type.name
    xp = _get_array_module(SMatrix)
    Z, X = SMatrix.Z, SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape mismatch: y={y.shape}, SMatrix T={SMatrix.T}, N={SMatrix.N}.")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)

    # Conjugate Gradient vectors: r (residual), d (direction)
    r = xp.zeros_like(lambda_flat)
    d = xp.zeros_like(lambda_flat)
    prev_r_dot = 0.0

    if alpha == "auto":
        if eta is None:
            print("Warning: eta is not set for power method estimation of step size. Using default value of 1.9.")
            eta = 1.9
        if eta >= 2.0 or eta <= 1.0:
            print(f"Warning: eta should be in (1.0, 2.0) for optimal convergence. Current value: {eta}.")
        
        # Estimate Lipschitz constant using power method
        L_estimate = estimate_operator_norm(SMatrix, num_iters=20)
        alpha = eta / L_estimate if L_estimate > 0 else 1.0
        print(f"Estimated Lipschitz constant: {L_estimate:.4f}, using step size alpha: {alpha:.5f}")

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

    description = f"AOT-BioMaps -- PGC ({matrix_type}) with {potential_type.name} (shape: {potential_shape.name}, r: {potential_radius}) β={beta} ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # 1. Forward model
        q_flat = forward_projection(SMatrix, lambda_flat)
        
        # 2. Compute potential gradient dynamically (Hessian is not used in PGC)
        grad_U, _, U_value = get_potential_function(
            potential_type, SMatrix, lambda_flat, 
            beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
            compute_grad=True, compute_hess=False, compute_energy=isCostFunction
        )

        if isCostFunction:
            q_safe = xp.maximum(q_flat, 1e-10)
            cost_history.append(float(xp.sum(q_safe - y_flat * xp.log(q_safe)) + U_value))

        # 3. Compute Gauss-Newton gradient: A^T * (Ax - y) + grad_U
        residual = q_flat - y_flat
        grad_f = backward_projection(SMatrix, residual) + grad_U
        
        # 4. Conjugate Gradient Update (Polak-Ribière)
        r = -grad_f
        r_dot = xp.sum(r * r)
        
        if it == 0:
            d = r
        else:
            # Conjugacy factor (beta_cg)
            beta_cg = xp.maximum(0, r_dot / (prev_r_dot + 1e-10))
            d = r + beta_cg * d
        
        # 5. Search step
        lambda_flat = lambda_flat + alpha * d
        lambda_flat = clamp_positive(SMatrix, lambda_flat)
        
        prev_r_dot = r_dot

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