"""
PIGD.py

Penalized Iterative Gradient Descent (PIGD) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import (
    check_gpu_available, estimate_operator_norm, forward_projection, backward_projection, 
    clamp_positive, get_potential_function, cost_function, _get_array_module
)
from AOT_biomaps.AOT_Recon.ReconEnums import OptimizerType, PotentialType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def PIGD(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: float = 1.0,      # Step size (learning rate)
    beta: float = 1.0,       # Regularization weight
    delta: float = 0.01,     # Parameter for potential function (e.g., Huber threshold)
    eta: Optional[float] = None, # Parameter for the Lipschitz constant estimation (must be < 2 for convergence and > 1 for faster convergence). Useless if alpha is a float.
    potential_type: PotentialType = PotentialType.QUADRATIC,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    Penalized Iterative Gradient Descent (PIGD) reconstruction algorithm.
    This algorithm iteratively updates the image using the gradient of the data 
    fidelity term and the gradient of the potential function.
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data (shape: (T, N))
        numIterations: Number of iterations
        alpha: Step size parameter
        beta: Regularization weight
        delta: Additional parameter for potential functions
        eta: Parameter for the Lipschitz constant estimation (must be < 2 for convergence and > 1 for faster convergence). Useless if alpha is a float.
        potential_type: Type of potential function (QUADRATIC, HUBER, RELATIVE_DIFFERENCE)
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
    Z, X = SMatrix.Z, SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions (T={SMatrix.T}, N={SMatrix.N}).")

    y_flat = xp.asarray(y.T.flatten().astype(xp.float32))
    lambda_flat = xp.full(ZX, 0.1, dtype=xp.float32)

    # Pre-calculate sensitivity image (A^T * 1) for diagonal preconditioning
    sens_img = backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32)+1e-10)
    inv_sens = 1.0 / xp.maximum(sens_img, 1e-8)

    if alpha == "auto":
        if eta is None:
            print("Warning: eta is not set for power method estimation of step size. Using default value of 1.9.")
            eta = 1.9
        if eta >= 2.0 or eta <= 1.0:
            print(f"Warning: For power method estimation of step size, eta should be in (1.0, 2.0) for convergence and faster convergence. Current value: {eta}. Proceeding with the given value, but consider adjusting it for better performance.")
        # Estimate Lipschitz constant using power method
        L_estimate = estimate_operator_norm(SMatrix, num_iters=20)
        alpha = eta / L_estimate if L_estimate > 0 else 1.0
        print(f"Estimated Lipschitz constant: {L_estimate:.4f}, using step size alpha: {alpha:.5f}")


    save_indices = list(range(0, numIterations, max(1, numIterations // max_saves)))
    if save_indices[-1] != numIterations - 1:
        save_indices.append(numIterations - 1)

    saved_lambda = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None

    description = f"AOT-BioMaps -- PIGD ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # 1. Forward projection
        q_flat = forward_projection(SMatrix, lambda_flat)
        
        # 2. Compute potential gradient (Hessian is not used in PIGD)
        # Gradient of potential U(lambda)
        grad_U, _ , U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta)
        if grad_U is None: 
            grad_U = xp.zeros_like(lambda_flat)
            U_value = 0.0

        # 3. Compute cost history
        if isCostFunction:
            q_safe = xp.maximum(q_flat, 1e-10)
            llh = xp.sum(q_safe - y_flat * xp.log(q_safe))
            cost_history.append(float(llh + U_value))

        # 4. Compute gradient of fidelity term: A^T * (A*lambda - y)
        # Note: (q_flat - y_flat) is the gradient of the Gaussian log-likelihood
        # For Poisson, we use the normalized residual (1 - y/Ax)
        residual = 1.0 - (y_flat / xp.maximum(q_flat, 1e-10))
        grad_fidelity = backward_projection(SMatrix, residual)

        # 5. PIGD Update: lambda = lambda - alpha * M^-1 * (Grad_Fidelity + Grad_Potential)
        # Using diagonal preconditioning (inv_sens) to normalize the gradient
        update_direction = (grad_fidelity + grad_U) * inv_sens
        lambda_flat = lambda_flat - alpha * update_direction

        # 6. Enforce non-negativity
        lambda_flat = clamp_positive(SMatrix, lambda_flat)

        if isSavingEachIteration and it in save_indices:
            saved_lambda.append(lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)

    final_result = lambda_flat.reshape(Z, X).get() if hasattr(lambda_flat, 'get') else lambda_flat.reshape(Z, X)

    return (saved_lambda, saved_indices_list, cost_history) if isSavingEachIteration else (final_result, None, cost_history)