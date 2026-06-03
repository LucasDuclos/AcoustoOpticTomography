"""
PPGMLEM.py

Penalized Preconditioned Gradient MLEM (PPGMLEM) reconstruction algorithm.
Uses unified SMatrix interface and ReconTools functions.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports spatial potential functions: QUADRATIC, HUBER, RELATIVE_DIFFERENCE
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import _get_array_module, estimate_operator_norm, forward_projection, backward_projection, clamp_positive, get_potential_function, check_gpu_available
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType, PotentialShapeType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def PPGMLEM(
    SMatrix: Union['SMatrix_DENSE', 'SMatrix_CSR', 'SMatrix_SELL'],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: Union[str, float] = "auto",     
    beta: float = 1.0,       
    delta: float = 1.0,      
    gamma: float = 0.01,     
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
    Penalized Preconditioned Gradient MLEM (PPGMLEM) reconstruction algorithm.
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports preconditioning:
    - NONE: No preconditioning
    - DIAGONAL: Diagonal preconditioning using A^T * 1
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data (shape: (T, N))
        numIterations: Number of iterations
        alpha: Step size parameter (float or 'auto' for power method estimation of Lipschitz constant)
        beta: Regularization weight
        delta: Parameter for Huber potential (threshold) or for RELATIVE_DIFFERENCE potential
        gamma: Preconditioning parameter
        eta: Parameter for Lipschitz estimation if alpha is "auto"
        potential_type: Type of potential function (QUADRATIC, HUBER, RELATIVE_DIFFERENCE)
        potential_shape: Neighborhood shape (PotentialShapeType enum)
        potential_radius: Neighborhood radius in pixels
        preconditioner_type: Type of preconditioner to use (default: NONE)
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
        
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
        - reconstructed_image: Final or list of images (Z, X)
        - saved_indices: List of saved iteration indices (None if not saving)
        - cost_history: List of cost function values (None if not requested)
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

    # Pre-compute sensitivity (A^T * 1)
    sens_img = backward_projection(SMatrix, xp.ones(SMatrix.N * SMatrix.T, dtype=xp.float32))
    sens_img = xp.maximum(sens_img, 1e-10)

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

    description = f"AOT-BioMaps -- PPGMLEM ({matrix_type}) with {potential_type.name} (shape: {potential_shape.name}, r: {potential_radius}) β={beta} ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

    for it in iterator:
        # 1. Forward projection
        q_flat = forward_projection(SMatrix, lambda_flat)

        # 2. Compute potential Gradient & Hessian dynamically
        grad_U, hess_U, U_value = get_potential_function(
            potential_type, SMatrix, lambda_flat, 
            beta=beta, delta=delta, shape=potential_shape, radius=potential_radius,
            compute_grad=True, compute_hess=True, compute_energy=isCostFunction
        )

        # 3. Track cost function (Negative Poisson LLH + Penalty)
        if isCostFunction:
            q_safe = xp.maximum(q_flat, 1e-10)
            nllh = xp.sum(q_safe - y_flat * xp.log(q_safe))
            cost_history.append(float(nllh + U_value))

        # 4. Compute ratio: y / (A*λ + ε)
        ratio = y_flat / xp.maximum(q_flat, 1e-10)

        # 5. Backward projection: c_flat = A^T * (y / Ax)
        c_flat = backward_projection(SMatrix, ratio)

        # 6. Correct mathematical gradient formulation for Poisson
        # EM Gradient direction = c_flat - sensitivity
        grad_EM = c_flat - sens_img
        
        # Total Gradient = grad_EM - grad_U (We want to maximize LLH and minimize U)
        total_grad = grad_EM - grad_U

        # 7. Preconditioned Update stabilized by Hessian and Gamma
        # denom = sensitivity + delta * hess_U + gamma
        denom = sens_img + delta * hess_U + gamma
        
        # lambda_new = lambda + alpha * (Total_Gradient / Denominator)
        lambda_flat = lambda_flat + alpha * (total_grad / xp.maximum(denom, 1e-10))

        # 8. Clamp to non-negative
        lambda_flat = clamp_positive(SMatrix, lambda_flat)

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