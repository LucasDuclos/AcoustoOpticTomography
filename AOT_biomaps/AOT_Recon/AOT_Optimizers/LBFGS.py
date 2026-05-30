"""
LBFGS.py

L-BFGS optimization algorithm with regularization support.
Uses unified SMatrix interface and ReconTools functions.
Manual implementation without scipy.minimize dependency.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
"""

import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

from AOT_biomaps.AOT_Recon.ReconTools import cost_function, forward_projection, backward_projection, clamp_positive, axpby, minus_axpy, dot_product, build_preconditioner, apply_diagonal_preconditioner, get_potential_function
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType, PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Non-differentiable potentials that LBFGS cannot handle
_NON_DIFFERENTIABLE_POTENTIALS = {PotentialType.TOTAL_VARIATION}


def LBFGS(
    SMatrix: Union[SMatrix_DENSE, SMatrix_CSR, SMatrix_SELL],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    beta: float = 1.0,
    delta: float = 0.01,
    potential_type: PotentialType = PotentialType.QUADRATIC,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    L-BFGS optimization algorithm with regularization support.
    Manual implementation without scipy.minimize dependency.
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Supports preconditioning:
    - NONE: No preconditioning
    - DIAGONAL: Diagonal preconditioning using A^T * 1
    
    Args:
        SMatrix: SMatrix instance (already allocated) must be SMatrix_DENSE, SMatrix_CSR, or SMatrix_SELL
        y: Measurement data (shape: (T, N))
        numIterations: Number of iterations
        potential_type: Type of potential function (QUADRATIC, HUBER, RELATIVE_DIFFERENCE)
        beta: Regularization weight
        delta: Parameter for Huber potential
        isSavingEachIteration: If True, saves intermediate results
        isCostFunction: If True, computes and saves cost function history
        withTumor: Boolean for description only
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
        preconditioner_type: Type of preconditioner to use (default: NONE)
        
    Returns:
        tuple: (reconstructed_image, saved_indices, cost_history)
        - reconstructed_image: Final or list of images (Z, X)
        - saved_indices: List of saved iteration indices (None if not saving)
        - cost_history: List of cost function values (None if not requested)
        
    Raises:
        ValueError: If potential_type is TV (non-differentiable, not compatible with LBFGS)
    """
    # Check if potential is differentiable
    if potential_type in _NON_DIFFERENTIABLE_POTENTIALS:
        raise ValueError(f"LBFGS cannot handle non-differentiable potentials like {potential_type}. Use PDHG instead.")
    
    tumor_str = "WITH" if withTumor else "WITHOUT"
    
    # Get device from SMatrix
    device = SMatrix.device
    matrix_type = SMatrix.matrix_type
    
    # Get dimensions
    Z = SMatrix.Z
    X = SMatrix.X
    ZX = Z * X

    if SMatrix.T != y.shape[0] or SMatrix.N != y.shape[1]:
        raise ValueError(f"Shape of y {y.shape} does not match SMatrix dimensions (T={SMatrix.T}, N={SMatrix.N}).")
    
    # Convert y to appropriate format
    if device == 'gpu' and CUPY_AVAILABLE:
        y_flat = cp.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = cp.full(ZX, 0.1, dtype=cp.float32)
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = np.full(ZX, 0.1, dtype=np.float32)

    
    # Compute preconditioner if requested
    preconditioner, preconditioner_inv = None, None
    if preconditioner_type != PreconditionerType.NONE:
        preconditioner, preconditioner_inv = build_preconditioner(SMatrix, preconditioner_type)
        
    # LBFGS parameters
    m = 10  # Memory size (number of previous steps to store)
    
    # Initialize LBFGS variables
    s_history = []  # List of s vectors (lambda_{k+1} - lambda_k)
    y_history = []  # List of y vectors (grad_{k+1} - grad_k)
    rho_history = []  # List of rho values (1 / y^T * s)
    
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
    
    description = f"AOT-BioMaps -- LBFGS ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Compute gradient: grad = A^T * (A * λ - y) + grad_U
        q_flat = forward_projection(SMatrix, lambda_flat)
        grad_f = backward_projection(SMatrix, q_flat - y_flat)  # Gradient of data fidelity (LS)
        grad_U, _, U_value = get_potential_function(potential_type, SMatrix, lambda_flat, beta=beta, delta=delta)  # Gradient of regularization
        grad_flat = grad_f + grad_U
        
        # Compute cost function if requested
        if isCostFunction:
            # LS cost + regularization
            cost = 0.5 * float(dot_product(SMatrix, q_flat - y_flat, q_flat - y_flat)) + float(U_value)
            cost_history.append(cost)
        
        # LBFGS two-loop recursion to compute search direction
        if it == 0:
            # First iteration: use negative gradient as search direction
            d_flat = -grad_flat
        else:
            # Two-loop recursion
            q = grad_flat.copy() if hasattr(grad_flat, 'copy') else grad_flat + 0
            alpha_list = []
            
            # First loop (backward)
            for i in range(len(s_history) - 1, -1, -1):
                s = s_history[i]
                y = y_history[i]
                rho = rho_history[i]
                alpha_i = rho * dot_product(SMatrix, s, q)
                alpha_list.append(alpha_i)
                q = minus_axpy(SMatrix, q, y, alpha_i)
            
            # Initial Hessian approximation
            if len(s_history) > 0:
                gamma_k = dot_product(SMatrix, s_history[-1], y_history[-1]) / dot_product(SMatrix, y_history[-1], y_history[-1])
            else:
                gamma_k = 1.0
            
            d_flat = gamma_k * q
            
            # Second loop (forward)
            for i in range(len(s_history)):
                s = s_history[i]
                y = y_history[i]
                rho = rho_history[i]
                beta_i = rho * dot_product(SMatrix, y, d_flat)
                d_flat = axpby(SMatrix, d_flat, s, 1.0, alpha_list[i] - beta_i)
        
        
        current_cost = cost_function(SMatrix, lambda_flat, y_flat, potential_type, beta=beta, delta=delta, array_module=cp if device == 'gpu' and CUPY_AVAILABLE else np)
        
        # Try step size = 1 initially
        step = 1.0
        lambda_new = axpby(SMatrix, lambda_flat, d_flat, 1.0, step)
        lambda_new = clamp_positive(SMatrix, lambda_new)
        new_cost = cost_function(SMatrix, lambda_new, y_flat, potential_type, beta=beta, delta=delta, array_module=cp if device == 'gpu' and CUPY_AVAILABLE else np)
        
        # Backtracking line search
        c1 = 1e-4
        max_ls_iter = 20
        ls_iter = 0
        while new_cost > current_cost + c1 * step * dot_product(SMatrix, grad_flat, d_flat) and ls_iter < max_ls_iter:
            step *= 0.5
            lambda_new = axpby(SMatrix, lambda_flat, d_flat, 1.0, step)
            lambda_new = clamp_positive(SMatrix, lambda_new)
            new_cost = cost_function(SMatrix, lambda_new, y_flat, potential_type, beta=beta, delta=delta, array_module=cp if device == 'gpu' and CUPY_AVAILABLE else np)
            ls_iter += 1
        
        # Update s and y for LBFGS
        s_k = lambda_new - lambda_flat
        grad_new_q = forward_projection(SMatrix, lambda_new)
        grad_new_f = backward_projection(SMatrix, grad_new_q - y_flat)
        grad_new_U, _, _ = get_potential_function(potential_type, SMatrix, lambda_new, beta=beta, delta=delta)
        grad_new_flat = grad_new_f + grad_new_U
        y_k = grad_new_flat - grad_flat
        
        # Update LBFGS history
        if len(s_history) >= m:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)
        
        s_history.append(s_k)
        y_history.append(y_k)
        rho_history.append(1.0 / (dot_product(SMatrix, y_k, s_k) + 1e-12))
        
        # Update lambda
        lambda_flat = lambda_new
        
        # Apply diagonal preconditioning if enabled
        if preconditioner_inv is not None:
            lambda_flat = apply_diagonal_preconditioner(lambda_flat, preconditioner_inv, SMatrix)
        
        if isSavingEachIteration and it in save_indices:
            if device == 'gpu' and CUPY_AVAILABLE:
                saved_lambda.append(cp.asnumpy(lambda_flat.reshape(Z, X)))
            else:
                saved_lambda.append(lambda_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)
    
    if device == 'gpu' and CUPY_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(lambda_flat.reshape(Z, X))
    else:
        final_result = lambda_flat.reshape(Z, X)
    
    if isSavingEachIteration:
        return saved_lambda, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history
