"""
LBFGS.py

L-BFGS optimization algorithm with regularization support.
Uses unified SMatrix interface and ReconTools functions.
Manual implementation without scipy.minimize dependency.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
"""

from AOT_biomaps.AOT_Recon.ReconTools import (
    projection, backprojection, clamp_positive, calculate_memory_requirement, 
    check_gpu_memory, zeros, ones, axpby, minus_axpy, dot_product, fill_array,
    quadratic_potential, huber_potential, relative_difference_potential, build_adjacency_indices
)
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.Config import config

import numpy as np
from tqdm import trange
import warnings

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Non-differentiable potentials that LBFGS cannot handle
_NON_DIFFERENTIABLE_POTENTIALS = {PotentialType.TV}


def LBFGS(
    SMatrix,
    y,
    numIterations=100,
    potential_type=PotentialType.QUADRATIC,
    alpha=1.0,
    beta=1.0,
    delta=0.01,
    gamma=0.01,
    sigma=0.01,
    isSavingEachIteration=True,
    isCostFunction=False,
    withTumor=True,
    max_saves=5000,
    show_logs=True,
):
    """
    L-BFGS optimization algorithm with regularization support.
    Manual implementation without scipy.minimize dependency.
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        potential_type: Type of potential function (QUADRATIC, HUBER_PIECEWISE, NUYTS_RELATIVE)
        alpha: Regularization weight
        beta: Additional parameter for potential functions
        delta: Parameter for Huber potential
        gamma: Parameter for LBFGS
        sigma: Parameter for LBFGS
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
    TN = SMatrix.N * SMatrix.T
    
    # Convert y to appropriate format
    if device == 'gpu' and CUPY_AVAILABLE:
        y_flat = cp.asarray(y.T.flatten().astype(np.float32))
        theta_flat = cp.full(ZX, 0.1, dtype=cp.float32)
        array_module = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        theta_flat = np.full(ZX, 0.1, dtype=np.float32)
        array_module = np
    
    # Select potential function
    def get_potential(U):
        if potential_type == PotentialType.QUADRATIC:
            return quadratic_potential(SMatrix, U, alpha)
        elif potential_type == PotentialType.HUBER_PIECEWISE:
            return huber_potential(SMatrix, U, alpha, delta)
        elif potential_type == PotentialType.NUYTS_RELATIVE:
            return relative_difference_potential(SMatrix, U, alpha, beta)
        else:
            raise ValueError(f"Unsupported potential type: {potential_type}")
    
    # LBFGS parameters
    m = 10  # Memory size (number of previous steps to store)
    
    # Initialize LBFGS variables
    s_history = []  # List of s vectors (theta_{k+1} - theta_k)
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
    
    saved_theta = []
    saved_indices_list = []
    cost_history = [] if isCostFunction else None
    
    description = f"AOT-BioMaps -- LBFGS ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Compute gradient: grad = A^T * (A * theta - y) + grad_U
        q_flat = projection(SMatrix, theta_flat)
        grad_f = backprojection(SMatrix, q_flat - y_flat)  # Gradient of data fidelity (LS)
        grad_U, _, U_value = get_potential(theta_flat)  # Gradient of regularization
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
        
        # Line search (simple backtracking)
        def compute_cost(theta):
            q = projection(SMatrix, theta)
            cost = 0.5 * float(dot_product(SMatrix, q - y_flat, q - y_flat))
            _, _, U_val = get_potential(theta)
            cost += float(U_val)
            return cost
        
        current_cost = compute_cost(theta_flat)
        
        # Try step size = 1 initially
        step = 1.0
        theta_new = axpby(SMatrix, theta_flat, d_flat, 1.0, step)
        theta_new = clamp_positive(SMatrix, theta_new)
        new_cost = compute_cost(theta_new)
        
        # Backtracking line search
        c1 = 1e-4
        max_ls_iter = 20
        ls_iter = 0
        while new_cost > current_cost + c1 * step * dot_product(SMatrix, grad_flat, d_flat) and ls_iter < max_ls_iter:
            step *= 0.5
            theta_new = axpby(SMatrix, theta_flat, d_flat, 1.0, step)
            theta_new = clamp_positive(SMatrix, theta_new)
            new_cost = compute_cost(theta_new)
            ls_iter += 1
        
        # Update s and y for LBFGS
        s_k = theta_new - theta_flat
        grad_new_q = projection(SMatrix, theta_new)
        grad_new_f = backprojection(SMatrix, grad_new_q - y_flat)
        grad_new_U, _, _ = get_potential(theta_new)
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
        
        # Update theta
        theta_flat = theta_new
        
        if isSavingEachIteration and it in save_indices:
            if device == 'gpu' and CUPY_AVAILABLE:
                saved_theta.append(cp.asnumpy(theta_flat.reshape(Z, X)))
            else:
                saved_theta.append(theta_flat.reshape(Z, X).copy())
            saved_indices_list.append(it)
    
    if device == 'gpu' and CUPY_AVAILABLE:
        cp.cuda.Stream.null.synchronize()
        final_result = cp.asnumpy(theta_flat.reshape(Z, X))
    else:
        final_result = theta_flat.reshape(Z, X)
    
    if isSavingEachIteration:
        return saved_theta, saved_indices_list, cost_history
    else:
        return final_result, None, cost_history
