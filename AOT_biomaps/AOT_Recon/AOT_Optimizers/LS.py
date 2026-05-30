"""
LS.py

Least Squares reconstruction algorithms.
Uses ReconTools functions for all matrix operations.
Single unified function that works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

Supports preconditioning:
- NONE: No preconditioning
- DIAGONAL: Diagonal preconditioning using A^T * 1
"""
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from AOT_biomaps.AOT_Recon.ReconTools import cost_function, forward_projection, backward_projection, minus_axpy, clamp_positive, build_preconditioner, apply_diagonal_preconditioner
from AOT_biomaps.AOT_Recon.ReconEnums import PreconditionerType
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SMatrix_SELL
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SMatrix_CSR
from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SMatrix_DENSE

def LS(
    SMatrix : Union[SMatrix_DENSE, SMatrix_CSR, SMatrix_SELL],
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    alpha: float = 0.01,
    preconditioner_type: PreconditionerType = PreconditionerType.NONE,
    isSavingEachIteration: bool = True,
    isCostFunction: bool = False,
    withTumor: bool = True,
    max_saves: int = 5000,
    show_logs: bool = True,
) -> Tuple[Union[np.ndarray, list], Optional[list], Optional[list]]:
    """
    Least Squares reconstruction using Projected Gradient Descent (PGD).
    
    Uses ReconTools functions for all matrix operations, so it works with
    any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).
    
    Args:
        SMatrix: SMatrix instance (already allocated)
        y: Measurement data
        numIterations: Number of iterations
        alpha: Step size parameter
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
    """   
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
        array_module = cp
    else:
        y_flat = np.asarray(y.T.flatten().astype(np.float32))
        lambda_flat = np.full(ZX, 0.1, dtype=np.float32)
        array_module = np
    
    # Compute preconditioner if requested
    preconditioner, preconditioner_inv = None, None
    if preconditioner_type != PreconditionerType.NONE:
        preconditioner, preconditioner_inv = build_preconditioner(SMatrix, preconditioner_type)
    
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
    
    description = f"AOT-BioMaps -- LS ({matrix_type}) ---- {tumor_str} TUMOR ---- {device.upper()}"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        # Compute gradient: g = A^T * (A * λ - y)
        q_flat = forward_projection(SMatrix, lambda_flat)
        r_flat = minus_axpy(SMatrix, y_flat, q_flat, -1.0)  # r = y - A*λ
        g_flat = backward_projection(SMatrix, r_flat)  # g = A^T * r
        
        # Compute cost function if requested
        if isCostFunction:
            cost_history.append(cost_function(SMatrix, lambda_flat, y_flat, array_module=array_module))

        # Update: λ = λ - alpha * g
        lambda_flat = minus_axpy(SMatrix, lambda_flat, g_flat, alpha)
        
        # Apply diagonal preconditioning if enabled
        if preconditioner_inv is not None:
            lambda_flat = apply_diagonal_preconditioner(lambda_flat, preconditioner_inv, SMatrix)
        
        # Clamp to non-negative
        lambda_flat = clamp_positive(SMatrix, lambda_flat)
        
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
