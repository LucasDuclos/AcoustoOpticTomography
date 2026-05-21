"""
MLEM.py

Maximum Likelihood Expectation Maximization (MLEM) reconstruction algorithm.
Supports both CPU (NumPy) and GPU (CuPy) implementations.

This module provides unified MLEM implementations that work with:
- Dense matrices
- CSR sparse matrices
- SELL-C-sigma sparse matrices
"""

import warnings
import numpy as np
from tqdm import trange
from typing import Optional, Union, Tuple
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Recon.ReconEnums import SMatrixType

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def MLEM(
    SMatrix,
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    withTumor: bool = True,
    device: Optional[str] = None,
    denominator_threshold: float = 1e-6,
    max_saves: int = 5000,
    show_logs: bool = True,
    smatrixType: SMatrixType = SMatrixType.SELL,
) -> Tuple[Union[np.ndarray, list], Optional[list]]:
    """
    Unified MLEM algorithm for Acousto-Optic Tomography.
    
    Supports both CPU and GPU implementations based on availability.
    
    Args:
        SMatrix: System matrix (SparseMatrix wrapper, CSR, SELL, or dense array)
        y: Measurement data
        numIterations: Number of iterations
        isSavingEachIteration: If True, saves intermediate results
        withTumor: Boolean for description only
        device: 'cpu' or 'gpu' (auto-detected if None)
        denominator_threshold: Threshold for denominator to avoid division by zero
        max_saves: Maximum number of intermediate saves
        show_logs: If True, shows progress bar
        smatrixType: Type of sparse matrix (SELL, CSR, DENSE)
        
    Returns:
        Tuple of (reconstructed image(s), iteration indices) if isSavingEachIteration
        or (final image, None) otherwise
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"
    
    # Determine device
    if device is None:
        device = 'gpu' if CUPY_AVAILABLE else 'cpu'
    
    # Check if SMatrix is a wrapper
    try:
        from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import SparseMatrix
        if isinstance(SMatrix, SparseMatrix):
            smatrixType = SMatrixType.SELL if SMatrix.matrix_type == 'SELL' else SMatrixType.CSR
            device = SMatrix.device
    except:
        pass
    
    # Route to appropriate implementation
    if smatrixType == SMatrixType.CSR:
        return MLEM_sparse(SMatrix, y, numIterations, isSavingEachIteration, 
                         tumor_str, max_saves, denominator_threshold, show_logs, device)
    elif smatrixType == SMatrixType.SELL:
        return MLEM_sparse(SMatrix, y, numIterations, isSavingEachIteration,
                         tumor_str, max_saves, denominator_threshold, show_logs, device)
    elif smatrixType == SMatrixType.DENSE:
        return MLEM_dense(SMatrix, y, numIterations, isSavingEachIteration,
                        tumor_str, max_saves, denominator_threshold, show_logs, device)
    else:
        raise ValueError(f"Unsupported SMatrixType: {smatrixType}")


def MLEM_dense(
    SMatrix: np.ndarray,
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    tumor_str: str = "WITH",
    max_saves: int = 5000,
    denominator_threshold: float = 1e-6,
    show_logs: bool = True,
    device: Optional[str] = None,
) -> Tuple[Union[np.ndarray, list], Optional[list]]:
    """
    MLEM implementation for dense matrices.
    
    Args:
        SMatrix: Dense system matrix (T, Z, X, N)
        y: Measurement data
        numIterations: Number of iterations
        isSavingEachIteration: If True, saves intermediate results
        tumor_str: Tumor string for description
        max_saves: Maximum number of intermediate saves
        denominator_threshold: Threshold for denominator
        show_logs: If True, shows progress bar
        device: 'cpu' or 'gpu'
        
    Returns:
        Tuple of (reconstructed image(s), iteration indices) or (final image, None)
    """
    try:
        if device == 'gpu' and CUPY_AVAILABLE:
            return _MLEM_dense_gpu(SMatrix, y, numIterations, isSavingEachIteration,
                                  tumor_str, max_saves, denominator_threshold, show_logs)
        else:
            return _MLEM_dense_cpu(SMatrix, y, numIterations, isSavingEachIteration,
                                  tumor_str, max_saves, denominator_threshold, show_logs)
    except Exception as e:
        warnings.warn(f"Error in dense MLEM: {type(e).__name__}: {e}")
        return None, None


def _MLEM_dense_gpu(
    SMatrix: np.ndarray,
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int,
    isSavingEachIteration: bool,
    tumor_str: str,
    max_saves: int,
    denominator_threshold: float,
    show_logs: bool,
) -> Tuple[Union[np.ndarray, list], Optional[list]]:
    """MLEM implementation for dense matrices using CuPy."""
    T, Z, X, N = SMatrix.shape
    ZX = Z * X
    TN = T * N
    
    # Convert to CuPy arrays
    A_flat = cp.asarray(SMatrix.astype(np.float32)).transpose(0, 3, 1, 2).reshape(TN, ZX)
    y_flat = cp.asarray(y.astype(np.float32).reshape(-1))
    theta_flat = cp.ones(ZX, dtype=cp.float32)
    norm_factor_flat = cp.sum(SMatrix.astype(np.float32), axis=(0, 3)).reshape(-1)
    
    description = f"AOT-BioMaps -- ML-EM ---- {tumor_str} TUMOR ---- GPU"
    
    # Setup save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)
    
    saved_theta = []
    saved_indices = []
    
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        q_flat = A_flat @ theta_flat
        mask = q_flat >= denominator_threshold
        e_flat = cp.where(mask, y_flat / (q_flat + np.finfo(np.float32).eps), cp.ones_like(q_flat))
        c_flat = A_flat.T @ e_flat
        theta_flat = (theta_flat / (norm_factor_flat + np.finfo(np.float32).eps)) * c_flat
        
        if isSavingEachIteration and it in save_indices:
            saved_theta.append(cp.asnumpy(theta_flat.reshape(Z, X)))
            saved_indices.append(it)
    
    cp.cuda.Stream.null.synchronize()
    final_result = cp.asnumpy(theta_flat.reshape(Z, X))
    
    if isSavingEachIteration:
        return saved_theta, saved_indices
    else:
        return final_result, None


def _MLEM_dense_cpu(
    SMatrix: np.ndarray,
    y: np.ndarray,
    numIterations: int,
    isSavingEachIteration: bool,
    tumor_str: str,
    max_saves: int,
    denominator_threshold: float,
    show_logs: bool,
) -> Tuple[Union[np.ndarray, list], Optional[list]]:
    """MLEM implementation for dense matrices using NumPy."""
    T, Z, X, N = SMatrix.shape
    A_flat = SMatrix.astype(np.float32).transpose(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.astype(np.float32).reshape(-1)
    theta_0 = np.ones((Z, X), dtype=np.float32)
    matrix_theta = [theta_0]
    saved_indices = [0]
    normalization_factor = np.sum(SMatrix, axis=(0, 3)).astype(np.float32)
    normalization_factor_flat = normalization_factor.reshape(-1)
    
    # Setup save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)
    
    description = f"AOT-BioMaps -- ML-EM ---- {tumor_str} TUMOR ---- CPU"
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    
    for it in iterator:
        theta_p = matrix_theta[-1]
        theta_p_flat = theta_p.reshape(-1)
        q_flat = A_flat @ theta_p_flat
        
        mask = q_flat >= denominator_threshold
        e_flat = np.where(mask, y_flat / (q_flat + np.finfo(np.float32).tiny), 1.0)
        
        c_flat = A_flat.T @ e_flat
        theta_p_plus_1_flat = theta_p_flat / (normalization_factor_flat + np.finfo(np.float32).tiny) * c_flat
        theta_p_plus_1 = theta_p_plus_1_flat.reshape(Z, X)
        
        if isSavingEachIteration and (it + 1) in save_indices:
            matrix_theta.append(theta_p_plus_1)
            saved_indices.append(it + 1)
        else:
            matrix_theta[-1] = theta_p_plus_1
    
    if not isSavingEachIteration:
        return matrix_theta[-1], None
    else:
        return matrix_theta, saved_indices


def MLEM_sparse(
    SMatrix,
    y: Union[np.ndarray, 'cp.ndarray'],
    numIterations: int = 100,
    isSavingEachIteration: bool = True,
    tumor_str: str = "WITH",
    max_saves: int = 5000,
    denominator_threshold: float = 1e-6,
    show_logs: bool = True,
    device: Optional[str] = None,
) -> Tuple[Union[np.ndarray, list], Optional[list]]:
    """
    MLEM implementation for sparse matrices (CSR or SELL).
    
    Args:
        SMatrix: Sparse matrix (SparseMatrix wrapper, CSR, or SELL)
        y: Measurement data
        numIterations: Number of iterations
        isSavingEachIteration: If True, saves intermediate results
        tumor_str: Tumor string for description
        max_saves: Maximum number of intermediate saves
        denominator_threshold: Threshold for denominator
        show_logs: If True, shows progress bar
        device: 'cpu' or 'gpu'
        
    Returns:
        Tuple of (reconstructed image(s), iteration indices) or (final image, None)
    """
    try:
        # Determine device from SMatrix or parameter
        if device is None:
            device = getattr(SMatrix, 'device', 'gpu' if CUPY_AVAILABLE else 'cpu')
        
        # Get matrix properties
        if hasattr(SMatrix, 'N'):
            N = SMatrix.N
            T = SMatrix.T
            Z = SMatrix.Z
            X = SMatrix.X
        else:
            # For dense matrices
            T, Z, X, N = SMatrix.shape
        
        TN = T * N
        ZX = Z * X
        
        # Convert y to appropriate format
        if device == 'gpu' and CUPY_AVAILABLE:
            y_flat = cp.asarray(y.T.flatten().astype(np.float32))
            theta_flat = cp.full(ZX, 0.1, dtype=cp.float32)
            norm_factor_inv = cp.asarray(SMatrix.norm_factor_inv) if hasattr(SMatrix, 'norm_factor_inv') else None
        else:
            y_flat = np.asarray(y.T.flatten().astype(np.float32))
            theta_flat = np.full(ZX, 0.1, dtype=np.float32)
            norm_factor_inv = SMatrix.norm_factor_inv if hasattr(SMatrix, 'norm_factor_inv') else None
        
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
        
        description = f"AOT-BioMaps -- ML-EM (Sparse) ---- {tumor_str} TUMOR ---- {device.upper()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
        
        for it in iterator:
            # Forward projection
            q_flat = SMatrix.projection(theta_flat)
            
            # Compute ratio
            if device == 'gpu' and CUPY_AVAILABLE:
                mask = q_flat >= denominator_threshold
                e_flat = cp.where(mask, y_flat / (q_flat + 1e-8), cp.ones_like(q_flat))
            else:
                mask = q_flat >= denominator_threshold
                e_flat = np.where(mask, y_flat / (q_flat + 1e-8), np.ones_like(q_flat))
            
            # Backprojection
            c_flat = SMatrix.backprojection(e_flat)
            
            # Update theta
            if norm_factor_inv is not None:
                if device == 'gpu' and CUPY_AVAILABLE:
                    theta_flat = theta_flat * norm_factor_inv * c_flat
                else:
                    theta_flat = theta_flat * norm_factor_inv * c_flat
            else:
                theta_flat = theta_flat * c_flat
            
            if isSavingEachIteration and it in save_indices:
                if device == 'gpu' and CUPY_AVAILABLE:
                    saved_theta.append(cp.asnumpy(theta_flat.reshape(Z, X)))
                else:
                    saved_theta.append(theta_flat.reshape(Z, X).copy())
                saved_indices_list.append(int(it))
        
        if device == 'gpu' and CUPY_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
            final_result = cp.asnumpy(theta_flat.reshape(Z, X))
        else:
            final_result = theta_flat.reshape(Z, X)
        
        if isSavingEachIteration:
            return saved_theta, saved_indices_list
        else:
            return final_result, None
            
    except Exception as e:
        warnings.warn(f"Error in sparse MLEM: {type(e).__name__}: {e}")
        return None, None


# Legacy functions for backward compatibility

def MLEM_sparseCSR_cupy(*args, **kwargs):
    """Legacy function - use MLEM_sparse instead."""
    warnings.warn("MLEM_sparseCSR_cupy is deprecated. Use MLEM_sparse instead.")
    return MLEM_sparse(*args, **kwargs)


def MLEM_sparseSELL_cupy(*args, **kwargs):
    """Legacy function - use MLEM_sparse instead."""
    warnings.warn("MLEM_sparseSELL_cupy is deprecated. Use MLEM_sparse instead.")
    return MLEM_sparse(*args, **kwargs)


def _MLEM_single_GPU(*args, **kwargs):
    """Legacy function - use MLEM_dense instead."""
    warnings.warn("_MLEM_single_GPU is deprecated. Use MLEM_dense instead.")
    return MLEM_dense(*args, **kwargs)


def _MLEM_CPU_opti(*args, **kwargs):
    """Legacy function - use MLEM_dense instead."""
    warnings.warn("_MLEM_CPU_opti is deprecated. Use MLEM_dense instead.")
    return MLEM_dense(*args, device='cpu', **kwargs)
