"""
AOT_Kernels.py

Centralized kernel management for AOT_biomaps library.
Provides both GPU (CuPy/CUDA) and CPU (NumPy) implementations for all operations.

Structure:
- Each operation has a CPU and GPU implementation
- Automatic fallback to CPU if GPU is not available
- Consistent API across all operations
"""

import warnings
import numpy as np

# Check for CuPy availability
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False





def _is_cupy_array(arr):
    """Check if an array is a CuPy array. Returns False if CuPy is not available."""
    if not CUPY_AVAILABLE:
        return False
    return isinstance(arr, cp.ndarray)


def _as_cupy_array(arr):
    """Convert array to CuPy array if CuPy is available, otherwise return as-is."""
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr


def _as_numpy_array(arr):
    """Convert CuPy array to NumPy array if it's a CuPy array and CuPy is available."""
    if CUPY_AVAILABLE and _is_cupy_array(arr):
        return cp.asnumpy(arr)
    return arr


# ============================================================================
# UTILITY OPERATIONS
# ============================================================================

def fill_array_value(arr, value, device=None):
    """
    Fill an array with a constant value.
    
    Args:
        arr: Input array (NumPy or CuPy)
        value: Value to fill with
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Filled array (same type as input)
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        if not _is_cupy_array(arr):
            arr = _as_cupy_array(arr)
        arr.fill(value)
        return arr
    else:
        arr = _as_numpy_array(arr)
        arr.fill(value)
        return arr


def fill_array_zero(arr, device=None):
    """
    Fill an array with zeros.
    
    Args:
        arr: Input array (NumPy or CuPy)
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Zero-filled array (same type as input)
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        if not _is_cupy_array(arr):
            arr = _as_cupy_array(arr)
        arr.fill(0.0)
        return arr
    else:
        arr = _as_numpy_array(arr)
        arr.fill(0.0)
        return arr


def clamp_positive(arr, device=None):
    """
    Clamp all values to be non-negative (max with 0).
    
    Args:
        arr: Input array
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Clamped array
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        arr = _as_cupy_array(arr)
        return cp.maximum(arr, 0.0)
    else:
        arr = _as_numpy_array(arr)
        return np.maximum(arr, 0.0)


def vector_axpby(z, x, y, alpha, beta, device=None):
    """
    Compute z = alpha * x + beta * y (element-wise).
    
    Args:
        z: Output array (can be same as x or y)
        x: First input array
        y: Second input array
        alpha: Scaling factor for x
        beta: Scaling factor for y
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Result array
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        x_gpu = _as_cupy_array(x)
        y_gpu = _as_cupy_array(y)
        return alpha * x_gpu + beta * y_gpu
    else:
        x_cpu = np.asarray(x) if not isinstance(x, np.ndarray) else x
        y_cpu = np.asarray(y) if not isinstance(y, np.ndarray) else y
        return alpha * x_cpu + beta * y_cpu


def vector_minus_axpy(r, z, alpha, device=None):
    """
    Compute r = r - alpha * z (in-place).
    
    Args:
        r: Input/output array
        z: Input array
        alpha: Scaling factor
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Modified r array
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        r_gpu = _as_cupy_array(r)
        z_gpu = _as_cupy_array(z)
        r_gpu -= alpha * z_gpu
        return r_gpu
    else:
        r_cpu = np.asarray(r) if not isinstance(r, np.ndarray) else r
        z_cpu = np.asarray(z) if not isinstance(z, np.ndarray) else z
        r_cpu = _as_numpy_array(r_cpu)
        r_cpu -= alpha * z_cpu
        return r_cpu


def invert_vector(vec, clip_min=1e-12, device=None):
    """
    Compute output = 1 / input with clipping to avoid division by zero.
    
    Args:
        vec: Input vector
        clip_min: Minimum value to clip to
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Inverted vector
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        vec_gpu = _as_cupy_array(vec)
        return 1.0 / cp.clip(vec_gpu, clip_min, None)
    else:
        vec_cpu = np.asarray(vec) if not isinstance(vec, np.ndarray) else vec
        vec_cpu = _as_numpy_array(vec_cpu)
        return 1.0 / np.clip(vec_cpu, clip_min, None)


# ============================================================================
# SPARSE MATRIX OPERATIONS
# ============================================================================

def sparse_matrix_vector_product_csr(data, indices, indptr, x, num_rows, device=None):
    """
    Compute y = A * x for CSR sparse matrix.
    
    Args:
        data: Non-zero values of the matrix
        indices: Column indices for each non-zero
        indptr: Pointer to the start of each row
        x: Input vector
        num_rows: Number of rows in matrix
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Result vector y
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        data_gpu = _as_cupy_array(data)
        indices_gpu = _as_cupy_array(indices)
        indptr_gpu = _as_cupy_array(indptr)
        x_gpu = _as_cupy_array(x)
        
        y = cp.zeros(num_rows, dtype=data_gpu.dtype)
        
        # Use CuPy's sparse matrix multiplication if available
        try:
            from cupyx.scipy.sparse import csr_matrix
            A = csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(num_rows, x_gpu.shape[0]))
            y = A.dot(x_gpu)
        except:
            # Fallback to manual implementation
            for i in range(num_rows):
                start = int(indptr_gpu[i])
                end = int(indptr_gpu[i + 1])
                y[i] = cp.sum(data_gpu[start:end] * x_gpu[indices_gpu[start:end]])
        
        return y
    else:
        data_cpu = np.asarray(data) if not isinstance(data, np.ndarray) else data
        indices_cpu = np.asarray(indices) if not isinstance(indices, np.ndarray) else indices
        indptr_cpu = np.asarray(indptr) if not isinstance(indptr, np.ndarray) else indptr
        x_cpu = np.asarray(x) if not isinstance(x, np.ndarray) else x
        
        y = np.zeros(num_rows, dtype=data_cpu.dtype)
        
        for i in range(num_rows):
            start = int(indptr_cpu[i])
            end = int(indptr_cpu[i + 1])
            y[i] = np.sum(data_cpu[start:end] * x_cpu[indices_cpu[start:end]])
        
        return y


# ============================================================================
# MLEM OPERATIONS
# ============================================================================

def ratio_kernel(y, q, threshold=1e-12, device=None):
    """
    Compute element-wise ratio e = y / max(q, threshold) for MLEM.
    
    Args:
        y: Numerator array
        q: Denominator array
        threshold: Minimum threshold for denominator
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Ratio array e
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        y_gpu = _as_cupy_array(y)
        q_gpu = _as_cupy_array(q)
        denom = cp.maximum(q_gpu, threshold)
        e = y_gpu / denom
        e = cp.where(cp.isfinite(e), e, 0.0)
        return e
    else:
        y_cpu = np.asarray(y) if not isinstance(y, np.ndarray) else y
        q_cpu = np.asarray(q) if not isinstance(q, np.ndarray) else q
        y_cpu = _as_numpy_array(y_cpu)
        q_cpu = _as_numpy_array(q_cpu)
        denom = np.maximum(q_cpu, threshold)
        e = y_cpu / denom
        e[~np.isfinite(e)] = 0.0
        return e


def update_theta(theta, c, norm_factor_inv, device=None):
    """
    Update theta values in MLEM: theta *= norm_inv * c.
    
    Args:
        theta: Current theta values
        c: Correction values
        norm_factor_inv: Inverse normalization factors
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Updated theta values
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        theta_gpu = _as_cupy_array(theta)
        c_gpu = _as_cupy_array(c)
        norm_gpu = _as_cupy_array(norm_factor_inv)
        theta_gpu *= norm_gpu * c_gpu
        theta_gpu = cp.where(cp.isfinite(theta_gpu), theta_gpu, 0.0)
        theta_gpu = cp.maximum(theta_gpu, 0.0)
        return theta_gpu
    else:
        theta_cpu = np.asarray(theta) if not isinstance(theta, np.ndarray) else theta
        c_cpu = np.asarray(c) if not isinstance(c, np.ndarray) else c
        norm_cpu = np.asarray(norm_factor_inv) if not isinstance(norm_factor_inv, np.ndarray) else norm_factor_inv
        theta_cpu = _as_numpy_array(theta_cpu)
        theta_cpu *= norm_cpu * c_cpu
        theta_cpu[~np.isfinite(theta_cpu)] = 0.0
        theta_cpu = np.maximum(theta_cpu, 0.0)
        return theta_cpu


# ============================================================================
# TOTAL VARIATION (TV) OPERATIONS
# ============================================================================

def gradient_2d(x, device=None):
    """
    Compute 2D gradient (forward differences) for TV regularization.
    
    Args:
        x: Input 2D array (Z, X)
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Gradient array with shape (2, Z, X) where:
        - p[0] = gradient in x direction
        - p[1] = gradient in z direction
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        x_gpu = _as_cupy_array(x)
        Z, X = x_gpu.shape
        p = cp.zeros((2, Z, X), dtype=x_gpu.dtype)
        
        # Gradient in x direction
        p[0, :, :-1] = x_gpu[:, 1:] - x_gpu[:, :-1]
        p[0, :, -1] = 0.0
        
        # Gradient in z direction
        p[1, :-1, :] = x_gpu[1:, :] - x_gpu[:-1, :]
        p[1, -1, :] = 0.0
        
        return p
    else:
        x_cpu = np.asarray(x) if not isinstance(x, np.ndarray) else x
        x_cpu = _as_numpy_array(x_cpu)
        Z, X = x_cpu.shape
        p = np.zeros((2, Z, X), dtype=x_cpu.dtype)
        
        # Gradient in x direction
        p[0, :, :-1] = x_cpu[:, 1:] - x_cpu[:, :-1]
        p[0, :, -1] = 0.0
        
        # Gradient in z direction
        p[1, :-1, :] = x_cpu[1:, :] - x_cpu[:-1, :]
        p[1, -1, :] = 0.0
        
        return p


def divergence_2d(p, device=None):
    """
    Compute 2D divergence (adjoint of gradient) for TV regularization.
    
    Args:
        p: Input gradient array with shape (2, Z, X) or (2*Z*X,)
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Divergence array with shape (Z, X)
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        p_gpu = cp.asarray(p) if not isinstance(p, cp.ndarray) else p
        
        # Handle both (2, Z, X) and flattened formats
        if p_gpu.ndim == 3:
            # p has shape (2, Z, X)
            Z, X = p_gpu.shape[1], p_gpu.shape[2]
            div = cp.zeros((Z, X), dtype=p_gpu.dtype)
            
            # Divergence in x direction
            div[:, :-1] += p_gpu[0, :, :-1]
            div[:, 1:] -= p_gpu[0, :, 1:]
            
            # Divergence in z direction
            div[:-1, :] += p_gpu[1, :-1, :]
            div[1:, :] -= p_gpu[1, 1:, :]
            
            return div
        else:
            # p has shape (2*Z*X,)
            N = p_gpu.shape[0] // 2
            div = cp.zeros(N, dtype=p_gpu.dtype)
            
            # Divergence in x direction
            div[:-1] += p_gpu[:N-1]
            div[1:] -= p_gpu[1:N]
            
            # Divergence in z direction
            div[:-X] += p_gpu[N:N-X]
            div[X:] -= p_gpu[N+X:2*N]
            
            return div
    else:
        p_cpu = np.asarray(p) if not isinstance(p, np.ndarray) else p
        p_cpu = _as_numpy_array(p_cpu)
        
        # Handle both (2, Z, X) and flattened formats
        if p_cpu.ndim == 3:
            # p has shape (2, Z, X)
            Z, X = p_cpu.shape[1], p_cpu.shape[2]
            div = np.zeros((Z, X), dtype=p_cpu.dtype)
            
            # Divergence in x direction
            div[:, :-1] += p_cpu[0, :, :-1]
            div[:, 1:] -= p_cpu[0, :, 1:]
            
            # Divergence in z direction
            div[:-1, :] += p_cpu[1, :-1, :]
            div[1:, :] -= p_cpu[1, 1:, :]
            
            return div
        else:
            # p has shape (2*Z*X,)
            N = p_cpu.shape[0] // 2
            div = np.zeros(N, dtype=p_cpu.dtype)
            
            # Divergence in x direction
            div[:-1] += p_cpu[:N-1]
            div[1:] -= p_cpu[1:N]
            
            # Divergence in z direction (assuming X is known or N is square)
            # This is a simplified version
            X = int(np.sqrt(N))
            if X * X == N:
                div[:-X] += p_cpu[N:N-X]
                div[X:] -= p_cpu[N+X:2*N]
            
            return div


def proj_tv(p, alpha, device=None):
    """
    Project onto TV constraint set (L2 ball with radius alpha).
    
    Args:
        p: Input array with shape (2, Z, X) or (2*N,)
        alpha: Radius of the L2 ball
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Projected array
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        p_gpu = cp.asarray(p) if not isinstance(p, cp.ndarray) else p
        
        if p_gpu.ndim == 3:
            # p has shape (2, Z, X)
            norm = cp.sqrt(p_gpu[0]**2 + p_gpu[1]**2)
            scale = cp.where(norm > alpha, alpha / norm, 1.0)
            p_gpu[0] *= scale
            p_gpu[1] *= scale
        else:
            # p has shape (2*N,)
            N = p_gpu.shape[0] // 2
            norm = cp.sqrt(p_gpu[:N]**2 + p_gpu[N:]**2)
            scale = cp.where(norm > alpha, alpha / norm, 1.0)
            p_gpu[:N] *= scale
            p_gpu[N:] *= scale
        
        return p_gpu
    else:
        p_cpu = np.asarray(p) if not isinstance(p, np.ndarray) else p
        p_cpu = _as_numpy_array(p_cpu)
        
        if p_cpu.ndim == 3:
            # p has shape (2, Z, X)
            norm = np.sqrt(p_cpu[0]**2 + p_cpu[1]**2)
            scale = np.where(norm > alpha, alpha / norm, 1.0)
            p_cpu[0] *= scale
            p_cpu[1] *= scale
        else:
            # p has shape (2*N,)
            N = p_cpu.shape[0] // 2
            norm = np.sqrt(p_cpu[:N]**2 + p_cpu[N:]**2)
            scale = np.where(norm > alpha, alpha / norm, 1.0)
            p_cpu[:N] *= scale
            p_cpu[N:] *= scale
        
        return p_cpu


# ============================================================================
# FFT-BASED OPERATIONS
# ============================================================================

def calculate_envelope_squared(field, device=None):
    """
    Compute the squared envelope of a signal using Hilbert transform.
    
    Args:
        field: Input signal (T, X, Z) or (T, X, Y, Z)
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Squared envelope of the signal
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        field_gpu = cp.asarray(field) if not isinstance(field, cp.ndarray) else field
        
        # Use CuPy's FFT for Hilbert transform
        from scipy.signal import hilbert as scipy_hilbert
        
        # For now, use CPU implementation as CuPy's hilbert may not be available
        # This is a placeholder - actual GPU implementation would use FFT
        analytic_signal = scipy_hilbert(field_gpu.get(), axis=0)
        envelope_sq = np.abs(analytic_signal) ** 2
        return cp.asarray(envelope_sq)
    else:
        field_cpu = np.asarray(field) if not isinstance(field, np.ndarray) else field
        if isinstance(field_cpu, cp.ndarray):
            field_cpu = cp.asnumpy(field_cpu)
        
        from scipy.signal import hilbert
        analytic_signal = hilbert(field_cpu, axis=0)
        envelope_sq = np.abs(analytic_signal) ** 2
        return envelope_sq.astype(np.float32)


# ============================================================================
# PRECONDITIONING OPERATIONS
# ============================================================================

def update_dual_data_precond(q, Ax, y, sigma_vec, device=None):
    """
    Update dual variable with vector preconditioning for data term.
    Formula: q = (q + sigma * (Ax - y)) / (1 + sigma)
    
    Args:
        q: Dual variable
        Ax: Forward projection
        y: Measurements
        sigma_vec: Preconditioning vector
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Updated dual variable
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        q_gpu = cp.asarray(q) if not isinstance(q, cp.ndarray) else q
        Ax_gpu = cp.asarray(Ax) if not isinstance(Ax, cp.ndarray) else Ax
        y_gpu = cp.asarray(y) if not isinstance(y, cp.ndarray) else y
        sigma_gpu = cp.asarray(sigma_vec) if not isinstance(sigma_vec, cp.ndarray) else sigma_vec
        
        q_gpu = (q_gpu + sigma_gpu * (Ax_gpu - y_gpu)) / (1.0 + sigma_gpu)
        return q_gpu
    else:
        q_cpu = np.asarray(q) if not isinstance(q, np.ndarray) else q
        Ax_cpu = np.asarray(Ax) if not isinstance(Ax, np.ndarray) else Ax
        y_cpu = np.asarray(y) if not isinstance(y, np.ndarray) else y
        sigma_cpu = np.asarray(sigma_vec) if not isinstance(sigma_vec, np.ndarray) else sigma_vec
        
        if isinstance(q_cpu, cp.ndarray):
            q_cpu = cp.asnumpy(q_cpu)
        
        q_cpu = (q_cpu + sigma_cpu * (Ax_cpu - y_cpu)) / (1.0 + sigma_cpu)
        return q_cpu


def update_primal_precond(x, gradient_combined, tau_vec, device=None):
    """
    Update primal variable with vector preconditioning.
    Formula: x = x - tau * gradient
    
    Args:
        x: Primal variable
        gradient_combined: Combined gradient
        tau_vec: Preconditioning vector
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Updated primal variable
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
        grad_gpu = cp.asarray(gradient_combined) if not isinstance(gradient_combined, cp.ndarray) else gradient_combined
        tau_gpu = cp.asarray(tau_vec) if not isinstance(tau_vec, cp.ndarray) else tau_vec
        
        x_gpu -= tau_gpu * grad_gpu
        return x_gpu
    else:
        x_cpu = np.asarray(x) if not isinstance(x, np.ndarray) else x
        grad_cpu = np.asarray(gradient_combined) if not isinstance(gradient_combined, np.ndarray) else gradient_combined
        tau_cpu = np.asarray(tau_vec) if not isinstance(tau_vec, np.ndarray) else tau_vec
        
        if isinstance(x_cpu, cp.ndarray):
            x_cpu = cp.asnumpy(x_cpu)
        
        x_cpu -= tau_cpu * grad_cpu
        return x_cpu


# ============================================================================
# DOWNSCALING OPERATIONS
# ============================================================================

def downsample_3d(field, mode='avg', device=None):
    """
    Downsample a 3D array by a factor of 2 in each dimension.
    
    Args:
        field: Input 3D array (T, X, Z)
        mode: Downsampling mode ('avg' or 'max')
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Downsampled array
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        field_gpu = cp.asarray(field) if not isinstance(field, cp.ndarray) else field
        
        if mode == 'avg':
            # Use pooling for average downsampling
            x = field_gpu[cp.newaxis, cp.newaxis, ...]
            x_down = cp.nn.pooling.avg_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            return cp.asnumpy(x_down.squeeze(0).squeeze(0))
        else:
            # Use pooling for max downsampling
            x = field_gpu[cp.newaxis, cp.newaxis, ...]
            x_down = cp.nn.pooling.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            return cp.asnumpy(x_down.squeeze(0).squeeze(0))
    else:
        field_cpu = np.asarray(field) if not isinstance(field, np.ndarray) else field
        field_cpu = _as_numpy_array(field_cpu)
        
        if mode == 'avg':
            # Simple slicing for CPU (not true averaging, but fast)
            return field_cpu[::2, ::2, ::2]
        else:
            # Max pooling on CPU
            T, X, Z = field_cpu.shape
            T_new, X_new, Z_new = T // 2, X // 2, Z // 2
            downsampled = np.zeros((T_new, X_new, Z_new), dtype=field_cpu.dtype)
            
            for t in range(T_new):
                for x in range(X_new):
                    for z in range(Z_new):
                        block = field_cpu[2*t:2*t+2, 2*x:2*x+2, 2*z:2*z+2]
                        if mode == 'max':
                            downsampled[t, x, z] = np.max(block)
                        else:
                            downsampled[t, x, z] = np.mean(block)
            
            return downsampled


# ============================================================================
# MEMORY UTILITIES
# ============================================================================

def get_device_memory_info(device=None):
    """
    Get available and total GPU memory in bytes.
    
    Args:
        device: 'cpu' or 'gpu' (auto-detected if None)
        
    Returns:
        Tuple of (free_memory, total_memory) in bytes
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        try:
            # Use modern CuPy API (compatible with CuPy 12+)
            free = cp.cuda.runtime.getFreeMem()
            total = cp.cuda.runtime.getTotalMem()
            return free, total
        except Exception:
            # Fallback for older CuPy versions
            try:
                free = cp.cuda.runtime.memoryGetInfo()[0]
                total = cp.cuda.runtime.memoryGetInfo()[1]
                return free, total
            except Exception:
                return 0, 0
    else:
        return 0, 0


def check_cuda_available():
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    return CUPY_AVAILABLE



