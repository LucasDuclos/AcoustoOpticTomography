"""
ReconTools.py

Unified reconstruction tools for AOT-BioMaps.
All operations (projection, backprojection, vector ops, potentials) implemented as standalone functions.
Works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

For GPU: Uses CUDA kernels from AOT_biomaps_kernels.cu when available.
For CPU: Uses NumPy operations.
"""

import os
import numpy as np
from scipy.signal.windows import hann
import warnings

# Optional cupy imports for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import SMatrix classes
try:
    from AOT_biomaps.AOT_Recon.AOT_SMatrix import SMatrix_CSR, SMatrix_SELL, SMatrix_DENSE
except ModuleNotFoundError:
    SMatrix_CSR = None
    SMatrix_SELL = None
    SMatrix_DENSE = None


def _get_array_module(device):
    """Get the appropriate array module based on device."""
    if device == 'gpu' and CUPY_AVAILABLE:
        return cp
    else:
        return np


def _get_array(x, device):
    """Convert x to appropriate array type based on device."""
    if device == 'gpu' and CUPY_AVAILABLE:
        if isinstance(x, np.ndarray):
            return cp.asarray(x)
        elif isinstance(x, cp.ndarray):
            return x
        else:
            return cp.array(x)
    else:
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)


# =============================================================================
# BASIC ARRAY OPERATIONS
# =============================================================================

def zeros(SMatrix, shape):
    """Create a zero array with the appropriate device and type."""
    device = SMatrix.device
    xp = _get_array_module(device)
    return xp.zeros(shape, dtype=xp.float32)


def ones(SMatrix, shape):
    """Create an array of ones with the appropriate device and type."""
    device = SMatrix.device
    xp = _get_array_module(device)
    return xp.ones(shape, dtype=xp.float32)


def fill_array(SMatrix, value, shape):
    """Create an array filled with a specific value."""
    device = SMatrix.device
    xp = _get_array_module(device)
    return xp.full(shape, value, dtype=xp.float32)


def clamp_positive(SMatrix, x):
    """Clamp array values to be non-negative. Uses CUDA kernel when available."""
    device = SMatrix.device
    
    if device == 'gpu' and CUPY_AVAILABLE and hasattr(SMatrix._data, 'sparse_mod'):
        sparse_mod = SMatrix._data.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                N = x_gpu.size
                threads = 256
                blocks = (N + threads - 1) // threads
                clamp_kernel = sparse_mod.get_function('clamp_positive_kernel')
                clamp_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                           args=[x_gpu.data.ptr, np.int32(N)])
                cp.cuda.Stream.null.synchronize()
                return x_gpu
            except Exception:
                pass
    
    xp = _get_array_module(device)
    return xp.maximum(x, 0.0)


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

def axpby(SMatrix, x, y, a, b):
    """Compute a*x + b*y. Uses CUDA kernel when available."""
    device = SMatrix.device
    
    if device == 'gpu' and CUPY_AVAILABLE and hasattr(SMatrix._data, 'sparse_mod'):
        sparse_mod = SMatrix._data.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                y_gpu = cp.asarray(y) if not isinstance(y, cp.ndarray) else y
                N = x_gpu.size
                z_gpu = cp.empty(N, dtype=cp.float32)
                threads = 256
                blocks = (N + threads - 1) // threads
                axpby_kernel = sparse_mod.get_function('vector_axpby_kernel')
                axpby_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                           args=[z_gpu.data.ptr, x_gpu.data.ptr, y_gpu.data.ptr,
                                 np.float32(a), np.float32(b), np.int32(N)])
                cp.cuda.Stream.null.synchronize()
                return z_gpu
            except Exception:
                pass
    
    return a * x + b * y


def minus_axpy(SMatrix, x, y, a):
    """Compute x - a*y. Uses CUDA kernel when available."""
    device = SMatrix.device
    
    if device == 'gpu' and CUPY_AVAILABLE and hasattr(SMatrix._data, 'sparse_mod'):
        sparse_mod = SMatrix._data.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                y_gpu = cp.asarray(y) if not isinstance(y, cp.ndarray) else y
                N = x_gpu.size
                threads = 256
                blocks = (N + threads - 1) // threads
                minus_axpy_kernel = sparse_mod.get_function('vector_minus_axpy_kernel')
                # Note: vector_minus_axpy_kernel does r = r - alpha * z
                # We need to copy x first since it modifies in place
                r_gpu = x_gpu.copy()
                minus_axpy_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                                args=[r_gpu.data.ptr, y_gpu.data.ptr, np.float32(a), np.int32(N)])
                cp.cuda.Stream.null.synchronize()
                return r_gpu
            except Exception:
                pass
    
    return x - a * y


def dot_product(SMatrix, x, y):
    """Compute dot product of two vectors."""
    device = SMatrix.device
    xp = _get_array_module(device)
    return xp.sum(x * y)


def vector_divide(SMatrix, x, y, epsilon=1e-12):
    """Element-wise division: x / y with protection against division by zero."""
    device = SMatrix.device
    
    if device == 'gpu' and CUPY_AVAILABLE and hasattr(SMatrix._data, 'sparse_mod'):
        sparse_mod = SMatrix._data.sparse_mod
        if sparse_mod is not None:
            try:
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                y_gpu = cp.asarray(y) if not isinstance(y, cp.ndarray) else y
                N = x_gpu.size
                result_gpu = cp.empty(N, dtype=cp.float32)
                threads = 256
                blocks = (N + threads - 1) // threads
                invert_kernel = sparse_mod.get_function('invert_vector_kernel')
                invert_kernel(grid=(blocks, 1), block=(threads, 1, 1),
                            args=[result_gpu.data.ptr, y_gpu.data.ptr, np.float32(epsilon), np.int32(N)])
                # Now multiply: result = x * (1/y)
                result_gpu = x_gpu * result_gpu
                cp.cuda.Stream.null.synchronize()
                return result_gpu
            except Exception:
                pass
    
    xp = _get_array_module(device)
    return x / (y + epsilon)


def apply_normalization(SMatrix, x, norm_factor):
    """Apply normalization to vector."""
    return x * norm_factor


# =============================================================================
# MATRIX-VECTOR OPERATIONS (delegated to SMatrix implementation)
# These use the CUDA kernels from SMatrix classes
# =============================================================================

def projection(SMatrix, theta):
    """Forward projection: q = A * theta. Uses SMatrix._data.projection() which calls CUDA kernels."""
    return SMatrix._data.projection(theta)


def backprojection(SMatrix, e):
    """Backprojection: c = A^T * e. Uses SMatrix._data.backprojection() which calls CUDA kernels."""
    return SMatrix._data.backprojection(e)


# =============================================================================
# ADJACENCY AND GRAPH OPERATIONS
# =============================================================================

def build_adjacency_indices(SMatrix):
    """Build adjacency indices for regularization (4-connectivity)."""
    device = SMatrix.device
    Z = SMatrix.Z
    X = SMatrix.X
    
    xp = _get_array_module(device)
    
    # Create adjacency list: each pixel has up to 4 neighbors
    # Format: (num_edges, 2) array where each row is (i, j) for edge between i and j
    edges = []
    
    for z in range(Z):
        for x in range(X):
            idx = z * X + x
            # Right neighbor
            if x < X - 1:
                edges.append((idx, idx + 1))
            # Down neighbor
            if z < Z - 1:
                edges.append((idx, idx + X))
    
    if device == 'gpu' and CUPY_AVAILABLE:
        return cp.array(edges, dtype=cp.int32)
    else:
        return np.array(edges, dtype=np.int32)


# =============================================================================
# POTENTIAL FUNCTIONS
# =============================================================================

def quadratic_potential(SMatrix, U, alpha):
    """
    Quadratic potential: 0.5 * alpha * ||x||^2
    Returns (grad_U, hess_U, U_value)
    """
    device = SMatrix.device
    xp = _get_array_module(device)
    
    grad_U = alpha * U
    hess_U = alpha * xp.ones_like(U)
    U_value = 0.5 * alpha * xp.sum(U**2)
    
    return grad_U, hess_U, U_value


def huber_potential(SMatrix, U, alpha, delta=0.01):
    """
    Huber potential for robust regularization.
    Returns (grad_U, hess_U, U_value)
    """
    device = SMatrix.device
    xp = _get_array_module(device)
    
    abs_U = xp.abs(U)
    
    # Huber function and derivatives
    mask_small = abs_U <= delta
    mask_large = ~mask_small
    
    grad_U = xp.zeros_like(U)
    grad_U[mask_small] = alpha * U[mask_small]
    grad_U[mask_large] = alpha * delta * xp.sign(U[mask_large])
    
    hess_U = xp.zeros_like(U)
    hess_U[mask_small] = alpha
    hess_U[mask_large] = 0.0
    
    U_value = xp.zeros((), dtype=xp.float32)
    U_value += 0.5 * alpha * xp.sum(U[mask_small]**2)
    U_value += alpha * delta * xp.sum(abs_U[mask_large] - 0.5 * delta)
    
    return grad_U, hess_U, U_value


def relative_difference_potential(SMatrix, U, alpha, beta=1.0):
    """
    Relative difference potential for edge-preserving regularization.
    Uses adjacency from build_adjacency_indices.
    Returns (grad_U, hess_U, U_value)
    """
    device = SMatrix.device
    xp = _get_array_module(device)
    
    adj_indices = build_adjacency_indices(SMatrix)
    
    grad_U = xp.zeros_like(U)
    U_value = xp.zeros((), dtype=xp.float32)
    
    for edge in adj_indices:
        i, j = edge
        diff = U[i] - U[j]
        
        # Relative difference potential
        denom = xp.sqrt(U[i]**2 + U[j]**2 + beta**2)
        
        # Gradient contributions
        grad_U[i] += alpha * diff / denom
        grad_U[j] -= alpha * diff / denom
        
        # Energy
        U_value += alpha * (xp.sqrt(U[i]**2 + U[j]**2 + beta**2) - beta)
    
    # Hessian is more complex, approximate as constant for now
    hess_U = alpha * xp.ones_like(U)
    
    return grad_U, hess_U, U_value


def tv_potential(SMatrix, U, alpha):
    """
    Total Variation potential (non-differentiable).
    Returns (subgradient, 0, U_value) - hess_U=0 because TV is non-differentiable.
    """
    device = SMatrix.device
    xp = _get_array_module(device)
    
    Z = SMatrix.Z
    X = SMatrix.X
    
    grad_U = xp.zeros_like(U)
    U_value = xp.zeros((), dtype=xp.float32)
    
    # Compute TV using finite differences
    for z in range(Z):
        for x in range(X):
            idx = z * X + x
            
            # Right neighbor difference
            if x < X - 1:
                diff_x = U[idx + 1] - U[idx]
                U_value += alpha * xp.abs(diff_x)
                # Subgradient for x component
                if diff_x > 0:
                    grad_U[idx] -= alpha
                    grad_U[idx + 1] += alpha
                elif diff_x < 0:
                    grad_U[idx] += alpha
                    grad_U[idx + 1] -= alpha
            
            # Down neighbor difference
            if z < Z - 1:
                diff_z = U[idx + X] - U[idx]
                U_value += alpha * xp.abs(diff_z)
                # Subgradient for z component
                if diff_z > 0:
                    grad_U[idx] -= alpha
                    grad_U[idx + X] += alpha
                elif diff_z < 0:
                    grad_U[idx] += alpha
                    grad_U[idx + X] -= alpha
    
    # TV is non-differentiable, hessian is 0
    hess_U = xp.zeros_like(U)
    
    return grad_U, hess_U, U_value


# =============================================================================
# FILE I/O
# =============================================================================

def load_recon(hdr_path):
    """
    Load an Interfile (.hdr) and its binary file (.img) to reconstruct an image as Vinci does.

    Parameters:
    -----------
    - hdr_path : full path to the .hdr file

    Returns:
    --------
    - image : NumPy array containing the image
    - header : dictionary containing metadata from the .hdr file
    """
    header = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            if ':=' in line:
                key, value = line.split(':=', 1)
                key = key.strip().lower().replace('!', '')
                value = value.strip()
                header[key] = value

    data_file = header.get('name of data file')
    if data_file is None:
        raise ValueError(f"Cannot find data file associated with header file {hdr_path}")

    img_path = os.path.join(os.path.dirname(hdr_path), data_file)

    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 4) if f'matrix size [{i}]' in header]
    if shape and shape[-1] == 1:
        shape = shape[:-1]

    if not shape:
        raise ValueError("Cannot determine image shape from metadata.")

    data_type = header.get('number format', 'short float').lower()
    dtype_map = {
        'short float': np.float32,
        'float': np.float32,
        'int16': np.int16,
        'int32': np.int32,
        'uint16': np.uint16,
        'uint8': np.uint8
    }
    dtype = dtype_map.get(data_type)
    if dtype is None:
        raise ValueError(f"Unsupported data type: {data_type}")

    byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
    endianess = '<' if 'little' in byte_order else '>'

    img_size = os.path.getsize(img_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize

    if img_size != expected_size:
        raise ValueError(f"Image file size ({img_size} bytes) does not match expected size ({expected_size} bytes).")

    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)

    image = data.reshape(shape[::-1])

    rescale_slope = float(header.get('data rescale slope', 1))
    rescale_offset = float(header.get('data rescale offset', 0))
    image = image * rescale_slope + rescale_offset

    return image


def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.
    Equivalent to sklearn.metrics.mean_squared_error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def calculate_memory_requirement(SMatrix, y):
    """
    Calculate the memory required (in GB) for SMatrix and y.
    
    Args:
        SMatrix: Matrix object (SMatrix, SMatrix_CSR, SMatrix_SELL, SMatrix_DENSE, np.ndarray, cp.ndarray)
        y: Vector (float32)
    """
    total_bytes = 0
    
    # Check if it's the unified SMatrix wrapper
    SMatrix_class_name = type(SMatrix).__name__
    
    # Custom Sparse Matrix (SELL/CSR/DENSE) - either wrapper or direct implementation
    if SMatrix_class_name == 'SMatrix' or isinstance(SMatrix, (SMatrix_CSR, SMatrix_SELL, SMatrix_DENSE)):
        try:
            matrix_size_gb = SMatrix.get_matrix_size() if SMatrix_class_name == 'SMatrix' else SMatrix.getMatrixSize()
            if isinstance(matrix_size_gb, dict) and 'error' in matrix_size_gb:
                raise ValueError(f"SMatrix allocation error: {matrix_size_gb['error']}")
            
            size_SMatrix = matrix_size_gb * (1024 ** 3)
            total_bytes += size_SMatrix
            print(f"SMatrix size: {matrix_size_gb:.3f} GB")
        except AttributeError:
            raise AttributeError("SMatrix must implement the get_matrix_size() or getMatrixSize() method.")
    elif isinstance(SMatrix, np.ndarray):
        size_SMatrix = SMatrix.nbytes
        total_bytes += size_SMatrix
        print(f"SMatrix (NumPy Dense) size: {size_SMatrix / (1024 ** 3):.3f} GB")
    elif CUPY_AVAILABLE and isinstance(SMatrix, cp.ndarray):
        size_SMatrix = SMatrix.nbytes
        total_bytes += size_SMatrix
        print(f"SMatrix (CuPy) size: {size_SMatrix / (1024 ** 3):.3f} GB")
    else:
        raise ValueError("SMatrix must be a SMatrix object, np.ndarray, or cp.ndarray.")
    
    # Vector y
    if hasattr(y, 'nbytes'):
        size_y = y.nbytes
        total_bytes += size_y
        print(f"Vector y size: {size_y / (1024 ** 3):.3f} GB")
    else:
        raise ValueError("Vector y must be an array type exposing the .nbytes attribute.")
    
    return total_bytes / (1024 ** 3)


def check_gpu_memory(device_index, required_memory, show_logs=True):
    """Check if enough memory is available on the specified GPU."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available. Cannot check GPU memory.")
    
    free_memory = cp.cuda.runtime.memoryGetInfo()[0]
    free_memory_gb = free_memory / 1024**3
    
    if show_logs:
        print(f"Free memory on GPU {device_index}: {free_memory_gb:.2f} GB, Required memory: {required_memory:.2f} GB")
    
    return free_memory_gb >= required_memory


# =============================================================================
# OLD CUDA/GPU FUNCTIONS (kept for backward compatibility with AnalyticRecon)
# These are CuPy-only functions used by analytic reconstruction methods
# =============================================================================

def fourierz_gpu(z, X):
    """Forward Fourier transform along z-axis (GPU)."""
    if CUPY_AVAILABLE:
        dz = float(z[1] - z[0])
        Nz = X.shape[0]
        return cp.fft.fftshift(
            cp.fft.fft(
                cp.fft.ifftshift(X, axes=0),
                axis=0
            ),
            axes=0
        ) * (Nz * dz)


def ifourierz_gpu(z, X):
    """Inverse Fourier transform along z-axis (GPU)."""
    if CUPY_AVAILABLE:
        dz = float(z[1] - z[0])
        Nz = X.shape[0]
        return cp.fft.ifftshift(
            cp.fft.ifft(
                cp.fft.fftshift(X, axes=0),
                axis=0
            ),
            axes=0
        ) * (1 / dz)


def ifourierx_gpu(F_fx_z, dx):
    """Inverse Fourier along X (axis=1), Matlab-compatible (GPU)."""
    if CUPY_AVAILABLE:
        return (
            cp.fft.ifftshift(
                cp.fft.ifft(
                    cp.fft.fftshift(F_fx_z, axes=1),
                    axis=1
                ),
                axes=1
            ) * (1.0 / dx)
        )


def EvalDelayLawOS_center(X_m, theta, DelayLAWS, ActiveLIST, c):
    """
    Return the rotation center C for each angle.
    """
    Nangle = DelayLAWS.shape[1]
    C = np.zeros((Nangle, 2))
    ct = DelayLAWS * c
    
    for i in range(Nangle):
        active_idx = np.where(ActiveLIST[:, i] == 1)[0]
        if len(active_idx) == 0:
            continue
        angle_i = np.round(theta[i], 5)
        u = np.array([np.sin(angle_i), np.cos(angle_i)])
        X0 = X_m - u[0] * ct[:, i]
        Z0 = 0 - u[1] * ct[:, i]
        if Z0[-1] - Z0[0] != 0:
            C[i, 0] = (Z0[-1]*X0[0] - Z0[0]*X0[-1]) / (Z0[-1] - Z0[0])
            C[i, 1] = 0
    return C


def rotate_theta_gpu(X, Z, Iin, theta, C):
    """GPU equivalent of RotateTheta.m"""
    if CUPY_AVAILABLE:
        X_rel = X - C[0]
        Z_rel = Z - C[1]
        c = cp.cos(theta)
        s = cp.sin(theta)
        Xout = c * X_rel + s * Z_rel
        Zout = -s * X_rel + c * Z_rel
        Xout += C[0]
        Zout += C[1]
        dx = X[0, 1] - X[0, 0]
        dz = Z[1, 0] - Z[0, 0]
        x0 = X[0, 0]
        z0 = Z[0, 0]
        ix = (Xout - x0) / dx
        iz = (Zout - z0) / dz
        coords = cp.stack([iz.ravel(), ix.ravel()])
        Iout = map_coordinates(
            Iin, coords, order=1, mode='constant', cval=0.0
        )
        return Iout.reshape(Iin.shape)


def filter_radon_gpu(fz, Fc):
    """Filter for filtered backprojection (GPU)."""
    if CUPY_AVAILABLE:
        FILTER = cp.abs(fz)
        FILTER = cp.where(cp.abs(fz) > Fc, 0, FILTER)
        FILTER *= cp.exp(-2 * cp.abs(fz / Fc)**10)
        return FILTER


def filter_radon(f, N, filter_type, Fc):
    """
    Implement filters for filtered backprojection (iRadon).
    Inspired by MATLAB FilterRadon function from Mamouna Bocoum.

    Parameters:
    -----------
    f : np.ndarray
        Frequency vector (e.g., f_t or f_z).
    N : int
        Filter size (length of f).
    filter_type : str
        Filter type: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'.
    Fc : float
        Cutoff frequency.

    Returns:
    -----------
    FILTER : np.ndarray
        Filter applied to frequencies.
    """
    FILTER = np.abs(f)

    if filter_type == 'ram-lak':
        pass
    elif filter_type == 'shepp-logan':
        with np.errstate(divide='ignore', invalid='ignore'):
            FILTER = FILTER * (np.sinc(2 * f / (2 * Fc)))
        FILTER[np.isnan(FILTER)] = 1.0
    elif filter_type == 'cosine':
        FILTER = FILTER * np.cos(2 * np.pi * f / (4 * Fc))
    elif filter_type == 'hamming':
        FILTER = FILTER * (0.54 + 0.46 * np.cos(2 * np.pi * f / Fc))
    elif filter_type == 'hann':
        FILTER = FILTER * (1 + np.cos(2 * np.pi * f / (4 * Fc))) / 2
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    FILTER[np.abs(f) > Fc] = 0
    FILTER = FILTER * np.exp(-2 * (np.abs(f) / Fc)**10)

    return FILTER
