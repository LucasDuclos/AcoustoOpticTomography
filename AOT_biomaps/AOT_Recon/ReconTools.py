"""
ReconTools.py

Unified reconstruction tools for AOT-BioMaps.
All operations (forward_projection, backward_projection, vector ops, potentials) implemented as standalone functions.
Works with any SMatrix type (CSR, SELL, DENSE) and any device (CPU, GPU).

For GPU: Uses CUDA kernels from AOT_biomaps_kernels.cu when available.
For CPU: Uses NumPy operations.
"""

import os
import numpy as np
import contextlib # <-- AJOUT POUR LE CONTEXTE HYBRIDE
from scipy.signal.windows import hann
import warnings
from tqdm import trange

from AOT_biomaps.AOT_Recon.ReconEnums import PotentialShapeType, PotentialType, StopCriterionType
from AOT_biomaps.AOT_Recon.AOT_Preconditioner.PreconditionerEnums import PreconditionerType

# Optional cupy imports for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# =============================================================================
# HARDWARE CONTEXT UTILITIES
# =============================================================================

def check_gpu_available(SMatrix) -> bool:
    """Check if GPU operations are available."""
    if not isinstance(getattr(SMatrix, 'device', ''), str) or "gpu" not in SMatrix.device:
        return False
    if not hasattr(SMatrix, 'sparse_mod') and SMatrix.matrix_type.name != 'DENSE':
        return False
    if not CUPY_AVAILABLE:
        print("[AOT-biomaps] Warning: CuPy is not available. Falling back to CPU.")
        SMatrix.device = 'cpu'
        return False
    return True

def get_array_module(SMatrix):
    """Get the appropriate array module based on device."""
    if check_gpu_available(SMatrix):
        return cp
    else:
        return np

def get_device_context(SMatrix):
    """
    Returns a context manager ensuring operations run on the correct GPU.
    Returns a nullcontext for CPU operations.
    """
    if check_gpu_available(SMatrix) and hasattr(SMatrix, 'gpu_index'):
        return cp.cuda.Device(SMatrix.gpu_index)
    return contextlib.nullcontext()

# =============================================================================
# BASIC ARRAY OPERATIONS
# =============================================================================

def clamp_positive(SMatrix, x):
    """Clamp array values to be non-negative. Uses CUDA kernel when available."""   
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)
        return xp.maximum(x, 0.0)

# =============================================================================
# MATRIX-VECTOR OPERATIONS (delegated to SMatrix implementation)
# =============================================================================

def forward_projection(SMatrix, theta):
    """Forward projection: q = A * theta. Uses SMatrix.forward_projection() which calls CUDA kernels."""
    return SMatrix.forward_projection(theta)

def backward_projection(SMatrix, e):
    """Backprojection: c = A^T * e. Uses SMatrix.backward_projection() which calls CUDA kernels."""
    return SMatrix.backward_projection(e)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mse(SMatrix, lambda_true, lambda_pred):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.
    Equivalent to sklearn.metrics.mean_squared_error.
    """
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix) if SMatrix is not None else np
        lambda_true = xp.asarray(lambda_true)
        lambda_pred = xp.asarray(lambda_pred)
        return xp.mean((lambda_true - lambda_pred) ** 2)

# =============================================================================
# ALGORITHM FUNCTIONS
# =============================================================================

def estimate_lipschitz_constant(SMatrix, preconditioner, num_iters=20):
    """
    Estimate the Lipschitz constant (spectral radius) of the gradient operator
    using the Power Iteration method.
    """
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)

        # Initialize a random vector v_0 on the unit sphere: ||v_0||_2 = 1
        v = xp.random.randn(SMatrix.Z * SMatrix.X).astype(xp.float32)
        v /= xp.linalg.norm(v)

        eps = 1e-12
        L_est = 0.0

        for i in range(num_iters):
            # Forward projection -> u = A * v_k
            Av = forward_projection(SMatrix, v)
            if float(xp.sum(xp.abs(Av))) < eps:
                raise RuntimeError("[AOT-biomaps] Forward projection returned all zeros.")
            
            # Backward projection -> w = Aᴴ * u = Aᴴ * A * v_k
            w = backward_projection(SMatrix, Av)
            
            # Map complex gradients to the real parameter space if necessary
            w = xp.real(w).astype(xp.float32) if SMatrix.isComplexSMatrix else w.astype(xp.float32)
            
            # Apply the preconditioner -> w = P⁻¹ * Aᴴ * A * v_k
            w = preconditioner.apply_inverse(w)
            
            norm = float(xp.linalg.norm(w))
            
            if norm < eps:
                raise RuntimeError("[AOT-biomaps] Power iteration collapsed to zero.")

            # Normalize for the next iteration -> v_{k+1} = w / ||w||_2
            v = w / norm
            L_est = norm 

        return max(L_est, 0.0)


def calculate_step_size_LS(SMatrix, preconditioner, eta, num_iters, show_logs):
    if eta is None:
        print("[AOT-biomaps] Warning: eta not set. Defaulting to 1.9.")
        eta = 1.9

    if not (1.0 < eta < 2.0):
        print(f"[AOT-biomaps] Warning: eta={eta} is outside (1,2).")

    L = estimate_lipschitz_constant(SMatrix, preconditioner=preconditioner, num_iters=num_iters)
    alpha = eta / L if L > 0 else 1.0

    if show_logs:
        print(f"[AOT-biomaps] Using step size alpha = {alpha:.3e}")

    return alpha

def calculate_step_size_PGC(SMatrix, preconditioner, eta, num_iters, show_logs):
    if eta is None:
        print("[AOT-biomaps] Warning: eta not set. Defaulting to 1.9.")
        eta = 1.9

    if not (1.0 < eta < 2.0):
        print(f"[AOT-biomaps] Warning: eta={eta} is outside (1,2).")
    
    L = estimate_lipschitz_constant(SMatrix, preconditioner=preconditioner, num_iters=num_iters)
    alpha = eta / L if L > 0 else 1.0

    if show_logs:
        print(f"[AOT-biomaps] Using step size alpha = {alpha:.3e}")

    return alpha
    
# =====================================================================
# GPU KERNELS (CuPy) - Zero Allocation & Gather Paradigm
# =====================================================================

if CUPY_AVAILABLE:
    mrf_potential_kernel = cp.ElementwiseKernel(
        'int32 Z, int32 X, raw float32 U, raw int32 dz, raw int32 dx, raw float32 w, int32 num_neighbors, float32 beta, float32 delta, bool is_huber, int32 hessian_mode',
        'float32 grad_out, float32 hess_out, float32 energy_out',
        '''
        int z = i / X;
        int x = i % X;
        float u_i = U[i];
        
        float g = 0.0f;
        float h = 0.0f;
        float e = 0.0f;
        
        for(int k = 0; k < num_neighbors; ++k) {
            int nz = z + dz[k];
            int nx = x + dx[k];
            
            if (nz >= 0 && nz < Z && nx >= 0 && nx < X) {
                float u_j = U[nz * X + nx];
                float diff = u_i - u_j;
                float abs_diff = abs(diff);
                float weight = w[k];
                
                if (is_huber) {
                    if (abs_diff <= delta) {
                        g += beta * weight * diff / delta;
                        h += beta * weight / delta;
                        e += 0.5f * beta * weight * diff * diff / delta;
                    } else {
                        g += beta * weight * (diff > 0.0f ? 1.0f : -1.0f);
                        if (hessian_mode == 1) {
                            h += beta * weight / abs_diff;
                        } else {
                            h += 0.0f;
                        }
                        e += beta * weight * (abs_diff - 0.5f * delta);
                    }
                } else { 
                    g += beta * weight * diff;
                    h += beta * weight;
                    e += 0.5f * beta * weight * diff * diff;
                }
            }
        }
        grad_out = g;
        hess_out = h;
        energy_out = e;
        ''',
        'fused_mrf_potential_kernel'
    )

    rdp_potential_kernel = cp.ElementwiseKernel(
        'int32 Z, int32 X, raw float32 U, raw int32 dz, raw int32 dx, raw float32 w, int32 num_neighbors, float32 beta, float32 delta, int32 hessian_mode',
        'float32 grad_out, float32 hess_out, float32 energy_out',
        '''
        int z = i / X;
        int x = i % X;
        float u_i = U[i];
        
        float gamma = 0.05f;
        
        float g = 0.0f;
        float h = 0.0f;
        float e = 0.0f;
        
        for(int k = 0; k < num_neighbors; ++k) {
            int nz = z + dz[k];
            int nx = x + dx[k];
            
            if (nz >= 0 && nz < Z && nx >= 0 && nx < X) {
                float u_j = U[nz * X + nx];
                float diff = u_i - u_j;
                float abs_diff = abs(diff);
                float sum_ij = u_i + u_j;
                float weight = w[k];
                
                float denom = sum_ij + gamma + delta * abs_diff;
                float denom_sq = denom * denom;
                
                float sign_diff = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                float d_denom = 1.0f + delta * sign_diff;
                
                g += beta * weight * (2.0f * diff * denom - (diff * diff) * d_denom) / denom_sq;
                
                h += beta * weight * 2.0f / denom;
                e += beta * weight * (diff * diff) / denom;
            }
        }
        grad_out = g;
        hess_out = h;
        energy_out = e;
        ''',
        'fused_rdp_potential_kernel'
    )

_OFFSET_CACHE = {}

def build_full_neighborhood_offsets(SMatrix, shape=PotentialShapeType.CROSS, radius=1):
    """Generates the FULL neighborhood (Gather) and caches it."""
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)
        if shape not in [PotentialShapeType.CROSS, PotentialShapeType.SQUARE, PotentialShapeType.CIRCLE]:
            raise ValueError(f"Unsupported neighborhood shape: {shape}")

        offsets_dz, offsets_dx, weights = [], [], []
        total_weight = 0.0
        
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz == 0 and dx == 0:
                    continue
                    
                dist_l2 = np.sqrt(dz**2 + dx**2)
                dist_l1 = abs(dz) + abs(dx)
                dist_linf = max(abs(dz), abs(dx))
                
                is_valid = False
                if shape == PotentialShapeType.CROSS: is_valid = (dist_l1 <= radius)
                elif shape == PotentialShapeType.SQUARE: is_valid = (dist_linf <= radius)
                elif shape == PotentialShapeType.CIRCLE: is_valid = (dist_l2 <= radius + 1e-5)

                if is_valid:
                    weight = 1.0 / dist_l2
                    offsets_dz.append(dz)
                    offsets_dx.append(dx)
                    weights.append(weight)
                    total_weight += weight
                    
        normalization_factor = 4.0 / (total_weight + 1e-10)
        weights = [w * normalization_factor for w in weights]
        
        return xp.array(offsets_dz, dtype=xp.int32), xp.array(offsets_dx, dtype=xp.int32), xp.array(weights, dtype=xp.float32)

def get_potential_function(
    potential_type: PotentialType, 
    SMatrix, 
    U, 
    beta: float, 
    shape: PotentialShapeType, 
    radius: int, 
    delta: float = 1.0, 
    compute_grad: bool = True, 
    compute_hess: bool = True, 
    compute_energy: bool = True,
    use_surrogate_hessian: bool = False
):
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)
        
        if potential_type == PotentialType.NONE:
            return (xp.zeros_like(U) if compute_grad else None, xp.zeros_like(U) if compute_hess else None, 0.0 if compute_energy else None)
        
        Z, X = SMatrix.Z, SMatrix.X
        is_gpu = (xp.__name__ == 'cupy')
        hessian_mode = 1 if use_surrogate_hessian else 0
        
        # FIXED: Inclus SMatrix.gpu_index pour éviter les croisements mémoires entre GPUs
        gpu_id = getattr(SMatrix, 'gpu_index', 0)
        cache_key = (shape, radius, xp.__name__, gpu_id)
        
        if cache_key not in _OFFSET_CACHE:
            _OFFSET_CACHE[cache_key] = build_full_neighborhood_offsets(SMatrix, shape, radius)
        dz_arr, dx_arr, w_arr = _OFFSET_CACHE[cache_key]

        # GPU Execution (Zero Allocation)
        if is_gpu:
            grad_out = xp.empty_like(U, dtype=xp.float32)
            hess_out = xp.empty_like(U, dtype=xp.float32)
            energy_out = xp.empty_like(U, dtype=xp.float32) if compute_energy else grad_out 
            
            if potential_type in [PotentialType.QUADRATIC, PotentialType.HUBER]:
                mrf_potential_kernel(
                    Z, X, U, dz_arr, dx_arr, w_arr, len(dz_arr), 
                    beta, delta, potential_type == PotentialType.HUBER, hessian_mode,
                    grad_out, hess_out, energy_out
                )
            elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
                rdp_potential_kernel(
                    Z, X, U, dz_arr, dx_arr, w_arr, len(dz_arr), 
                    beta, delta, hessian_mode,
                    grad_out, hess_out, energy_out
                )
            else:
                raise ValueError(f"[AOT-biomaps] Unsupported potential: {potential_type}")

            U_value = float(xp.sum(energy_out) / 2.0) if compute_energy else 0.0
            
            return (
                grad_out if compute_grad else None,
                hess_out if compute_hess else None,
                U_value
            )
        
        # CPU Fallback (Numpy)
        else:
            U_img = U.reshape(Z, X)
            grad_img = xp.zeros_like(U_img, dtype=xp.float32) if compute_grad else None
            hess_img = xp.zeros_like(U_img, dtype=xp.float32) if compute_hess else None
            U_value = 0.0
            
            for k in range(len(dz_arr)):
                dz, dx, w = int(dz_arr[k]), int(dx_arr[k]), float(w_arr[k])
                
                slice_c_z = slice(max(0, -dz), min(Z, Z - dz))
                slice_n_z = slice(max(0, dz), min(Z, Z + dz))
                slice_c_x = slice(max(0, -dx), min(X, X - dx))
                slice_n_x = slice(max(0, dx), min(X, X + dx))
                
                u_i = U_img[slice_c_z, slice_c_x]
                u_j = U_img[slice_n_z, slice_n_x]
                
                diff = u_i - u_j
                
                if potential_type == PotentialType.QUADRATIC:
                    g = beta * w * diff
                    h = beta * w
                    e = 0.5 * beta * w * diff**2
                    
                elif potential_type == PotentialType.HUBER:
                    abs_diff = xp.abs(diff)
                    mask_quad = abs_diff <= delta
                    mask_lin = ~mask_quad
                    
                    g = xp.zeros_like(diff)
                    g[mask_quad] = beta * w * diff[mask_quad] / delta
                    g[mask_lin] = beta * w * xp.sign(diff[mask_lin])
                    
                    h = xp.zeros_like(diff)
                    h[mask_quad] = beta * w / delta
                    if hessian_mode == 1:
                        h[mask_lin] = beta * w / (abs_diff[mask_lin] + 1e-8)
                        
                    e = xp.zeros_like(diff)
                    e[mask_quad] = 0.5 * beta * w * (diff[mask_quad]**2) / delta
                    e[mask_lin] = beta * w * (abs_diff[mask_lin] - 0.5 * delta)
                    
                elif potential_type == PotentialType.RELATIVE_DIFFERENCE:
                    gamma = 0.05
                    denom = u_i + u_j + gamma + delta * xp.abs(diff)
                    d_denom = 1.0 + delta * xp.sign(diff)
                    
                    g = beta * w * (2.0 * diff * denom - (diff**2) * d_denom) / (denom**2)
                    h = beta * w * 2.0 / denom
                    e = beta * w * (diff**2) / denom

                if compute_grad: grad_img[slice_c_z, slice_c_x] += g
                if compute_hess: hess_img[slice_c_z, slice_c_x] += h
                if compute_energy: U_value += float(xp.sum(e))

            if compute_energy: U_value /= 2.0
            return (grad_img.flatten() if compute_grad else None, 
                    hess_img.flatten() if compute_hess else None, 
                    U_value)   

# =============================================================================
# STOPPING CRITERIA
# =============================================================================

def check_stopping_criterion(SMatrix, current_lambda, prev_lambda, criterion_type, threshold, window_size, history=None, ground_truth=None, gradient=None, window_history=None):
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)

        if window_history is None:
            window_history = []

        if criterion_type == StopCriterionType.MAX_ITERATIONS:
            return False, 0.0

        raw_metric = 0.0

        if criterion_type == StopCriterionType.RELATIVE_CHANGE:
            raw_metric = float(xp.linalg.norm(current_lambda - prev_lambda) / (xp.linalg.norm(current_lambda) + 1e-10))
            
        elif criterion_type == StopCriterionType.COST_FUNCTION:
            if not history: return False, 0.0
            raw_metric = float(history[-1])
            
        elif criterion_type == StopCriterionType.MSE:
            if ground_truth is None:
                raise ValueError(f"[AOT-biomaps] Ground truth image required for MSE stopping criterion.")
            raw_metric = float(mse(SMatrix, ground_truth, current_lambda))
            
        elif criterion_type == StopCriterionType.GRADIENT_NORM:
            if gradient is None:
                raise ValueError(f"[AOT-biomaps] Gradient stop criterion is not supported with this optimizer.")
            raw_metric = float(xp.linalg.norm(gradient))
            
        else:
            raise ValueError(f"[AOT-biomaps] Unsupported stopping criterion type: {criterion_type}")

        window_history.append(raw_metric)
        
        if len(window_history) > window_size + 1:
            window_history.pop(0)

        if len(window_history) < 2:
            return False, 0.0

        if criterion_type in [StopCriterionType.RELATIVE_CHANGE, StopCriterionType.GRADIENT_NORM]:
            avg_diff = sum(window_history[1:]) / window_size
        else:
            diffs = [
                abs(window_history[i] - window_history[i-1]) / (abs(window_history[i-1]) + 1e-10) 
                for i in range(1, len(window_history))
            ]
            avg_diff = sum(diffs) / len(diffs)

        return bool(avg_diff < threshold), avg_diff    

# =============================================================================
# GRADIENT AND DIVERGENCE OPERATIONS (for TV regularization)
# =============================================================================

def gradient_2d(SMatrix, x):
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)
        Z = SMatrix.Z
        X = SMatrix.X
        
        x_img = x.reshape(Z, X)
        
        grad_x_img = xp.zeros_like(x_img, dtype=xp.float32)
        grad_z_img = xp.zeros_like(x_img, dtype=xp.float32)
        
        grad_x_img[:, :-1] = x_img[:, 1:] - x_img[:, :-1]
        grad_z_img[:-1, :] = x_img[1:, :] - x_img[:-1, :]
        
        return grad_x_img.flatten(), grad_z_img.flatten()

def divergence_2d(SMatrix, p_x, p_z):
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)
        Z = SMatrix.Z
        X = SMatrix.X
        
        p_x_img = p_x.reshape(Z, X)
        p_z_img = p_z.reshape(Z, X)
        
        div_x = xp.zeros_like(p_x_img, dtype=xp.float32)
        div_z = xp.zeros_like(p_z_img, dtype=xp.float32)
        
        div_x[:, 0] = -p_x_img[:, 0]
        div_x[:, 1:-1] = p_x_img[:, :-2] - p_x_img[:, 1:-1]
        div_x[:, -1] = p_x_img[:, -2]
        
        div_z[0, :] = -p_z_img[0, :]
        div_z[1:-1, :] = p_z_img[:-2, :] - p_z_img[1:-1, :]
        div_z[-1, :] = p_z_img[-2, :]
        
        return (div_x + div_z).flatten()

def proj_tv(SMatrix, p, radius=1.0):
    with get_device_context(SMatrix):
        xp = get_array_module(SMatrix)
        ZX = SMatrix.Z * SMatrix.X
        
        p_x = p[:ZX]
        p_z = p[ZX:]
        
        norm_p = xp.sqrt(p_x**2 + p_z**2 + 1e-12)
        mask = norm_p > radius
        
        p_x_proj = xp.where(mask, p_x * radius / norm_p, p_x)
        p_z_proj = xp.where(mask, p_z * radius / norm_p, p_z)
        
        return xp.concatenate([p_x_proj, p_z_proj])

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
        raise ValueError(f"[AOT-biomaps] Cannot find data file associated with header file {hdr_path}")

    img_path = os.path.join(os.path.dirname(hdr_path), data_file)

    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 4) if f'matrix size [{i}]' in header]
    if shape and shape[-1] == 1:
        shape = shape[:-1]

    if not shape:
        raise ValueError(f"[AOT-biomaps] Cannot determine image shape from metadata.")
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
        raise ValueError(f"[AOT-biomaps] Unsupported data type: {data_type}")

    byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
    endianess = '<' if 'little' in byte_order else '>'

    img_size = os.path.getsize(img_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize

    if img_size != expected_size:
        raise ValueError(f"[AOT-biomaps] Image file size ({img_size} bytes) does not match expected size ({expected_size} bytes).")

    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)

    image = data.reshape(shape[::-1])

    rescale_slope = float(header.get('data rescale slope', 1))
    rescale_offset = float(header.get('data rescale offset', 0))
    image = image * rescale_slope + rescale_offset

    return image

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

    try:
        matrix_size_gb = SMatrix.get_matrix_size()
        if isinstance(matrix_size_gb, dict) and 'error' in matrix_size_gb:
            raise ValueError(f"[AOT-biomaps] SMatrix allocation error: {matrix_size_gb['error']}")
        
        size_SMatrix = matrix_size_gb * (1024 ** 3)
        total_bytes += size_SMatrix
        print(f"[AOT-biomaps] SMatrix size: {matrix_size_gb:.3f} GB")
    except AttributeError:
        raise AttributeError(f"[AOT-biomaps] SMatrix must implement the get_matrix_size() method.")
    
    # Vector y
    if hasattr(y, 'nbytes'):
        size_y = y.nbytes
        total_bytes += size_y
        print(f"[AOT-biomaps] Vector y size: {size_y / (1024 ** 3):.3f} GB")
    else:
        raise ValueError(f"[AOT-biomaps] Vector y must be an array type exposing the .nbytes attribute.")

    return total_bytes / (1024 ** 3)

def check_gpu_memory(device_index, required_memory, show_logs=True):
    """Check if enough memory is available on the specified GPU."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available. Cannot check GPU memory.")
    
    free_memory = cp.cuda.runtime.memoryGetInfo()[0]
    free_memory_gb = free_memory / 1024**3
    
    if show_logs:
        print(f"[AOT-biomaps] Free memory on GPU {device_index}: {free_memory_gb:.2f} GB, Required memory: {required_memory:.2f} GB")
    
    return free_memory_gb >= required_memory

def check_gpu_available(SMatrix) -> bool:
    """Check if GPU operations are available."""
    if not isinstance(SMatrix.device, str) or "gpu" not in SMatrix.device:
        return False
    if not hasattr(SMatrix, 'sparse_mod'):
        return False
    if not CUPY_AVAILABLE:
        print("[AOT-biomaps] Warning: CuPy is not available. Falling back to CPU.")
        SMatrix.device = 'cpu'
        return False
    return True

def get_array_module(SMatrix):
    """Get the appropriate array module based on device."""
    if check_gpu_available(SMatrix):
        return cp
    else:
        return np

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
        raise ValueError(f"[AOT-biomaps] Unknown filter type: {filter_type}")

    FILTER[np.abs(f) > Fc] = 0
    FILTER = FILTER * np.exp(-2 * (np.abs(f) / Fc)**10)

    return FILTER
