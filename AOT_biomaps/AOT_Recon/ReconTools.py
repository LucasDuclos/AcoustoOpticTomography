import os
import numpy as np
from scipy.signal.windows import hann
from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix import SparseSMatrix_CSR, SparseSMatrix_SELL
import warnings

# Optional cupy imports for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

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
    Calculate la Mean Squared Error (MSE) entre deux tableaux.
    Équivalent à sklearn.metrics.mean_squared_error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def calculate_memory_requirement(SMatrix, y):
    """
    Calculate la mémoire requise (en Go) pour :
    - SMatrix : Matrice (np.ndarray, CuPy CSR, SparseSMatrix_CSR ou SparseSMatrix_SELL)
    - y : vecteur (NumPy ou CuPy, float32)

    Args:
        SMatrix: Matrix object (np.ndarray, cpsparse.csr_matrix, SparseSMatrix_CSR, or SparseSMatrix_SELL)
        y: Vector (float32)
    """
    total_bytes = 0

    # ---
    
    # 1.1. Custom Sparse Matrix (SELL/CSR)
    if isinstance(SMatrix, (SparseSMatrix_SELL, SparseSMatrix_CSR)):
        # We rely on the getMatrixSize method, which we fixed to track all host/GPU bytes.
        # This is the most reliable way to estimate memory for custom GPU-backed structures.
        try:
            matrix_size_gb = SMatrix.getMatrixSize()
            if isinstance(matrix_size_gb, dict) and 'error' in matrix_size_gb:
                raise ValueError(f"SMatrix allocation error: {matrix_size_gb['error']}")
            
            # Convert GB back to bytes (1 GB = 1024^3 bytes)
            size_SMatrix = matrix_size_gb * (1024 ** 3)
            total_bytes += size_SMatrix
            print(f"SMatrix (Custom Sparse) size: {matrix_size_gb:.3f} GB")

        except AttributeError:
            raise AttributeError("Custom Sparse Matrix must implement the getMatrixSize() method.")
    
    # 1.2. NumPy Dense Array (Standard)
    elif isinstance(SMatrix, np.ndarray):
        # Dense NumPy array (float32)
        size_SMatrix = SMatrix.nbytes
        total_bytes += size_SMatrix
        print(f"SMatrix (NumPy Dense) size: {size_SMatrix / (1024 ** 3):.3f} GB")

    # 1.3. CuPy CSR Matrix (Standard Sparse CuPy)
    # Note: Requires CuPy to be imported, which is usually done outside this function.
    # Assuming 'cpsparse.csr_matrix' is available in the environment if this path is taken.
    elif 'cupy.sparse' in str(type(SMatrix)): # Using string check for type safety outside CuPy context
        # CuPy CSR matrix structure: data (float32), indices (int32), indptr (int32)
        nnz = SMatrix.nnz
        num_rows = SMatrix.shape[0]
        size_data = nnz * 4        # float32 = 4 bytes
        size_indices = nnz * 4     # int32 = 4 bytes
        size_indptr = (num_rows + 1) * 4 # int32 = 4 bytes
        size_SMatrix = size_data + size_indices + size_indptr
        total_bytes += size_SMatrix
        print(f"SMatrix (CuPy CSR) size: {size_SMatrix / (1024 ** 3):.3f} GB")

    else:
        raise ValueError("SMatrix must be a np.ndarray, cpsparse.csr_matrix, or a custom SparseSMatrix object (CSR/SELL).")

    # ---
    
    # Check if y is a CuPy array or NumPy array (assuming float32 based on docstring)
    if hasattr(y, 'nbytes'):
        size_y = y.nbytes
        total_bytes += size_y
        print(f"Vector y size: {size_y / (1024 ** 3):.3f} GB")
    else:
        # Fallback if object doesn't expose nbytes (e.g., custom buffer), but usually array objects do.
        raise ValueError("Vector y must be an array type exposing the .nbytes attribute.")

    # ---
    return total_bytes / (1024 ** 3)

def check_gpu_memory(device_index, required_memory, show_logs=True):
    """Check if enough memory is available on the specified GPU."""
    import cupy as cp
    free_memory = cp.cuda.runtime.memoryGetInfo()[0]
    free_memory_gb = free_memory / 1024**3
    if show_logs:
        print(f"Free memory on GPU {device_index}: {free_memory_gb:.2f} GB, Required memory: {required_memory:.2f} GB")
    return free_memory_gb >= required_memory

def _forward_projection_cupy(SMatrix, theta_p, q_p):
    """Forward projection using CuPy."""
    q_p[:] = cp.dot(SMatrix.reshape(q_p.shape[0], -1), theta_p.ravel())

def _backward_projection_cupy(SMatrix, e_p, c_p):
    """Backward projection using CuPy."""
    c_p[:] = cp.dot(SMatrix.reshape(e_p.shape[0], -1).T, e_p.ravel())

def _build_adjacency_sparse(Z, X, corner=(0.5 - np.sqrt(2) / 4) / np.sqrt(2), face=0.5 - np.sqrt(2) / 4, dtype=np.float32):
    """Build adjacency matrix for sparse operations."""
    rows, cols, weights = [], [], []
    for z in range(Z):
        for x in range(X):
            j = z * X + x
            for dz, dx in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),   (1, 0), (1, 1)]:
                nz, nx = z + dz, x + dx
                if 0 <= nz < Z and 0 <= nx < X:
                    k = nz * X + nx
                    weight = corner if abs(dz) + abs(dx) == 2 else face
                    rows.append(j)
                    cols.append(k)
                    weights.append(weight)
    return cp.sparse.coo_matrix((cp.array(weights, dtype=dtype), (cp.array(rows), cp.array(cols))), shape=(Z*X, Z*X)).tocsr()

def power_method(P, PT, data, Z, X, n_it=10):
    """Power method for spectral radius estimation using CuPy."""
    x = cp.random.randn(Z * X)
    x = x / cp.linalg.norm(x)
    for _ in range(n_it):
        Ax = P(x)
        ATax = PT(Ax)
        x = ATax / cp.linalg.norm(ATax)
    ATax = PT(P(x))
    return cp.sqrt(cp.dot(x, ATax))

def proj_l2(p, alpha):
    """L2 projection using CuPy."""
    if alpha <= 0:
        return cp.zeros_like(p)
    norm = cp.sqrt(cp.sum(p**2, axis=0, keepdims=True) + 1e-12)
    return p * cp.minimum(norm, alpha) / (norm + 1e-12)

def gradient(x):
    """Compute gradient using CuPy."""
    grad_x = cp.zeros_like(x)
    grad_y = cp.zeros_like(x)
    grad_x[:, :-1] = x[:, 1:] - x[:, :-1]
    grad_y[:-1, :] = x[1:, :] - x[:-1, :]
    return cp.stack((grad_x, grad_y), axis=0)

def div(x):
    """Compute divergence using CuPy."""
    if x.ndim == 3:
        x = cp.expand_dims(x, axis=0)

    gx = x[:, 0, :, :]
    gy = x[:, 1, :, :]

    div_x = cp.zeros_like(gx)
    div_x[:, :, 1:] += gx[:, :, :-1]
    div_x[:, :, :-1] -= gx[:, :, :-1]

    div_y = cp.zeros_like(gy)
    div_y[:, 1:, :] += gy[:, :-1, :]
    div_y[:, :-1, :] -= gy[:, :-1, :]

    return -(div_x + div_y)

def norm2sq(x):
    """Squared L2 norm using CuPy."""
    return cp.sum(x**2)

def norm1(x):
    """L1 norm using CuPy."""
    return cp.sum(cp.abs(x))

def KL_divergence(Ax, y):
    """KL divergence using CuPy."""
    return cp.sum(Ax - y * cp.log(Ax + 1e-10))

def gradient_KL(Ax, y):
    """KL gradient using CuPy."""
    return 1 - y / (Ax + 1e-10)

def prox_F_star(y, sigma, a):
    """Proximal operator for F* using CuPy."""
    return 0.5 * (y - cp.sqrt(y**2 + 4 * sigma * a))

def prox_G(x, tau, K):
    """Proximal operator for G using CuPy."""
    return cp.clip(x - tau * K, 0, None)

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

def compute_TV_cpu(x, Z, X, isotropic=False):
    """
    Compute total variation of x (1D flattened of shape Z*X).
    isotropic=False -> anisotropic (sum |dx| + |dy|)
    isotropic=True -> isotropic sqrt(dx^2 + dy^2)
    """
    x2d = x.reshape(Z, X)
    dx = np.diff(x2d, axis=1)
    dy = np.diff(x2d, axis=0)
    if isotropic:
        mags = np.sqrt(dx**2 + dy**2)
        return float(np.sum(mags))
    else:
        return float(np.sum(np.abs(dx)) + np.sum(np.abs(dy)))

def get_apodization_vector_gpu(matrix_sparse_obj):
    """
    Generate a 2D Hanning window vector for apodization of system matrix A.
    This vector should be multiplied by the columns of A (Z*X pixels).
    """
    Z = matrix_sparse_obj.Z
    X = matrix_sparse_obj.X

    window_x = hann(X).astype(np.float32)
    window_z = np.ones(Z, dtype=np.float32)

    window_2d = np.outer(window_z, window_x)
    window_vector = window_2d.flatten()

    window_gpu = cp.asarray(window_vector)

    print(f"Window vector (Z*X={Z*X}) generated and transferred to GPU.")

    return window_gpu

def power_method_estimate_L__SELL(SMatrix, n_it=20):
    """Estimate ||A||^2 using power method with CuPy."""
    TN = int(SMatrix.N * SMatrix.T)
    ZX = int(SMatrix.Z * SMatrix.X)

    x = cp.random.randn(ZX).astype(cp.float32)
    x /= cp.linalg.norm(x) + 1e-12

    for _ in range(n_it):
        q = SMatrix.projection(x)
        ATq = SMatrix.backprojection(q)
        norm = cp.linalg.norm(ATq)
        if norm < 1e-12:
            break
        x = ATq / norm

    L_sq = float(cp.dot(x, SMatrix.backprojection(SMatrix.projection(x))))
    return max(L_sq, 1e-6)

def fourierz_gpu(z, X):
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
    """
    Inverse Fourier along X (axis=1), Matlab-compatible
    F_fx_z : (Nz, Nx) complex cupy array
    dx : scalar (spacing in x)
    """

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

    Parameters:
    -----------
    X_m : probe element positions
    DelayLAWS : delays in seconds (each column = angle, each row = element)
    ActiveLIST : mask of active elements (1 = active)
    c : speed of sound

    Returns:
    --------
    C : rotation centers for each angle
    """
    Nangle = DelayLAWS.shape[1]
    C = np.zeros((Nangle, 2))

    ct = DelayLAWS * c  # convert seconds to distance

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
    """
    GPU equivalent of RotateTheta.m
    X, Z, Iin : cupy arrays (Nz, Nx)
    theta : scalar (float)
    C : (2,) array-like
    """

    # ---
    X_rel = X - C[0]
    Z_rel = Z - C[1]

    c = cp.cos(theta)
    s = cp.sin(theta)

    # ---
    Xout = c * X_rel + s * Z_rel
    Zout = -s * X_rel + c * Z_rel

    # Back to original frame
    Xout += C[0]
    Zout += C[1]

    # ---
    # Grille régulière supposée
    dx = X[0, 1] - X[0, 0]
    dz = Z[1, 0] - Z[0, 0]

    x0 = X[0, 0]
    z0 = Z[0, 0]

    ix = (Xout - x0) / dx
    iz = (Zout - z0) / dz

    # ---
    # map_coordinates attend (ndim, Npoints)
    coords = cp.stack([iz.ravel(), ix.ravel()])

    Iout = map_coordinates(
        Iin,
        coords,
        order=1,          # bilinear
        mode='constant',
        cval=0.0
    )

    return Iout.reshape(Iin.shape)

def filter_radon_gpu(fz, Fc):
    FILTER = cp.abs(fz)
    FILTER = cp.where(cp.abs(fz) > Fc, 0, FILTER)
    FILTER *= cp.exp(-2 * cp.abs(fz / Fc)**10)
    return FILTER

