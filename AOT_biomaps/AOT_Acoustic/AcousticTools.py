from scipy.signal import hilbert
from scipy.ndimage import zoom
from scipy.io import loadmat as scipy_loadmat
import os
import numpy as np
from scipy.stats import linregress
from numba import njit, prange

# Optional cupy import for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def loadmat(param_path_mat):
    """
    Load a .mat file (MATLAB format).

    Args:
        param_path_mat: Path to the .mat file.

    Returns:
        Dictionary containing the variables from the file.
    """
    try:
        return scipy_loadmat(param_path_mat)
    except Exception:
        raise ValueError(f"[AOT-biomaps] Could not load {param_path_mat}. Consider using scipy.io.loadmat or h5py for HDF5 files.")

def reshape_field_gpu(field, factor, GPUdevice):
    """
    Downsample a 3D or 4D field on GPU using PyTorch or CuPy.
    Args:
        field: Input field (numpy array or torch/cupy array).
        factor: Downsampling factor (tuple of ints).
        GPUdevice: GPU device (e.g., "cuda:0").
    Returns:
        Downsampled field (numpy array).
    """
    cp.cuda.Device(GPUdevice).use()  # Set the GPU device
    
    if field is None:
        raise ValueError(f"[AOT-biomaps] Acoustic field is not generated.")

    if not isinstance(field, cp.ndarray):
        field = cp.asarray(field, dtype=cp.float32)

    if len(factor) == 3:
        if field.ndim != 3:
            raise ValueError(f"[AOT-biomaps] Expected a 3D field (T, Z, X).")
    elif len(factor) == 4:
        if field.ndim != 4:
            raise ValueError(f"[AOT-biomaps] Expected a 4D field (T, Y, Z, X).")
    else:
        raise ValueError(f"[AOT-biomaps] Unsupported dimensions. Only 3D and 4D fields are supported.")

    if not all(isinstance(f, int) and f >= 1 for f in factor):
        raise ValueError(f"[AOT-biomaps] Downsampling factors must be integers >= 1.")

    new_shape = tuple(s // f for s, f in zip(field.shape, factor))
    zoom_factors = tuple(new_s / old_s for new_s, old_s in zip(new_shape, field.shape))
    downsampled = cupyx.scipy.ndimage.zoom(field, zoom_factors, order=1)

    return cp.asnumpy(downsampled).astype(np.float32)


def reshape_field_cpu(field, factor):
    """
    Downsample a 3D or 4D field on CPU using scipy (optimized).
    Args:
        field: Input field (numpy array).
        factor: Downsampling factor (tuple of ints).
    Returns:
        Downsampled field (numpy array).
    """
    if field is None:
        raise ValueError(f"[AOT-biomaps] Acoustic field is not generated.")

    if not isinstance(field, np.ndarray):
        field = np.asarray(field, dtype=np.float32)

    # Validate factor (must be integers >= 1)
    if not all(isinstance(f, int) and f >= 1 for f in factor):
        raise ValueError(f"[AOT-biomaps] Downsampling factors must be integers >= 1.")

    # Calculate new shape
    new_shape = [s // f for s, f in zip(field.shape, factor)]

    # Use zoom with order=1 (linear) for downsampling
    zoom_factors = [s_new / s_orig for s_new, s_orig in zip(new_shape, field.shape)]
    downsampled = zoom(field, zoom_factors, order=1)

    return downsampled.astype(np.float32)

def calculate_envelope_squared_cpu(field):
    """
    Compute the squared envelope of the acoustic field on CPU in a vectorized way.
    Optimized for 3D (T, X, Z) or 4D (T, X, Y, Z) arrays.

    Args:
        field: Acoustic field (numpy.ndarray). Expected shape: (T, X, Z) or (T, X, Y, Z).

    Returns:
        envelope_sq (numpy.ndarray): Squared envelope of the acoustic field.
    """
    try:
        if field is None:
            raise ValueError(f"[AOT-biomaps] Acoustic field is not generated.")

        if not isinstance(field, np.ndarray):
            field = np.asarray(field, dtype=np.float32)

        if len(field.shape) not in [3, 4]:
            raise ValueError(f"[AOT-biomaps] Field must be 3D (T, X, Z) or 4D (T, X, Y, Z).")

        # Vectorized Hilbert transform along the time axis (axis=0)
        analytic_signal = hilbert(field, axis=0)
        envelope_sq = np.abs(analytic_signal) ** 2

        return envelope_sq.astype(np.float32)

    except Exception as e:
        print(f"[AOT-biomaps] Error in calculate_envelope_squared_cpu: {e}")
        raise

def calculate_envelope_squared_gpu(field, GPUdevice, chunk_size=100):
    """
    Compute the squared envelope of the acoustic field on GPU using CuPy.
    Returns the result on CPU (numpy.ndarray) and frees GPU memory.

    Args:
        field: Acoustic field (numpy.ndarray or cupy.ndarray) with shape (T, X, Z) or (T, X, Y, Z).
        GPUdevice: The GPU device to use.
        chunk_size: Number of spatial elements to process at once (to avoid OOM).

    Returns:
        envelope_sq (numpy.ndarray): Squared envelope on CPU.
    """
    if not CUPY_AVAILABLE:
        print("[AOT-biomaps] Warning: CuPy not available. Falling back to CPU.")
        return calculate_envelope_squared_cpu(field)
    
    try:
        cp.cuda.Device(GPUdevice).use()
        field_gpu = cp.asarray(field, dtype=cp.float32) 

        T = field_gpu.shape[0]
        field_flat = field_gpu.reshape(T, -1)

        n_fft = T
        h = cp.zeros(n_fft, dtype=cp.float32)
        if n_fft % 2 == 0:
            h[0] = h[n_fft // 2] = 1
            h[1:n_fft // 2] = 2
        else:
            h[0] = 1
            h[1:(n_fft + 1) // 2] = 2
        h = h[:, cp.newaxis]  # (T, 1)

        field_fft = cp.fft.fft(field_flat, axis=0)  
        analytic_signal = cp.fft.ifft(field_fft * h, axis=0)
        envelope_sq = cp.abs(analytic_signal) ** 2

        return cp.asnumpy(envelope_sq.reshape(T, *field.shape[1:]))

    except cp.cuda.memory.OutOfMemoryError:
        print(f"[AOT-biomaps] Insufficient GPU memory. Falling back to CPU.")
        return calculate_envelope_squared_cpu(field)
    except Exception as e:
        print(f"[AOT-biomaps] Error in calculate_envelope_squared_gpu: {e}")
        raise

def calculate_envelope_cpu(field):
    """
    Compute the envelope of the acoustic field on CPU in a vectorized way.
    Optimized for 3D (T, X, Z) or 4D (T, X, Y, Z) arrays.

    Args:
        field: Acoustic field (numpy.ndarray). Expected shape: (T, X, Z) or (T, X, Y, Z).

    Returns:
        envelope (numpy.ndarray): Envelope of the acoustic field.
    """
    try:
        if field is None:
            raise ValueError(f"[AOT-biomaps] Acoustic field is not generated.")

        if not isinstance(field, np.ndarray):
            field = np.asarray(field, dtype=np.float32)

        if len(field.shape) not in [3, 4]:
            raise ValueError(f"[AOT-biomaps] Field must be 3D (T, X, Z) or 4D (T, X, Y, Z).")

        # Vectorized Hilbert transform along the time axis (axis=0)
        analytic_signal = hilbert(field, axis=0)
        envelope = np.abs(analytic_signal)

        return envelope.astype(np.float32)

    except Exception as e:
        print(f"[AOT-biomaps] Error in calculate_envelope_cpu: {e}")
        raise

def calculate_envelope_gpu(field, GPUdevice, chunk_size=100):
    """
    Compute the envelope of the acoustic field on GPU using CuPy.
    Returns the result on CPU (numpy.ndarray) and frees GPU memory.

    Args:
        field: Acoustic field (numpy.ndarray or cupy.ndarray) with shape (T, X, Z) or (T, X, Y, Z).
        GPUdevice: The GPU device to use.
        chunk_size: Number of spatial elements to process at once (to avoid OOM).

    Returns:
        envelope (numpy.ndarray): Envelope on CPU.
    """
    if not CUPY_AVAILABLE:
        print(f"[AOT-biomaps] Warning: CuPy not available. Falling back to CPU.")
        return calculate_envelope_cpu(field)
    
    try:
        cp.cuda.Device(GPUdevice).use()
        field_gpu = cp.asarray(field, dtype=cp.float32) 

        T = field_gpu.shape[0]
        field_flat = field_gpu.reshape(T, -1)

        n_fft = T
        h = cp.zeros(n_fft, dtype=cp.float32)
        if n_fft % 2 == 0:
            h[0] = h[n_fft // 2] = 1
            h[1:n_fft // 2] = 2
        else:
            h[0] = 1
            h[1:(n_fft + 1) // 2] = 2
        h = h[:, cp.newaxis]  # (T, 1)

        field_fft = cp.fft.fft(field_flat, axis=0)  
        analytic_signal = cp.fft.ifft(field_fft * h, axis=0)
        envelope = cp.abs(analytic_signal)

        return cp.asnumpy(envelope.reshape(T, *field.shape[1:]))

    except cp.cuda.memory.OutOfMemoryError:
        print(f"[AOT-biomaps] Insufficient GPU memory. Falling back to CPU.")
        return calculate_envelope_cpu(field)
    except Exception as e:
        print(f"[AOT-biomaps] Error in calculate_envelope_gpu: {e}")
        raise

def calculate_envelope_squared(field, isGPU=None, GPUdevice=None, chunk_size=100):
    """
    Compute the squared envelope of the acoustic field.
    Automatically uses GPU if available and requested, otherwise falls back to CPU.

    Args:
        field: Acoustic field (numpy.ndarray or cupy.ndarray) with shape (T, X, Z) or (T, X, Y, Z).
        isGPU: Whether to use GPU for computation. (Default is None, which uses CPU.)
        GPUdevice: The GPU device to use. (Default is None, which uses the default GPU.)
        chunk_size: Number of spatial elements to process at once (to avoid OOM).

    Returns:
        envelope_sq (numpy.ndarray): Squared envelope of the acoustic field.
    """
    if isGPU is True and CUPY_AVAILABLE:
        return calculate_envelope_squared_gpu(field=field, GPUdevice = GPUdevice, chunk_size=chunk_size)
    else:
        return calculate_envelope_squared_cpu(field=field)

def calculate_envelope(field, isGPU=None, GPUdevice=None, chunk_size=100):
    """
    Compute the envelope of the acoustic field.
    Automatically uses GPU if available and requested, otherwise falls back to CPU.

    Args:
        field: Acoustic field (numpy.ndarray or cupy.ndarray) with shape (T, X, Z) or (T, X, Y, Z).
        isGPU: Whether to use GPU for computation. (Default is None, which uses CPU.)
        GPUdevice: The GPU device to use. (Default is None, which uses the default GPU.)
        chunk_size: Number of spatial elements to process at once (to avoid OOM).

    Returns:
        envelope (numpy.ndarray): Envelope of the acoustic field.
    """
    if isGPU is True and CUPY_AVAILABLE:
        return calculate_envelope_gpu(field=field, GPUdevice = GPUdevice, chunk_size=chunk_size)
    else:
        return calculate_envelope_cpu(field=field)

def get_pattern(pathFile):
    """
    Extract the pattern from a file path.

    Args:
        pathFile (str): Path to the file containing the pattern.

    Returns:
        str: The pattern string.
    """
    try:
        # Pattern between first _ and last _
        pattern = os.path.basename(pathFile).split('_')[1:-1]
        pattern_str = ''.join(pattern)
        return pattern_str
    except Exception as e:
        print(f"[AOT-biomaps] Error reading pattern from file: {e}")
        return None

def detect_space_0_and_space_1(hex_string):
    """
    Detect the longest sequences of 0s and 1s in a hex string.

    Args:
        hex_string: Hexadecimal string.

    Returns:
        tuple: (space_0, space_1) where space_0 is the length of the longest 0 sequence,
               and space_1 is the length of the longest 1 sequence.
    """
    binary_string = bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)

    # Find longest sequence of consecutive 0s
    zeros_groups = [len(s) for s in binary_string.split('1')]
    space_0 = max(zeros_groups) if zeros_groups else 0

    # Find longest sequence of consecutive 1s
    ones_groups = [len(s) for s in binary_string.split('0')]
    space_1 = max(ones_groups) if ones_groups else 0

    return space_0, space_1

def get_angle(pathFile):
    """
    Extract the angle from a file path.

    Args:
        pathFile (str): Path to the file containing the angle.

    Returns:
        int: The angle in degrees.
    """
    try:
        # Angle between last _ and .
        angle_str = os.path.basename(pathFile).split('_')[-1].replace('.', '')
        if angle_str.startswith('0'):
            angle_str = angle_str[1:]
        elif angle_str.startswith('1'):
            angle_str = '-' + angle_str[1:]
        else:
            raise ValueError(f"[AOT-biomaps] Invalid angle format in file name: {pathFile}")
        return int(angle_str)
    except Exception as e:
        print(f"[AOT-biomaps] Error reading angle from file: {e}")
        return None

def get_frequency(fileName, num_elements, dx):
    """
    Calculate the spatial frequency from a file name.

    Args:
        fileName: File name containing the pattern.
        num_elements: Number of elements in the probe.
        dx: Element spacing in meters.

    Returns:
        int: Spatial frequency in mm^-1.
    """
    profile = hex_to_binary_profile(fileName[6:-4], num_elements)

    if set(fileName[6:-4].lower().replace(" ", "")) == {'f'}:
        fs_key = 0.0  # fs_key in mm^-1 (0.0 mm^-1)
    else:
        ft_prof = np.fft.fft(profile)
        idx_max = np.argmax(np.abs(ft_prof[1:len(profile)//2])) + 1
        freqs = np.fft.fftfreq(len(profile), d=dx)

        # freqs is in m^-1 because dx is in meters
        fs_m_inv = abs(freqs[idx_max])

        fs_key = fs_m_inv  # Spatial frequency in mm^-1
    return int(fs_key / (1/(len(profile)*dx)))

def format_angle(a):
    """
    Format an angle as a string for file naming.

    Args:
        a: Angle in degrees.

    Returns:
        str: Formatted angle string (e.g., "045" or "145" for -45).
    """
    return f"{'1' if a < 0 else '0'}{abs(int(a)):02d}"

def next_power_of_2(n):
    """
    Calculate the next power of 2 greater than or equal to n.

    Args:
        n: Input integer.

    Returns:
        int: Next power of 2 >= n.
    """
    return int(2 ** np.ceil(np.log2(n)))

def hex_to_binary_profile(hex_string, n_piezos=192):
    """
    Convert a hex string to a binary profile array.

    Args:
        hex_string: Hexadecimal string representing the pattern.
        n_piezos: Number of piezos in the probe (default: 192).

    Returns:
        numpy.ndarray: Binary profile as an array of 0s and 1s.
    """
    hex_string = hex_string.strip().replace(" ", "").replace("\n", "")
    if set(hex_string.lower()) == {'f'}:
        return np.ones(n_piezos, dtype=int)

    try:
        n_char = len(hex_string)
        n_bits = n_char * 4
        binary_str = bin(int(hex_string, 16))[2:].zfill(n_bits)
        if len(binary_str) < n_piezos:
            # Pad or truncate to match the actual probe size
            binary_str = binary_str.ljust(n_piezos, '0')
        elif len(binary_str) > n_piezos:
            binary_str = binary_str[:n_piezos]
        return np.array([int(b) for b in binary_str])
    except ValueError:
        return np.zeros(n_piezos, dtype=int)

def calculate_angle_from_delays(delays, c=1540):
    """
    Calculate the angle of incidence θ (in degrees) from an array of 192 delays.
    Uses linear regression to estimate the slope of the delays.

    Args:
        delays: Array of 192 delays (in seconds).
        c: Speed of sound (m/s).

    Returns:
        theta: Angle in degrees (positive to the right, negative to the left).
    """
    pitch = 0.2e-3  # Element spacing (m)
    x = np.linspace(-(192-1)/2 * pitch, (192-1)/2 * pitch, 192)  # Element positions (m)

    # Linear regression to estimate the slope (sinθ / c)
    slope, _, _, _, _ = linregress(x, delays)

    # Calculate the angle (in degrees)
    theta = np.rad2deg(np.arcsin(slope * c))

    # Determine the sign based on the position of the maximum delay
    max_index = np.argmax(delays)
    if max_index < 95:  # Left
        theta = -abs(theta)
    elif max_index > 95:  # Right
        theta = abs(theta)
    else:  # Center (θ ≈ 0)
        theta = 0.0

    return int(np.round(theta, 0))

@njit(parallel=True, fastmath=True)
def compute_field_numba(field, t, active_indices, apod_window, weight_base, 
                        x_start_probe_fine, x_pivot_px_fine, dx_fine, c0, angle_rad, 
                        n_t_burst, enveloppe_t, el_width_px_fine, cos_a, sin_a, 
                        factor, Nt, Nz, Nx, Nx_fine, Nz_fine):
    """
    Kernel compilé en C pour le calcul intensif de la propagation acoustique.
    """
    for idx in prange(len(active_indices)):
        i = active_indices[idx]
        val_i = weight_base * apod_window[i]

        x_i_px_fine = x_start_probe_fine + (i * el_width_px_fine)
        dist_to_pivot = (x_i_px_fine - x_pivot_px_fine) * dx_fine
        delay_i = (abs(dist_to_pivot) * np.sin(abs(angle_rad))) / c0

        for t_idx in range(Nt):
            t_eff = t[t_idx] - delay_i
            if t_eff <= 0 or t_eff >= t[-1]:
                continue
                
            dist_travelled = c0 * t_eff
            z_px_fine = int((dist_travelled * cos_a) / dx_fine)
            x_px_fine_base = int((x_i_px_fine * dx_fine + dist_travelled * sin_a) / dx_fine)

            for b_shift in range(n_t_burst):
                st = t_idx + b_shift
                if st >= Nt:
                    continue
                    
                val_final = enveloppe_t[b_shift] * val_i

                for offset_x in range(el_width_px_fine):
                    curr_x = x_px_fine_base + offset_x
                    
                    if 0 <= z_px_fine < Nz_fine and 0 <= curr_x < Nx_fine:
                        zf = z_px_fine // factor
                        xf = curr_x // factor
                        
                        field[st, zf, xf] += val_final