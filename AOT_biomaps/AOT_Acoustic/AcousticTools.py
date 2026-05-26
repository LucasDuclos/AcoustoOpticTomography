from scipy.signal import hilbert
from scipy.ndimage import zoom
from scipy.io import loadmat as scipy_loadmat
import os
import numpy as np
from scipy.stats import linregress

# Optional cupy import for GPU acceleration
try:
    import cupy as cp
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
        raise ValueError(f"Could not load {param_path_mat}. Consider using scipy.io.loadmat or h5py for HDF5 files.")

def reshape_field(field, factor, device=None):
    """
    Downsample a 3D or 4D field using scipy interpolation.
    Args:
        field: Input field (numpy array).
        factor: Downsampling factor (tuple of ints).
        device: Ignored (kept for backward compatibility).
    Returns:
        Downsampled field (numpy array).
    """
    if field is None:
        raise ValueError("Acoustic field is not generated. Please generate the field first.")

    if not isinstance(field, np.ndarray):
        field = np.asarray(field, dtype=np.float32)

    if len(factor) == 3:
        if field.ndim != 3:
            raise ValueError("Expected 3D field.")
        # Use scipy.ndimage.zoom for downsampling
        zoom_factors = [1.0 / f for f in factor]
        downsampled = zoom(field, zoom_factors, order=1)  # order=1 for linear interpolation

    elif len(factor) == 4:
        if field.ndim != 4:
            raise ValueError("Expected 4D field.")
        zoom_factors = [1.0 / f for f in factor]
        downsampled = zoom(field, zoom_factors, order=1)

    else:
        raise ValueError("Unsupported dimension. Only 3D and 4D fields are supported.")

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
            raise ValueError("Acoustic field is not generated.")

        if not isinstance(field, np.ndarray):
            field = np.asarray(field, dtype=np.float32)

        if len(field.shape) not in [3, 4]:
            raise ValueError("Field must be 3D (T, X, Z) or 4D (T, X, Y, Z).")

        # Vectorized Hilbert transform along the time axis (axis=0)
        analytic_signal = hilbert(field, axis=0)
        envelope_sq = np.abs(analytic_signal) ** 2

        return envelope_sq.astype(np.float32)

    except Exception as e:
        print(f"Error in calculate_envelope_squared_cpu: {e}")
        raise

def calculate_envelope_squared_gpu(field, chunk_size=100):
    """
    Compute the squared envelope of the acoustic field on GPU using CuPy.
    Returns the result on CPU (numpy.ndarray) and frees GPU memory.

    Args:
        field: Acoustic field (numpy.ndarray or cupy.ndarray) with shape (T, X, Z) or (T, X, Y, Z).
        chunk_size: Number of spatial elements to process at once (to avoid OOM).

    Returns:
        envelope_sq (numpy.ndarray): Squared envelope on CPU.
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available. Falling back to CPU.")
        return calculate_envelope_squared_cpu(field)
    
    try:
        # 1. Dimensions
        T = field.shape[0]
        spatial_dims = field.shape[1:]
        total_spatial_size = int(cp.prod(cp.array(spatial_dims)))

        # Prepare Hilbert filter (once)
        n_fft = T
        h = cp.zeros(n_fft, dtype=cp.float32)
        if n_fft % 2 == 0:
            h[0] = h[n_fft // 2] = 1
            h[1:n_fft // 2] = 2
        else:
            h[0] = 1
            h[1:(n_fft + 1) // 2] = 2
        h = h[:, cp.newaxis]  # For broadcasting

        # Flatten spatial dimensions for easy iteration
        field_flat = field.reshape(T, -1)
        n_spatial = field_flat.shape[1]

        envelope_sq = cp.empty((T, n_spatial), dtype=cp.float32)

        # 2. Process in chunks
        for i in range(0, n_spatial, chunk_size):
            end = min(i + chunk_size, n_spatial)

            # Transfer only the current chunk
            chunk = cp.asarray(field_flat[:, i:end], dtype=cp.float32)

            # FFT, filter, IFFT
            chunk_fft = cp.fft.fft(chunk, axis=0)
            chunk_analytic = cp.fft.ifft(chunk_fft * h, axis=0)

            # Store squared envelope directly
            envelope_sq[:, i:end] = cp.abs(chunk_analytic) ** 2

            # Free chunk memory immediately
            del chunk, chunk_fft, chunk_analytic

        # 3. Reshape and return to CPU
        return cp.asnumpy(envelope_sq.reshape(T, *spatial_dims))

    except cp.cuda.memory.OutOfMemoryError:
        print("⚠️ Insufficient GPU memory. Falling back to CPU.")
        return calculate_envelope_squared_cpu(field)
    except Exception as e:
        print(f"Error in calculate_envelope_squared_gpu: {e}")
        raise

def calculate_envelope_squared(field, device=None):
    """
    Compute the squared envelope of the acoustic field.
    Automatically uses GPU if available and requested, otherwise falls back to CPU.

    Args:
        field: Acoustic field (numpy.ndarray or cupy.ndarray) with shape (T, X, Z) or (T, X, Y, Z).
        device: 'gpu' to use GPU (if available), 'cpu' to force CPU, None for auto-detection.

    Returns:
        envelope_sq (numpy.ndarray): Squared envelope of the acoustic field.
    """
    if device == 'gpu' and CUPY_AVAILABLE:
        return calculate_envelope_squared_gpu(field)
    else:
        return calculate_envelope_squared_cpu(field)

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
        print(f"Error reading pattern from file: {e}")
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
            raise ValueError("Invalid angle format in file name.")
        return int(angle_str)
    except Exception as e:
        print(f"Error reading angle from file: {e}")
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