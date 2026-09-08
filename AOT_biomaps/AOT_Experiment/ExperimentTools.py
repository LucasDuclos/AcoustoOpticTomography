import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import groupby
from tqdm import trange
import os
import random
from typing import List, Tuple

def select_random_activeList(input_file, output_file, N):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)

    if N > total_lines:
        raise ValueError(f"[AOT-biomaps] N ({N}) cannot be greater than the total number of lines in the file ({total_lines}).")

    selected_indices = random.sample(range(total_lines), N)
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in):
            if i in selected_indices:
                f_out.write(line)

def get_selected_indices(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"[AOT-biomaps] Input file {input_file} does not exist.")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"[AOT-biomaps] File {output_file} does not exist. Run select_random_activeList first.")

    with open(output_file, 'r') as f_out:
        output_lines = f_out.readlines()

    selected_indices = []
    with open(input_file, 'r') as f_in:
        for i, line in enumerate(f_in):
            if line in output_lines:
                selected_indices.append(i)
                output_lines.remove(line)
    if len(output_lines) !=0:
        raise ValueError(f"[AOT-biomaps] File {output_file} contains lines not in {input_file}. Ensure it was created with select_random_activeList.")

    return selected_indices

def calc_mat_os(xm, fx, bool_active_list, signal_type):
    """
    Generate a binary mask matrix based on spatial frequency and element positions.

    Parameters:
        xm: Array of real element positions (in meters)
        fx: Spatial frequency (in m^-1)
        signal_type: 'cos' or 'sin' for the signal type
        bool_active_list: Binary matrix to match the shape of the output

    Returns:
        numpy.ndarray: Mask matrix with shape (num_els, num_cols)
    """
    num_els = len(xm)
    num_cols = bool_active_list.shape[1]

    if signal_type == 'cos':
        mask = (np.cos(2 * np.pi * fx * xm) > 0).astype(float)
    elif signal_type == 'sin':
        mask = (np.sin(2 * np.pi * fx * xm) > 0).astype(float)
    else:
        mask = np.ones(num_els)  # Fallback

    return np.tile(mask[:, np.newaxis], (1, num_cols))

def convert_to_hex_list(matrix):
    """
    Convert a binary matrix to a list of hexadecimal strings (4-bit chunks).
    Each column becomes a string.

    Parameters:
        matrix: Binary numpy array (2D)

    Returns:
        list: List of hexadecimal strings, one per column
    """
    n_els, n_scans = matrix.shape

    # 1. Padding to ensure n_els is a multiple of 4
    remainder = n_els % 4
    if remainder != 0:
        padding = np.zeros((4 - remainder, n_scans))
        matrix = np.vstack([matrix, padding])

    # 2. Reshape to isolate 4-bit blocks (nibbles)
    # Resulting shape: (Number of blocks, 4 bits, Number of scans)
    blocks = matrix.reshape(-1, 4, n_scans)

    # 3. Calculate decimal value of each block (0 to 15)
    # First element is considered as LSB
    weights = np.array([1, 2, 4, 8]).reshape(1, 4, 1)
    dec_values = np.sum(blocks * weights, axis=1).astype(int)

    # 4. Convert to hexadecimal characters
    hex_table = np.array(list("0123456789abcdef"))
    hex_matrix = hex_table[dec_values]

    # 5. Assemble strings (from element N to 0 for standard Shift Register order)
    return ["".join(hex_matrix[::-1, col]) for col in range(n_scans)]

def hex_to_binary_profile(hex_string, n_piezos=192):
    """
    Convert a hexadecimal string to a binary profile array.

    Parameters:
        hex_string: Hexadecimal string representing the pattern
        n_piezos: Number of piezo elements in the probe (default: 192)

    Returns:
        numpy.ndarray: Binary profile as an array of 0s and 1s
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

def binary_to_hex_profile(bits):
    """
    Convert a binary profile to a hexadecimal string.

    Parameters:
        bits: Array or list of binary values (0s and 1s)

    Returns:
        str: Hexadecimal string representation
    """
    bit_string = ''.join(str(b) for b in bits)
    bit_string = bit_string.zfill(len(bits))
    hex_string = ''.join([f"{int(bit_string[i:i+4], 2):x}" for i in range(0, len(bit_string), 4)])
    return hex_string

def get_phase_deterministic(profile):
    """
    Determine the phase based on the initial value (0 or 1) and the shift state
    of the binary sequence.

    WARNING: This function is kept for compatibility, but in practice the logic is often
    simplified if labels guarantee phases 0, pi/2, pi, 3pi/2.

    Parameters:
        profile: Binary profile array

    Returns:
        float: Phase value in radians
    """
    runs = [(k, sum(1 for _ in g)) for k, g in groupby(profile)]
    if not runs: return 0.0

    nominal_half_period = max([r[1] for r in runs])
    if nominal_half_period == 0: return 0.0

    first_val = runs[0][0]  # 0 or 1
    first_len = runs[0][1]
    # Detect 50% duty cycle
    is_shifted = (0.3 < first_len / nominal_half_period < 0.7)

    # ---
    if first_val == 0:
        if is_shifted:
            idx = 3  # C1/C3 shifted (phi_1 or phi_3)
        else:
            idx = 4  # C2/C4 not shifted
    else:  # first_val == 1
        if is_shifted:
            idx = 1  # C1/C3 shifted (phi_1 or phi_3)
        else:
            idx = 2  # C2/C4 not shifted

    # Use quadrature phases 0, pi/2, pi, 3pi/2
    if idx == 1:
        phase = 0
    elif idx == 2:
        phase = np.pi/2
    elif idx == 3:
        phase = np.pi
    elif idx == 4:
        phase = 3*np.pi/2

    return phase

def add_sincos_cpu(R, decimation, theta):
    """
    Add sine/cosine components to the input matrix R based on decimation and theta parameters.

    Parameters:
        R: Input complex matrix
        decimation: Decimation factors
        theta: Angle parameters

    Returns:
        tuple: (Iout, theta_u, decim_u) where Iout is the output complex matrix
    """
    decimation = np.asarray(decimation)
    theta = np.asarray(theta)

    ScanParam = np.stack([decimation, theta], axis=1)
    uniq, ia, ib = np.unique(ScanParam, axis=0, return_index=True, return_inverse=True)

    theta_u = uniq[:,1]
    decim_u = uniq[:,0]

    theta0 = np.unique(theta_u)
    N0 = len(theta0)

    Rg = np.asarray(R)
    Nz = Rg.shape[0]
    Nk = N0 + (Rg.shape[1] - N0)//4

    Iout = np.zeros((Nz, Nk), dtype=np.complex64)
    # fx = 0 (plane wave)
    Iout[:, :N0] = Rg[:, :N0]

    k = N0
    for i in range(N0, len(ia)):
        idx = np.where(ib == i)[0]
        h1, h2, h3, h4 = Rg[:, idx].T
        Iout[:, k] = ((h1 - h2) - 1j*(h3 - h4)) / 2
        k += 1

    return Iout, theta_u, decim_u

def load_AOsignal(AOsignalPath):
    """
    Load AO signal from a file.

    Parameters:
        AOsignalPath: Path to the AO signal file (.cdh/.cdf or .npy)

    Returns:
        numpy.ndarray: Loaded AO signal

    Raises:
        ValueError: If the file format is not supported
    """
    if AOsignalPath.endswith(".cdh"):
        with open(AOsignalPath, "r") as file:
            cdh_content = file.readlines()

        cdf_path = AOsignalPath.replace(".cdh", ".cdf")

        # Extract parameters from .cdh file
        n_scans = int([line.split(":")[1].strip() for line in cdh_content if "Number of events" in line][0])
        n_acquisitions_per_event = int([line.split(":")[1].strip() for line in cdh_content if "Number of acquisitions per event" in line][0])
        num_elements = int([line.split(":")[1].strip() for line in cdh_content if "Number of US transducers" in line][0])

        # Initialize structures
        AO_signal = np.zeros((n_acquisitions_per_event, n_scans), dtype=np.float32)
        active_lists = []
        angles = []

        # Read binary file
        with open(cdf_path, "rb") as file:
            for j in trange(n_scans, desc="[AOT-biomaps] Reading events"):
                # Read activeList: 48 hex chars = 24 bytes
                active_list_bytes = file.read(24)
                active_list_hex = active_list_bytes.hex()
                active_lists.append(active_list_hex)

                # Read angle (1 signed byte)
                angle_byte = file.read(1)
                angle = np.frombuffer(angle_byte, dtype=np.int8)[0]
                angles.append(angle)

                # Read AO signal (float32)
                data = np.frombuffer(file.read(n_acquisitions_per_event * 4), dtype=np.float32)
                if len(data) != n_acquisitions_per_event:
                    raise ValueError(f"[AOT-biomaps] Error at event {j}: expected {n_acquisitions_per_event}, got {len(data)}")
                AO_signal[:, j] = data

        return AO_signal
    elif AOsignalPath.endswith(".npy"):
        return np.load(AOsignalPath)  # Assumed to be in the correct format
    else:
        raise ValueError("[AOT-biomaps] Unsupported file format. Use .cdh/.cdf or .npy.")

def create_dark_transparent_hot_cmap(vmin=0.0, opacity=1.0):
    n_colors = 256
    hot_cmap = plt.cm.get_cmap('hot', n_colors)
    colors = [hot_cmap(i)[:3] + (opacity,) for i in range(n_colors)]
    colors = [
        (0, 0, 0, 0) if i/n_colors < vmin else colors[i]
        for i in range(n_colors)
    ]
    return LinearSegmentedColormap.from_list('dark_transparent_hot', colors)

def flip_probe(active_list_lines: List[str], y: np.ndarray) -> Tuple[np.ndarray, List[int], List[tuple]]:
    """
    Rearranges the columns of the Acousto-Optic (AO) measurement matrix y (dimension: Times x N) 
    to match a full geometric flip of the probe (inverting both the piezoelectric activation 
    patterns and the acoustic steering angles) by performing all inline transformations.
    """
    config_to_idx = {line.strip(): i for i, line in enumerate(active_list_lines) if line.strip()}
    
    new_indices = []
    missing_configs = []
    
    for i, line in enumerate(active_list_lines):
        line = line.strip()
        if not line:
            continue
            
        pattern, angle = line.split('_')
        
        # Inline 1: Spatial mirror inversion of the 192-bit piezoelectric pattern
        bin_str = bin(int(pattern, 16))[2:].zfill(192)
        flipped_pattern = hex(int(bin_str[::-1], 2))[2:].zfill(48)
        
        # Inline 2: Sign inversion of the acoustic beam angle
        if angle == "000":
            flipped_angle = "000"
        elif angle.startswith("0"):
            flipped_angle = "1" + angle[1:]
        elif angle.startswith("1"):
            flipped_angle = "0" + angle[1:]
        else:
            raise ValueError(f"Unknown angle format: {angle}")
            
        flipped_config = f"{flipped_pattern}_{flipped_angle}"
        
        if flipped_config in config_to_idx:
            new_indices.append(config_to_idx[flipped_config])
        else:
            new_indices.append(i)
            missing_configs.append((i, line, flipped_config))
            
    y_flipped = y[:, new_indices]
    return y_flipped, new_indices, missing_configs

def flip_only_angles(active_list_lines: List[str], y: np.ndarray) -> Tuple[np.ndarray, List[int], List[tuple]]:
    """
    Rearranges the columns of the Acousto-Optic (AO) measurement matrix y by re-indexing 
    and routing the signals to invert ONLY the acoustic angles while keeping the 
    piezoelectric activation patterns unchanged, with inline angle inversion logic.
    """
    config_to_idx = {line.strip(): i for i, line in enumerate(active_list_lines) if line.strip()}
    new_indices = []
    missing_configs = []
    
    for i, line in enumerate(active_list_lines):
        line = line.strip()
        if not line:
            continue
            
        pattern, angle = line.split('_')
        
        # Inline: Sign inversion of the acoustic beam angle
        if angle == "000":
            flipped_angle = "000"
        elif angle.startswith("0"):
            flipped_angle = "1" + angle[1:]
        elif angle.startswith("1"):
            flipped_angle = "0" + angle[1:]
        else:
            raise ValueError(f"Unknown angle format: {angle}")
            
        target_config = f"{pattern}_{flipped_angle}"
        
        if target_config in config_to_idx:
            new_indices.append(config_to_idx[target_config])
        else:
            new_indices.append(i)
            missing_configs.append((i, line, target_config))
            
    return y[:, new_indices], new_indices, missing_configs

def flip_only_piezos(active_list_lines: List[str], y: np.ndarray) -> Tuple[np.ndarray, List[int], List[tuple]]:
    """
    Rearranges the columns of the Acousto-Optic (AO) measurement matrix y by re-indexing 
    and routing the signals to invert ONLY the piezoelectric activations (spatial mirror) 
    while keeping the acoustic angles unchanged, with inline bit-reversal logic.
    """
    config_to_idx = {line.strip(): i for i, line in enumerate(active_list_lines) if line.strip()}
    new_indices = []
    missing_configs = []
    
    for i, line in enumerate(active_list_lines):
        line = line.strip()
        if not line:
            continue
            
        pattern, angle = line.split('_')
        
        # Inline: Spatial mirror inversion of the 192-bit piezoelectric pattern
        bin_str = bin(int(pattern, 16))[2:].zfill(192)
        flipped_pattern = hex(int(bin_str[::-1], 2))[2:].zfill(48)
        
        target_config = f"{flipped_pattern}_{angle}"
        
        if target_config in config_to_idx:
            new_indices.append(config_to_idx[target_config])
        else:
            new_indices.append(i)
            missing_configs.append((i, line, target_config))
            
    return y[:, new_indices], new_indices, missing_configs