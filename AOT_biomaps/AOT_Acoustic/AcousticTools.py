from AOT_biomaps.Config import config

import os
import h5py
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

def loadmat(param_path_mat):
    """
    Charge un fichier .mat (format HDF5) sans SciPy.
    Args:
        param_path_mat: Chemin vers le fichier .mat.
    Returns:
        Dictionnaire contenant les variables du fichier.
    """
    with h5py.File(param_path_mat, 'r') as f:
        data = {}
        for key in f.keys():
            # Récupère les données et convertit en numpy array si nécessaire
            item = f[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]  # Convertit en numpy array
            elif isinstance(item, h5py.Group):
                # Pour les structures MATLAB (nested)
                data[key] = {}
                for subkey in item:
                    data[key][subkey] = item[subkey][()]
    return data

def reshape_field(field, factor, device=None):
    """
    Downsample a 3D or 4D field using PyTorch interpolation (auto-detects GPU/CPU).
    Args:
        field: Input field (numpy array or torch.Tensor).
        factor: Downsampling factor (tuple of ints).
        device: Force device ('cpu' or 'cuda'). If None, auto-detects GPU.
    Returns:
        Downsampled field (same type as input: numpy array or torch.Tensor).
    """
    # Check input
    if field is None:
        raise ValueError("Acoustic field is not generated. Please generate the field first.")

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device.lower()
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'.")

    # Convert to torch.Tensor if needed
    if isinstance(field, np.ndarray):
        field = torch.from_numpy(field)
    elif not isinstance(field, torch.Tensor):
        raise TypeError("Input must be a numpy array or torch.Tensor.")

    # Move to the target device
    field = field.to(device)

    # Add batch and channel dimensions (required by torch.interpolate)
    if len(factor) == 3:
        if field.dim() != 3:
            raise ValueError("Expected 3D field.")
        field = field.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Calculate new shape
        new_shape = [
            field.shape[2] // factor[0],
            field.shape[3] // factor[1],
            field.shape[4] // factor[2]
        ]

        # Trilinear interpolation
        downsampled = torch.nn.functional.interpolate(
            field,
            size=new_shape,
            mode='trilinear',
            align_corners=True
        )
        downsampled = downsampled.squeeze(0).squeeze(0)  # Remove batch/channel dims

    elif len(factor) == 4:
        if field.dim() != 4:
            raise ValueError("Expected 4D field.")
        field = field.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D, H, W)

        new_shape = [
            field.shape[2] // factor[0],
            field.shape[3] // factor[1],
            field.shape[4] // factor[2],
            field.shape[5] // factor[3]
        ]

        # Tetra-linear interpolation
        downsampled = torch.nn.functional.interpolate(
            field,
            size=new_shape,
            mode='trilinear',  # PyTorch uses 'trilinear' for both 3D and 4D
            align_corners=True
        )
        downsampled = downsampled.squeeze(0).squeeze(0)

    else:
        raise ValueError("Unsupported dimension. Only 3D and 4D fields are supported.")

    # Convert back to numpy if input was numpy
    if isinstance(field, np.ndarray):
        return downsampled.cpu().numpy()
    else:
        return downsampled


def CPU_hilbert(signal, axis=0):
    """
    Compute the Hilbert transform of a real signal using NumPy.

    Parameters:
    - signal: Input real signal (numpy.ndarray).
    - axis: Axis along which to compute the Hilbert transform.

    Returns:
    - analytic_signal: The analytic signal of the input.
    """
    fft_signal = np.fft.fftn(signal, axes=[axis])
    h = np.zeros_like(signal)

    if axis == 0:
        h[0 : signal.shape[0] // 2 + 1, ...] = 1
        h[signal.shape[0] // 2 + 1 :, ...] = 2
    else:
        raise ValueError("Axis not supported for this implementation.")

    analytic_signal = np.fft.ifftn(fft_signal * h, axes=[axis])
    return analytic_signal

def GPU_hilbert(signal, axis=0):
    """
    Compute the Hilbert transform of a real signal using PyTorch.

    Parameters:
    - signal: Input real signal (torch.Tensor).
    - axis: Axis along which to compute the Hilbert transform.

    Returns:
    - analytic_signal: The analytic signal of the input.
    """
    fft_signal = torch.fft.fftn(signal, dim=axis)
    h = torch.zeros_like(signal)
    if axis == 0:
        h[0 : signal.shape[0] // 2 + 1, ...] = 1
        h[signal.shape[0] // 2 + 1 :, ...] = 2
    else:
        raise ValueError("Axis not supported for this implementation.")

    analytic_signal = torch.fft.ifftn(fft_signal * h, dim=axis)
    return analytic_signal

def calculate_envelope_squared(field, isGPU):
    """
    Calculate the analytic envelope of the acoustic field using either CPU or GPU with PyTorch.
    Parameters:
        - field: Input acoustic field (numpy.ndarray or torch.Tensor).
        - isGPU (bool): If True, use GPU for computation. Otherwise, use CPU.
    Returns:
        - envelope (numpy.ndarray): The squared analytic envelope of the acoustic field.
    """
    try:
        if field is None:
            raise ValueError("Acoustic field is not generated. Please generate the field first.")

        # Convert input to tensor (handle both numpy arrays and tensors)
        if isinstance(field, np.ndarray):
            acoustic_field = torch.from_numpy(field).to(dtype=torch.float32)
        else:
            acoustic_field = field.detach().clone().to(dtype=torch.float32)

        # Handle GPU/CPU transfer
        if isGPU:
            if not torch.cuda.is_available():
                print("CUDA is not available, falling back to CPU.")
                isGPU = False
            else:
                # Check GPU memory
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                required_memory = acoustic_field.numel() * acoustic_field.element_size()
                if free_memory < required_memory:
                    print(f"GPU memory insufficient ({required_memory / (1024 ** 2):.2f} MB required, {free_memory / (1024 ** 2):.2f} MB free), falling back to CPU.")
                    isGPU = False
                else:
                    acoustic_field = acoustic_field.cuda()

        if len(acoustic_field.shape) not in [3, 4]:
            raise ValueError("Input acoustic field must be a 3D or 4D array.")

        def process_slice(slice_index):
            """Calculate the envelope for a given slice of the acoustic field."""
            slice_data = acoustic_field[slice_index]

            if len(acoustic_field.shape) == 3:
                if isGPU:
                    return torch.abs(GPU_hilbert(slice_data, axis=0))**2
                else:
                    return torch.from_numpy(np.abs(CPU_hilbert(slice_data.cpu().numpy(), axis=0))**2).to(dtype=torch.float32)

            elif len(acoustic_field.shape) == 4:
                if isGPU:
                    return torch.stack([
                        torch.abs(GPU_hilbert(slice_data[:, y, z], axis=0))**2
                        for y in range(slice_data.shape[1])
                        for z in range(slice_data.shape[2])
                    ]).reshape(slice_data.shape[1], slice_data.shape[2], -1).permute(2, 0, 1)
                else:
                    envelope = torch.zeros_like(slice_data)
                    for y in range(slice_data.shape[1]):
                        for z in range(slice_data.shape[2]):
                            envelope[:, y, z] = torch.from_numpy(
                                np.abs(CPU_hilbert(slice_data[:, y, z].cpu().numpy(), axis=0))**2
                            )
                    return envelope

        # Process slices
        num_slices = acoustic_field.shape[0]
        slice_indices = range(num_slices)

        if isGPU:
            envelopes = [process_slice(i) for i in slice_indices]
        else:
            with ThreadPoolExecutor() as executor:
                envelopes = list(executor.map(process_slice, slice_indices))

        # Combine results
        envelope = torch.stack(envelopes, axis=0)
        return envelope.cpu().numpy() if isGPU else envelope.numpy()

    except Exception as e:
        print(f"Error in calculate_envelope_squared method: {e}")
        raise


def getPattern(pathFile):
    """
    Get the pattern from a file path.

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
    binary_string = bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)
    
    # Trouver la plus longue séquence de 0 consécutifs
    zeros_groups = [len(s) for s in binary_string.split('1')]
    space_0 = max(zeros_groups) if zeros_groups else 0

    # Trouver la plus longue séquence de 1 consécutifs
    ones_groups = [len(s) for s in binary_string.split('0')]
    space_1 = max(ones_groups) if ones_groups else 0

    return space_0, space_1

def getAngle(pathFile):
    """
    Get the angle from a file path.

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

def next_power_of_2(n):
    """Calculate the next power of 2 greater than or equal to n."""
    return int(2 ** np.ceil(np.log2(n)))
        
