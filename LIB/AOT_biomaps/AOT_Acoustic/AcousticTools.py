from AOT_biomaps.Config import config

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RegularGridInterpolator
import torch
import numpy as np
import re


def reshape_field(field,factor):
    """
    Downsample the acoustic field using interpolation to reduce its size for faster processing.
    This method uses interpolation to estimate values on a coarser grid.
    """
    try:
        if field is None:
            raise ValueError("Acoustic field is not generated. Please generate the field first.")

        if len(factor) == 3:
            # Create new grid for 3D field
            x = np.arange(field.shape[0])
            y = np.arange(field.shape[1])
            z = np.arange(field.shape[2])

            # Create interpolating function
            interpolator = RegularGridInterpolator((x, y, z), field)

            # Create new coarser grid points
            x_new = np.linspace(0, field.shape[0] - 1, field.shape[0] // factor[0])
            y_new = np.linspace(0, field.shape[1] - 1, field.shape[1] // factor[1])
            z_new = np.linspace(0, field.shape[2] - 1, field.shape[2] // factor[2])

            # Create meshgrid for new points
            x_grid, y_grid, z_grid = np.meshgrid(x_new, y_new, z_new, indexing='ij')

            # Interpolate values
            points = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()), axis=-1)
            smoothed_field = interpolator(points).reshape(x_grid.shape)

            return smoothed_field

        elif len(factor) == 4:
            # Create new grid for 4D field
            x = np.arange(field.shape[0])
            y = np.arange(field.shape[1])
            z = np.arange(field.shape[2])
            w = np.arange(field.shape[3])

            # Create interpolating function
            interpolator = RegularGridInterpolator((x, y, z, w), field)

            # Create new coarser grid points
            x_new = np.linspace(0, field.shape[0] - 1, field.shape[0] // factor[0])
            y_new = np.linspace(0, field.shape[1] - 1, field.shape[1] // factor[1])
            z_new = np.linspace(0, field.shape[2] - 1, field.shape[2] // factor[2])
            w_new = np.linspace(0, field.shape[3] - 1, field.shape[3] // factor[3])

            # Create meshgrid for new points
            x_grid, y_grid, z_grid, w_grid = np.meshgrid(x_new, y_new, z_new, w_new, indexing='ij')

            # Interpolate values
            points = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), w_grid.flatten()), axis=-1)
            smoothed_field = interpolator(points).reshape(x_grid.shape)

            return smoothed_field

        else:
            raise ValueError("Invalid dimension for downsampling. Supported dimensions are: 3D, 4D.")

    except Exception as e:
        print(f"Error in interpolate_reshape_field method: {e}")
        raise

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
    - field: Input acoustic field.
    - isGPU (bool): If True, use GPU for computation. Otherwise, use CPU.
    Returns:
    - envelope (numpy.ndarray): The squared analytic envelope of the acoustic field.
    """
    try:
        if field is None:
            raise ValueError("Acoustic field is not generated. Please generate the field first.")

        if isGPU:
            # Check GPU memory
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                required_memory = field.nbytes
                if free_memory < required_memory:
                    print(f"GPU memory insufficient {required_memory / (1024 ** 2)} MB, Free GPU memory: {free_memory / (1024 ** 2)} MB, falling back to CPU.")
                    isGPU = False
                    acoustic_field = torch.tensor(field, dtype=torch.float32)
                else:
                    acoustic_field = torch.tensor(field, dtype=torch.float32).cuda()
            else:
                print("CUDA is not available, falling back to CPU.")
                isGPU = False
                acoustic_field = torch.tensor(field, dtype=torch.float32)
        else:
            acoustic_field = torch.tensor(field, dtype=torch.float32)

        if len(acoustic_field.shape) not in [3, 4]:
            raise ValueError("Input acoustic field must be a 3D or 4D array.")

        def process_slice(slice_index, isGPU):
            """Calculate the envelope for a given slice of the acoustic field."""
            slice_data = acoustic_field[slice_index]

            if isGPU:
                # Use GPU_hilbert for GPU computation
                envelope_slice = torch.abs(GPU_hilbert(slice_data, axis=0))**2
            else:
                # Move to CPU for CPU computation
                slice_data = slice_data.cpu()
                envelope_slice = torch.tensor(np.abs(CPU_hilbert(slice_data.numpy(), axis=0))**2, dtype=torch.float32)

            if len(acoustic_field.shape) == 3:
                return envelope_slice
            elif len(acoustic_field.shape) == 4:
                envelope = torch.zeros_like(slice_data)
                for y in range(slice_data.shape[1]):
                    for z in range(slice_data.shape[2]):
                        if isGPU:
                            envelope[:, y, z] = torch.abs(GPU_hilbert(slice_data[:, y, z], axis=0))**2
                        else:
                            envelope[:, y, z] = torch.tensor(np.abs(CPU_hilbert(slice_data[:, y, z].cpu().numpy(), axis=0))**2, dtype=torch.float32)
                return envelope

        # Determine the number of slices to process in parallel
        num_slices = acoustic_field.shape[0]
        slice_indices = range(num_slices)

        if isGPU:
            # Use GPU directly without multithreading
            envelopes = [process_slice(slice_index, isGPU) for slice_index in slice_indices]
        else:
            # Use ThreadPoolExecutor to parallelize the computation on CPU
            with ThreadPoolExecutor() as executor:
                envelopes = list(executor.map(lambda index: process_slice(index, isGPU), slice_indices))

        # Combine the results into a single array
        envelope = torch.stack(envelopes, axis=0)
        return envelope.numpy() if not isGPU else envelope.cpu().numpy()

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
        
