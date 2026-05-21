from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.AOT_Recon.ReconTools import _build_adjacency_sparse, calculate_memory_requirement, check_gpu_memory
from AOT_biomaps.Config import config

import numpy as np
from tqdm import trange

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

def DEPIERRO(
        SMatrix,
        y,
        numIterations,
        beta,
        sigma,
        isSavingEachIteration,
        withTumor,
        max_saves,
        show_logs):
    """
    This method implements the DEPIERRO algorithm using either CPU or single-GPU CuPy acceleration.
    Multi-GPU and Multi-CPU modes are not implemented for this algorithm.
    """
    try:
        tumor_str = "WITH" if withTumor else "WITHOUT"
        # Auto-select device and method
        use_gpu = check_gpu_memory(config.select_best_gpu(), calculate_memory_requirement(SMatrix, y), show_logs=show_logs)
        # Dispatch to the appropriate implementation
        if use_gpu:
            return _DEPIERRO_GPU(SMatrix, y, numIterations, beta, sigma, isSavingEachIteration, tumor_str, max_saves, show_logs)
        else:
            return _DEPIERRO_CPU(SMatrix, y, numIterations, beta, sigma, isSavingEachIteration, tumor_str, max_saves, show_logs)
    except Exception as e:
        print(f"Error in DEPIERRO: {type(e).__name__}: {e}")
        return None, None

def _DEPIERRO_GPU(SMatrix, y, numIterations, beta, sigma, isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    # Convert data to CuPy arrays (float64)
    A_matrix_cp = cp.array(SMatrix, dtype=cp.float64)
    y_cp = cp.array(y, dtype=cp.float64)
    # Dimensions
    T, Z, X, N = SMatrix.shape
    J = Z * X
    # Reshape matrices
    A_flat = A_matrix_cp.transpose(0, 3, 1, 2).reshape(T * N, J)
    y_flat = y_cp.reshape(-1)
    # Initialize theta
    theta_0 = cp.ones((Z, X), dtype=cp.float64)
    matrix_theta_cp = [theta_0.copy()]  # Copy to avoid references
    I_reconMatrix = [theta_0.get()]  # Get numpy array for CPU storage
    # Normalization factor
    normalization_factor = A_matrix_cp.sum(axis=(0, 3))
    normalization_factor_flat = normalization_factor.reshape(-1)
    # Build adjacency matrix
    adj_index, adj_values = _build_adjacency_sparse(Z, X, dtype=cp.float64)
    # Ensure arrays are CuPy arrays
    if isinstance(adj_index, list):
        adj_index = [cp.array(idx) for idx in adj_index]
    else:
        adj_index = cp.array(adj_index)
    adj_values = cp.array(adj_values)

    # Description for progress bar
    description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: DE PIERRO (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {tumor_str} TUMOR ---- processing on single GPU no.{cp.cuda.runtime.getDevice()}"
    # Configuration for saving iterations
    saved_indices = [0]

    # Calculate save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)

    # Main MAP-EM loop
    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    for it in iterator:
        theta_p = matrix_theta_cp[-1]
        theta_p_flat = theta_p.reshape(-1)
        # Step 1: Forward projection
        q_flat = A_flat @ theta_p_flat
        q_flat = q_flat + cp.finfo(cp.float64).tiny  # Avoid division by zero
        # Step 2: Error estimation
        e_flat = y_flat / q_flat
        # Step 3: Backprojection of error
        c_flat = A_flat.T @ e_flat
        # Step 4: Multiplicative update (EM)
        theta_EM_p_flat = theta_p_flat * c_flat
        # Step 5: Calculate W_j and gamma_j
        # Use bincount for scatter sum operation
        W_j = cp.bincount(adj_index[0].get(), weights=adj_values.get(), minlength=J) * (1.0 / (sigma**2))
        W_j = cp.array(W_j)
        theta_k = theta_p_flat[adj_index[1].get()]
        weighted_theta_k = theta_k * adj_values
        gamma_j = theta_p_flat * W_j + cp.bincount(adj_index[0].get(), weights=weighted_theta_k.get(), minlength=J)
        gamma_j = cp.array(gamma_j)
        # Step 6: De Pierro update (quadratic resolution)
        A_coeff = 2 * beta * W_j
        B = -beta * gamma_j + normalization_factor_flat
        C = -theta_EM_p_flat
        discriminant = B**2 - 4 * A_coeff * C
        discriminant = cp.clip(discriminant, a_min=0, a_max=None)
        theta_p_plus_1_flat = (-B + cp.sqrt(discriminant)) / (2 * A_coeff + cp.finfo(cp.float64).tiny)
        theta_p_plus_1_flat = cp.clip(theta_p_plus_1_flat, a_min=0, a_max=None)
        # Step 7: Update theta
        theta_next = theta_p_plus_1_flat.reshape(Z, X)
        matrix_theta_cp.append(theta_next)  # Add new iteration
        # Conditional saving
        if isSavingEachIteration and it in save_indices:
            I_reconMatrix.append(theta_next.get())
            saved_indices.append(it)
        # Partial memory cleanup
        del theta_p_flat, q_flat, e_flat, c_flat, theta_EM_p_flat, theta_p_plus_1_flat
        cp.cuda.Stream.null.synchronize()

    # Final cleanup
    del A_matrix_cp, y_cp, A_flat, y_flat, normalization_factor, normalization_factor_flat
    del adj_index, adj_values
    cp.cuda.Stream.null.synchronize()
    # Return result
    if isSavingEachIteration:
        return I_reconMatrix, saved_indices
    else:
        return matrix_theta_cp[-1].get(), None

def _DEPIERRO_CPU(SMatrix, y, numIterations, beta, sigma, isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    CPU implementation of the DEPIERRO algorithm using NumPy.
    """
    try:
        if beta is None or sigma is None:
            raise ValueError("DEPIERRO optimizer requires beta and sigma parameters.")

        A_matrix = np.array(SMatrix, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        T, Z, X, N = SMatrix.shape
        J = Z * X
        A_flat = A_matrix.transpose(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_array.reshape(-1)
        theta_0 = np.ones((Z, X), dtype=np.float32)
        matrix_theta = [theta_0]
        I_reconMatrix = [theta_0.copy()]
        saved_indices = [0]
        normalization_factor = A_matrix.sum(axis=(0, 3))
        normalization_factor_flat = normalization_factor.reshape(-1)
        adj_index, adj_values = _build_adjacency_sparse(Z, X)

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: DE PIERRO (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {tumor_str} TUMOR ---- processing on single CPU"

        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
        for it in iterator:
            theta_p = matrix_theta[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = np.dot(A_flat, theta_p_flat)
            e_flat = y_flat / (q_flat + np.finfo(np.float32).tiny)
            c_flat = np.dot(A_flat.T, e_flat)
            theta_EM_p_flat = theta_p_flat * c_flat
            alpha_j = normalization_factor_flat
            W_j = np.bincount(adj_index[0], weights=adj_values, minlength=J) * (1.0 / sigma**2)
            theta_k = theta_p_flat[adj_index[1]]
            weighted_theta_k = theta_k * adj_values
            gamma_j = theta_p_flat * W_j + np.bincount(adj_index[0], weights=weighted_theta_k, minlength=J)
            A = 2 * beta * W_j
            B = -beta * gamma_j + alpha_j
            C = -theta_EM_p_flat
            theta_p_plus_1_flat = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A + np.finfo(np.float32).tiny)
            theta_p_plus_1_flat = np.clip(theta_p_plus_1_flat, a_min=0, a_max=None)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta[-1] = theta_next
            if isSavingEachIteration and it in save_indices:
                I_reconMatrix.append(theta_next.copy())
                saved_indices.append(it)

        if isSavingEachIteration:
            return I_reconMatrix, saved_indices
        else:
            return I_reconMatrix[-1], None
    except Exception as e:
        print(f"An error occurred in _DEPIERRO_CPU: {e}")
        return None, None