from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic import _Omega_QUADRATIC_CPU, _Omega_QUADRATIC_GPU
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.RelativeDifferences import _Omega_RELATIVE_DIFFERENCE_CPU, _Omega_RELATIVE_DIFFERENCE_GPU
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Huber import _Omega_HUBER_PIECEWISE_CPU, _Omega_HUBER_PIECEWISE_GPU
from AOT_biomaps.AOT_Recon.ReconTools import _build_adjacency_sparse, check_gpu_memory, calculate_memory_requirement
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

def MAPEM(
    SMatrix,
    y,
    Omega,
    beta,
    delta=None,
    gamma=None,
    sigma=None,
    numIterations=100,
    isSavingEachIteration=True,
    withTumor=True,
    max_saves=5000,
    show_logs=True
):
    """
    This method implements the MAPEM algorithm using CuPy for GPU acceleration.
    Multi-GPU and Multi-CPU modes are not implemented for this algorithm.
    """
    try:
        tumor_str = "WITH" if withTumor else "WITHOUT"
        # Auto-select device and method
        use_gpu = check_gpu_memory(config.select_best_gpu(), calculate_memory_requirement(SMatrix, y), show_logs=show_logs)
        # Dispatch to the appropriate implementation
        if use_gpu:
            return _MAPEM_GPU(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
                             isSavingEachIteration, tumor_str, max_saves, show_logs)
        else:
            return _MAPEM_CPU(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
                             isSavingEachIteration, tumor_str, max_saves, show_logs)
    except Exception as e:
        print(f"Error in MAPEM: {type(e).__name__}: {e}")
        return None, None

def MAPEM_STOP(
    SMatrix,
    y,
    Omega,
    beta,
    delta=None,
    gamma=None,
    sigma=None,
    numIterations=100,
    isSavingEachIteration=True,
    withTumor=True,
    max_saves=5000,
    show_logs=True
):
    """
    This method implements the MAPEM_STOP algorithm using CuPy for GPU acceleration.
    Stops when penalized log-likelihood stops increasing.
    """
    try:
        tumor_str = "WITH" if withTumor else "WITHOUT"
        use_gpu = check_gpu_memory(config.select_best_gpu(), calculate_memory_requirement(SMatrix, y), show_logs=show_logs)
        if use_gpu:
            return _MAPEM_GPU_STOP(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
                                  isSavingEachIteration, tumor_str, max_saves, show_logs)
        else:
            return _MAPEM_CPU_STOP(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
                                  isSavingEachIteration, tumor_str, max_saves, show_logs)
    except Exception as e:
        print(f"Error in MAPEM_STOP: {type(e).__name__}: {e}")
        return None, None

def _MAPEM_CPU_STOP(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
                   isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    CPU implementation of MAPEM with STOP condition using CuPy arrays.
    """
    try:
        # Validate potential type and parameters
        if not isinstance(Omega, PotentialType):
            raise TypeError(f"Omega must be of type PotentialType, got {type(Omega)}")

        if Omega == PotentialType.HUBER_PIECEWISE:
            if delta is None:
                raise ValueError("delta must be specified for HUBER_PIECEWISE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for HUBER_PIECEWISE potential type.")
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            if gamma is None:
                raise ValueError("gamma must be specified for RELATIVE_DIFFERENCE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for RELATIVE_DIFFERENCE potential type.")
        elif Omega == PotentialType.QUADRATIC:
            if sigma is None:
                raise ValueError("sigma must be specified for QUADRATIC potential type.")
            if beta is None:
                raise ValueError("beta must be specified for QUADRATIC potential type.")
        else:
            raise ValueError(f"Unknown potential type: {Omega}")

        SMatrix_cp = cp.asarray(SMatrix, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        T, Z, X, N = SMatrix.shape
        A_flat = SMatrix_cp.transpose(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_cp.reshape(-1)
        I_0 = cp.ones((Z, X), dtype=cp.float32)
        theta_list = [I_0]
        results = [I_0.get()]
        saved_indices = [0]
        normalization_factor = SMatrix_cp.sum(axis=(0, 3)).reshape(-1)
        adj_index, adj_values = _build_adjacency_sparse(Z, X)

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        # Set description based on potential type
        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) + STOP condition (penalized log-likelihood) ---- {tumor_str} TUMOR ---- processing on single CPU"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) + STOP condition (penalized log-likelihood) ---- {tumor_str} TUMOR ---- processing on single CPU"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) + STOP condition (penalized log-likelihood) ---- {tumor_str} TUMOR ---- processing on single CPU"

        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
        for it in iterator:
            theta_p = theta_list[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + cp.finfo(cp.float32).tiny)
            c_flat = A_flat.T @ e_flat

            # Select potential function
            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, U_value = _Omega_HUBER_PIECEWISE_CPU(theta_p_flat, adj_index, adj_values, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, U_value = _Omega_RELATIVE_DIFFERENCE_CPU(theta_p_flat, adj_index, adj_values, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, U_value = _Omega_QUADRATIC_CPU(theta_p_flat, adj_index, adj_values, sigma=sigma)

            denom = normalization_factor + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_next_flat = theta_p_flat + num / (denom + cp.finfo(cp.float32).tiny)
            theta_next_flat = cp.clip(theta_next_flat, a_min=0, a_max=None)
            theta_next = theta_next_flat.reshape(Z, X)
            theta_list[-1] = theta_next

            if isSavingEachIteration and it in save_indices:
                results.append(theta_next.get())
                saved_indices.append(it+1)

            log_likelihood = (y_flat * cp.log(q_flat + cp.finfo(cp.float32).tiny) - (q_flat + cp.finfo(cp.float32).tiny)).sum()
            penalized_log_likelihood = log_likelihood - beta * U_value

            if (it + 1) % 100 == 0:
                print(f"Iter {it+1}: logL={log_likelihood.get():.3e}, U={U_value.get():.3e}, penalized logL={penalized_log_likelihood.get():.3e}")

        if isSavingEachIteration:
            return results, saved_indices
        else:
            return results[-1], None
    except Exception as e:
        print(f"An error occurred in _MAPEM_CPU_STOP: {e}")
        return None, None

def _MAPEM_GPU_STOP(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
                   isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    GPU implementation of MAPEM with STOP condition using CuPy.
    """
    try:
        # Validate potential type and parameters
        if not isinstance(Omega, PotentialType):
            raise TypeError(f"Omega must be of type PotentialType, got {type(Omega)}")

        if Omega == PotentialType.HUBER_PIECEWISE:
            if delta is None:
                raise ValueError("delta must be specified for HUBER_PIECEWISE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for HUBER_PIECEWISE potential type.")
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            if gamma is None:
                raise ValueError("gamma must be specified for RELATIVE_DIFFERENCE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for RELATIVE_DIFFERENCE potential type.")
        elif Omega == PotentialType.QUADRATIC:
            if sigma is None:
                raise ValueError("sigma must be specified for QUADRATIC potential type.")
            if beta is None:
                raise ValueError("beta must be specified for QUADRATIC potential type.")
        else:
            raise ValueError(f"Unknown potential type: {Omega}")

        SMatrix_cp = cp.asarray(SMatrix, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        T, Z, X, N = SMatrix.shape
        J = Z * X
        A_flat = SMatrix_cp.transpose(0, 3, 1, 2).reshape(T * N, J)
        y_flat = y_cp.reshape(-1)
        theta_0 = cp.ones((Z, X), dtype=cp.float32)
        matrix_theta_cp = [theta_0]
        matrix_theta_from_gpu_MAPEM = [theta_0.get()]
        saved_indices = [0]
        normalization_factor = SMatrix_cp.sum(axis=(0, 3))
        normalization_factor_flat = normalization_factor.reshape(-1)
        previous = -np.inf
        nb_false_successive = 0
        adj_index, adj_values = _build_adjacency_sparse(Z, X)

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        # Set description based on potential type
        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) + STOP condition (penalized log-likelihood) ---- {tumor_str} TUMOR ---- processing on single GPU"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) + STOP condition (penalized log-likelihood) ---- {tumor_str} TUMOR ---- processing on single GPU"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) + STOP condition (penalized log-likelihood) ---- {tumor_str} TUMOR ---- processing on single GPU"

        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
        for it in iterator:
            theta_p = matrix_theta_cp[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + cp.finfo(cp.float32).tiny)
            c_flat = A_flat.T @ e_flat

            # Select potential function
            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, U_value = _Omega_HUBER_PIECEWISE_GPU(theta_p_flat, adj_index, adj_values, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, U_value = _Omega_RELATIVE_DIFFERENCE_GPU(theta_p_flat, adj_index, adj_values, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, U_value = _Omega_QUADRATIC_GPU(theta_p_flat, adj_index, adj_values, sigma=sigma)

            denom = normalization_factor_flat + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_p_plus_1_flat = theta_p_flat + num / (denom + cp.finfo(cp.float32).tiny)
            theta_p_plus_1_flat = cp.clip(theta_p_plus_1_flat, a_min=0, a_max=None)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_cp[-1] = theta_next

            if isSavingEachIteration and it in save_indices:
                matrix_theta_from_gpu_MAPEM.append(theta_next.get())
                saved_indices.append(it+1)

            log_likelihood = (y_flat * cp.log(q_flat + cp.finfo(cp.float32).tiny) - (q_flat + cp.finfo(cp.float32).tiny)).sum()
            penalized_log_likelihood = log_likelihood - beta * U_value

            if it == 0 or (it + 1) % 100 == 0:
                current = penalized_log_likelihood.get()
                if current <= previous:
                    nb_false_successive += 1
                else:
                    nb_false_successive = 0
                print(f"Iter {it + 1}: lnL without term ln(m_i !) inside={log_likelihood.get():.8e}, Gibbs energy function U={U_value.get():.4e}, penalized lnL without term ln(m_i !) inside={penalized_log_likelihood.get():.8e}, p lnL (current {current:.8e} - previous {previous:.8e} > 0)={(current - previous > 0)}, nb_false_successive={nb_false_successive}")
                previous = current

        del SMatrix_cp, y_cp, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        cp.cuda.Stream.null.synchronize()
        if isSavingEachIteration:
            return matrix_theta_from_gpu_MAPEM, saved_indices
        else:
            return matrix_theta_from_gpu_MAPEM[-1], None
    except Exception as e:
        print(f"An error occurred in _MAPEM_GPU_STOP: {e}")
        del SMatrix_cp, y_cp, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        cp.cuda.Stream.null.synchronize()
        return None, None

def _MAPEM_CPU(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
               isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    CPU implementation of MAPEM using CuPy arrays.
    """
    try:
        # Validate potential type and parameters
        if not isinstance(Omega, PotentialType):
            raise TypeError(f"Omega must be of type PotentialType, got {type(Omega)}")

        if Omega == PotentialType.HUBER_PIECEWISE:
            if delta is None:
                raise ValueError("delta must be specified for HUBER_PIECEWISE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for HUBER_PIECEWISE potential type.")
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            if gamma is None:
                raise ValueError("gamma must be specified for RELATIVE_DIFFERENCE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for RELATIVE_DIFFERENCE potential type.")
        elif Omega == PotentialType.QUADRATIC:
            if sigma is None:
                raise ValueError("sigma must be specified for QUADRATIC potential type.")
            if beta is None:
                raise ValueError("beta must be specified for QUADRATIC potential type.")
        else:
            raise ValueError(f"Unknown potential type: {Omega}")

        T, Z, X, N = SMatrix.shape
        A_flat = cp.asarray(SMatrix, dtype=cp.float32).transpose(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = cp.asarray(y, dtype=cp.float32).reshape(-1)
        theta_0 = cp.ones((Z, X), dtype=cp.float32)
        matrix_theta_np = [theta_0]
        I_reconMatrix = [theta_0.get()]
        saved_indices = [0]
        normalization_factor = SMatrix.sum(axis=(0, 3))
        normalization_factor_flat = cp.asarray(normalization_factor.reshape(-1), dtype=cp.float32)
        adj_index, adj_values = _build_adjacency_sparse(Z, X)

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        # Set description based on potential type
        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) ---- {tumor_str} TUMOR ---- processing on single CPU"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) ---- {tumor_str} TUMOR ---- processing on single CPU"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {tumor_str} TUMOR ---- processing on single CPU"

        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
        for it in iterator:
            theta_p = matrix_theta_np[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + cp.finfo(cp.float32).tiny)
            c_flat = A_flat.T @ e_flat

            # Select potential function
            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, _ = _Omega_HUBER_PIECEWISE_CPU(theta_p_flat, adj_index, adj_values, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, _ = _Omega_RELATIVE_DIFFERENCE_CPU(theta_p_flat, adj_index, adj_values, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, _ = _Omega_QUADRATIC_CPU(theta_p_flat, adj_index, adj_values, sigma=sigma)

            denom = normalization_factor_flat + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_p_plus_1_flat = theta_p_flat + num / (denom + cp.finfo(cp.float32).tiny)
            theta_p_plus_1_flat = cp.clip(theta_p_plus_1_flat, a_min=0, a_max=None)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_np.append(theta_next)
            if isSavingEachIteration and it in save_indices:
                I_reconMatrix.append(theta_next.get())
                saved_indices.append(it+1)

        if isSavingEachIteration:
            return I_reconMatrix, saved_indices
        else:
            return I_reconMatrix[-1], None
    except Exception as e:
        print(f"An error occurred in _MAPEM_CPU: {e}")
        return None, None

def _MAPEM_GPU(SMatrix, y, Omega, beta, delta, gamma, sigma, numIterations,
               isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    GPU implementation of MAPEM using CuPy.
    """
    try:
        # Validate potential type and parameters
        if not isinstance(Omega, PotentialType):
            raise TypeError(f"Omega must be of type PotentialType, got {type(Omega)}")

        if Omega == PotentialType.HUBER_PIECEWISE:
            if delta is None:
                raise ValueError("delta must be specified for HUBER_PIECEWISE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for HUBER_PIECEWISE potential type.")
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            if gamma is None:
                raise ValueError("gamma must be specified for RELATIVE_DIFFERENCE potential type.")
            if beta is None:
                raise ValueError("beta must be specified for RELATIVE_DIFFERENCE potential type.")
        elif Omega == PotentialType.QUADRATIC:
            if sigma is None:
                raise ValueError("sigma must be specified for QUADRATIC potential type.")
            if beta is None:
                raise ValueError("beta must be specified for QUADRATIC potential type.")
        else:
            raise ValueError(f"Unknown potential type: {Omega}")

        SMatrix_cp = cp.asarray(SMatrix, dtype=cp.float32)
        y_cp = cp.asarray(y, dtype=cp.float32)
        T, Z, X, N = SMatrix.shape
        J = Z * X
        A_flat = SMatrix_cp.transpose(0, 3, 1, 2).reshape(T * N, J)
        y_flat = y_cp.reshape(-1)
        theta_0 = cp.ones((Z, X), dtype=cp.float32)
        matrix_theta_cp = [theta_0]
        I_reconMatrix = [theta_0.get()]
        saved_indices = [0]
        normalization_factor = SMatrix_cp.sum(axis=(0, 3))
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

        # Set description based on potential type
        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) ---- {tumor_str} TUMOR ---- processing on single GPU"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) ---- {tumor_str} TUMOR ---- processing on single GPU"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC σ:{sigma:.4f}) ---- {tumor_str} TUMOR ---- processing on single GPU"

        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
        for it in iterator:
            theta_p = matrix_theta_cp[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + cp.finfo(cp.float32).tiny)
            c_flat = A_flat.T @ e_flat

            # Select potential function
            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, _ = _Omega_HUBER_PIECEWISE_GPU(theta_p_flat, adj_index, adj_values, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, _ = _Omega_RELATIVE_DIFFERENCE_GPU(theta_p_flat, adj_index, adj_values, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, _ = _Omega_QUADRATIC_GPU(theta_p_flat, adj_index, adj_values, sigma=sigma)

            denom = normalization_factor_flat + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_p_plus_1_flat = theta_p_flat + num / (denom + cp.finfo(cp.float32).tiny)
            theta_p_plus_1_flat = cp.clip(theta_p_plus_1_flat, a_min=0, a_max=None)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_cp.append(theta_next)
            if isSavingEachIteration and it in save_indices:
                I_reconMatrix.append(theta_next.get())
                saved_indices.append(it+1)

        del SMatrix_cp, y_cp, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        cp.cuda.Stream.null.synchronize()
        if isSavingEachIteration:
            return I_reconMatrix, saved_indices
        else:
            return I_reconMatrix[-1], None
    except Exception as e:
        print(f"An error occurred in _MAPEM_GPU: {e}")
        del SMatrix_cp, y_cp, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        cp.cuda.Stream.null.synchronize()
        return None, None