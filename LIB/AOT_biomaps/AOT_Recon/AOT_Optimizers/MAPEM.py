
from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Quadratic import _Omega_QUADRATIC_CPU, _Omega_QUADRATIC_GPU
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.RelativeDifferences import _Omega_RELATIVE_DIFFERENCE_CPU, _Omega_RELATIVE_DIFFERENCE_GPU
from AOT_biomaps.AOT_Recon.AOT_PotentialFunctions.Huber import _Omega_HUBER_PIECEWISE_CPU, _Omega_HUBER_PIECEWISE_GPU
from AOT_biomaps.AOT_Recon.ReconTools import _build_adjacency_sparse_CPU, _build_adjacency_sparse_GPU
from AOT_biomaps.Config import config

import numpy as np
import torch
from tqdm import trange


def _MAPEM_CPU_STOP(SMatrix, y, Omega, numIterations, beta, delta, gamma, sigma, isSavingEachIteration, withTumor):
    """
    MAPEM version CPU simple - sans GPU - torch uniquement
    """
    try:
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

        SMatrix = torch.tensor(SMatrix, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        T, Z, X, N = SMatrix.shape
        A_flat = SMatrix.permute(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_tensor.reshape(-1)
        I_0 = torch.ones((Z, X), dtype=torch.float32)
        theta_list = [I_0]
        results = [I_0.numpy()]
        previous = -np.inf
        normalization_factor = SMatrix.sum(dim=(0, 3)).reshape(-1)
        adj_index, adj_values = _build_adjacency_sparse_CPU(Z, X)

        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) + STOP condition (penalized log-likelihood) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) + STOP condition (penalized log-likelihood) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) + STOP condition (penalized log-likelihood) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"

        for p in trange(numIterations, desc=description):
            theta_p = theta_list[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + torch.finfo(torch.float32).tiny)
            c_flat = A_flat.T @ e_flat

            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, U_value = _Omega_HUBER_PIECEWISE_CPU(theta_p_flat, adj_index, adj_values, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, U_value = _Omega_RELATIVE_DIFFERENCE_CPU(theta_p_flat, adj_index, adj_values, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, U_value = _Omega_QUADRATIC_CPU(theta_p_flat, adj_index, adj_values, sigma=sigma)

            denom = normalization_factor + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_next_flat = theta_p_flat + num / (denom + torch.finfo(torch.float32).tiny)
            theta_next_flat = torch.clamp(theta_next_flat, min=0)
            theta_next = theta_next_flat.reshape(Z, X)
            theta_list[-1] = theta_next
            results.append(theta_next.numpy())

            log_likelihood = (y_flat * torch.log(q_flat + 1e-8) - (q_flat + 1e-8)).sum()
            penalized_log_likelihood = log_likelihood - beta * U_value
            current = penalized_log_likelihood.item()

            if (p + 1) % 100 == 0:
                print(f"Iter {p+1}: logL={log_likelihood:.3e}, U={U_value:.3e}, penalized logL={penalized_log_likelihood:.3e}")

        if isSavingEachIteration:
            return results[-1]
        else:
            return results
    except Exception as e:
        print(f"An error occurred in _MAPEM_CPU_STOP: {e}")
        return None

def _MAPEM_GPU_STOP(SMatrix, y, Omega, numIterations, beta, delta, gamma, sigma, isSavingEachIteration, withTumor):
    """
    Maximum A Posteriori (MAP) estimation for Bayesian reconstruction.
    This method computes the MAP estimate of the parameters given the data.
    """
    try:
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

        device = torch.device(f"cuda:{config.select_best_gpu()}")
        A_matrix_torch = torch.tensor(SMatrix, dtype=torch.float32).to(device)
        y_torch = torch.tensor(y, dtype=torch.float32).to(device)
        T, Z, X, N = SMatrix.shape
        J = Z * X
        A_flat = A_matrix_torch.permute(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_torch.reshape(-1)
        I_0 = torch.ones((Z, X), dtype=torch.float32, device=device)
        matrix_theta_torch = [I_0]
        matrix_theta_from_gpu_MAPEM = [I_0.cpu().numpy()]
        normalization_factor = A_matrix_torch.sum(dim=(0, 3))
        normalization_factor_flat = normalization_factor.reshape(-1)
        previous = -np.inf
        nb_false_successive = 0
        adj_index, adj_values = _build_adjacency_sparse_GPU(Z, X, device=device)

        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) + STOP condition (penalized log-likelihood) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) + STOP condition (penalized log-likelihood) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) + STOP condition (penalized log-likelihood) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"

        for p in trange(numIterations, desc=description):
            theta_p = matrix_theta_torch[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + torch.finfo(torch.float32).tiny)
            c_flat = A_flat.T @ e_flat

            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, U_value = _Omega_HUBER_PIECEWISE_GPU(theta_p_flat, adj_index, adj_values, device=device, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, U_value = _Omega_RELATIVE_DIFFERENCE_GPU(theta_p_flat, adj_index, adj_values, device=device, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, U_value = _Omega_QUADRATIC_GPU(theta_p_flat, adj_index, adj_values, device=device, sigma=sigma)
            else:
                raise ValueError(f"Unknown potential type: {Omega}")

            denom = normalization_factor_flat + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_p_plus_1_flat = theta_p_flat + num / (denom + torch.finfo(torch.float32).tiny)
            theta_p_plus_1_flat = torch.clamp(theta_p_plus_1_flat, min=0)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_torch[-1] = theta_next

            if p % 1 == 0:
                matrix_theta_from_gpu_MAPEM.append(theta_next.cpu().numpy())

            log_likelihood = (y_flat * (torch.log(q_flat + torch.finfo(torch.float32).tiny)) - (q_flat + torch.finfo(torch.float32).tiny)).sum()
            penalized_log_likelihood = log_likelihood - beta * U_value

            if p == 0 or (p + 1) % 100 == 0:
                current = penalized_log_likelihood.item()
                if current <= previous:
                    nb_false_successive += 1
                else:
                    nb_false_successive = 0
                print(f"Iter {p + 1}: lnL without term ln(m_i !) inside={log_likelihood.item():.8e}, Gibbs energy function U={U_value.item():.4e}, penalized lnL without term ln(m_i !) inside={penalized_log_likelihood.item():.8e}, p lnL (current {current:.8e} - previous {previous:.8e} > 0)={(current - previous > 0)}, nb_false_successive={nb_false_successive}")
                previous = current

        del A_matrix_torch, y_torch, A_flat, y_flat, I_0, normalization_factor, normalization_factor_flat
        torch.cuda.empty_cache()

        if isSavingEachIteration:
            return matrix_theta_from_gpu_MAPEM
        else:
            return matrix_theta_from_gpu_MAPEM[-1]
    except Exception as e:
        print(f"An error occurred in _MAPEM_GPU_STOP: {e}")
        del A_matrix_torch, y_torch, A_flat, y_flat, I_0, normalization_factor, normalization_factor_flat
        torch.cuda.empty_cache()
        return None

def _MAPEM_CPU(SMatrix, y, Omega, numIterations, beta, delta, gamma, sigma, isSavingEachIteration, withTumor):
    try:
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
        J = Z * X
        A_flat = np.transpose(SMatrix, (0, 3, 1, 2)).reshape(T * N, Z * X)
        y_flat = y.reshape(-1)
        theta_0 = np.ones((Z, X), dtype=np.float32)
        matrix_theta_np = [theta_0]
        I_reconMatrix = [theta_0.copy()]
        normalization_factor = SMatrix.sum(axis=(0, 3))
        normalization_factor_flat = normalization_factor.reshape(-1)
        adj_index, adj_values = _build_adjacency_sparse_CPU(Z, X)

        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"

        for p in trange(numIterations, desc=description):
            theta_p = matrix_theta_np[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + np.finfo(np.float32).tiny)
            c_flat = A_flat.T @ e_flat

            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, _ = _Omega_HUBER_PIECEWISE_CPU(theta_p_flat, adj_index, adj_values, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, _ = _Omega_RELATIVE_DIFFERENCE_CPU(theta_p_flat, adj_index, adj_values, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, _ = _Omega_QUADRATIC_CPU(theta_p_flat, adj_index, adj_values, sigma=sigma)

            denom = normalization_factor_flat + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_p_plus_1_flat = theta_p_flat + num / (denom + np.finfo(np.float32).tiny)
            theta_p_plus_1_flat = np.clip(theta_p_plus_1_flat, 0, None)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_np.append(theta_next)

            if p % 1 == 0:
                I_reconMatrix.append(theta_next.copy())

        if isSavingEachIteration:
            return I_reconMatrix
        else:
            return I_reconMatrix[-1]
    except Exception as e:
        print(f"An error occurred in _MAPEM_CPU: {e}")
        return None

def _MAPEM_GPU(SMatrix, y, Omega, numIterations, beta, delta, gamma, sigma, isSavingEachIteration, withTumor):
    """
    Maximum A Posteriori (MAP) estimation for Bayesian reconstruction using GPU.
    This method computes the MAP estimate of the parameters given the data.
    """
    try:
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

        device = torch.device(f"cuda:{config.select_best_gpu()}")
        A_matrix_torch = torch.tensor(SMatrix, dtype=torch.float32).to(device)
        y_torch = torch.tensor(y, dtype=torch.float32).to(device)
        T, Z, X, N = SMatrix.shape
        J = Z * X
        A_flat = A_matrix_torch.permute(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_torch.reshape(-1)
        theta_0 = torch.ones((Z, X), dtype=torch.float32, device=device)
        matrix_theta_torch = [theta_0]
        I_reconMatrix = [theta_0.cpu().numpy()]
        normalization_factor = A_matrix_torch.sum(dim=(0, 3))
        normalization_factor_flat = normalization_factor.reshape(-1)
        adj_index, adj_values = _build_adjacency_sparse_GPU(Z, X, device=device)

        if Omega == PotentialType.HUBER_PIECEWISE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse HUBER β:{beta:.4f}, δ:{delta:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"
        elif Omega == PotentialType.RELATIVE_DIFFERENCE:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse RD β:{beta:.4f}, γ:{gamma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"
        elif Omega == PotentialType.QUADRATIC:
            description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: MAP-EM (Sparse QUADRATIC σ:{sigma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"

        for p in trange(numIterations, desc=description):
            theta_p = matrix_theta_torch[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = (y_flat - q_flat) / (q_flat + torch.finfo(torch.float32).tiny)
            c_flat = A_flat.T @ e_flat

            if Omega == PotentialType.HUBER_PIECEWISE:
                grad_U, hess_U, _ = _Omega_HUBER_PIECEWISE_GPU(theta_p_flat, adj_index, adj_values, device=device, delta=delta)
            elif Omega == PotentialType.RELATIVE_DIFFERENCE:
                grad_U, hess_U, _ = _Omega_RELATIVE_DIFFERENCE_GPU(theta_p_flat, adj_index, adj_values, device=device, gamma=gamma)
            elif Omega == PotentialType.QUADRATIC:
                grad_U, hess_U, _ = _Omega_QUADRATIC_GPU(theta_p_flat, adj_index, adj_values, device=device, sigma=sigma)
            else:
                raise ValueError(f"Unknown potential type: {Omega}")

            denom = normalization_factor_flat + theta_p_flat * beta * hess_U
            num = theta_p_flat * (c_flat - beta * grad_U)
            theta_p_plus_1_flat = theta_p_flat + num / (denom + torch.finfo(torch.float32).tiny)
            theta_p_plus_1_flat = torch.clamp(theta_p_plus_1_flat, min=0)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_torch.append(theta_next)

            if p % 1 == 0:
                I_reconMatrix.append(theta_next.cpu().numpy())

        del A_matrix_torch, y_torch, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        torch.cuda.empty_cache()

        if isSavingEachIteration:
            return I_reconMatrix
        else:
            return I_reconMatrix[-1]
    except Exception as e:
        print(f"An error occurred in _MAPEM_GPU: {e}")
        del A_matrix_torch, y_torch, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        torch.cuda.empty_cache()
        return None
