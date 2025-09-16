from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.AOT_Recon.ReconTools import _build_adjacency_sparse_CPU, _build_adjacency_sparse_GPU
from AOT_biomaps.Config import config

import numpy as np
import torch
from tqdm import trange
if config.get_process()  == 'gpu':
    try:
        from torch_scatter import scatter
    except ImportError:
        raise ImportError("torch_scatter and torch_sparse are required for GPU processing. Please install them using 'pip install torch-scatter torch-sparse' with correct link (follow instructions https://github.com/LucasDuclos/AcoustoOpticTomography/edit/main/README.md).")


def _DEPIERRO_GPU(SMatrix, y, Omega, numIterations, beta, sigma, isSavingEachIteration, withTumor):
    try:
        if Omega != PotentialType.QUADRATIC:
            raise ValueError("Depierro95 optimizer only supports QUADRATIC potential function.")
        if beta is None or sigma is None:
            raise ValueError("Depierro95 optimizer requires beta and sigma parameters.")

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

        description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: DE PIERRO (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"


        for p in trange(numIterations, desc=description):
            theta_p = matrix_theta_torch[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = y_flat / (q_flat + torch.finfo(torch.float32).tiny)
            c_flat = A_flat.T @ e_flat
            theta_EM_p_flat = theta_p_flat * c_flat
            alpha_j = normalization_factor_flat
            W_j = scatter(adj_values, adj_index[0], dim=0, dim_size=J, reduce='sum') * (1.0 / sigma**2)
            theta_k = theta_p_flat[adj_index[1]]
            weighted_theta_k = theta_k * adj_values
            gamma_j = theta_p_flat * W_j + scatter(weighted_theta_k, adj_index[0], dim=0, dim_size=J, reduce='sum')
            A = 2 * beta * W_j
            B = -beta * gamma_j + alpha_j
            C = -theta_EM_p_flat
            theta_p_plus_1_flat = (-B + torch.sqrt(B ** 2 - 4 * A * C)) / (2 * A + torch.finfo(torch.float32).tiny)
            theta_p_plus_1_flat = torch.clamp(theta_p_plus_1_flat, min=0)
            theta_next = theta_p_plus_1_flat.reshape(Z, X)
            matrix_theta_torch[-1] = theta_next

            if p % 1 == 0:
                I_reconMatrix.append(theta_next.cpu().numpy())

            del theta_p_flat, q_flat, e_flat, c_flat, theta_EM_p_flat, theta_p_plus_1_flat, theta_next
            torch.cuda.empty_cache()

        del A_matrix_torch, y_torch, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        torch.cuda.empty_cache()

        if isSavingEachIteration:
            return I_reconMatrix
        else:
            return I_reconMatrix[-1]

    except Exception as e:
        print(f"An error occurred in _DEPIERRO_GPU: {e}")
        del A_matrix_torch, y_torch, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
        torch.cuda.empty_cache()
        return None

def _DEPIERRO_CPU(SMatrix, y, Omega, numIterations, beta, sigma, isSavingEachIteration, withTumor):
    try:
        if Omega != PotentialType.QUADRATIC:
            raise ValueError("Depierro95 optimizer only supports QUADRATIC potential function.")
        if beta is None or sigma is None:
            raise ValueError("Depierro95 optimizer requires beta and sigma parameters.")

        A_matrix = np.array(SMatrix, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        T, Z, X, N = SMatrix.shape
        J = Z * X
        A_flat = A_matrix.transpose(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_array.reshape(-1)
        theta_0 = np.ones((Z, X), dtype=np.float32)
        matrix_theta = [theta_0]
        I_reconMatrix = [theta_0.copy()]
        normalization_factor = A_matrix.sum(axis=(0, 3))
        normalization_factor_flat = normalization_factor.reshape(-1)
        adj_index, adj_values = _build_adjacency_sparse_CPU(Z, X)

        description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: DE PIERRO (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU ----"


        for p in trange(numIterations, desc=description):
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

            if p % 1 == 0:
                I_reconMatrix.append(theta_next.copy())

        if isSavingEachIteration:
            return I_reconMatrix
        else:
            return I_reconMatrix[-1]

    except Exception as e:
        print(f"An error occurred in _DEPIERRO_CPU: {e}")
        return None
