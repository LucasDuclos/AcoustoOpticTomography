from AOT_biomaps.AOT_Recon.ReconTools import _forward_projection, _backward_projection
from AOT_biomaps.Config import config

import numba
import torch
import numpy as np
import os
from tqdm import trange

# def _MLEM_GPU_basic(SMatrix, y, numIterations, isSavingEachIteration, withTumor):
#     """
#     Maximum Likelihood Expectation-Maximization (MLEM) for GPU.
#     """
#     try:
#         device = torch.device(f"cuda:{config.select_best_gpu()}")
#         A_matrix_torch = torch.tensor(SMatrix, dtype=torch.float32).to(device)
#         y_torch = torch.tensor(y, dtype=torch.float32).to(device)
#         T, Z, X, N = SMatrix.shape
#         theta_0 = torch.ones((Z, X), dtype=torch.float32, device=device)

#         A_flat = A_matrix_torch.permute(0, 3, 1, 2).reshape(T * N, Z * X)
#         y_flat = y_torch.reshape(-1)
#         matrix_theta_torch = [theta_0]
#         normalization_factor = A_matrix_torch.sum(dim=(0, 3))
#         normalization_factor_flat = normalization_factor.reshape(-1)

#         description = f"AOT-BioMaps -- Algebraic Reconstruction Tomography: ML-EM ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()} ----"

#         for _ in trange(numIterations, desc=description):
#             theta_p = matrix_theta_torch[-1]
#             theta_p_flat = theta_p.reshape(-1)
#             q_flat = A_flat @ theta_p_flat
#             e_flat = y_flat / (q_flat + torch.finfo(torch.float32).tiny)
#             c_flat = A_flat.T @ e_flat
#             theta_p_plus_1_flat = (theta_p_flat / (normalization_factor_flat + torch.finfo(torch.float32).tiny)) * c_flat
#             matrix_theta_torch.append(theta_p_plus_1_flat.reshape(Z, X))

#         print("MLEM completed successfully on single GPU, freeing memory.")
#         del A_matrix_torch, y_torch, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
#         torch.cuda.empty_cache()

#         if not isSavingEachIteration:
#             return matrix_theta_torch[-1].cpu().numpy()
#         else:
#             return [theta.cpu().numpy() for theta in matrix_theta_torch]

#     except Exception as e:
#         print("Error in single GPU MLEM:", type(e).__name__, ":", e)
#         print("Cleaning up resources due to error...")
#         del A_matrix_torch, y_torch, A_flat, y_flat, theta_0, normalization_factor, normalization_factor_flat
#         torch.cuda.empty_cache()
#         return None

def _MLEM_GPU_basic(SMatrix, y, numIterations, isSavingEachIteration, withTumor):
    """
    Optimized MLEM using GPU — fully flattened variables, minimal memory.
    """
    try:
        device = torch.device(f"cuda:{config.select_best_gpu()}")
        eps = torch.finfo(torch.float32).eps

        T, Z, X, N = SMatrix.shape
        ZX = Z * X
        TN = T * N

        # Conversion directe en A_flat
        A_flat = (
            torch.from_numpy(SMatrix)
            .to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)     # (T, N, Z, X)
            .contiguous()
            .reshape(TN, ZX)
        )

        # Données observées
        y_flat = torch.from_numpy(y).to(device=device, dtype=torch.float32).reshape(-1)

        # Initialisation
        theta_flat = torch.ones(ZX, dtype=torch.float32, device=device)

        # Normalisation directement depuis SMatrix
        norm_factor_flat = (
            torch.from_numpy(SMatrix)
            .to(device=device, dtype=torch.float32)
            .sum(dim=(0, 3))        # (Z, X)
            .reshape(-1)
        )

        # Affichage
        description = f"AOT-BioMaps -- Algebraic Reconstruction Tomography: ML-EM ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- GPU {torch.cuda.current_device()}"

        if isSavingEachIteration:
            matrix_theta_torch = [theta_flat.reshape(Z, X).clone()]

        with torch.no_grad():
            for _ in trange(numIterations, desc=description):
                q_flat = A_flat @ theta_flat
                e_flat = y_flat / (q_flat + eps)
                c_flat = A_flat.T @ e_flat
                theta_flat = (theta_flat / (norm_factor_flat + eps)) * c_flat

                if isSavingEachIteration:
                    matrix_theta_torch.append(theta_flat.reshape(Z, X).clone())

        # Libération mémoire
        del A_flat, y_flat, norm_factor_flat
        torch.cuda.empty_cache()

        if isSavingEachIteration:
            return [theta.reshape(Z, X).cpu().numpy() for theta in matrix_theta_torch]
        else:
            return theta_flat.reshape(Z, X).cpu().numpy()

    except Exception as e:
        print("Error in MLEM:", type(e).__name__, ":", e)
        torch.cuda.empty_cache()
        return None


def _MLEM_CPU_basic(SMatrix, y, numIterations, isSavingEachIteration, withTumor):
    try:
        q_p = np.zeros((SMatrix.shape[0], SMatrix.shape[3]))
        c_p = np.zeros((SMatrix.shape[1], SMatrix.shape[2]))
        theta_p_0 = np.ones((SMatrix.shape[1], SMatrix.shape[2]))
        matrix_theta = [theta_p_0]
        normalization_factor = np.sum(SMatrix, axis=(0, 3))

        description = f"AOT-BioMaps -- Algebraic Reconstruction Tomography: ML-EM ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU (basic) ----"   

        for _ in trange(numIterations, desc=description):
            theta_p = matrix_theta[-1]
            for _t in range(SMatrix.shape[0]):
                for _n in range(SMatrix.shape[3]):
                    q_p[_t, _n] = np.sum(SMatrix[_t, :, :, _n] * theta_p)

            e_p = y / (q_p + 1e-8)

            for _z in range(SMatrix.shape[1]):
                for _x in range(SMatrix.shape[2]):
                    c_p[_z, _x] = np.sum(SMatrix[:, _z, _x, :] * e_p)

            theta_p_plus_1 = theta_p / (normalization_factor + 1e-8) * c_p
            matrix_theta.append(theta_p_plus_1)

        if not isSavingEachIteration:
            return matrix_theta[-1]
        else:
            return matrix_theta

    except Exception as e:
        print("Error in basic CPU MLEM:", type(e).__name__, ":", e)
        return None

    def _MLEM_CPU_multi(SMatrix, y, numIterations, isSavingEachIteration, withTumor):
        try:
            numba.set_num_threads(os.cpu_count())
            q_p = np.zeros((SMatrix.shape[0], SMatrix.shape[3]))
            c_p = np.zeros((SMatrix.shape[1], SMatrix.shape[2]))
            theta_p_0 = np.ones((SMatrix.shape[1], SMatrix.shape[2]))
            matrix_theta = [theta_p_0]
            normalization_factor = np.sum(SMatrix, axis=(0, 3))

            description = f"AOT-BioMaps -- Algebraic Reconstruction Tomography: ML-EM ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on multithread CPU ({numba.config.NUMBA_DEFAULT_NUM_THREADS} threads)----"

            for _ in trange(numIterations, desc=description):
                theta_p = matrix_theta[-1]
                _forward_projection(SMatrix, theta_p, q_p)
                e_p = y / (q_p + 1e-8)
                _backward_projection(SMatrix, e_p, c_p)
                theta_p_plus_1 = theta_p / (normalization_factor + 1e-8) * c_p
                matrix_theta.append(theta_p_plus_1)

            if not isSavingEachIteration:
                return matrix_theta[-1]
            else:
                return matrix_theta

        except Exception as e:
            print("Error in multi-threaded CPU MLEM:", type(e).__name__, ":", e)
            return None
        
def _MLEM_CPU_opti(SMatrix, y, numIterations, isSavingEachIteration, withTumor):
    try:
        T, Z, X, N = SMatrix.shape
        A_flat = SMatrix.astype(np.float32).transpose(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y.astype(np.float32).reshape(-1)
        theta_0 = np.ones((Z, X), dtype=np.float32)
        matrix_theta = [theta_0]
        normalization_factor = np.sum(SMatrix, axis=(0, 3)).astype(np.float32)
        normalization_factor_flat = normalization_factor.reshape(-1)

        description = f"AOT-BioMaps -- Algebraic Reconstruction Tomography: ML-EM ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single CPU (optimized) ----"

        for _ in trange(numIterations, desc=description):
            theta_p = matrix_theta[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = y_flat / (q_flat + np.finfo(np.float32).tiny)
            c_flat = A_flat.T @ e_flat
            theta_p_plus_1_flat = theta_p_flat / (normalization_factor_flat + np.finfo(np.float32).tiny) * c_flat
            matrix_theta.append(theta_p_plus_1_flat.reshape(Z, X))

        if not isSavingEachIteration:
            return matrix_theta[-1]
        else:
            return matrix_theta

    except Exception as e:
        print("Error in optimized CPU MLEM:", type(e).__name__, ":", e)
        return None

def _MLEM_GPU_multi(SMatrix, y, numIterations, isSavingEachIteration, withTumor):
    try:
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda:0')
        A_matrix_torch = torch.tensor(SMatrix, dtype=torch.float32).to(device)
        y_torch = torch.tensor(y, dtype=torch.float32).to(device)
        T, Z, X, N = SMatrix.shape

        A_matrix_torch = A_matrix_torch.permute(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y_torch.reshape(-1)

        A_split = torch.chunk(A_matrix_torch, num_gpus, dim=0)
        y_split = torch.chunk(y_flat, num_gpus)

        theta_0 = torch.ones((Z, X), dtype=torch.float32, device=device)
        theta_list = [theta_0.clone().to(device) for _ in range(num_gpus)]

        normalization_factor = A_matrix_torch.sum(dim=0).reshape(Z, X).to(device)

        description = f"AOT-BioMaps -- Algebraic Reconstruction Tomography: ML-EM ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on multi-GPU ----"

        for _ in trange(numIterations, desc=description):
            theta_p_list = [theta_list[i].to(f'cuda:{i}') for i in range(num_gpus)]

            for i in range(num_gpus):
                with torch.cuda.device(f'cuda:{i}'):
                    A_i = A_split[i].to(f'cuda:{i}')
                    y_i = y_split[i].to(f'cuda:{i}')
                    theta_p = theta_p_list[i]

                    q_flat = A_i @ theta_p.reshape(-1)
                    e_flat = y_i / (q_flat + torch.finfo(torch.float32).tiny)
                    c_flat = A_i.T @ e_flat
                    theta_p_plus_1_flat = (theta_p.reshape(-1) / (normalization_factor.to(f'cuda:{i}').reshape(-1) + torch.finfo(torch.float32).tiny)) * c_flat

                    theta_list[i] = theta_p_plus_1_flat.reshape(Z, X).to('cuda:0')

                    del A_i, y_i, theta_p, q_flat, e_flat, c_flat, theta_p_plus_1_flat
                    torch.cuda.empty_cache()

        print("MLEM completed successfully on multi-GPU, freeing memory.")

        del A_matrix_torch, y_torch, A_split, y_split, theta_0, normalization_factor
        torch.cuda.empty_cache()

        for i in range(num_gpus):
            torch.cuda.empty_cache()

        if not isSavingEachIteration:
            return torch.stack(theta_list).mean(dim=0).cpu().numpy()
        else:
            return [theta.cpu().numpy() for theta in theta_list]

    except Exception as e:
        print("Error in multi-GPU MLEM:", type(e).__name__, ":", e)
        print("Cleaning up resources due to error...")

        del A_matrix_torch, y_torch, A_split, y_split, theta_0, normalization_factor
        torch.cuda.empty_cache()

        for i in range(num_gpus):
            torch.cuda.empty_cache()

        return None
