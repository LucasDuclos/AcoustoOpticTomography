from AOT_biomaps.AOT_Recon.ReconTools import _forward_projection, _backward_projection, check_gpu_memory, calculate_memory_requirement
from AOT_biomaps.Config import config
import numba
import torch
import numpy as np
import os
from tqdm import trange

def MLEM(
    SMatrix,
    y,
    numIterations=100,
    isSavingEachIteration=True,
    withTumor=True,
    device=None,
    use_multi_gpu=False,
    use_numba=False,
    max_saves=5000,
):
    """
    Unified MLEM algorithm for Acousto-Optic Tomography.
    Works on CPU (basic, multithread, optimized) and GPU (single or multi-GPU).
    Args:
        SMatrix: System matrix (shape: T, Z, X, N)
        y: Measurement data (shape: T, N)
        numIterations: Number of iterations
        isSavingEachIteration: If True, saves intermediate results
        withTumor: Boolean for description only
        device: Torch device (auto-selected if None)
        use_multi_gpu: If True and GPU available, uses all GPUs
        use_numba: If True and on CPU, uses multithreaded Numba
        max_saves: Maximum number of intermediate saves (default: 5000)
    Returns:
        Reconstructed image(s) and iteration indices (if isSavingEachIteration)
    """
    try:
        tumor_str = "WITH" if withTumor else "WITHOUT"
        # Auto-select device and method
        if device is None:
            if torch.cuda.is_available() and check_gpu_memory(config.select_best_gpu(), calculate_memory_requirement(SMatrix, y)):
                device = torch.device(f"cuda:{config.select_best_gpu()}")
                use_gpu = True
            else:
                device = torch.device("cpu")
                use_gpu = False
        else:
            use_gpu = device.type == "cuda"
        # Dispatch to the appropriate implementation
        if use_gpu:
            if use_multi_gpu and torch.cuda.device_count() > 1:
                return _MLEM_multi_GPU(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, max_saves)
            else:
                return _MLEM_single_GPU(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, device, max_saves)
        else:
            if use_numba:
                return _MLEM_CPU_numba(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, max_saves)
            else:
                return _MLEM_CPU_opti(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, max_saves)
    except Exception as e:
        print(f"Error in MLEM: {type(e).__name__}: {e}")
        return None, None

def _MLEM_single_GPU(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, device, max_saves=5000):
    try:
        eps = torch.finfo(torch.float32).eps
        T, Z, X, N = SMatrix.shape
        ZX = Z * X
        TN = T * N
        A_flat = (
            torch.from_numpy(SMatrix)
            .to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .contiguous()
            .reshape(TN, ZX)
        )
        y_flat = torch.from_numpy(y).to(device=device, dtype=torch.float32).reshape(-1)
        theta_flat = torch.ones(ZX, dtype=torch.float32, device=device)
        norm_factor_flat = (
            torch.from_numpy(SMatrix)
            .to(device=device, dtype=torch.float32)
            .sum(dim=(0, 3))
            .reshape(-1)
        )
        description = f"AOT-BioMaps -- ML-EM ---- {tumor_str} TUMOR ---- GPU {torch.cuda.current_device()}"

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        saved_theta = []
        saved_indices = []

        with torch.no_grad():
            for it in trange(numIterations, desc=description):
                q_flat = A_flat @ theta_flat
                e_flat = y_flat / (q_flat + eps)
                c_flat = A_flat.T @ e_flat
                theta_flat = (theta_flat / (norm_factor_flat + eps)) * c_flat
                if isSavingEachIteration and it in save_indices:
                    saved_theta.append(theta_flat.reshape(Z, X).clone())
                    saved_indices.append(it)

        # Free memory
        del A_flat, y_flat, norm_factor_flat
        torch.cuda.empty_cache()
        if isSavingEachIteration:
            return [t.cpu().numpy() for t in saved_theta], saved_indices
        else:
            return theta_flat.reshape(Z, X).cpu().numpy(), None
    except Exception as e:
        print(f"Error in single-GPU MLEM: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return None, None

def _MLEM_multi_GPU(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, max_saves=5000):
    try:
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda:0')
        T, Z, X, N = SMatrix.shape
        A_matrix_torch = torch.tensor(SMatrix, dtype=torch.float32).to(device).permute(0, 3, 1, 2).reshape(T * N, Z * X)
        y_torch = torch.tensor(y, dtype=torch.float32).to(device).reshape(-1)
        A_split = torch.chunk(A_matrix_torch, num_gpus, dim=0)
        y_split = torch.chunk(y_torch, num_gpus)
        theta_0 = torch.ones((Z, X), dtype=torch.float32, device=device)
        theta_list = [theta_0.clone().to(device) for _ in range(num_gpus)]
        normalization_factor = A_matrix_torch.sum(dim=0).reshape(Z, X).to(device)

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        saved_theta = [theta_0.cpu().numpy()]
        saved_indices = [0]
        description = f"AOT-BioMaps -- ML-EM ---- {tumor_str} TUMOR ---- processing on multi-GPU ({num_gpus} GPUs) ----"

        for it in trange(numIterations, desc=description):
            theta_p_list = []
            for i in range(num_gpus):
                with torch.cuda.device(f'cuda:{i}'):
                    theta_p = theta_list[i].to(f'cuda:{i}')
                    A_i = A_split[i].to(f'cuda:{i}')
                    y_i = y_split[i].to(f'cuda:{i}')
                    q_flat = A_i @ theta_p.reshape(-1)
                    e_flat = y_i / (q_flat + torch.finfo(torch.float32).tiny)
                    c_flat = A_i.T @ e_flat
                    theta_p_plus_1_flat = (theta_p.reshape(-1) / (normalization_factor.to(f'cuda:{i}').reshape(-1) + torch.finfo(torch.float32).tiny)) * c_flat
                    theta_p_plus_1 = theta_p_plus_1_flat.reshape(Z, X)
                    theta_p_list.append(theta_p_plus_1)
            for i in range(num_gpus):
                theta_list[i] = theta_p_list[i].to('cuda:0')
            if isSavingEachIteration and it in save_indices:
                saved_theta.append(torch.stack(theta_p_list).mean(dim=0).cpu().numpy())
                saved_indices.append(it + 1)

        del A_matrix_torch, y_torch, A_split, y_split, theta_0, normalization_factor
        for i in range(num_gpus):
            torch.cuda.empty_cache()
        if not isSavingEachIteration:
            return torch.stack(theta_p_list).mean(dim=0).cpu().numpy(), None
        else:
            return saved_theta, saved_indices
    except Exception as e:
        print(f"Error in multi-GPU MLEM: {type(e).__name__}: {e}")
        del A_matrix_torch, y_torch, A_split, y_split, theta_0, normalization_factor
        for i in range(num_gpus):
            torch.cuda.empty_cache()
        return None, None

def _MLEM_CPU_numba(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, max_saves=5000):
    try:
        numba.set_num_threads(os.cpu_count())
        q_p = np.zeros((SMatrix.shape[0], SMatrix.shape[3]))
        c_p = np.zeros((SMatrix.shape[1], SMatrix.shape[2]))
        theta_p_0 = np.ones((SMatrix.shape[1], SMatrix.shape[2]))
        matrix_theta = [theta_p_0]
        saved_indices = [0]
        normalization_factor = np.sum(SMatrix, axis=(0, 3))

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        description = f"AOT-BioMaps -- ML-EM ---- {tumor_str} TUMOR ---- processing on multithread CPU ({numba.config.NUMBA_DEFAULT_NUM_THREADS} threads) ----"

        for it in trange(numIterations, desc=description):
            theta_p = matrix_theta[-1]
            _forward_projection(SMatrix, theta_p, q_p)
            e_p = y / (q_p + 1e-8)
            _backward_projection(SMatrix, e_p, c_p)
            theta_p_plus_1 = theta_p / (normalization_factor + 1e-8) * c_p
            if isSavingEachIteration and it in save_indices:
                matrix_theta.append(theta_p_plus_1)
                saved_indices.append(it + 1)
            else:
                matrix_theta[-1] = theta_p_plus_1

        if not isSavingEachIteration:
            return matrix_theta[-1], None
        else:
            return matrix_theta, saved_indices
    except Exception as e:
        print(f"Error in Numba CPU MLEM: {type(e).__name__}: {e}")
        return None, None

def _MLEM_CPU_opti(SMatrix, y, numIterations, isSavingEachIteration, tumor_str, max_saves=5000):
    try:
        T, Z, X, N = SMatrix.shape
        A_flat = SMatrix.astype(np.float32).transpose(0, 3, 1, 2).reshape(T * N, Z * X)
        y_flat = y.astype(np.float32).reshape(-1)
        theta_0 = np.ones((Z, X), dtype=np.float32)
        matrix_theta = [theta_0]
        saved_indices = [0]
        normalization_factor = np.sum(SMatrix, axis=(0, 3)).astype(np.float32)
        normalization_factor_flat = normalization_factor.reshape(-1)

        # Calculate save indices
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            step = numIterations // max_saves
            save_indices = list(range(0, numIterations, step))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        description = f"AOT-BioMaps -- ML-EM ---- {tumor_str} TUMOR ---- processing on single CPU (optimized) ----"

        for it in trange(numIterations, desc=description):
            theta_p = matrix_theta[-1]
            theta_p_flat = theta_p.reshape(-1)
            q_flat = A_flat @ theta_p_flat
            e_flat = y_flat / (q_flat + np.finfo(np.float32).tiny)
            c_flat = A_flat.T @ e_flat
            theta_p_plus_1_flat = theta_p_flat / (normalization_factor_flat + np.finfo(np.float32).tiny) * c_flat
            theta_p_plus_1 = theta_p_plus_1_flat.reshape(Z, X)
            if isSavingEachIteration and it in save_indices:
                matrix_theta.append(theta_p_plus_1)
                saved_indices.append(it + 1)
            else:
                matrix_theta[-1] = theta_p_plus_1

        if not isSavingEachIteration:
            return matrix_theta[-1], None
        else:
            return matrix_theta, saved_indices
    except Exception as e:
        print(f"Error in optimized CPU MLEM: {type(e).__name__}: {e}")
        return None, None
