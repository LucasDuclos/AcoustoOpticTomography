from AOT_biomaps.Config import config
import torch
import numpy as np
from tqdm import trange

def LS(
    SMatrix,
    y,
    numIterations=5000,
    alpha=1e-3,
    isSavingEachIteration=True,
    withTumor=True,
    device=None,
    max_saves=5000,
):
    """
    Least Squares reconstruction using Projected Gradient Descent (PGD) with non-negativity constraint.
    Currently only implements the stable GPU version.
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"
    # Force GPU usage for now
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this implementation.")
        device = torch.device(f"cuda:{config.select_best_gpu()}")
    else:
        if device.type != "cuda":
            raise RuntimeError("Only GPU implementation is available for now.")
    return _LS_GPU_stable(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves)

def _LS_GPU_stable(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves=5000):
    """
    Stable GPU implementation of LS using projected gradient descent with diagonal preconditioner.
    """
    device = torch.device(f"cuda:{config.select_best_gpu()}")
    T, Z, X, N = SMatrix.shape
    ZX = Z * X
    TN = T * N
    # 1. Conversion et normalisation
    A_flat = torch.from_numpy(SMatrix).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).reshape(TN, ZX)
    y_flat = torch.from_numpy(y).to(device=device, dtype=torch.float32).reshape(TN)
    norm_A = A_flat.max()
    norm_y = y_flat.max()
    A_flat.div_(norm_A + 1e-8)
    y_flat.div_(norm_y + 1e-8)
    # 2. Initialisation
    lambda_k = torch.zeros(ZX, device=device)
    lambda_history = [] if isSavingEachIteration else None
    saved_indices = []  # Pour stocker les indices des itérations sauvegardées

    # Calculate save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)

    # Préconditionneur diagonal
    diag_AAT = torch.sum(A_flat ** 2, dim=0)
    M_inv = 1.0 / torch.clamp(diag_AAT, min=1e-6)
    # Pré-allocation des tenseurs
    r_k = torch.empty_like(y_flat)
    AT_r = torch.empty(ZX, device=device)
    description = f"AOT-BioMaps -- Stable LS Reconstruction ---- {tumor_str} TUMOR ---- GPU {torch.cuda.current_device()}"

    for k in trange(numIterations, desc=description):
        # Calcul du résidu (inplace)
        torch.matmul(A_flat, lambda_k, out=r_k)
        r_k = y_flat - r_k
        if isSavingEachIteration and k in save_indices:
            lambda_history.append(lambda_k.clone().reshape(Z, X) * (norm_y / norm_A))
            saved_indices.append(k)

        # Gradient préconditionné (inplace)
        torch.matmul(A_flat.T, r_k, out=AT_r)
        AT_r *= M_inv
        # Mise à jour avec pas fixe et projection (inplace)
        lambda_k.add_(AT_r, alpha=alpha)
        lambda_k.clamp_(min=0)

    # 3. Dénormalisation
    lambda_final = lambda_k.reshape(Z, X) * (norm_y / norm_A)
    # Free memory
    del A_flat, y_flat, r_k, AT_r
    torch.cuda.empty_cache()
    if isSavingEachIteration:
        return [t.cpu().numpy() for t in lambda_history], saved_indices
    else:
        return lambda_final.cpu().numpy(), None

def _LS_GPU_opti(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_GPU_multi(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_CPU_opti(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_CPU_basic(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")
