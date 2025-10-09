from AOT_biomaps.AOT_Recon.ReconTools import power_method, gradient, div, proj_l2, prox_G, prox_F_star
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Recon.ReconEnums import NoiseType
import torch
from tqdm import trange

'''
This module implements Primal-Dual Hybrid Gradient (PDHG) methods for solving inverse problems in Acousto-Optic Tomography.
It includes Chambolle-Pock algorithms for Total Variation (TV) and Kullback-Leibler (KL) divergence regularization.
The methods can run on both CPU and GPU, with configurations set in the AOT_biomaps.Config module.
'''

def CP_TV(
    SMatrix,
    y,
    alpha=1e-1,
    theta=1.0,
    numIterations=5000,
    isSavingEachIteration=True,
    L=None,
    withTumor=True,
    device=None,
    max_saves=5000,
):
    """
    Chambolle-Pock algorithm for Total Variation (TV) regularization.
    Works on both CPU and GPU.
    Args:
        SMatrix: System matrix (shape: T, Z, X, N)
        y: Measurement data (shape: T, X, N)
        alpha: Regularization parameter for TV
        theta: Relaxation parameter (1.0 for standard Chambolle-Pock)
        numIterations: Number of iterations
        isSavingEachIteration: If True, returns selected intermediate reconstructions
        L: Lipschitz constant (estimated if None)
        withTumor: Boolean for description only
        device: Torch device (auto-selected if None)
        max_saves: Maximum number of intermediate saves (default: 5000)
    """
    # Auto-select device if not provided
    if device is None:
        device = torch.device(f"cuda:{config.select_best_gpu()}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Convert data to tensors and move to device
    A = torch.tensor(SMatrix, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    T, Z, X, N = SMatrix.shape
    A_flat = A.permute(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.reshape(-1)

    # Robust normalization
    norm_A = A_flat.abs().max().clamp(min=1e-8)
    norm_y = y_flat.abs().max().clamp(min=1e-8)
    A_flat = A_flat / norm_A
    y_flat = y_flat / norm_y

    # Define forward/backward operators
    P = lambda x: torch.matmul(A_flat, x)
    PT = lambda y: torch.matmul(A_flat.T, y)

    # Estimate Lipschitz constant if needed
    if L is None:
        try:
            L = power_method(P, PT, y_flat, Z, X)
            L = max(L, 1e-3)
        except:
            L = 1.0

    sigma = 1.0 / L
    tau = 1.0 / L

    # Initialize variables
    x = torch.zeros(Z * X, device=device)
    p = torch.zeros((2, Z, X), device=device)
    q = torch.zeros_like(y_flat)
    x_tilde = x.clone()

    # Calculate save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)

    I_reconMatrix = []
    saved_indices = []

    # Description for progress bar
    tumor_str = "WITH TUMOR" if withTumor else "WITHOUT TUMOR"
    device_str = f"GPU no.{torch.cuda.current_device()}" if device.type == "cuda" else "CPU"
    description = f"AOT-BioMaps -- Primal/Dual Reconstruction (TV) α:{alpha:.4f} L:{L:.4f} -- {tumor_str} -- {device_str}"

    # Main loop
    for iteration in trange(numIterations, desc=description):
        # Update p (TV proximal step)
        grad_x = gradient(x_tilde.reshape(Z, X))
        p = proj_l2(p + sigma * grad_x, alpha)

        # Update q (data fidelity)
        q = (q + sigma * (P(x_tilde) - y_flat)) / (1 + sigma)

        # Update x
        x_old = x.clone()
        div_p = div(p).ravel()  # Divergence calculation
        ATq = PT(q)
        x = (x - tau * (ATq - div_p)) / (1 + tau * 1e-6)  # Light L2 regularization

        # Update x_tilde
        x_tilde = x + theta * (x - x_old)

        # Save intermediate result if needed
        if isSavingEachIteration and iteration in save_indices:
            I_reconMatrix.append(x.reshape(Z, X).clone() * (norm_y / norm_A))
            saved_indices.append(iteration)

    # Return results
    if isSavingEachIteration:
        return [tensor.cpu().numpy() for tensor in I_reconMatrix], saved_indices
    else:
        return (x.reshape(Z, X) * (norm_y / norm_A)).cpu().numpy(), None


def CP_KL(
    SMatrix,
    y,
    alpha=1e-9,
    theta=1.0,
    numIterations=5000,
    isSavingEachIteration=True,
    L=None,
    withTumor=True,
    device=None,
    max_saves=5000,
):
    """
    Chambolle-Pock algorithm for Kullback-Leibler (KL) divergence regularization.
    Works on both CPU and GPU.
    Args:
        SMatrix: System matrix (shape: T, Z, X, N)
        y: Measurement data (shape: T, X, N)
        alpha: Regularization parameter
        theta: Relaxation parameter (1.0 for standard Chambolle-Pock)
        numIterations: Number of iterations
        isSavingEachIteration: If True, returns selected intermediate reconstructions
        L: Lipschitz constant (estimated if None)
        withTumor: Boolean for description only
        device: Torch device (auto-selected if None)
        max_saves: Maximum number of intermediate saves (default: 5000)
    """
    # Auto-select device if not provided
    if device is None:
        device = torch.device(f"cuda:{config.select_best_gpu()}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Convert data to tensors and move to device
    A = torch.tensor(SMatrix, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    T, Z, X, N = SMatrix.shape
    A_flat = A.permute(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.reshape(-1)

    # Define forward/backward operators
    P = lambda x: torch.matmul(A_flat, x.ravel())
    PT = lambda y: torch.matmul(A_flat.T, y)

    # Estimate Lipschitz constant if needed
    if L is None:
        L = power_method(P, PT, y_flat, Z, X)

    sigma = 1.0 / L
    tau = 1.0 / L

    # Initialize variables
    x = torch.zeros(Z * X, device=device)
    q = torch.zeros_like(y_flat)
    x_tilde = x.clone()

    # Calculate save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)

    I_reconMatrix = [x.reshape(Z, X).cpu().numpy()]
    saved_indices = [0]

    # Description for progress bar
    tumor_str = "WITH TUMOR" if withTumor else "WITHOUT TUMOR"
    device_str = f"GPU no.{torch.cuda.current_device()}" if device.type == "cuda" else "CPU"
    description = f"AOT-BioMaps -- Primal/Dual Reconstruction (KL) α:{alpha:.4f} L:{L:.4f} -- {tumor_str} -- {device_str}"

    # Main loop
    for iteration in trange(numIterations, desc=description):
        # Update q (proximal step for F*)
        q = prox_F_star(q + sigma * P(x_tilde) - sigma * y_flat, sigma, y_flat)

        # Update x (proximal step for G)
        x_old = x.clone()
        x = prox_G(x - tau * PT(q), tau, PT(torch.ones_like(y_flat)))

        # Update x_tilde
        x_tilde = x + theta * (x - x_old)

        # Save intermediate result if needed
        if isSavingEachIteration and iteration in save_indices:
            I_reconMatrix.append(x.reshape(Z, X).cpu().numpy())
            saved_indices.append(iteration)

    # Return results
    if isSavingEachIteration:
        return I_reconMatrix, saved_indices
    else:
        return I_reconMatrix[-1], None