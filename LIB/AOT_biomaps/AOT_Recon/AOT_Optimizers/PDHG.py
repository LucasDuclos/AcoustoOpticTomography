from AOT_biomaps.AOT_Recon.ReconTools import power_method, gradient_cpu, gradient_gpu, div_cpu, div_gpu, proj_l2, prox_G, prox_F_star
from AOT_biomaps.Config import config

import torch
from tqdm import trange

'''
This module implements Primal-Dual Hybrid Gradient (PDHG) methods for solving inverse problems in Acousto-Optic Tomography.
It includes Chambolle-Pock algorithms for Total Variation (TV) and Kullback-Leibler (KL) divergence regularization.
The methods can run on both CPU and GPU, with configurations set in the AOT_biomaps.Config module.
'''

def chambolle_pock_TV_cpu(SMatrix, y, alpha, theta, numIterations, isSavingEachIteration, L, withTumor):
    device = torch.device("cpu")
    A = torch.tensor(SMatrix, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    T, Z, X, N = SMatrix.shape
    A_flat = A.permute(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.reshape(-1)

    P = lambda x: torch.matmul(A_flat, x.ravel())
    PT = lambda y: torch.matmul(A_flat.T, y)

    if L is None:
        L = power_method(P, PT, y_flat, Z, X, isGPU=True)

    sigma = 1.0 / L
    tau = 1.0 / L

    x = torch.zeros(Z * X, device=device)
    p = torch.zeros((2, Z, X), device=device)
    q = torch.zeros_like(y_flat)
    x_tilde = x.clone()

    I_reconMatrix = [x.reshape(Z, X).cpu().numpy()]

    if withTumor:
        description = f"AOT-BioMaps -- Primal/Dual Recontruction Tomography : Chambolle-Pock (TV : Gaussian Noise) α:{alpha:.4f} L: {L:.4f} ---- WITH TUMOR ---- processing on single CPU ----"
    else:
        description = f"AOT-BioMaps -- Primal/Dual Recontruction Tomography : Chambolle-Pock (TV : Gaussian Noise) α:{alpha:.4f} L: {L:.4f} ---- WITHOUT TUMOR ---- processing on single CPU ----"

    for iteration in trange(numIterations, desc=description):
        p = proj_l2(p + sigma * gradient_cpu(x_tilde.reshape(Z, X)), alpha)
        q = (q + sigma * P(x_tilde) - sigma * y_flat) / (1.0 + sigma)

        x_old = x
        x = x + tau * div_cpu(p).ravel() - tau * PT(q)
        x_tilde = x + theta * (x - x_old)

        if iteration % 1 == 0:
            I_reconMatrix.append(x.reshape(Z, X).cpu().numpy())

    return I_reconMatrix if isSavingEachIteration else I_reconMatrix[-1]

def chambolle_pock_TV_gpu(SMatrix, y, alpha, theta, numIterations, isSavingEachIteration, L, withTumor):

    device = torch.device(f"cuda:{config.select_best_gpu()}")
    A = torch.tensor(SMatrix, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    T, Z, X, N = SMatrix.shape
    A_flat = A.permute(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.reshape(-1)
    P = lambda x: torch.matmul(A_flat, x.ravel())
    PT = lambda y: torch.matmul(A_flat.T, y)

    if L is None:
        L = power_method(P, PT, y_flat, Z, X, isGPU=True)

    sigma = 1.0 / L
    tau = 1.0 / L
    x = torch.zeros(Z * X, device=device)
    p = torch.zeros((2, Z, X), device=device)
    q = torch.zeros_like(y_flat)
    x_tilde = x.clone()
    I_reconMatrix = [x.reshape(Z, X).cpu().numpy()]

    if withTumor:
        description = f"AOT-BioMaps -- Primal/Dual Recontruction Tomography : Chambolle-Pock (TV : Gaussian Noise) α:{alpha:.4f} L: {L:.4f} ---- WITH TUMOR ---- processing on GPU no.{torch.cuda.current_device()} ----"
    else:
        description = f"AOT-BioMaps -- Primal/Dual Recontruction Tomography : Chambolle-Pock (TV: Gaussian Noise) α:{alpha:.4f} L: {L:.4f} ---- WITHOUT TUMOR ---- processing on GPU no.{torch.cuda.current_device()} ----"

    for iteration in trange(numIterations, desc=description):
        p = proj_l2(p + sigma * gradient_gpu(x_tilde.reshape(Z, X)), alpha)
        q = (q + sigma * P(x_tilde) - sigma * y_flat) / (1.0 + sigma)
        x_old = x
        x = x + tau * div_gpu(p).ravel() - tau * PT(q)
        x_tilde = x + theta * (x - x_old)
        if iteration % 1 == 0:
            I_reconMatrix.append(x.reshape(Z, X).cpu().numpy())
    return I_reconMatrix if isSavingEachIteration else I_reconMatrix[-1]

def chambolle_pock_KL_cpu(SMatrix, y, alpha, theta, numIterations, isSavingEachIteration, L, withTumor):
    device = torch.device("cpu")
    A = torch.tensor(SMatrix, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    T, Z, X, N = SMatrix.shape
    A_flat = A.permute(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.reshape(-1)

    P = lambda x: torch.matmul(A_flat, x.ravel())
    PT = lambda y: torch.matmul(A_flat.T, y)

    if L is None:
        L = power_method(P, PT, y_flat, Z, X, isGPU=False)

    sigma = 1.0 / L
    tau = 1.0 / L

    x = torch.zeros(Z * X, device=device)
    q = torch.zeros_like(y_flat)
    x_tilde = x.clone()

    I_reconMatrix = [x.reshape(Z, X).cpu().numpy()]

    if withTumor:
        description = f"AOT-BioMaps -- Primal/Dual Reconstruction Tomography: Chambolle-Pock (KL) α:{alpha:.4f} L: {L:.4f} ---- WITH TUMOR ---- processing on single CPU ----"
    else:
        description = f"AOT-BioMaps -- Primal/Dual Reconstruction Tomography: Chambolle-Pock (KL) α:{alpha:.4f} L: {L:.4f} ---- WITHOUT TUMOR ---- processing on single CPU ----"

    for iteration in trange(numIterations, desc=description):
        # Mise à jour de q avec l'opérateur proximal pour F*
        q = prox_F_star(q + sigma * P(x_tilde) - sigma * y_flat, sigma, y_flat)

        # Mise à jour de x avec l'opérateur proximal pour G
        x_old = x
        x = prox_G(x - tau * PT(q), tau, PT(torch.ones_like(y_flat)))

        x_tilde = x + theta * (x - x_old)

        if iteration % 1 == 0:
            I_reconMatrix.append(x.reshape(Z, X).cpu().numpy())

    return I_reconMatrix if isSavingEachIteration else I_reconMatrix[-1]

def chambolle_pock_KL_gpu(SMatrix, y, alpha, theta, numIterations, isSavingEachIteration, L, withTumor):
    # Sélection du GPU
    device = torch.device(f"cuda:{config.select_best_gpu()}")

    # Conversion des données en tenseurs et déplacement vers le GPU
    A = torch.tensor(SMatrix, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    T, Z, X, N = SMatrix.shape
    A_flat = A.permute(0, 3, 1, 2).reshape(T * N, Z * X)
    y_flat = y.reshape(-1)

    P = lambda x: torch.matmul(A_flat, x.ravel())
    PT = lambda y: torch.matmul(A_flat.T, y)

    if L is None:
        L = power_method(P, PT, y_flat, Z, X, isGPU=True)

    sigma = 1.0 / L
    tau = 1.0 / L

    x = torch.zeros(Z * X, device=device)
    q = torch.zeros_like(y_flat)
    x_tilde = x.clone()

    I_reconMatrix = [x.reshape(Z, X).cpu().numpy()]

    if withTumor:
        description = f"AOT-BioMaps -- Primal/Dual Reconstruction Tomography: Chambolle-Pock (KL) α:{alpha:.4f} L: {L:.4f} ---- WITH TUMOR ---- processing on GPU no.{torch.cuda.current_device()} ----"
    else:
        description = f"AOT-BioMaps -- Primal/Dual Reconstruction Tomography: Chambolle-Pock (KL) α:{alpha:.4f} L: {L:.4f} ---- WITHOUT TUMOR ---- processing on GPU no.{torch.cuda.current_device()} ----"

    for iteration in trange(numIterations, desc=description):
        # Mise à jour de q avec l'opérateur proximal pour F*
        q = prox_F_star(q + sigma * P(x_tilde) - sigma * y_flat, sigma, y_flat)

        # Mise à jour de x avec l'opérateur proximal pour G
        x_old = x
        x = prox_G(x - tau * PT(q), tau, PT(torch.ones_like(y_flat)))

        x_tilde = x + theta * (x - x_old)

        if iteration % 1 == 0:
            I_reconMatrix.append(x.reshape(Z, X).cpu().numpy())

    return I_reconMatrix if isSavingEachIteration else I_reconMatrix[-1]