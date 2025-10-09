from AOT_biomaps.AOT_Recon.ReconEnums import PotentialType
from AOT_biomaps.AOT_Recon.ReconTools import _build_adjacency_sparse
from AOT_biomaps.Config import config
import numpy as np
import torch
from tqdm import trange

if config.get_process() == 'gpu':
    try:
        from torch_scatter import scatter
    except ImportError:
        raise ImportError("torch_scatter and torch_sparse are required for GPU processing. Please install them using 'pip install torch-scatter torch-sparse' with correct link (follow instructions https://github.com/LucasDuclos/AcoustoOpticTomography/edit/main/README.md).")

def _DEPIERRO_GPU(SMatrix, y, numIterations, beta, sigma, isSavingEachIteration, withTumor, max_saves=5000):
    # Initialisation du device
    device = torch.device(f"cuda:{config.select_best_gpu()}")
    # Conversion des données en tenseurs PyTorch (float64)
    A_matrix_torch = torch.tensor(SMatrix, dtype=torch.float64, device=device)
    y_torch = torch.tensor(y, dtype=torch.float64, device=device)
    # Dimensions
    T, Z, X, N = SMatrix.shape
    J = Z * X
    # Redimensionnement des matrices
    A_flat = A_matrix_torch.permute(0, 3, 1, 2).reshape(T * N, J)
    y_flat = y_torch.reshape(-1)
    # Initialisation de theta
    theta_0 = torch.ones((Z, X), dtype=torch.float64, device=device)
    matrix_theta_torch = [theta_0.clone()]  # Clone pour éviter les références
    I_reconMatrix = [theta_0.cpu().numpy()]
    # Facteur de normalisation
    normalization_factor = A_matrix_torch.sum(dim=(0, 3))
    normalization_factor_flat = normalization_factor.reshape(-1)
    # Construction de la matrice d'adjacence
    adj_index, adj_values = _build_adjacency_sparse(Z, X, device=device, dtype=torch.float64)
    # Description pour la barre de progression
    description = f"AOT-BioMaps -- Bayesian Reconstruction Tomography: DE PIERRO (Sparse QUADRATIC β:{beta:.4f}, σ:{sigma:.4f}) ---- {'WITH' if withTumor else 'WITHOUT'} TUMOR ---- processing on single GPU no.{torch.cuda.current_device()}"
    # Configuration pour la sauvegarde des itérations
    saved_indices = [0]

    # Calculate save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)

    # Boucle principale MAP-EM
    for it in trange(numIterations, desc=description):
        theta_p = matrix_theta_torch[-1]
        theta_p_flat = theta_p.reshape(-1)
        # Étape 1 : Projection avant
        q_flat = A_flat @ theta_p_flat
        q_flat = q_flat + torch.finfo(torch.float64).tiny  # Évite la division par zéro
        # Étape 2 : Estimation de l'erreur
        e_flat = y_flat / q_flat
        # Étape 3 : Rétroprojection de l'erreur
        c_flat = A_flat.T @ e_flat
        # Étape 4 : Mise à jour multiplicative (EM)
        theta_EM_p_flat = theta_p_flat * c_flat
        # Étape 5 : Calcul de W_j et gamma_j
        W_j = scatter(adj_values, adj_index[0], dim=0, dim_size=J, reduce='sum') * (1.0 / (sigma**2))
        theta_k = theta_p_flat[adj_index[1]]
        weighted_theta_k = theta_k * adj_values
        gamma_j = theta_p_flat * W_j + scatter(weighted_theta_k, adj_index[0], dim=0, dim_size=J, reduce='sum')
        # Étape 6 : Mise à jour de De Pierro (résolution quadratique)
        A_coeff = 2 * beta * W_j
        B = -beta * gamma_j + normalization_factor_flat
        C = -theta_EM_p_flat
        discriminant = B**2 - 4 * A_coeff * C
        discriminant = torch.clamp(discriminant, min=0)
        theta_p_plus_1_flat = (-B + torch.sqrt(discriminant)) / (2 * A_coeff + torch.finfo(torch.float64).tiny)
        theta_p_plus_1_flat = torch.clamp(theta_p_plus_1_flat, min=0)
        # Étape 7 : Mise à jour de theta
        theta_next = theta_p_plus_1_flat.reshape(Z, X)
        matrix_theta_torch.append(theta_next)  # Ajoute la nouvelle itération
        # Sauvegarde conditionnelle
        if isSavingEachIteration and it in save_indices:
            I_reconMatrix.append(theta_next.cpu().numpy())
            saved_indices.append(it)
        # Libération mémoire partielle (optionnel, à ajuster selon besoin)
        del theta_p_flat, q_flat, e_flat, c_flat, theta_EM_p_flat, theta_p_plus_1_flat
        torch.cuda.empty_cache()

    # Libération finale des tenseurs GPU
    del A_matrix_torch, y_torch, A_flat, y_flat, normalization_factor, normalization_factor_flat
    torch.cuda.empty_cache()
    # Retour du résultat
    if isSavingEachIteration:
        return I_reconMatrix, saved_indices
    else:
        return matrix_theta_torch[-1].cpu().numpy(), None

def _DEPIERRO_CPU(SMatrix, y, Omega, numIterations, beta, sigma, isSavingEachIteration, withTumor, max_saves=5000):
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
            if isSavingEachIteration and p in save_indices:
                I_reconMatrix.append(theta_next.copy())
                saved_indices.append(p)

        if isSavingEachIteration:
            return I_reconMatrix, saved_indices
        else:
            return I_reconMatrix[-1], None
    except Exception as e:
        print(f"An error occurred in _DEPIERRO_CPU: {e}")
        return None, None
