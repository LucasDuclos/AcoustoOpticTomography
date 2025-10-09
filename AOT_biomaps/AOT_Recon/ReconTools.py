import os
import torch
import numpy as np
from numba import njit, prange
from torch_sparse import coalesce
import torch.nn.functional as F

def load_recon(hdr_path):
    """
    Lit un fichier Interfile (.hdr) et son fichier binaire (.img) pour reconstruire une image comme le fait Vinci.
    
    Paramètres :
    ------------
    - hdr_path : chemin complet du fichier .hdr
    
    Retour :
    --------
    - image : tableau NumPy contenant l'image
    - header : dictionnaire contenant les métadonnées du fichier .hdr
    """
    header = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            if ':=' in line:
                key, value = line.split(':=', 1)  # s'assurer qu'on ne coupe que la première occurrence de ':='
                key = key.strip().lower().replace('!', '')  # Nettoyage des caractères
                value = value.strip()
                header[key] = value
    
    # 📘 Obtenez le nom du fichier de données associé (le .img)
    data_file = header.get('name of data file')
    if data_file is None:
        raise ValueError(f"Impossible de trouver le fichier de données associé au fichier header {hdr_path}")
    
    img_path = os.path.join(os.path.dirname(hdr_path), data_file)
    
    # 📘 Récupérer la taille de l'image à partir des métadonnées
    shape = [int(header[f'matrix size [{i}]']) for i in range(1, 4) if f'matrix size [{i}]' in header]
    if shape and shape[-1] == 1:  # Si la 3e dimension est 1, on la supprime
        shape = shape[:-1]  # On garde (192, 240) par exemple
    
    if not shape:
        raise ValueError("Impossible de déterminer la forme de l'image à partir des métadonnées.")
    
    # 📘 Déterminez le type de données à utiliser
    data_type = header.get('number format', 'short float').lower()
    dtype_map = {
        'short float': np.float32,
        'float': np.float32,
        'int16': np.int16,
        'int32': np.int32,
        'uint16': np.uint16,
        'uint8': np.uint8
    }
    dtype = dtype_map.get(data_type)
    if dtype is None:
        raise ValueError(f"Type de données non pris en charge : {data_type}")
    
    # 📘 Ordre des octets (endianness)
    byte_order = header.get('imagedata byte order', 'LITTLEENDIAN').lower()
    endianess = '<' if 'little' in byte_order else '>'
    
    # 📘 Vérifie la taille réelle du fichier .img
    img_size = os.path.getsize(img_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize
    
    if img_size != expected_size:
        raise ValueError(f"La taille du fichier img ({img_size} octets) ne correspond pas à la taille attendue ({expected_size} octets).")
    
    # 📘 Lire les données binaires et les reformater
    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=endianess + np.dtype(dtype).char)
    
    image =  data.reshape(shape[::-1]) 
    
    # 📘 Rescale l'image si nécessaire
    rescale_slope = float(header.get('data rescale slope', 1))
    rescale_offset = float(header.get('data rescale offset', 0))
    image = image * rescale_slope + rescale_offset
    
    return image.T

def mse(y_true, y_pred):
    """
    Calcule la Mean Squared Error (MSE) entre deux tableaux.
    Équivalent à sklearn.metrics.mean_squared_error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def ssim(img1, img2, win_size=7, k1=0.01, k2=0.03, L=1.0):
    """
    Calcule l'SSIM entre deux images 2D (niveaux de gris).
    Équivalent à skimage.metrics.structural_similarity avec :
    - data_range=1.0 (images normalisées entre 0 et 1)
    - gaussian_weights=True (fenêtre gaussienne)
    - multichannel=False (1 canal)

    Args:
        img1, img2: Images 2D (numpy arrays) de même taille.
        win_size: Taille de la fenêtre de comparaison (doit être impair).
        k1, k2: Constantes pour stabiliser la division (valeurs typiques : 0.01, 0.03).
        L: Dynamique des pixels (1.0 si images dans [0,1], 255 si dans [0,255]).
    Returns:
        SSIM moyen sur l'image (float entre -1 et 1).
    """
    if img1.shape != img2.shape:
        raise ValueError("Les images doivent avoir la même taille.")
    if win_size % 2 == 0:
        raise ValueError("win_size doit être impair.")

    # Constantes
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Fenêtre gaussienne
    window = np.ones((win_size, win_size)) / (win_size ** 2)  # Approximation (skimage utilise une gaussienne)
    window = window / np.sum(window)  # Normalisation

    # Pad les images pour éviter les bords
    pad = win_size // 2
    img1_pad = np.pad(img1, pad, mode='reflect')
    img2_pad = np.pad(img2, pad, mode='reflect')

    # Calcul des statistiques locales
    mu1 = np.zeros_like(img1, dtype=np.float64)
    mu2 = np.zeros_like(img1, dtype=np.float64)
    sigma1_sq = np.zeros_like(img1, dtype=np.float64)
    sigma2_sq = np.zeros_like(img1, dtype=np.float64)
    sigma12 = np.zeros_like(img1, dtype=np.float64)

    # Itère sur chaque pixel (optimisé avec des convolutions)
    for i in range(pad, img1_pad.shape[0] - pad):
        for j in range(pad, img1_pad.shape[1] - pad):
            patch1 = img1_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            patch2 = img2_pad[i-pad:i+pad+1, j-pad:j+pad+1]

            mu1[i-pad, j-pad] = np.sum(patch1 * window)
            mu2[i-pad, j-pad] = np.sum(patch2 * window)
            sigma1_sq[i-pad, j-pad] = np.sum(window * (patch1 - mu1[i-pad, j-pad]) ** 2)
            sigma2_sq[i-pad, j-pad] = np.sum(window * (patch2 - mu2[i-pad, j-pad]) ** 2)
            sigma12[i-pad, j-pad] = np.sum(window * (patch1 - mu1[i-pad, j-pad]) * (patch2 - mu2[i-pad, j-pad]))

    # SSIM locale
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return np.mean(ssim_map)

def calculate_memory_requirement(SMatrix, y):
    """Calculate the memory requirement for the given matrices in GB."""
    num_elements_SMatrix = SMatrix.size
    num_elements_y = y.size
    num_elements_theta = SMatrix.shape[1] * SMatrix.shape[2]  # Assuming theta has shape (Z, X)

    # Calculate total memory requirement in GB
    total_memory = (num_elements_SMatrix + num_elements_y + num_elements_theta) * 32 / 8 / 1024**3
    return total_memory

def check_gpu_memory(device_index, required_memory):
    """Check if enough memory is available on the specified GPU."""
    free_memory, total_memory = torch.cuda.mem_get_info(f"cuda:{device_index}")
    free_memory_gb = free_memory / 1024**3
    print(f"Free memory on GPU {device_index}: {free_memory_gb:.2f} GB, Required memory: {required_memory:.2f} GB")
    return free_memory_gb >= required_memory

@njit(parallel=True)
def _forward_projection(SMatrix, theta_p, q_p):
    t_dim, z_dim, x_dim, i_dim = SMatrix.shape
    for _t in prange(t_dim):
        for _n in range(i_dim):
            total = 0.0
            for _z in range(z_dim):
                for _x in range(x_dim):
                    total += SMatrix[_t, _z, _x, _n] * theta_p[_z, _x]
            q_p[_t, _n] = total

@njit(parallel=True)
def _backward_projection(SMatrix, e_p, c_p):
    t_dim, z_dim, x_dim, n_dim = SMatrix.shape
    for _z in prange(z_dim):
        for _x in range(x_dim):
            total = 0.0
            for _t in range(t_dim):
                for _n in range(n_dim):
                    total += SMatrix[_t, _z, _x, _n] * e_p[_t, _n]
            c_p[_z, _x] = total


def _build_adjacency_sparse(Z, X, device, corner=(0.5 - np.sqrt(2) / 4) / np.sqrt(2), face=0.5 - np.sqrt(2) / 4,dtype=torch.float32):
    rows, cols, weights = [], [], []
    for z in range(Z):
        for x in range(X):
            j = z * X + x
            for dz, dx in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),   (1, 0), (1, 1)]:
                nz, nx = z + dz, x + dx
                if 0 <= nz < Z and 0 <= nx < X:
                    k = nz * X + nx
                    weight = corner if abs(dz) + abs(dx) == 2 else face
                    rows.append(j)
                    cols.append(k)
                    weights.append(weight)
    index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(weights, dtype=dtype, device=device)
    index, values = coalesce(index, values, m=Z*X, n=Z*X)
    return index, values


def power_method(P, PT, data, Z, X, n_it=10):
    x = torch.randn(Z * X, device=data.device)
    x = x / torch.norm(x)
    for _ in range(n_it):
        Ax = P(x)
        ATax = PT(Ax)
        x = ATax / torch.norm(ATax)
    ATax = PT(P(x))
    return torch.sqrt(torch.dot(x, ATax))

def proj_l2(p, alpha):
    if alpha <= 0:
        return torch.zeros_like(p)
    norm = torch.sqrt(torch.sum(p**2, dim=0, keepdim=True) + 1e-12)
    return p * torch.min(norm, torch.tensor(alpha, device=p.device)) / (norm + 1e-12)

def gradient(x):
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(x)
    grad_x[:, :-1] = x[:, 1:] - x[:, :-1]  # Gradient horizontal
    grad_y[:-1, :] = x[1:, :] - x[:-1, :]   # Gradient vertical
    return torch.stack((grad_x, grad_y), dim=0)

def div(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Ajoute une dimension batch si nécessaire

    gx = x[:, 0, :, :]  # Gradient horizontal (shape: [1, H, W] ou [H, W])
    gy = x[:, 1, :, :]  # Gradient vertical   (shape: [1, H, W] ou [H, W])

    # Divergence du gradient horizontal (gx)
    div_x = torch.zeros_like(gx)
    div_x[:, :, 1:] += gx[:, :, :-1]  # Contribution positive (gauche)
    div_x[:, :, :-1] -= gx[:, :, :-1] # Contribution négative (droite)

    # Divergence du gradient vertical (gy)
    div_y = torch.zeros_like(gy)
    div_y[:, 1:, :] += gy[:, :-1, :]  # Contribution positive (haut)
    div_y[:, :-1, :] -= gy[:, :-1, :] # Contribution négative (bas)

    return -(div_x + div_y)

def norm2sq(x):
    return torch.sum(x**2)

def norm1(x):
    return torch.sum(torch.abs(x))

def KL_divergence(Ax, y):
    return torch.sum(Ax - y * torch.log(Ax + 1e-10))

def gradient_KL(Ax, y):
    return 1 - y / (Ax + 1e-10)

def prox_F_star(y, sigma, a):
    return 0.5 * (y - torch.sqrt(y**2 + 4 * sigma * a))

def prox_G(x, tau, K):
    return torch.clamp(x - tau * K, min=0)

