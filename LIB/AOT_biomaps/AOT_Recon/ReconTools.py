import os
import torch
import numpy as np
from numba import njit, prange
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

@njit
def _build_adjacency_sparse_CPU(Z, X,corner = (0.5-np.sqrt(2)/4)/np.sqrt(2),face = 0.5-np.sqrt(2)/4):
    rows = []
    cols = []
    weights = []

    for z in range(Z):
        for x in range(X):
            j = z * X + x
            for dz in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dx == 0:
                        continue
                    nz, nx = z + dz, x + dx
                    if 0 <= nz < Z and 0 <= nx < X:
                        k = nz * X + nx
                        weight = corner if abs(dz) + abs(dx) == 2 else face
                        rows.append(j)
                        cols.append(k)
                        weights.append(weight)

    index = (np.array(rows), np.array(cols))
    values = np.array(weights, dtype=np.float32)
    return index, values 

def _build_adjacency_sparse_GPU(Z, X, device, corner=(0.5 - np.sqrt(2) / 4) / np.sqrt(2), face=0.5 - np.sqrt(2) / 4):
    weight_dict = {}

    for z in range(Z):
        for x in range(X):
            j = z * X + x
            for dz in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dx == 0:
                        continue
                    nz, nx = z + dz, x + dx
                    if 0 <= nz < Z and 0 <= nx < X:
                        k = nz * X + nx
                        weight = corner if abs(dz) + abs(dx) == 2 else face
                        if (j, k) in weight_dict:
                            weight_dict[(j, k)] += weight
                        else:
                            weight_dict[(j, k)] = weight

    rows = []
    cols = []
    weights = []
    for (j, k), weight in weight_dict.items():
        rows.append(j)
        cols.append(k)
        weights.append(weight)

    index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(weights, dtype=torch.float32, device=device)

    return index, values



def power_method(P, PT, data, Z, X, n_it=10, isGPU=False):
    x = PT(data)
    x = x.reshape(Z, X)
    for _ in range(n_it):
        grad = gradient_gpu(x) if isGPU else gradient_cpu(x)
        div = div_gpu(grad) if isGPU else div_cpu(grad)
        x = PT(P(x.ravel())) - div.ravel()
        s = torch.sqrt(torch.sum(x**2))
        x /= s
        x = x.reshape(Z, X)
    return torch.sqrt(s)

def proj_l2(p, alpha):
    norm = torch.sqrt(torch.sum(p**2, dim=0, keepdim=True))
    return p * alpha / torch.max(norm, torch.tensor(alpha, device=p.device))

def norm2sq(x):
    return torch.sum(x**2)

def norm1(x):
    return torch.sum(torch.abs(x))

def gradient_cpu(x):
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(x)

    grad_x[:-1, :] = x[1:, :] - x[:-1, :]
    grad_y[:, :-1] = x[:, 1:] - x[:, :-1]

    return torch.stack((grad_x, grad_y), dim=0)

def div_cpu(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Devient [1, 2, H, W]

    gx = x[:, 0:1, :, :]  # gradient horizontal
    gy = x[:, 1:2, :, :]  # gradient vertical

    # Définition des noyaux de divergence
    kernel_x = torch.tensor([[[[1.0], [-1.0]]]], dtype=torch.float32)
    kernel_y = torch.tensor([[[[1.0, -1.0]]]], dtype=torch.float32)

    # Appliquer la convolution
    div_x = F.conv2d(gx, kernel_x, padding=(1, 0))
    div_y = F.conv2d(gy, kernel_y, padding=(0, 1))

    # Rogner pour avoir la même taille (H, W)
    H, W = x.shape[2:]
    div_x = div_x[:, :, :H, :]
    div_y = div_y[:, :, :, :W]

    return -(div_x + div_y).squeeze()

def gradient_gpu(x):
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(x)
    grad_x[:-1, :] = x[1:, :] - x[:-1, :]
    grad_y[:, :-1] = x[:, 1:] - x[:, :-1]
    return torch.stack((grad_x, grad_y), dim=0)

def div_gpu(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Devient [1, 2, H, W]
    gx = x[:, 0:1, :, :]  # gradient horizontal
    gy = x[:, 1:2, :, :]  # gradient vertical

    # Définition des noyaux de divergence
    kernel_x = torch.tensor([[[[1.0], [-1.0]]]], dtype=torch.float32, device=x.device)
    kernel_y = torch.tensor([[[[1.0, -1.0]]]], dtype=torch.float32, device=x.device)

    # Appliquer la convolution
    div_x = F.conv2d(gx, kernel_x, padding=(1, 0))
    div_y = F.conv2d(gy, kernel_y, padding=(0, 1))

    # Rogner pour avoir la même taille (H, W)
    H, W = x.shape[2:]
    div_x = div_x[:, :, :H, :]
    div_y = div_y[:, :, :, :W]

    return -(div_x + div_y).squeeze()

def KL_divergence(Ax, y):
    return torch.sum(Ax - y * torch.log(Ax + 1e-10))

def gradient_KL(Ax, y):
    return 1 - y / (Ax + 1e-10)

def prox_F_star(y, sigma, a):
    return 0.5 * (y - torch.sqrt(y**2 + 4 * sigma * a))

def prox_G(x, tau, K):
    return torch.clamp(x - tau * K, min=0)

