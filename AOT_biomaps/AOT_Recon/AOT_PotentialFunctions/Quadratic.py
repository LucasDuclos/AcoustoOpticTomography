import numpy as np

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _Omega_QUADRATIC_CPU(theta_flat, j_idx, k_idx, values, sigma=1.0):
    """
    CPU implementation of the quadratic potential.

    Parameters:
        theta_flat (np.ndarray): shape (J,)
        j_idx (np.ndarray): indices j (shape N_edges,)
        k_idx (np.ndarray): indices k (shape N_edges,)
        values (np.ndarray): edge weights (shape N_edges,)
        sigma (float): standard deviation (scalar)

    Returns:
        grad_U (np.ndarray): shape (J,)
        hess_U (np.ndarray): shape (J,)
        U_value (float): scalar
    """
    n_nodes = theta_flat.shape[0]
    n_edges = j_idx.shape[0]

    grad_U = np.zeros(n_nodes)
    hess_U = np.zeros(n_nodes)
    U_value = 0.0

    for i in range(n_edges):
        j = j_idx[i]
        k = k_idx[i]
        v = values[i]
        diff = theta_flat[j] - theta_flat[k]

        psi = 0.5 * (diff / sigma) ** 2
        psi *= v
        U_value += psi

        grad = -v * diff / sigma**2
        hess = v / sigma**2

        grad_U[j] += grad
        hess_U[j] += hess

    U_value *= 0.5
    return grad_U, hess_U, U_value


def _Omega_QUADRATIC_GPU(theta_flat, index, values, sigma=1.0):
    """
    GPU implementation of the quadratic potential function, gradient and Hessian.

    Parameters:
        theta_flat (cupy.ndarray): (J,) array on GPU
        index (tuple): (j_idx, k_idx), indices of adjacent pixels
        values (cupy.ndarray): (N_edges,) weights, typically 1 or distance-based
        sigma (float): smoothness hyperparameter

    Returns:
        grad_U (cupy.ndarray): gradient of the potential function, shape (J,)
        hess_U (cupy.ndarray): diagonal of the Hessian, shape (J,)
        U_value (float): scalar, energy
    """
    j_idx, k_idx = index
    diff = theta_flat[j_idx] - theta_flat[k_idx]

    psi_pair = 0.5 * (diff / sigma) ** 2
    psi_pair = values * psi_pair

    grad_pair = values * (-diff / sigma**2)

    hess_pair = values * (1.0 / sigma**2)

    grad_U = cp.zeros_like(theta_flat)
    hess_U = cp.zeros_like(theta_flat)

    grad_U[j_idx] += grad_pair
    hess_U[j_idx] += hess_pair

    U_value = 0.5 * float(cp.sum(psi_pair))

    return grad_U, hess_U, U_value
