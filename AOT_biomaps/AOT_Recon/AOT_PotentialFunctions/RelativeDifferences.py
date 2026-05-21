import numpy as np

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _Omega_RELATIVE_DIFFERENCE_CPU(theta_flat, index, values, gamma):
    """
    CPU implementation of the relative difference potential function.

    Parameters:
        theta_flat (np.ndarray): Flattened parameter vector.
        index (tuple): Indices (j_idx, k_idx) of adjacent pixels.
        values (np.ndarray): Edge weights.
        gamma (float): Regularization parameter.

    Returns:
        grad_U (np.ndarray): Gradient of the potential function.
        hess_U (np.ndarray): Hessian of the potential function.
        U_value (float): Value of the potential function.
    """
    j_idx, k_idx = index
    theta_j = theta_flat[j_idx]
    theta_k = theta_flat[k_idx]
    diff = theta_k - theta_j
    abs_diff = np.abs(diff)
    denom = theta_k + theta_j + gamma * abs_diff + 1e-8
    num = diff ** 2
    psi_pair = num / denom
    psi_pair = values * psi_pair

    dpsi = (2 * diff * denom - num * (1 + gamma * np.sign(diff))) / (denom ** 2)
    grad_pair = values * (-dpsi)

    d2psi = (2 * denom ** 2 - 4 * diff * denom * (1 + gamma * np.sign(diff))
                + 2 * num * (1 + gamma * np.sign(diff)) ** 2) / (denom ** 3 + 1e-8)
    hess_pair = values * d2psi

    grad_U = np.zeros_like(theta_flat)
    hess_U = np.zeros_like(theta_flat)
    np.add.at(grad_U, j_idx, grad_pair)
    np.add.at(hess_U, j_idx, hess_pair)

    U_value = 0.5 * np.sum(psi_pair)
    return grad_U, hess_U, U_value


def _Omega_RELATIVE_DIFFERENCE_GPU(theta_flat, index, values, gamma):
    """
    GPU implementation of the relative difference potential function.

    Parameters:
        theta_flat (cupy.ndarray): Flattened parameter vector.
        index (tuple): Indices (j_idx, k_idx) of adjacent pixels.
        values (cupy.ndarray): Edge weights.
        gamma (float): Regularization parameter.

    Returns:
        grad_U (cupy.ndarray): Gradient of the potential function.
        hess_U (cupy.ndarray): Hessian of the potential function.
        U_value (float): Value of the potential function.
    """
    j_idx, k_idx = index
    theta_j = theta_flat[j_idx]
    theta_k = theta_flat[k_idx]
    diff = theta_k - theta_j
    abs_diff = cp.abs(diff)
    denom = theta_k + theta_j + gamma * abs_diff + 1e-8
    num = diff ** 2
    psi_pair = num / denom
    psi_pair = values * psi_pair

    dpsi = (2 * diff * denom - num * (1 + gamma * cp.sign(diff))) / (denom ** 2)
    grad_pair = values * (-dpsi)

    d2psi = (2 * denom ** 2 - 4 * diff * denom * (1 + gamma * cp.sign(diff))
             + 2 * num * (1 + gamma * cp.sign(diff)) ** 2) / (denom ** 3 + 1e-8)
    hess_pair = values * d2psi

    grad_U = cp.zeros_like(theta_flat)
    hess_U = cp.zeros_like(theta_flat)

    grad_U[j_idx] += grad_pair
    grad_U[k_idx] -= grad_pair

    hess_U[j_idx] += hess_pair
    hess_U[k_idx] += hess_pair

    U_value = 0.5 * float(cp.sum(psi_pair))
    return grad_U, hess_U, U_value
