import numpy as np

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _Omega_HUBER_PIECEWISE_CPU(theta_flat, index, values, delta):
    """
    Compute the gradient and Hessian of the Huber penalty function for sparse data.

    Parameters:
        theta_flat (numpy.ndarray): Flattened parameter vector.
        index (tuple): Indices of the sparse matrix in COO format.
        values (numpy.ndarray): Values of the sparse matrix in COO format.
        delta (float): Threshold for the Huber penalty.

    Returns:
        grad_U (numpy.ndarray): Gradient of the penalty function.
        hess_U (numpy.ndarray): Hessian of the penalty function.
        U_value (float): Value of the penalty function.
    """
    j_idx, k_idx = index
    diff = theta_flat[j_idx] - theta_flat[k_idx]
    abs_diff = np.abs(diff)

    psi_pair = np.where(abs_diff > delta,
                        delta * abs_diff - 0.5 * delta ** 2,
                        0.5 * diff ** 2)
    psi_pair = values * psi_pair

    grad_pair = np.where(abs_diff > delta,
                            delta * np.sign(diff),
                            diff)
    grad_pair = values * grad_pair

    hess_pair = np.where(abs_diff > delta,
                            np.zeros_like(diff),
                            np.ones_like(diff))
    hess_pair = values * hess_pair

    grad_U = np.zeros_like(theta_flat)
    hess_U = np.zeros_like(theta_flat)

    np.add.at(grad_U, j_idx, grad_pair)
    np.add.at(hess_U, j_idx, hess_pair)

    U_value = 0.5 * np.sum(psi_pair)

    return grad_U, hess_U, U_value


def _Omega_HUBER_PIECEWISE_GPU(theta_flat, index, values, delta):
    """
    Compute the gradient and Hessian of the Huber penalty function for sparse data on GPU.

    Parameters:
        theta_flat (cupy.ndarray): Flattened parameter vector.
        index (tuple): Indices of the sparse matrix in COO format.
        values (cupy.ndarray): Values of the sparse matrix in COO format.
        delta (float): Threshold for the Huber penalty.

    Returns:
        grad_U (cupy.ndarray): Gradient of the penalty function.
        hess_U (cupy.ndarray): Hessian of the penalty function.
        U_value (float): Value of the penalty function.
    """
    j_idx, k_idx = index
    diff = theta_flat[j_idx] - theta_flat[k_idx]
    abs_diff = cp.abs(diff)

    psi_pair = cp.where(abs_diff > delta,
                        delta * abs_diff - 0.5 * delta ** 2,
                        0.5 * diff ** 2)
    psi_pair = values * psi_pair

    grad_pair = cp.where(abs_diff > delta,
                        delta * cp.sign(diff),
                        diff)
    grad_pair = values * grad_pair

    hess_pair = cp.where(abs_diff > delta,
                            cp.zeros_like(diff),
                            cp.ones_like(diff))
    hess_pair = values * hess_pair

    grad_U = cp.zeros_like(theta_flat)
    hess_U = cp.zeros_like(theta_flat)

    grad_U[j_idx] += grad_pair
    grad_U[k_idx] -= grad_pair

    hess_U[j_idx] += hess_pair
    hess_U[k_idx] += hess_pair

    U_value = 0.5 * float(cp.sum(psi_pair))

    return grad_U, hess_U, U_value
