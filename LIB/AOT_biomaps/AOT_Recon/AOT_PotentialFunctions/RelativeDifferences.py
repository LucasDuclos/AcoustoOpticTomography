import numpy as np
import torch
from numba import njit

@njit
def _Omega_RELATIVE_DIFFERENCE_CPU(theta_flat, index, values, gamma):
    j_idx, k_idx = index
    theta_j = theta_flat[j_idx]
    theta_k = theta_flat[k_idx]
    diff = theta_k - theta_j
    abs_diff = np.abs(diff)

    denom = theta_k + theta_j + gamma * abs_diff + 1e-8
    num = diff ** 2

    # First derivative ∂U/∂θ_j
    dpsi = (2 * diff * denom - num * (1 + gamma * np.sign(diff))) / (denom ** 2)
    grad_pair = values * (-dpsi)  # Note the negative sign: U contains ψ(θ_k, θ_j), seeking ∂/∂θ_j

    # Second derivative ∂²U/∂θ_j² (numerically stable, approximate treatment)
    d2psi = (2 * denom ** 2 - 4 * diff * denom * (1 + gamma * np.sign(diff))
                + 2 * num * (1 + gamma * np.sign(diff)) ** 2) / (denom ** 3 + 1e-8)
    hess_pair = values * d2psi

    grad_U = np.zeros_like(theta_flat)
    hess_U = np.zeros_like(theta_flat)

    np.add.at(grad_U, j_idx, grad_pair)
    np.add.at(hess_U, j_idx, hess_pair)

    return grad_U, hess_U

def _Omega_RELATIVE_DIFFERENCE_GPU(theta_flat, index, values, device, gamma):
    j_idx, k_idx = index
    theta_j = theta_flat[j_idx]
    theta_k = theta_flat[k_idx]
    diff = theta_k - theta_j
    abs_diff = torch.abs(diff)
    denom = theta_k + theta_j + gamma * abs_diff + 1e-8
    num = diff ** 2

    # Compute gradient contributions
    dpsi = (2 * diff * denom - num * (1 + gamma * torch.sign(diff))) / (denom ** 2)
    grad_pair = values * (-dpsi)

    # Compute Hessian contributions
    d2psi = (2 * denom ** 2 - 4 * diff * denom * (1 + gamma * torch.sign(diff))
             + 2 * num * (1 + gamma * torch.sign(diff)) ** 2) / (denom ** 3 + 1e-8)
    hess_pair = values * d2psi

    # Initialize gradient and Hessian on the correct device
    grad_U = torch.zeros_like(theta_flat, device=device)
    hess_U = torch.zeros_like(theta_flat, device=device)

    # Accumulate gradient contributions
    grad_U.index_add_(0, j_idx, grad_pair)
    grad_U.index_add_(0, k_idx, -grad_pair)

    # Accumulate Hessian contributions
    hess_U.index_add_(0, j_idx, hess_pair)
    hess_U.index_add_(0, k_idx, hess_pair)

    return grad_U, hess_U