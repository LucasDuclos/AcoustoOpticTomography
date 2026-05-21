import numpy as np
from tqdm import trange
from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Recon.ReconTools import (
    power_method, gradient, div, proj_l2, prox_G, prox_F_star,
    compute_TV_cpu, check_gpu_memory, calculate_memory_requirement
)
from AOT_biomaps.AOT_Recon.ReconEnums import NoiseType, SMatrixType

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def CP_TV(
    SMatrix,
    y,
    alpha=None,
    beta=1e-4,
    theta=1.0,
    numIterations=5000,
    isSavingEachIteration=True,
    L=None,
    withTumor=True,
    device=None,
    max_saves=5000,
    show_logs=True,
    smatrixType=SMatrixType.SELL,
    k_security=0.8,
    use_power_method=True,
    auto_alpha_gamma=0.05,
    apply_positivity_clamp=True,
    tikhonov_as_gradient=False,
    use_laplacian=True,
    laplacian_beta_scale=1.0
):
    """
    Chambolle-Pock algorithm for Total Variation (TV) regularization using CuPy.
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"

    if device is None:
        if config.numGPUs > 0 and check_gpu_memory(config.bestGPU, calculate_memory_requirement(SMatrix, y), show_logs=show_logs):
            device = config.bestGPU
            use_gpu = True
        else:
            device = None
            use_gpu = False

    if use_gpu:
        if smatrixType == SMatrixType.SELL:
            return CP_TV_cupy(SMatrix, y, alpha, beta, theta, numIterations, isSavingEachIteration,
                            L, tumor_str, device, max_saves, show_logs, k_security, use_power_method,
                            auto_alpha_gamma, apply_positivity_clamp, tikhonov_as_gradient,
                            use_laplacian, laplacian_beta_scale)
        elif smatrixType == SMatrixType.DENSE:
            return CP_TV_dense_cupy(SMatrix, y, alpha, theta, numIterations, isSavingEachIteration,
                                  L, tumor_str, device, max_saves, show_logs)
        else:
            raise ValueError("Unsupported SMatrixType for GPU Chambolle Pock (LS-TV).")
    else:
        raise NotImplementedError("CPU Chambolle Pock (LS-TV) not implemented.")


def CP_KL(
    SMatrix,
    y,
    alpha=None,
    beta=1e-4,
    theta=1.0,
    numIterations=5000,
    isSavingEachIteration=True,
    L=None,
    withTumor=True,
    device=None,
    max_saves=5000,
    show_logs=True,
    smatrixType=SMatrixType.SELL,
    k_security=0.8,
    use_power_method=True,
    auto_alpha_gamma=0.05,
    apply_positivity_clamp=True,
    tikhonov_as_gradient=False,
    use_laplacian=True,
    laplacian_beta_scale=1.0
):
    """
    Chambolle-Pock algorithm for KL divergence regularization using CuPy.
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"

    if device is None:
        if config.numGPUs > 0 and check_gpu_memory(config.bestGPU, calculate_memory_requirement(SMatrix, y), show_logs=show_logs):
            device = config.bestGPU
            use_gpu = True
        else:
            device = None
            use_gpu = False

    if use_gpu:
        if smatrixType == SMatrixType.DENSE:
            return CP_KL_dense_cupy(SMatrix, y, alpha, theta, numIterations, isSavingEachIteration,
                                   L, tumor_str, device, max_saves, show_logs)
        else:
            raise NotImplementedError("GPU Chambolle Pock (LS-KL) with sparse matrices not implemented.")
    else:
        raise NotImplementedError("CPU Chambolle Pock (LS-KL) not implemented.")


def CP_TV_dense_cupy(
    SMatrix,
    y,
    alpha=1e-1,
    theta=1.0,
    numIterations=5000,
    isSavingEachIteration=True,
    L=None,
    tumor_str="",
    device=None,
    max_saves=5000,
    show_logs=True,
):
    """Chambolle-Pock algorithm for TV regularization with dense matrices using CuPy."""
    try:
        T, Z, X, N = SMatrix.shape
        with cp.cuda.Device(device):
            A = cp.asarray(SMatrix.astype(np.float32))
            y_arr = cp.asarray(y.astype(np.float32))

            A_flat = A.transpose(0, 3, 1, 2).reshape(T * N, Z * X)
            y_flat = y_arr.reshape(-1)

            norm_A = cp.abs(A_flat).max().clip(min=1e-8)
            norm_y = cp.abs(y_flat).max().clip(min=1e-8)
            A_flat = A_flat / norm_A
            y_flat = y_flat / norm_y

            P = lambda x: A_flat @ x
            PT = lambda y: A_flat.T @ y

            if L is None:
                L = power_method(P, PT, y_flat, Z, X)
                L = max(float(L), 1e-3)

            sigma = 1.0 / L
            tau = 1.0 / L

            x = cp.zeros(Z * X)
            p = cp.zeros((2, Z, X))
            q = cp.zeros_like(y_flat)
            x_tilde = x.copy()

            if numIterations <= max_saves:
                save_indices = list(range(numIterations))
            else:
                step = numIterations // max_saves
                save_indices = list(range(0, numIterations, step))
                if save_indices[-1] != numIterations - 1:
                    save_indices.append(numIterations - 1)

            I_reconMatrix = []
            saved_indices = []

            device_str = f"GPU {device}"
            description = f"AOT-BioMaps -- Primal/Dual Reconstruction (LS-TV) alpha:{alpha:.4f} L:{L:.4f} -- {tumor_str} -- {device_str}"
            iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

            for it in iterator:
                grad_x = gradient(x_tilde.reshape(Z, X))
                p = proj_l2(p + sigma * grad_x, alpha)

                q = (q + sigma * (P(x_tilde) - y_flat)) / (1 + sigma)

                x_old = x.copy()
                div_p = div(p).ravel()
                ATq = PT(q)
                x = (x - tau * (ATq - div_p)) / (1 + tau * 1e-6)

                x_tilde = x + theta * (x - x_old)

                if isSavingEachIteration and it in save_indices:
                    I_reconMatrix.append(cp.asnumpy(x.reshape(Z, X) * (norm_y / norm_A)))
                    saved_indices.append(it)

            cp.cuda.Stream.null.synchronize()

            if isSavingEachIteration:
                return I_reconMatrix, saved_indices
            else:
                return cp.asnumpy(x.reshape(Z, X) * (norm_y / norm_A)), None

    except Exception as e:
        print(f"Error in CP_TV_dense_cupy: {type(e).__name__}: {e}")
        return None, None


def CP_KL_dense_cupy(
    SMatrix,
    y,
    alpha=1e-9,
    theta=1.0,
    numIterations=5000,
    isSavingEachIteration=True,
    L=None,
    tumor_str="",
    device=None,
    max_saves=5000,
    show_logs=True,
):
    """Chambolle-Pock algorithm for KL divergence with dense matrices using CuPy."""
    try:
        T, Z, X, N = SMatrix.shape
        with cp.cuda.Device(device):
            A = cp.asarray(SMatrix.astype(np.float32))
            y_arr = cp.asarray(y.astype(np.float32))

            A_flat = A.transpose(0, 3, 1, 2).reshape(T * N, Z * X)
            y_flat = y_arr.reshape(-1)

            P = lambda x: A_flat @ x.ravel()
            PT = lambda y: A_flat.T @ y

            if L is None:
                L = power_method(P, PT, y_flat, Z, X)

            sigma = 1.0 / L
            tau = 1.0 / L

            x = cp.zeros(Z * X)
            q = cp.zeros_like(y_flat)
            x_tilde = x.copy()

            if numIterations <= max_saves:
                save_indices = list(range(numIterations))
            else:
                step = numIterations // max_saves
                save_indices = list(range(0, numIterations, step))
                if save_indices[-1] != numIterations - 1:
                    save_indices.append(numIterations - 1)

            I_reconMatrix = [cp.asnumpy(x.reshape(Z, X))]
            saved_indices = [0]

            device_str = f"GPU {device}"
            description = f"AOT-BioMaps -- Primal/Dual Reconstruction (KL) alpha:{alpha:.4f} L:{L:.4f} -- {tumor_str} -- {device_str}"
            iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

            for iteration in iterator:
                q = prox_F_star(q + sigma * P(x_tilde) - sigma * y_flat, sigma, y_flat)

                x_old = x.copy()
                x = prox_G(x - tau * PT(q), tau, PT(cp.ones_like(y_flat)))

                x_tilde = x + theta * (x - x_old)

                if isSavingEachIteration and iteration in save_indices:
                    I_reconMatrix.append(cp.asnumpy(x.reshape(Z, X)))
                    saved_indices.append(iteration)

            cp.cuda.Stream.null.synchronize()

            if isSavingEachIteration:
                return I_reconMatrix, saved_indices
            else:
                return I_reconMatrix[-1], None

    except Exception as e:
        print(f"Error in CP_KL_dense_cupy: {type(e).__name__}: {e}")
        return None, None


def CP_TV_cupy(
    SMatrix,
    y,
    alpha=None,
    beta=1e-4,
    theta=1.0,
    numIterations=2000,
    isSavingEachIteration=True,
    L=None,
    tumor_str="",
    device=None,
    max_saves=2000,
    show_logs=True,
    k_security=0.8,
    use_power_method=True,
    auto_alpha_gamma=0.05,
    apply_positivity_clamp=True,
    tikhonov_as_gradient=False,
    use_laplacian=True,
    laplacian_beta_scale=1.0
):
    """Chambolle-Pock algorithm for TV with sparse matrices using CuPy."""
    try:
        with cp.cuda.Device(device):
            TN = int(SMatrix.N * SMatrix.T)
            ZX = int(SMatrix.Z * SMatrix.X)
            Z = int(SMatrix.Z)
            X = int(SMatrix.X)

            y_flat = cp.asarray(y.T.flatten().astype(np.float32))
            maxy = float(cp.max(cp.abs(y_flat))) if y_flat.size > 0 else 0.0
            if maxy > 0:
                y_normed = y_flat / maxy
            else:
                y_normed = y_flat.copy()

            if use_power_method or L is None:
                L_LS_sq = float(SMatrix.power_method_estimate_L())
                L_nabla_sq = 8.0
                L_op_norm = np.sqrt(L_LS_sq + L_nabla_sq)
                if L_op_norm < 1e-6:
                    L_op_norm = 1.0
            else:
                L_op_norm = L

            tau = np.float32(k_security / L_op_norm)
            sigma = np.float32(k_security / L_op_norm)

            x = cp.zeros(ZX)
            x_old = cp.zeros(ZX)
            x_tilde = cp.zeros(ZX)
            p = cp.zeros(2 * ZX)
            q = cp.zeros(TN)
            grad = cp.zeros(2 * ZX)
            div_p = cp.zeros(ZX)
            Ax = cp.zeros(TN)
            ATq = cp.zeros(ZX)

            if auto_alpha_gamma and alpha is None:
                ATy = SMatrix.backprojection(y_normed)
                x_host = cp.asnumpy(ATy)
                Ax_host = cp.asnumpy(SMatrix.projection(ATy))
                resid = Ax_host - cp.asnumpy(y_normed[:TN])
                data_term = 0.5 * float(np.dot(resid, resid))
                tv_term = float(compute_TV_cpu(x_host, Z, X)) + 1e-12
                alpha = float(auto_alpha_gamma * data_term / tv_term)
                if show_logs:
                    print(f"[auto-alpha] data_term={data_term:.6e}, tv_term={tv_term:.6e}, alpha_set={alpha:.6e}")

            if numIterations <= max_saves:
                save_indices_all = list(range(0, numIterations + 1))
            else:
                step = max(1, numIterations // max_saves)
                save_indices_all = list(range(0, numIterations + 1, step))

            device_str = f"GPU {device}"
            description = f"AOT-BioMaps -- Primal/Dual Reconstruction (LS-TV) -- {tumor_str} -- {device_str}"
            iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

            I_reconMatrix = []
            saved_indices = []

            if isSavingEachIteration and 0 in save_indices_all:
                I_reconMatrix.append(cp.asnumpy(x.reshape((Z, X)) * maxy))
                saved_indices.append(0)

            for it in iterator:
                grad = gradient(x_tilde.reshape(Z, X)).ravel()
                p = proj_l2(p + sigma * grad, alpha)

                Ax = SMatrix.projection(x_tilde)
                q = (q + sigma * (Ax - y_normed)) / (1 + sigma)

                x_old = x.copy()
                div_p = div(p.reshape(2, Z, X)).ravel()
                ATq = SMatrix.backprojection(q)
                x = x - tau * (ATq - div_p)

                if apply_positivity_clamp:
                    x = cp.maximum(x, 0.0)

                if beta > 0:
                    if tikhonov_as_gradient:
                        x = x * (1.0 - 2.0 * tau * beta)
                    else:
                        x = x / (1.0 + 2.0 * tau * beta)

                x_tilde = x + theta * (x - x_old)

                if isSavingEachIteration and (it + 1) in save_indices_all:
                    I_reconMatrix.append(cp.asnumpy(x.reshape((Z, X)) * maxy))
                    saved_indices.append(it + 1)

            cp.cuda.Stream.null.synchronize()

            if isSavingEachIteration:
                return I_reconMatrix, saved_indices
            else:
                return cp.asnumpy(x.reshape((Z, X)) * maxy), None

    except Exception as e:
        print(f"Error in CP_TV_cupy: {type(e).__name__}: {e}")
        return None, None
