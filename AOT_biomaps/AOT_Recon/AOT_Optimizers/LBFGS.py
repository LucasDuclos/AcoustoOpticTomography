import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import time

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def lbfgs_aniso_tv(SMatrix, y, n_epochs=100, window=5, tol=1e-5, alpha_x=0.01, alpha_z=0.05, eps=1e-4):
    """
    L-BFGS optimization with anisotropic TV regularization for acousto-optic tomography.
    Uses CuPy for GPU acceleration and custom CUDA kernels for sparse matrix operations.

    Parameters:
        SMatrix: Sparse matrix object with loaded CUDA module and GPU arrays.
        y: Measured data (2D array).
        n_epochs: Maximum number of iterations.
        window: Window size for smoothed convergence detection.
        tol: Tolerance for convergence detection.
        alpha_x: Regularization weight for x-direction (anisotropic TV).
        alpha_z: Regularization weight for z-direction (anisotropic TV).
        eps: Small constant for numerical stability in TV regularization.

    Returns:
        tuple: (history_x, history_cost) where:
            - history_x: List of reconstructed images at each saved iteration
            - history_cost: List of cost function values at each iteration
    """
    try:
        SMatrix.sparse_mod = cp.cuda.runtime.moduleFromFile(SMatrix.module_path)

        calc_q_k = SMatrix.sparse_mod.get_function("lbfgs_calc_q_kernel")
        back_k = SMatrix.sparse_mod.get_function("lbfgs_backprojection_kernel")
        aniso_tv_k = SMatrix.sparse_mod.get_function("lbfgs_aniso_tv_eval_kernel")
        div_k = SMatrix.sparse_mod.get_function("lbfgs_divergence_kernel")
        proj_op_full = SMatrix.sparse_mod.get_function("projection_kernel__SELL")

        Nx, Nz = SMatrix.X, SMatrix.Z
        ZX, TN = Nz * Nx, int(SMatrix.N * SMatrix.T)
        block = 256
        grid_ZX = ((ZX + block - 1) // block, 1, 1)
        grid_TN = ((TN + block - 1) // block, 1, 1)

        y_raw = y.T.flatten().astype(np.float32)
        y_scale = np.max(np.abs(y_raw)) + 1e-9
        y_gpu = cp.asarray(y_raw / y_scale)

        x_gpu = cp.cuda.alloc(ZX * 4)
        Ax_gpu = cp.cuda.alloc(TN * 4)
        q_gpu = cp.cuda.alloc(TN * 4)
        grad_data_gpu = cp.cuda.alloc(ZX * 4)

        p_gpu = cp.cuda.alloc(2 * ZX * 4)
        cost_reg_gpu = cp.cuda.alloc(ZX * 4)
        grad_reg_gpu = cp.cuda.alloc(ZX * 4)

        q_cpu = np.empty(TN, dtype=np.float32)
        cost_reg_cpu = np.empty(ZX, dtype=np.float32)
        grad_data_cpu = np.empty(ZX, dtype=np.float32)
        grad_reg_cpu = np.empty(ZX, dtype=np.float32)

        a_x_f32 = np.float32(alpha_x)
        a_z_f32 = np.float32(alpha_z)
        eps_f32 = np.float32(eps)

        history_x = []
        history_cost = []
        last_evaluated_cost = 0.0
        converged = False
        pbar = tqdm(total=n_epochs, desc="L-BFGS Anisotropic TV")

        def cost_and_grad(x_flat_cpu):
            nonlocal last_evaluated_cost, converged

            cp.cuda.memcpy_htod(x_gpu, x_flat_cpu.astype(np.float32))

            proj_op_full(
                grid=grid_TN, block=(block, 1, 1),
                args=[Ax_gpu, SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu,
                      SMatrix.slice_ptr_gpu, SMatrix.slice_len_gpu,
                      x_gpu, np.int32(TN), np.int32(SMatrix.slice_height)]
            )

            calc_q_k(
                grid=grid_TN, block=(block, 1, 1),
                args=[q_gpu, Ax_gpu, y_gpu, np.int32(TN)]
            )

            cp.cuda.memset_d32(grad_data_gpu, 0, ZX)
            back_k(
                grid=grid_TN, block=(block, 1, 1),
                args=[SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu, SMatrix.slice_ptr_gpu,
                      SMatrix.slice_len_gpu, q_gpu, grad_data_gpu, np.int32(TN), np.int32(SMatrix.slice_height)]
            )

            aniso_tv_k(
                grid=grid_ZX, block=(block, 1, 1),
                args=[p_gpu, cost_reg_gpu, x_gpu, a_x_f32, a_z_f32, eps_f32,
                      np.int32(Nz), np.int32(Nx), np.int32(ZX)]
            )

            div_k(
                grid=grid_ZX, block=(block, 1, 1),
                args=[grad_reg_gpu, p_gpu, np.int32(Nz), np.int32(Nx), np.int32(ZX)]
            )

            cp.cuda.memcpy_dtoh(q_cpu, q_gpu)
            cp.cuda.memcpy_dtoh(cost_reg_cpu, cost_reg_gpu)
            cp.cuda.memcpy_dtoh(grad_data_cpu, grad_data_gpu)
            cp.cuda.memcpy_dtoh(grad_reg_cpu, grad_reg_gpu)

            data_cost = 0.5 * np.sum(q_cpu**2)
            reg_cost = np.sum(cost_reg_cpu)
            total_cost = data_cost + reg_cost

            total_grad = (grad_data_cpu + grad_reg_cpu).astype(np.float64)
            last_evaluated_cost = total_cost

            return np.float64(total_cost), total_grad

        def epoch_callback(xk):
            history_x.append(xk.reshape((Nz, Nx)).copy() * y_scale)
            history_cost.append(last_evaluated_cost)

            if len(history_cost) >= 2 * window:
                mean_cost_curr = np.mean(history_cost[-window:])
                mean_cost_prev = np.mean(history_cost[-2*window:-window])

                cost_variation = np.abs(mean_cost_curr - mean_cost_prev) / (np.abs(mean_cost_prev) + 1e-12)
                pbar.set_postfix(dCost=f"{cost_variation:.2e}")

                if cost_variation < tol:
                    pbar.write(f"\nConvergence detected at epoch {len(history_cost)} (Variation: {cost_variation:.2e} < {tol})")
                    nonlocal converged
                    converged = True
            else:
                pbar.set_postfix(Cost=f"{last_evaluated_cost:.3e}")

            pbar.update(1)

        x0 = np.zeros(ZX, dtype=np.float64)
        bounds = [(0, None) for _ in range(ZX)]

        res = minimize(
            fun=cost_and_grad,
            x0=x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            callback=epoch_callback,
            options={
                'maxiter': n_epochs,
                'ftol': 1e-12,
                'gtol': 1e-12,
                'disp': False
            }
        )

        if converged:
            res.message = "Stopping condition reached (Tolerance on sliding window)."

        pbar.close()
        return history_x, history_cost

    finally:
        del x_gpu, Ax_gpu, q_gpu, grad_data_gpu, p_gpu, cost_reg_gpu, grad_reg_gpu
        cp.cuda.Stream.null.synchronize()