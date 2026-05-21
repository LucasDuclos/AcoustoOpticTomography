from AOT_biomaps.Config import config
from AOT_biomaps.AOT_Recon.ReconTools import calculate_memory_requirement, check_gpu_memory
from AOT_biomaps.AOT_Recon.ReconEnums import SMatrixType

import numpy as np
from tqdm import trange
import gc

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

def LS(
    SMatrix,
    y,
    numIterations=100,
    alpha=0.01,
    isSavingEachIteration=True,
    withTumor=True,
    max_saves=5000,
    show_logs=True,
    smatrixType=SMatrixType.SELL
):
    """
    Least Squares reconstruction using Projected Gradient Descent (PGD) with non-negativity constraint.
    Currently only implements GPU versions with CuPy.
    """
    tumor_str = "WITH" if withTumor else "WITHOUT"
    # Auto-select device and method
    use_gpu = check_gpu_memory(config.select_best_gpu(), calculate_memory_requirement(SMatrix, y), show_logs=show_logs)
    # Dispatch to the appropriate implementation
    if use_gpu:
        if smatrixType == SMatrixType.CSR:
            return _LS_CG_sparseCSR_cupy(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves, show_logs)
        elif smatrixType == SMatrixType.SELL:
            return _LS_CG_sparseSELL_cupy(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves, show_logs)
        elif smatrixType == SMatrixType.DENSE:
            return _LS_GPU_stable(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves, show_logs)
        else:
            raise ValueError("Unsupported SMatrixType for GPU LS.")
    else:
        raise NotImplementedError("Only GPU implementations are currently available for LS.")

def _LS_GPU_stable(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves=5000, show_logs=True):
    """
    Stable GPU implementation of LS using projected gradient descent with diagonal preconditioner.
    """
    T, Z, X, N = SMatrix.shape
    ZX = Z * X
    TN = T * N
    # 1. Conversion and normalization
    A_flat = cp.asarray(SMatrix, dtype=cp.float32).transpose(0, 3, 1, 2).reshape(TN, ZX)
    y_flat = cp.asarray(y, dtype=cp.float32).reshape(TN)
    norm_A = A_flat.max()
    norm_y = y_flat.max()
    A_flat /= (norm_A + 1e-8)
    y_flat /= (norm_y + 1e-8)
    # 2. Initialization
    lambda_k = cp.zeros(ZX, dtype=cp.float32)
    lambda_history = [] if isSavingEachIteration else None
    saved_indices = []  # For storing saved iteration indices

    # Calculate save indices
    if numIterations <= max_saves:
        save_indices = list(range(numIterations))
    else:
        step = numIterations // max_saves
        save_indices = list(range(0, numIterations, step))
        if save_indices[-1] != numIterations - 1:
            save_indices.append(numIterations - 1)

    # Diagonal preconditioner
    diag_AAT = cp.sum(A_flat ** 2, axis=0)
    M_inv = 1.0 / cp.clip(diag_AAT, a_min=1e-6, a_max=None)
    # Pre-allocate arrays
    r_k = cp.empty_like(y_flat)
    AT_r = cp.empty(ZX, dtype=cp.float32)
    description = f"AOT-BioMaps -- Stable LS Reconstruction ---- {tumor_str} TUMOR ---- GPU {cp.cuda.runtime.getDevice()}"

    iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)
    for it in iterator:
        # Compute residual
        r_k = A_flat @ lambda_k
        r_k = y_flat - r_k
        if isSavingEachIteration and it in save_indices:
            lambda_history.append((lambda_k.reshape(Z, X) * (norm_y / norm_A)).get())
            saved_indices.append(it)

        # Preconditioned gradient
        AT_r = A_flat.T @ r_k
        AT_r *= M_inv
        # Update with fixed step and projection
        lambda_k += alpha * AT_r
        lambda_k = cp.clip(lambda_k, a_min=0, a_max=None)

    # 3. Denormalization
    lambda_final = (lambda_k.reshape(Z, X) * (norm_y / norm_A)).get()
    # Free memory
    del A_flat, y_flat, r_k, AT_r
    cp.cuda.Stream.null.synchronize()
    if isSavingEachIteration:
        return lambda_history, saved_indices
    else:
        return lambda_final, None

def _LS_GPU_opti(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_GPU_multi(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_CPU_opti(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_CPU_basic(*args, **kwargs):
    raise NotImplementedError("Only _LS_GPU_stable is implemented for now.")

def _LS_CG_sparseCSR_cupy(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    Least Squares (LS) reconstruction using Conjugate Gradient (CG) with CSR sparse matrix format.
    Uses CuPy for GPU acceleration.

    Parameters:
        SMatrix: Instance of SparseSMatrix_CSR (already allocated with CuPy arrays)
        y: Measured data (1D np.float32 of size TN)
        numIterations: Maximum number of CG iterations
        alpha: Step size parameter (not used in pure CG, kept for compatibility)
        isSavingEachIteration: Whether to save all iterations
        tumor_str: String indicating tumor presence ("WITH" or "WITHOUT")
        max_saves: Maximum number of iterations to save
        show_logs: Whether to show progress bar
    """
    final_result = None

    # ---
    def _dot_product_gpu(a_ptr, b_ptr, N_int, stream):
        block_size = 256
        grid_size = (N_int + block_size - 1) // block_size

        reduction_host = cp.empty(grid_size, dtype=cp.float32)
        reduction_buffer = cp.cuda.alloc(reduction_host.nbytes)

        dot_kernel = SMatrix.sparse_mod.get_function("dot_product_reduction_kernel")

        dot_kernel(grid=(grid_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                  args=[reduction_buffer, a_ptr, b_ptr, np.int32(N_int)])

        cp.cuda.memcpy_dtoh(reduction_host, reduction_buffer)
        total_dot = cp.sum(reduction_host).get()

        reduction_buffer.free()
        return total_dot

    try:
        if not hasattr(SMatrix, 'sparse_mod'):
            raise TypeError("SMatrix must have a sparse_mod attribute with loaded CUDA kernels")

        dtype = np.float32
        TN = SMatrix.N * SMatrix.T
        ZX = SMatrix.Z * SMatrix.X
        Z = SMatrix.Z
        X = SMatrix.X
        block_size = 256
        tolerance = 1e-12

        if show_logs:
            print(f"Executing on GPU device: {cp.cuda.runtime.getDevice()}")
            print(f"Dim X: {X}, Dim Z: {Z}, TN: {TN}, ZX: {ZX}")

        stream = cp.cuda.Stream()

        # Allocate buffers
        y = y.T.flatten().astype(dtype)
        y_gpu = cp.cuda.alloc(y.nbytes)
        cp.cuda.memcpy_htod_async(y_gpu, y, stream)

        theta_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        cp.cuda.memcpy_htod_async(theta_flat_gpu, cp.full(ZX, 0.1, dtype=dtype), stream)

        q_flat_gpu = cp.cuda.alloc(TN * np.dtype(dtype).itemsize)
        r_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        p_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        z_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        ATy_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)

        # ---
        # 1. ATy = A^T * y
        cp.cuda.memset_d32_async(ATy_flat_gpu, 0, ZX, stream)
        backprojection_kernel = SMatrix.sparse_mod.get_function('backprojection_kernel__CSR')
        backprojection_kernel(
            grid=((TN + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
            args=[ATy_flat_gpu, SMatrix.values_gpu, SMatrix.row_ptr_gpu, SMatrix.col_ind_gpu,
                  y_gpu, np.int32(TN)]
        )

        # 2. q = A * theta_0
        projection_kernel = SMatrix.sparse_mod.get_function('projection_kernel__CSR')
        projection_kernel(
            grid=((TN + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
            args=[q_flat_gpu, SMatrix.values_gpu, SMatrix.row_ptr_gpu, SMatrix.col_ind_gpu,
                  theta_flat_gpu, np.int32(TN)]
        )

        # 3. r_temp = A^T * q = A^T A theta_0
        cp.cuda.memset_d32_async(r_flat_gpu, 0, ZX, stream)
        backprojection_kernel(
            grid=((TN + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
            args=[r_flat_gpu, SMatrix.values_gpu, SMatrix.row_ptr_gpu, SMatrix.col_ind_gpu,
                  q_flat_gpu, np.int32(TN)]
        )

        # 4. r_0 = ATy - r_temp (r = ATy + (-1)*r_temp)
        axpby_kernel = SMatrix.sparse_mod.get_function("vector_axpby_kernel")
        axpby_kernel(
            grid=((ZX + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
            args=[r_flat_gpu, ATy_flat_gpu, r_flat_gpu, np.float32(1.0), np.float32(-1.0), np.int32(ZX)]
        )

        # 5. p_0 = r_0
        cp.cuda.memcpy_dtod_async(p_flat_gpu, r_flat_gpu, ZX * np.dtype(dtype).itemsize, stream)

        # 6. rho_prev = ||r_0||^2
        rho_prev = _dot_product_gpu(r_flat_gpu, r_flat_gpu, ZX, stream)

        # ---
        saved_theta, saved_indices_list = [], []
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            save_indices = list(range(0, numIterations, max(1, numIterations // max_saves)))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        description = f"AOT-BioMaps -- LS-CG (CSR-sparse SMatrix) ---- {tumor_str} TUMOR ---- GPU {cp.cuda.runtime.getDevice()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        for it in iterator:
            # a. q = A * p
            projection_kernel(
                grid=((TN + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                args=[q_flat_gpu, SMatrix.values_gpu, SMatrix.row_ptr_gpu, SMatrix.col_ind_gpu,
                      p_flat_gpu, np.int32(TN)]
            )

            # b. z = A^T * q = A^T A p
            cp.cuda.memset_d32_async(z_flat_gpu, 0, ZX, stream)
            backprojection_kernel(
                grid=((TN + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                args=[z_flat_gpu, SMatrix.values_gpu, SMatrix.row_ptr_gpu, SMatrix.col_ind_gpu,
                      q_flat_gpu, np.int32(TN)]
            )

            # c. alpha = rho_prev / <p, z>
            pAp = _dot_product_gpu(p_flat_gpu, z_flat_gpu, ZX, stream)

            if abs(pAp) < 1e-15: break
            alpha_val = rho_prev / pAp

            # d. theta = theta + alpha * p
            axpby_kernel(
                grid=((ZX + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                args=[theta_flat_gpu, theta_flat_gpu, p_flat_gpu, np.float32(1.0), alpha_val, np.int32(ZX)]
            )

            # e. r = r - alpha * z
            minus_axpy_kernel = SMatrix.sparse_mod.get_function("vector_minus_axpy_kernel")
            minus_axpy_kernel(
                grid=((ZX + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                args=[r_flat_gpu, z_flat_gpu, alpha_val, np.int32(ZX)]
            )

            # f. rho_curr = ||r||^2
            rho_curr = _dot_product_gpu(r_flat_gpu, r_flat_gpu, ZX, stream)

            if rho_curr < tolerance: break

            # g. beta = rho_curr / rho_prev
            beta = rho_curr / rho_prev

            # h. p = r + beta * p
            axpby_kernel(
                grid=((ZX + block_size - 1) // block_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                args=[p_flat_gpu, r_flat_gpu, p_flat_gpu, np.float32(1.0), beta, np.int32(ZX)]
            )

            rho_prev = rho_curr

            if show_logs and (it % 10 == 0 or it == numIterations - 1):
                stream.synchronize()

            if isSavingEachIteration and it in save_indices:
                theta_host = cp.empty(ZX, dtype=dtype)
                cp.cuda.memcpy_dtoh(theta_host, theta_flat_gpu)
                saved_theta.append(theta_host.get().reshape(Z, X))
                saved_indices_list.append(it)

        stream.synchronize()

        final_result = cp.empty(ZX, dtype=dtype)
        cp.cuda.memcpy_dtoh(final_result, theta_flat_gpu)
        final_result = final_result.get().reshape(Z, X)

        # Free temporaries
        y_gpu.free()
        q_flat_gpu.free()
        r_flat_gpu.free()
        p_flat_gpu.free()
        z_flat_gpu.free()
        theta_flat_gpu.free()
        ATy_flat_gpu.free()

        return (saved_theta, saved_indices_list) if isSavingEachIteration else (final_result, None)

    except Exception as e:
        print(f"Error in LS_CG_sparseCSR_cupy: {type(e).__name__}: {e}")
        gc.collect()
        return None, None

def _LS_CG_sparseSELL_cupy(SMatrix, y, numIterations, alpha, isSavingEachIteration, tumor_str, max_saves, show_logs=True):
    """
    Least Squares (LS) reconstruction using Conjugate Gradient (CG) with SELL-C-sigma sparse matrix format.
    Uses CuPy for GPU acceleration.

    Parameters:
        SMatrix: Instance of SparseSMatrix_SELL (already allocated with CuPy arrays)
        y: Measured data (1D np.float32 of size TN)
        numIterations: Maximum number of CG iterations
        alpha: Step size parameter (not used in pure CG, kept for compatibility)
        isSavingEachIteration: Whether to save all iterations
        tumor_str: String indicating tumor presence ("WITH" or "WITHOUT")
        max_saves: Maximum number of iterations to save
        show_logs: Whether to show progress bar
    """
    final_result = None

    # ---
    def _dot_product_gpu(a_ptr, b_ptr, N_int, stream):
        block_size = 256
        grid_size = (N_int + block_size - 1) // block_size

        reduction_host = cp.empty(grid_size, dtype=cp.float32)
        reduction_buffer = cp.cuda.alloc(reduction_host.nbytes)

        dot_kernel = SMatrix.sparse_mod.get_function("dot_product_reduction_kernel")

        dot_kernel(grid=(grid_size, 1, 1), block=(block_size, 1, 1), stream=stream,
                  args=[reduction_buffer, a_ptr, b_ptr, np.int32(N_int)])

        cp.cuda.memcpy_dtoh(reduction_host, reduction_buffer)
        total_dot = cp.sum(reduction_host).get()

        reduction_buffer.free()
        return total_dot

    try:
        if not hasattr(SMatrix, 'sparse_mod'):
            raise TypeError("SMatrix must have a sparse_mod attribute with loaded CUDA kernels")
        if SMatrix.sell_values_gpu is None:
            raise RuntimeError("SELL not built. Call allocate_sell_c_sigma_direct() first.")

        dtype = np.float32
        TN = int(SMatrix.N * SMatrix.T)
        ZX = int(SMatrix.Z * SMatrix.X)
        Z = SMatrix.Z
        X = SMatrix.X
        block_size = 256
        tolerance = 1e-12

        # Access SELL parameters
        projection_kernel = SMatrix.sparse_mod.get_function("projection_kernel__SELL")
        backprojection_kernel = SMatrix.sparse_mod.get_function("backprojection_kernel__SELL")
        axpby_kernel = SMatrix.sparse_mod.get_function("vector_axpby_kernel")
        minus_axpy_kernel = SMatrix.sparse_mod.get_function("vector_minus_axpy_kernel")
        slice_height = np.int32(SMatrix.slice_height)
        grid_rows = ((TN + block_size - 1) // block_size, 1, 1)

        stream = cp.cuda.Stream()

        # Allocate buffers
        y = y.T.flatten().astype(dtype)
        y_gpu = cp.cuda.alloc(y.nbytes)
        cp.cuda.memcpy_htod_async(y_gpu, y, stream)

        theta_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        cp.cuda.memcpy_htod_async(theta_flat_gpu, cp.full(ZX, 0.1, dtype=dtype), stream)

        q_flat_gpu = cp.cuda.alloc(TN * np.dtype(dtype).itemsize)
        r_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        p_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        z_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)
        ATy_flat_gpu = cp.cuda.alloc(ZX * np.dtype(dtype).itemsize)

        # ---
        # 1. ATy = A^T * y
        cp.cuda.memset_d32_async(ATy_flat_gpu, 0, ZX, stream)
        backprojection_kernel(
            SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu, SMatrix.slice_ptr_gpu, SMatrix.slice_len_gpu,
            y_gpu, ATy_flat_gpu, np.int32(TN), slice_height,
            block=(block_size, 1, 1), grid=grid_rows, stream=stream
        )

        # 2. q = A * theta_0
        projection_kernel(
            q_flat_gpu, SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu, SMatrix.slice_ptr_gpu, SMatrix.slice_len_gpu,
            theta_flat_gpu, np.int32(TN), slice_height,
            block=(block_size, 1, 1), grid=grid_rows, stream=stream
        )

        # 3. r_temp = A^T * q = A^T A theta_0
        cp.cuda.memset_d32_async(r_flat_gpu, 0, ZX, stream)
        backprojection_kernel(
            SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu, SMatrix.slice_ptr_gpu, SMatrix.slice_len_gpu,
            q_flat_gpu, r_flat_gpu, np.int32(TN), slice_height,
            block=(block_size, 1, 1), grid=grid_rows, stream=stream
        )

        # 4. r_0 = ATy - r_temp
        axpby_kernel(
            r_flat_gpu, ATy_flat_gpu, r_flat_gpu,
            np.float32(1.0), np.float32(-1.0), np.int32(ZX),
            block=(block_size, 1, 1), grid=((ZX + block_size - 1) // block_size, 1, 1), stream=stream
        )

        # 5. p_0 = r_0
        cp.cuda.memcpy_dtod_async(p_flat_gpu, r_flat_gpu, ZX * np.dtype(dtype).itemsize, stream)

        # 6. rho_prev = ||r_0||^2
        rho_prev = _dot_product_gpu(r_flat_gpu, r_flat_gpu, ZX, stream)

        # ---
        saved_theta, saved_indices_list = [], []
        if numIterations <= max_saves:
            save_indices = list(range(numIterations))
        else:
            save_indices = list(range(0, numIterations, max(1, numIterations // max_saves)))
            if save_indices[-1] != numIterations - 1:
                save_indices.append(numIterations - 1)

        description = f"AOT-BioMaps -- LS-CG (SELL-c-σ-sparse SMatrix) ---- {tumor_str} TUMOR ---- GPU {cp.cuda.runtime.getDevice()}"
        iterator = trange(numIterations, desc=description) if show_logs else range(numIterations)

        for it in iterator:
            # a. q = A * p
            projection_kernel(
                q_flat_gpu, SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu, SMatrix.slice_ptr_gpu, SMatrix.slice_len_gpu,
                p_flat_gpu, np.int32(TN), slice_height,
                block=(block_size, 1, 1), grid=grid_rows, stream=stream
            )

            # b. z = A^T * q = A^T A p
            cp.cuda.memset_d32_async(z_flat_gpu, 0, ZX, stream)
            backprojection_kernel(
                SMatrix.sell_values_gpu, SMatrix.sell_colinds_gpu, SMatrix.slice_ptr_gpu, SMatrix.slice_len_gpu,
                q_flat_gpu, z_flat_gpu, np.int32(TN), slice_height,
                block=(block_size, 1, 1), grid=grid_rows, stream=stream
            )

            # c. alpha = rho_prev / <p, z>
            pAp = _dot_product_gpu(p_flat_gpu, z_flat_gpu, ZX, stream)

            if abs(pAp) < 1e-15: break
            alpha_val = rho_prev / pAp

            # d. theta = theta + alpha * p
            axpby_kernel(
                theta_flat_gpu, theta_flat_gpu, p_flat_gpu,
                np.float32(1.0), alpha_val, np.int32(ZX),
                block=(block_size, 1, 1), grid=((ZX + block_size - 1) // block_size, 1, 1), stream=stream
            )

            # e. r = r - alpha * z
            minus_axpy_kernel(
                r_flat_gpu, z_flat_gpu, alpha_val, np.int32(ZX),
                block=(block_size, 1, 1), grid=((ZX + block_size - 1) // block_size, 1, 1), stream=stream
            )

            # f. rho_curr = ||r||^2
            rho_curr = _dot_product_gpu(r_flat_gpu, r_flat_gpu, ZX, stream)

            if rho_curr < tolerance: break

            # g. beta = rho_curr / rho_prev
            beta = rho_curr / rho_prev

            # h. p = r + beta * p
            axpby_kernel(
                p_flat_gpu, r_flat_gpu, p_flat_gpu,
                np.float32(1.0), beta, np.int32(ZX),
                block=(block_size, 1, 1), grid=((ZX + block_size - 1) // block_size, 1, 1), stream=stream
            )

            rho_prev = rho_curr

            stream.synchronize()
            if isSavingEachIteration and it in save_indices:
                out = cp.empty(ZX, dtype=dtype)
                cp.cuda.memcpy_dtoh(out, theta_flat_gpu)
                saved_theta.append(out.get().reshape((Z, X)))
                saved_indices_list.append(it)

        # Final copy
        res = cp.empty(ZX, dtype=np.float32)
        cp.cuda.memcpy_dtoh(res, theta_flat_gpu)
        final_result = res.get().reshape((Z, X))

        # Free temporaries
        y_gpu.free()
        q_flat_gpu.free()
        r_flat_gpu.free()
        p_flat_gpu.free()
        z_flat_gpu.free()
        theta_flat_gpu.free()
        ATy_flat_gpu.free()

        return (saved_theta, saved_indices_list) if isSavingEachIteration else (final_result, None)

    except Exception as e:
        print(f"Error in LS_CG_sparseSELL_cupy: {type(e).__name__}: {e}")
        gc.collect()
        return None, None