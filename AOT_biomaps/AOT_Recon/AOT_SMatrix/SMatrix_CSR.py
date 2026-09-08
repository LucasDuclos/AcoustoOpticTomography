import os
import warnings
import numpy as np
from tqdm import trange
from typing import Optional, Union

from AOT_biomaps.AOT_Recon.AOT_SMatrix._mainSMatrix import SMatrix
from AOT_biomaps.AOT_Recon.ReconEnums import SMatrixType
from AOT_biomaps.AOT_Recon.ReconTools import check_gpu_available

# Check for CuPy availability
try:
    import cupy as cp
    import cupyx
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

class SMatrix_CSR(SMatrix):
    """
    Construction of a CSR matrix from a `experiment` object.
    Supports both REAL and COMPLEX fields via `isComplexSMatrix`.
    """

    def __init__(self, block_rows: int = 128, relative_threshold: float = 0.01, **kwargs):
        """
        Initialize CSR sparse matrix.
        Args:
            block_rows (int): Number of rows to process per block when building on GPU.
            relative_threshold (float): Relative threshold for sparsity.
            **kwargs: Arguments passed to base SMatrix class.
        """
        super().__init__(**kwargs)
        self.matrix_type = SMatrixType.CSR
  
        self.block_rows = block_rows
        self.relative_threshold = relative_threshold

        self.row_ptr = None
        self.h_col_ind = None
        self.h_values = None
        self.total_nnz = 0

        self.row_ptr_gpu = None
        self.col_ind_gpu = None
        self.values_gpu = None

    def _allocate_gpu(self):
        """Allocate and fill the CSR matrix on GPU using 1-Pass PCIe strategy."""
        with cp.cuda.Device(self.gpu_index):
            num_rows = self.N * self.T
            num_cols = self.Z * self.X
            br = self.block_rows
            dtype = self._get_dtype()
            cp_dtype = self._get_cp_dtype()

            # Initialize global row pointer
            self.row_ptr = np.zeros(num_rows + 1, dtype=np.int64)

            # Temporary lists to hold local blocks of sparse data
            col_ind_list = []
            values_list = []

            dense_block_host = np.empty((br, num_cols), dtype=dtype)
            count_nnz_kernel_name = "count_nnz_rows_kernel__COMPLEX" if self.isComplexSMatrix else "count_nnz_rows_kernel__REAL"
            fill_csr_kernel_name = "fill_kernel__CSR__COMPLEX" if self.isComplexSMatrix else "fill_kernel__CSR__REAL"
            count_nnz_kernel = self.sparse_mod.get_function(count_nnz_kernel_name)
            fill_csr_kernel = self.sparse_mod.get_function(fill_csr_kernel_name)
            block_size = 256

            for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Filling CSR ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: {self.device.upper()}'):
                current_rows = min(br, num_rows - b)

                sorted_keys = sorted(list(self.experiment.AcousticFields_demodulated.keys()))

                for r in range(current_rows):
                    global_row = b + r
                    if self.isComplexSMatrix:
                        n_idx = global_row // self.T
                        key = sorted_keys[n_idx]
                        dense_block_host[r] = self.experiment.AcousticFields_demodulated[key][global_row % self.T].flatten()
                    else:
                        n_idx = global_row // self.T
                        t_idx = global_row % self.T
                        dense_block_host[r] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

                # 1. Send dense data to GPU ONCE
                dense_block_gpu = cp.asarray(dense_block_host[:current_rows], dtype=cp_dtype)
                row_nnz_gpu = cp.zeros(current_rows, dtype=np.int32)

                grid = ((current_rows + block_size - 1) // block_size, 1, 1)

                # 2. Count NNZ
                count_nnz_kernel(
                    grid=grid, block=(block_size, 1, 1),
                    args=[dense_block_gpu, row_nnz_gpu, np.int32(current_rows), np.int32(num_cols),
                          np.float32(self.relative_threshold)]
                )
                cp.cuda.Stream.null.synchronize()

                # 3. Compute local offsets
                row_nnz_host = cp.asnumpy(row_nnz_gpu)
                local_row_ptr = np.zeros(current_rows + 1, dtype=np.int64)
                local_row_ptr[1:] = np.cumsum(row_nnz_host)
                local_nnz = int(local_row_ptr[-1])

                # Accumulate into global row_ptr
                self.row_ptr[b + 1 : b + current_rows + 1] = self.row_ptr[b] + local_row_ptr[1:]

                if local_nnz > 0:
                    # 4. Fill local CSR exactly where the data lives
                    local_row_ptr_gpu = cp.asarray(local_row_ptr)
                    local_col_ind_gpu = cp.empty(local_nnz, dtype=np.uint32)
                    local_values_gpu = cp.empty(local_nnz, dtype=cp_dtype)

                    fill_csr_kernel(
                        grid=grid, block=(block_size, 1, 1),
                        args=[dense_block_gpu, local_row_ptr_gpu, local_col_ind_gpu, local_values_gpu,
                              np.int32(current_rows), np.int32(num_cols),
                              np.float32(self.relative_threshold), np.int64(local_nnz)]
                    )
                    cp.cuda.Stream.null.synchronize()

                    # 5. Bring compressed data back to CPU
                    col_ind_list.append(cp.asnumpy(local_col_ind_gpu))
                    values_list.append(cp.asnumpy(local_values_gpu))

            self.total_nnz = int(self.row_ptr[-1])

            # 6. Concatenate locally constructed CSR blocks
            if self.total_nnz > 0:
                self.h_col_ind = np.concatenate(col_ind_list)
                self.h_values = np.concatenate(values_list)
            else:
                self.h_col_ind = np.array([], dtype=np.uint32)
                self.h_values = np.array([], dtype=dtype)

            # 7. Final unified GPU transfer for operations
            self.row_ptr_gpu = cp.asarray(self.row_ptr)
            self.col_ind_gpu = cp.asarray(self.h_col_ind)
            self.values_gpu = cp.asarray(self.h_values, dtype=cp_dtype)

            del self.h_col_ind
            del self.h_values
            self.h_col_ind = None
            self.h_values = None

    def _allocate_cpu(self):
        """Allocate and fill the CSR matrix on CPU with vectorized block processing."""
        num_rows = self.N * self.T
        num_cols = self.Z * self.X
        dtype = self._get_dtype()
        br = self.block_rows

        self.row_ptr = np.zeros(num_rows + 1, dtype=np.int64)
    
        col_ind_list = []
        values_list = []

        sorted_keys = sorted(list(self.experiment.AcousticFields_demodulated.keys())) if self.isComplexSMatrix else None

        for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Building CSR ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: CPU'):
            current_rows = min(br, num_rows - b)
            dense_block_host = np.empty((current_rows, num_cols), dtype=dtype)

            for r in range(current_rows):
                global_row = b + r
                n_idx = global_row // self.T
                t_idx = global_row % self.T
                if self.isComplexSMatrix:
                    key = sorted_keys[n_idx]
                    dense_block_host[r] = self.experiment.AcousticFields_demodulated[key][t_idx].flatten()
                else:
                    dense_block_host[r] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

            abs_block = np.abs(dense_block_host)
            row_max = np.max(abs_block, axis=1, keepdims=True)
            thr = row_max * self.relative_threshold
            mask = abs_block > thr

            row_nnz = np.count_nonzero(mask, axis=1)
            local_row_ptr = np.zeros(current_rows + 1, dtype=np.int64)
            local_row_ptr[1:] = np.cumsum(row_nnz)
            local_nnz = int(local_row_ptr[-1])

            self.row_ptr[b + 1 : b + current_rows + 1] = self.row_ptr[b] + local_row_ptr[1:]

            if local_nnz > 0:
                rows_idx, cols_idx = np.nonzero(mask)
                col_ind_list.append(cols_idx.astype(np.uint32))
                values_list.append(dense_block_host[rows_idx, cols_idx])

        self.total_nnz = int(self.row_ptr[-1])

        if self.total_nnz > 0:
            self.h_col_ind = np.concatenate(col_ind_list)
            self.h_values = np.concatenate(values_list)
        else:
            self.h_col_ind = np.array([], dtype=np.uint32)
            self.h_values = np.array([], dtype=dtype)
        
        from scipy.sparse import csr_matrix
        self.scipy_csr = csr_matrix((self.h_values, self.h_col_ind, self.row_ptr), shape=(num_rows, num_cols))

    def compute_norm_factor(self):
        """Compute normalization factor from CSR matrix by summing absolute values."""
        ZX = self.Z * self.X

        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                col_sum_gpu = cp.zeros(ZX, dtype=np.float32)
                acc_kernel_name = "accumulate_columns_atomic__COMPLEX" if self.isComplexSMatrix else "accumulate_columns_atomic__REAL"
                acc_kernel = self.sparse_mod.get_function(acc_kernel_name)
                threads = 256
                blocks = (self.total_nnz + threads - 1) // threads

                acc_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[self.values_gpu, self.col_ind_gpu, np.int64(self.total_nnz), col_sum_gpu]
                )
                cp.cuda.Stream.null.synchronize()
                norm = cp.asnumpy(col_sum_gpu)
        else:
            norm = np.zeros(ZX, dtype=np.float32)
            for i in range(self.total_nnz):
                col = int(self.h_col_ind[i])
                norm[col] += np.abs(self.h_values[i])

        norm = np.maximum(norm.astype(np.float64), 1e-6)
        self.norm_factor_inv = (1.0 / norm).astype(np.float32)

        if CUPY_AVAILABLE:
            with cp.cuda.Device(self.gpu_index):
                self.norm_factor_inv_gpu = cp.asarray(self.norm_factor_inv)

    def forward_projection(self, theta: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Perform forward projection: q = A * theta."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()

        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                theta_gpu = cp.asarray(theta, dtype=cp_dtype) if not isinstance(theta, cp.ndarray) else theta
                if theta_gpu.dtype != cp_dtype:
                    theta_gpu = theta_gpu.astype(cp_dtype)
                q_gpu = cp.zeros(self.N * self.T, dtype=cp_dtype)

                proj_kernel_name = "forward_projection_kernel__CSR__COMPLEX" if self.isComplexSMatrix else "forward_projection_kernel__CSR__REAL"
                proj_kernel = self.sparse_mod.get_function(proj_kernel_name)
                threads = 256
                blocks = (self.N * self.T + threads - 1) // threads

                proj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[q_gpu.data.ptr, self.values_gpu, self.row_ptr_gpu, self.col_ind_gpu,
                          theta_gpu.data.ptr, np.int32(self.N * self.T)]
                )
                cp.cuda.Stream.null.synchronize()
                return q_gpu
        else:
            theta_cpu = np.asarray(theta, dtype=dtype) if not isinstance(theta, np.ndarray) else theta
            return self.scipy_csr.dot(theta_cpu)

    def backward_projection(self, e: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Perform backward projection: c = A^T * e."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()

        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                e_gpu = cp.asarray(e, dtype=cp_dtype) if not isinstance(e, cp.ndarray) else e
                if e_gpu.dtype != cp_dtype:
                    e_gpu = e_gpu.astype(cp_dtype)
                c_gpu = cp.zeros(self.Z * self.X, dtype=cp_dtype)

                backproj_kernel_name = "backward_projection_kernel__CSR__COMPLEX" if self.isComplexSMatrix else "backward_projection_kernel__CSR__REAL"
                backproj_kernel = self.sparse_mod.get_function(backproj_kernel_name)
                threads = 256
                blocks = (self.N * self.T + threads - 1) // threads

                backproj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[c_gpu.data.ptr, self.values_gpu, self.row_ptr_gpu, self.col_ind_gpu,
                          e_gpu.data.ptr, np.int32(self.N * self.T)]
                )
                cp.cuda.Stream.null.synchronize()
                return c_gpu
        else:
            e_cpu = np.asarray(e, dtype=dtype) if not isinstance(e, np.ndarray) else e
            return self.scipy_csr.T.dot(e_cpu)
        
    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        """Apply apodization window to the matrix values."""
        raise NotImplementedError("[AOT-biomaps] Apodization not implemented for CSR matrix.")

    def compute_density(self) -> float:
        """Returns the actual density of the CSR matrix in percentage."""
        if self.row_ptr is None and self.row_ptr_gpu is None:
            raise RuntimeError("[AOT-biomaps] Sparse matrix not allocated yet.")
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        density_ratio = self.total_nnz / (num_rows * num_cols)
        return density_ratio * 100.0

    def get_matrix_size(self) -> dict:
        """Returns the total size of the CSR matrix in GB."""
        if self.row_ptr is None and self.row_ptr_gpu is None:
            return {"error": "[AOT-biomaps] Sparse matrix not allocated yet."}

        total_bytes = 0

        if self.row_ptr is not None: total_bytes += self.row_ptr.nbytes
        if self.h_col_ind is not None: total_bytes += self.h_col_ind.nbytes
        if self.h_values is not None: total_bytes += self.h_values.nbytes
        if self.norm_factor_inv is not None: total_bytes += self.norm_factor_inv.nbytes

        if self.row_ptr_gpu is not None: total_bytes += self.row_ptr_gpu.nbytes
        if self.col_ind_gpu is not None: total_bytes += self.col_ind_gpu.nbytes
        if self.values_gpu is not None: total_bytes += self.values_gpu.nbytes
        if self.norm_factor_inv_gpu is not None: total_bytes += self.norm_factor_inv_gpu.nbytes

        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024**3),
            "device": self.device
        }

    def _free_specific(self):
        """Free all GPU memory allocated by CSR."""
        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                attrs = ['col_ind_gpu', 'values_gpu', 'row_ptr_gpu', 'norm_factor_inv_gpu']
                for attr in attrs:
                    gpu_mem = getattr(self, attr, None)
                    if gpu_mem is not None:
                        try:
                            setattr(self, attr, None)
                            if hasattr(gpu_mem, 'free'):
                                gpu_mem.free()
                            del gpu_mem
                        except Exception as e:
                            warnings.warn(f"[AOT-biomaps] Error freeing {attr}: {e}")

                if CUPY_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
        
    def compute_hessian_diagonal(self):
        """
        Compute diag(A^H A).
        """
        ZX = self.Z * self.X

        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                diag = cp.zeros(ZX, dtype=cp.float32)
                cupyx.scatter_add(diag, self.col_ind_gpu.astype(cp.int32), cp.abs(self.values_gpu) ** 2)
                return diag
        else:
            diag = np.zeros(ZX, dtype=np.float32)
            np.add.at(diag, self.h_col_ind.astype(np.int64), np.abs(self.h_values) ** 2)
            return diag
        
    def normalize_matrix(self):
        """
        Normalizes the CSR matrix by its maximum absolute value.
        Restores the system conditioning for Primal-Dual solvers.
        """
        max_val = 0.0
        
        if check_gpu_available(self) and self.values_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                max_val = float(cp.max(cp.abs(self.values_gpu)))
                if max_val > 0:
                    self.values_gpu /= max_val
                    # Check for the existence of the CPU cache (which is sometimes deleted in _allocate_gpu)
                    if getattr(self, 'h_values', None) is not None:
                        self.h_values /= max_val
        elif getattr(self, 'h_values', None) is not None:
            max_val = float(np.max(np.abs(self.h_values)))
            if max_val > 0:
                self.h_values /= max_val
        else:
            warnings.warn("[AOT-biomaps] CSR Matrix not allocated, normalization impossible.")
            return
        self.normalization_factor = max_val
        
        print(f"[AOT-biomaps] CSR Matrix normalized (Original absolute max: {max_val:.2e})")
        
        # Critical update of the normalization factors (preconditioners)
        self.compute_norm_factor()

    def compute_absolute_row_col_sums(self):
        """
        Computes row and column sums of absolute values (|A| * 1 and |A|^T * 1) 
        without phase cancellation for complex matrices in CSR format.
        """
        is_gpu = check_gpu_available(self)
        ZX = int(self.Z * self.X)
        NT = int(self.N * self.T)
        
        if is_gpu:
            with cp.cuda.Device(self.gpu_index):
                col_sums = cp.zeros(ZX, dtype=cp.float32)
                row_sums = cp.zeros(NT, dtype=cp.float32)
                
                abs_vals = cp.abs(self.values_gpu)
                valid = abs_vals != 0
                
                # Column sums (|A|^T * 1)
                cupyx.scatter_add(col_sums, self.col_ind_gpu[valid].astype(cp.int32), abs_vals[valid].astype(cp.float32))
                
                # Row sums (|A| * 1) via segment reduction over row_ptr
                row_ptr_host = cp.asnumpy(self.row_ptr_gpu)
                abs_vals_host = cp.asnumpy(abs_vals)
                row_sums_host = np.zeros(NT, dtype=np.float32)
                for i in trange(NT, desc="[AOT-biomaps] Computing row and column sums (GPU)"):
                    start = row_ptr_host[i]
                    end = row_ptr_host[i+1]
                    if end > start:
                        row_sums_host[i] = np.sum(abs_vals_host[start:end])
                row_sums = cp.asarray(row_sums_host)
                
                return row_sums, col_sums
        else:
            col_sums = np.zeros(ZX, dtype=np.float32)
            row_sums = np.zeros(NT, dtype=np.float32)
            
            for i in trange(NT, desc="[AOT-biomaps] Computing row and column sums (CPU)"):
                start = int(self.row_ptr[i])
                end = int(self.row_ptr[i+1])
                row_acc = 0.0
                for j in range(start, end):
                    val = self.h_values[j]
                    if val != 0:
                        col = int(self.h_col_ind[j])
                        abs_val = float(np.abs(val))
                        col_sums[col] += abs_val
                        row_acc += abs_val
                row_sums[i] = row_acc
                
            return row_sums, col_sums