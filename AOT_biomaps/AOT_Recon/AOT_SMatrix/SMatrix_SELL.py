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

class SMatrix_SELL(SMatrix):
    """
    Sparse matrix in SELL-C-sigma format for efficient GPU operations.
    Supports both REAL and COMPLEX fields via `isComplexSMatrix`.
    """

    def __init__(self, block_rows: int = 256, relative_threshold: float = 0.01,
                 slice_height: int = 32, sigma: int = 4096, **kwargs):
        """
        Initialize SELL sparse matrix.

        Args:
            block_rows (int): Number of rows to process per block when building on GPU.
            relative_threshold (float): Relative threshold for sparsity.
            slice_height (int): Number of rows per slice in SELL format (32 for NVIDIA warps).
            sigma (int): Sorting window size for padding minimization (must be a multiple of slice_height).
            **kwargs: Arguments passed to base SMatrix class.
        """
        super().__init__(**kwargs)
        self.matrix_type = SMatrixType.SELL

        # Hyperparameters
        self.block_rows = block_rows
        self.relative_threshold = relative_threshold
        self.slice_height = slice_height
        self.sigma = sigma

        # Attributes specific to SELL
        self.sell_values = None
        self.sell_colinds = None
        self.slice_ptr = None
        self.slice_len = None
        self.total_storage = 0
        self.total_nnz = 0

        # Permutation arrays for SELL-C-sigma
        self.row_perm = None
        self.inv_row_perm = None

        # GPU arrays
        self.sell_values_gpu = None
        self.sell_colinds_gpu = None
        self.slice_ptr_gpu = None
        self.slice_len_gpu = None
        self.row_perm_gpu = None
        self.inv_row_perm_gpu = None

    def _apply_sigma_sorting(self, row_nnz: np.ndarray, num_rows: int) -> np.ndarray:
        """Applies local sorting within blocks of size sigma to minimize padding."""
        self.row_perm = np.arange(num_rows, dtype=np.int32)

        for start_idx in range(0, num_rows, self.sigma):
            end_idx = min(start_idx + self.sigma, num_rows)
            chunk_nnz = row_nnz[start_idx:end_idx]
            local_perm = np.argsort(chunk_nnz)[::-1]
            self.row_perm[start_idx:end_idx] = self.row_perm[start_idx:end_idx][local_perm]

        self.inv_row_perm = np.empty_like(self.row_perm)
        self.inv_row_perm[self.row_perm] = np.arange(num_rows)

        if CUPY_AVAILABLE:
            with cp.cuda.Device(self.gpu_index):
                self.row_perm_gpu = cp.asarray(self.row_perm)
                self.inv_row_perm_gpu = cp.asarray(self.inv_row_perm)

        return row_nnz[self.row_perm]

    def _allocate_gpu(self):
        """Allocate and fill the SELL matrix on GPU using PCIe block-streaming to prevent OOM."""
        with cp.cuda.Device(self.gpu_index):
            num_rows = int(self.N * self.T)
            num_cols = int(self.Z * self.X)
            self.total_nnz = 0
            C = int(self.slice_height)
            br = int(self.block_rows)
            dtype = self._get_dtype()
            cp_dtype = self._get_cp_dtype()

            # Temporary CPU buffer
            dense_block_host = np.empty((br, num_cols), dtype=dtype)

            # 1) Count NNZ per physical row
            row_nnz_gpu = cp.zeros(num_rows, dtype=np.int32)
            count_kernel_name = "count_nnz_rows_kernel__COMPLEX" if self.isComplexSMatrix else "count_nnz_rows_kernel__REAL"
            count_kernel = self.sparse_mod.get_function(count_kernel_name)
            threads = 256

            for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Count NNZ ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: {self.device.upper()}'):
                current_rows = min(br, num_rows - b)

                # Fetch a small block from CPU RAM
                for r in range(current_rows):
                    global_row = b + r
                    n_idx = global_row // self.T
                    t_idx = global_row % self.T
                    if self.isComplexSMatrix:
                        # For complex: use demodulated_fields
                        key = list(self.experiment.AcousticFields_demodulated.keys())[n_idx]
                        dense_block_host[r] = self.experiment.AcousticFields_demodulated[key][t_idx].flatten()
                    else:
                        # For real: use AcousticFields
                        dense_block_host[r] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

                # Transfer only this specific block to the GPU
                dense_gpu = cp.asarray(dense_block_host[:current_rows], dtype=cp_dtype)
                grid = ((current_rows + threads - 1) // threads, 1, 1)

                count_kernel(
                    grid=grid, block=(threads, 1, 1),
                    args=[dense_gpu, row_nnz_gpu[b:], np.int32(current_rows), np.int32(num_cols),
                        np.float32(self.relative_threshold)]
                )
                cp.cuda.Stream.null.synchronize()

            row_nnz = cp.asnumpy(row_nnz_gpu)

            # 2) Apply SELL-C-sigma sorting (on the CPU)
            row_nnz = self._apply_sigma_sorting(row_nnz, num_rows)

            # 3) Compute per-slice maxlen and slice_ptr based on sorted rows
            num_slices = (num_rows + C - 1) // C
            self.slice_len = np.zeros(num_slices, dtype=np.int32)
            self.slice_ptr = np.zeros(num_slices + 1, dtype=np.int64)

            for s in range(num_slices):
                r0 = s * C
                r1 = min(num_rows, r0 + C)
                self.slice_len[s] = int(np.max(row_nnz[r0:r1])) if (r1 > r0) else 0
                self.total_nnz += self.slice_len[s] * C 

            if np.all(self.slice_len == 0):
                raise ValueError("[AOT-biomaps] slice_len contains only zeros. Check row_nnz.")

            self.slice_ptr[0] = 0
            for s in range(num_slices):
                self.slice_ptr[s+1] = self.slice_ptr[s] + (self.slice_len[s] * C)
            self.total_storage = int(self.slice_ptr[-1])

            # Allocate final sparse arrays on GPU
            self.sell_values_gpu = cp.zeros(self.total_storage, dtype=cp_dtype)
            self.sell_colinds_gpu = cp.zeros(self.total_storage, dtype=np.uint32)
            self.slice_ptr_gpu = cp.asarray(self.slice_ptr)
            self.slice_len_gpu = cp.asarray(self.slice_len)

            # 4) Fill SELL arrays (Stream 2 - fetching dense rows according to permutation)
            fill_kernel_name = "fill_kernel__SELL__COMPLEX" if self.isComplexSMatrix else "fill_kernel__SELL__REAL"
            fill_kernel = self.sparse_mod.get_function(fill_kernel_name)

            for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Fill SELL ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: {self.device.upper()}'):
                current_rows = min(br, num_rows - b)

                # Fetch sorted blocks from CPU RAM
                for r in range(current_rows):
                    sorted_row = b + r
                    physical_row = int(self.row_perm[sorted_row])
                    if self.isComplexSMatrix:
                        n_idx = physical_row // self.T
                        key = list(self.experiment.AcousticFields_demodulated.keys())[n_idx]
                        dense_block_host[r] = self.experiment.AcousticFields_demodulated[key][physical_row % self.T].flatten()
                    else:
                        n_idx = physical_row // self.T
                        t_idx = physical_row % self.T
                        dense_block_host[r] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

                # Transfer the chunk to the GPU
                dense_gpu = cp.asarray(dense_block_host[:current_rows], dtype=cp_dtype)
                grid = ((current_rows + threads - 1) // threads, 1, 1)
                count_offset = b

                fill_kernel(
                    grid=grid, block=(threads, 1, 1),
                    args=[
                        dense_gpu,
                        row_nnz_gpu,
                        self.slice_ptr_gpu,
                        self.slice_len_gpu,
                        self.sell_colinds_gpu,
                        self.sell_values_gpu,
                        np.int32(current_rows),
                        np.int32(num_cols),
                        np.int32(count_offset),
                        np.int32(C),
                        np.float32(self.relative_threshold)
                    ]
                )
                cp.cuda.Stream.null.synchronize()

    def _allocate_cpu(self):
        """Allocate and fill the SELL matrix on CPU with vectorized block processing."""
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        self.total_nnz = 0
        C = int(self.slice_height)
        dtype = self._get_dtype()
        br = getattr(self, 'block_rows', 128)

        # 1) Count NNZ per physical row (par blocs vectorisés)
        row_nnz = np.zeros(num_rows, dtype=np.int32)
        sorted_keys = sorted(list(self.experiment.AcousticFields_demodulated.keys())) if self.isComplexSMatrix else None

        for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Count NNZ per row ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: CPU'):
            current_rows = min(br, num_rows - b)
            dense_block = np.empty((current_rows, num_cols), dtype=dtype)

            for r in range(current_rows):
                global_row = b + r
                n_idx = global_row // self.T
                t_idx = global_row % self.T
                if self.isComplexSMatrix:
                    key = sorted_keys[n_idx]
                    dense_block[r] = self.experiment.AcousticFields_demodulated[key][t_idx].flatten()
                else:
                    dense_block[r] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

            abs_block = np.abs(dense_block)
            row_max = np.max(abs_block, axis=1, keepdims=True)
            thr = row_max * self.relative_threshold
            row_nnz[b : b + current_rows] = np.count_nonzero(abs_block > thr, axis=1)

        # 2) Apply SELL-C-sigma sorting
        row_nnz = self._apply_sigma_sorting(row_nnz, num_rows)

        # 3) Compute per-slice maxlen and slice_ptr
        num_slices = (num_rows + C - 1) // C
        self.slice_len = np.zeros(num_slices, dtype=np.int32)

        for s in range(num_slices):
            r0 = s * C
            r1 = min(num_rows, r0 + C)
            self.slice_len[s] = int(np.max(row_nnz[r0:r1])) if (r1 > r0) else 0

        if np.all(self.slice_len == 0):
            raise ValueError("[AOT-biomaps] slice_len contains only zeros. Check row_nnz.")

        self.slice_ptr = np.zeros(num_slices + 1, dtype=np.int64)
        for s in range(num_slices):
            self.slice_ptr[s+1] = self.slice_ptr[s] + (self.slice_len[s] * C)
            self.total_nnz += self.slice_len[s] * C
        self.total_storage = int(self.slice_ptr[-1])

        # Allocate CPU arrays
        self.sell_values = np.zeros(self.total_storage, dtype=dtype)
        self.sell_colinds = np.zeros(self.total_storage, dtype=np.uint32)

        # 4) Fill SELL arrays using permuted order via batched processing
        for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Fill SELL ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: CPU'):
            current_rows = min(br, num_rows - b)
            dense_block = np.empty((current_rows, num_cols), dtype=dtype)

            for r in range(current_rows):
                sorted_row = b + r
                physical_row = int(self.row_perm[sorted_row])
                n_idx = physical_row // self.T
                t_idx = physical_row % self.T
                if self.isComplexSMatrix:
                    key = sorted_keys[n_idx]
                    dense_block[r] = self.experiment.AcousticFields_demodulated[key][t_idx].flatten()
                else:
                    dense_block[r] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

            abs_block = np.abs(dense_block)
            row_max = np.max(abs_block, axis=1, keepdims=True)
            thr_block = row_max * self.relative_threshold

            for r in range(current_rows):
                sorted_row = b + r
                row = dense_block[r]
                thr = thr_block[r, 0]

                slice_id = sorted_row // C
                row_in_slice = sorted_row % C
                base = int(self.slice_ptr[slice_id])
                len_slice = int(self.slice_len[slice_id])

                valid_cols = np.flatnonzero(np.abs(row) > thr)
                k = len(valid_cols)

                if k > 0:
                    pos_indices = base + row_in_slice + np.arange(k) * C
                    valid_mask = pos_indices < self.total_storage
                    self.sell_values[pos_indices[valid_mask]] = row[valid_cols[valid_mask]]
                    self.sell_colinds[pos_indices[valid_mask]] = valid_cols[valid_mask]

                if k < len_slice:
                    pad_indices = base + row_in_slice + np.arange(k, len_slice) * C
                    pad_mask = pad_indices < self.total_storage
                    self.sell_values[pad_indices[pad_mask]] = 0.0
                    self.sell_colinds[pad_indices[pad_mask]] = 0

        self.sell_rowinds = np.zeros(self.total_storage, dtype=np.int32)
        for s in range(num_slices):
            base = int(self.slice_ptr[s])
            length = int(self.slice_len[s])
            if length > 0:
                rows_in_slice = np.arange(s * C, s * C + C, dtype=np.int32)
                self.sell_rowinds[base:base + length * C] = np.tile(rows_in_slice, length)

    def forward_projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Perform forward projection: q = P^-1 * (A_sell * theta)."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()

        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                theta_gpu = cp.asarray(theta, dtype=cp_dtype)
                theta_gpu = cp.ascontiguousarray(theta_gpu.view(cp.float32)) if self.isComplexSMatrix else cp.ascontiguousarray(theta_gpu)
                q_gpu_permuted = cp.zeros(self.N * self.T, dtype=cp_dtype)

                proj_kernel_name = "forward_projection_kernel__SELL__COMPLEX" if self.isComplexSMatrix else "forward_projection_kernel__SELL__REAL"
                proj_kernel = self.sparse_mod.get_function(proj_kernel_name)
                threads = 256
                blocks = (self.N * self.T + threads - 1) // threads

                proj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[q_gpu_permuted.data.ptr, self.sell_values_gpu.data.ptr, self.sell_colinds_gpu.data.ptr,
                          self.slice_ptr_gpu.data.ptr, self.slice_len_gpu.data.ptr, theta_gpu.data.ptr,
                          np.int32(self.N * self.T), np.int32(self.slice_height)]
                )
                cp.cuda.Stream.null.synchronize()
                return q_gpu_permuted[self.inv_row_perm_gpu]
        else:
            theta_cpu = np.asarray(theta, dtype=dtype) if not isinstance(theta, np.ndarray) else theta
            
            valid = self.sell_values != 0
            v_vals = self.sell_values[valid]
            v_cols = self.sell_colinds[valid]
            v_rows = self.sell_rowinds[valid]

            q_permuted = np.zeros(self.N * self.T, dtype=dtype)
            np.add.at(q_permuted, v_rows, v_vals * theta_cpu[v_cols])
            
            return q_permuted[self.inv_row_perm]

    def backward_projection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Perform backprojection: c = A_sell^T * (P * e)."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()

        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                e_gpu = cp.asarray(e, dtype=cp_dtype)
                e_gpu = e_gpu[self.row_perm_gpu]
                e_gpu = cp.ascontiguousarray(e_gpu)

                if self.isComplexSMatrix:
                    c_gpu = cp.zeros(self.Z * self.X, dtype=cp.complex64)
                else:
                    c_gpu = cp.zeros(self.Z * self.X, dtype=cp.float32)

                bp_kernel_name = "backward_projection_kernel__SELL__COMPLEX" if self.isComplexSMatrix else "backward_projection_kernel__SELL__REAL"
                bp_kernel = self.sparse_mod.get_function(bp_kernel_name)
                threads = 256
                blocks = (self.N * self.T + threads - 1) // threads

                bp_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[self.sell_values_gpu, self.sell_colinds_gpu, self.slice_ptr_gpu,
                        self.slice_len_gpu, e_gpu.data.ptr, c_gpu.data.ptr,
                        np.int32(self.N * self.T), np.int32(self.slice_height)]
                )
                cp.cuda.Stream.null.synchronize()
                return c_gpu
        else:
            e_cpu = np.asarray(e, dtype=dtype) if not isinstance(e, np.ndarray) else e
            e_cpu_permuted = e_cpu[self.row_perm]
            c = np.zeros(self.Z * self.X, dtype=dtype)

            valid = self.sell_values != 0
            v_vals = self.sell_values[valid]
            v_cols = self.sell_colinds[valid]
            v_rows = self.sell_rowinds[valid]

            np.add.at(c, v_cols.astype(np.int64), v_vals * e_cpu_permuted[v_rows])
            return c

    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        """Apply apodization window to the matrix values."""
        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                window_gpu = cp.asarray(window_vector) if not isinstance(window_vector, cp.ndarray) else window_vector
                if isinstance(window_gpu, np.ndarray):
                    window_gpu = cp.asarray(window_gpu)
                apodize_kernel_name = "apply_apodization_kernel__SELL__COMPLEX" if self.isComplexSMatrix else "apply_apodization_kernel__SELL__REAL"
                apodize_kernel = self.sparse_mod.get_function(apodize_kernel_name)
                threads = 128
                blocks = (self.total_storage + threads - 1) // threads

                apodize_kernel(
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    args=[self.sell_values_gpu, self.sell_colinds_gpu, window_gpu.data.ptr,
                          np.int64(self.total_storage), np.uint32(self.Z * self.X)]
                )
                cp.cuda.Stream.null.synchronize()
                self.sell_values = cp.asnumpy(self.sell_values_gpu)
        else:
            window_cpu = np.asarray(window_vector) if not isinstance(window_vector, np.ndarray) else window_vector
            if isinstance(window_cpu, cp.ndarray):
                window_cpu = cp.asnumpy(window_cpu)

            for i in trange(self.total_storage, desc="[AOT-biomaps] Applying apodization (CPU)"):
                col = int(self.sell_colinds[i])
                if col < len(window_cpu):
                    self.sell_values[i] *= window_cpu[col]

    def compute_norm_factor(self):
        """Compute normalization factor for SELL matrix using GPU kernels or vectorized CPU."""
        ZX = self.Z * self.X

        if check_gpu_available(self) and getattr(self, 'sell_values_gpu', None) is not None:
            with cp.cuda.Device(self.gpu_index):
                # Allocate GPU memory for column sums
                col_sum_gpu = cp.zeros(ZX, dtype=cp.float32)

                # Select the appropriate kernel
                kernel_name = "accumulate_columns_atomic__COMPLEX" if self.isComplexSMatrix else "accumulate_columns_atomic__REAL"
                acc_kernel = self.sparse_mod.get_function(kernel_name)

                # Configure kernel launch
                threads = 256
                blocks = (self.total_storage + threads - 1) // threads

                acc_kernel(
                    grid=(blocks, 1, 1),
                    block=(threads, 1, 1),
                    args=[
                        self.sell_values_gpu,  # float or float2 array
                        self.sell_colinds_gpu,
                        np.int64(self.total_storage),
                        col_sum_gpu  # Output: float array (norms)
                    ]
                )

                cp.cuda.Stream.null.synchronize()
                self.norm_factor_inv_gpu = 1.0 / (col_sum_gpu + 1e-10)
                self.norm_factor_inv = cp.asnumpy(self.norm_factor_inv_gpu)
        else:
            col_sums = np.zeros(ZX, dtype=cp.float32)
            if self.isComplexSMatrix:
                np.add.at(col_sums, self.sell_colinds.astype(np.int64), np.abs(self.sell_values))
            else:
                np.add.at(col_sums, self.sell_colinds.astype(np.int64), np.abs(self.sell_values))

            self.norm_factor_inv = 1.0 / (col_sums + 1e-10)

    def compute_density(self) -> float:
        """
        Returns the actual density of the SELL-C-sigma matrix in percentage.
        Density = (total_nnz) / (Total elements) * 100.
        """
        if self.slice_ptr is None:
            raise RuntimeError("[AOT-biomaps] The SELL-C-sigma matrix is not allocated yet.")

        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        total_elements = num_rows * num_cols

        density_ratio = self.total_nnz / total_elements
        return density_ratio * 100.0

    def get_matrix_size(self) -> dict:
        """Returns the total size of the SELL-C-sigma matrix in GB."""
        if self.sell_values is None and self.sell_values_gpu is None:
            return {"error": "[AOT-biomaps] The SELL-C-sigma matrix is not yet allocated."}

        total_bytes = 0
        if self.slice_ptr is not None: total_bytes += self.slice_ptr.nbytes
        if self.slice_len is not None: total_bytes += self.slice_len.nbytes
        if self.sell_values is not None: total_bytes += self.sell_values.nbytes
        if self.sell_colinds is not None: total_bytes += self.sell_colinds.nbytes
        if getattr(self, 'norm_factor_inv', None) is not None: total_bytes += self.norm_factor_inv.nbytes
        if getattr(self, 'row_perm', None) is not None: total_bytes += self.row_perm.nbytes * 2
        if getattr(self, 'inv_row_perm', None) is not None: total_bytes += self.inv_row_perm.nbytes
        if self.sell_values_gpu is not None: total_bytes += self.sell_values_gpu.nbytes
        if self.sell_colinds_gpu is not None: total_bytes += self.sell_colinds_gpu.nbytes
        if getattr(self, 'slice_ptr_gpu', None) is not None: total_bytes += self.slice_ptr_gpu.nbytes
        if getattr(self, 'slice_len_gpu', None) is not None: total_bytes += self.slice_len_gpu.nbytes
        if getattr(self, 'norm_factor_inv_gpu', None) is not None: total_bytes += self.norm_factor_inv_gpu.nbytes
        if getattr(self, 'row_perm_gpu', None) is not None: total_bytes += self.row_perm_gpu.nbytes
        if getattr(self, 'inv_row_perm_gpu', None) is not None: total_bytes += self.inv_row_perm_gpu.nbytes

        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024 ** 3),
            "device": self.device
        }

    def _free_specific(self):
        """Free all GPU memory allocated by SELL."""
        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                attrs = ["sell_values_gpu", "sell_colinds_gpu", "slice_ptr_gpu", "slice_len_gpu",
                        "row_perm_gpu", "inv_row_perm_gpu", "norm_factor_inv_gpu"]

                for a in attrs:
                    gpu_mem = getattr(self, a, None)
                    if gpu_mem is not None:
                        try:
                            setattr(self, a, None)
                            if hasattr(gpu_mem, 'free'):
                                gpu_mem.free()
                            del gpu_mem
                        except Exception as e:
                            warnings.warn(f"[AOT-biomaps] Error freeing {a}: {e}")

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
                valid = self.sell_values_gpu != 0
                cupyx.scatter_add(diag, self.sell_colinds_gpu[valid].astype(cp.int32), cp.abs(self.sell_values_gpu[valid]) ** 2)
                return diag
        else:
            diag = np.zeros(ZX, dtype=np.float32)
            valid = self.sell_values != 0
            np.add.at(diag, self.sell_colinds[valid].astype(np.int64), np.abs(self.sell_values[valid]) ** 2)
            return diag
    
    def normalize_matrix(self):
        """
        Normalizes the SELL matrix by its maximum absolute value.
        Restores the system conditioning for Primal-Dual solvers.
        """
        max_val = 0.0
        
        if check_gpu_available(self) and self.sell_values_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                max_val = float(cp.max(cp.abs(self.sell_values_gpu)))
                if max_val > 0:
                    self.sell_values_gpu /= max_val
                    if self.sell_values is not None:
                        self.sell_values /= max_val
        elif self.sell_values is not None:
            max_val = float(np.max(np.abs(self.sell_values)))
            if max_val > 0:
                self.sell_values /= max_val
        else:
            warnings.warn("[AOT-biomaps] SELL Matrix not allocated, normalization impossible.")
            return
            
        self.normalization_factor = max_val

        print(f"[AOT-biomaps] SELL Matrix normalized (Original absolute max: {max_val:.2e})")
        
        # Critical update of the normalization factors (preconditioners)
        self.compute_norm_factor()
    
    def compute_absolute_row_col_sums(self):
        """
        Compute row and column sums of absolute values for the Ehrhardt diagonal preconditioner,
        taking into account the SELL-C-sigma row permutation.
        """
        ZX = self.Z * self.X
        NT = self.N * self.T

        # ==========================================================
        # GPU
        # ==========================================================
        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                row_sums_sorted = cp.zeros(NT, dtype=cp.float32)
                col_sums = cp.zeros(ZX, dtype=cp.float32)
                threads = 256

                # -----------------------------
                # Column sums
                # -----------------------------
                blocks_col = (self.total_storage + threads - 1) // threads
                kernel_col = self.sparse_mod.get_function(
                    "accumulate_abs_columns_atomic__COMPLEX"
                    if self.isComplexSMatrix
                    else "accumulate_abs_columns_atomic__REAL"
                )
                kernel_col(
                    grid=(blocks_col, 1, 1),
                    block=(threads, 1, 1),
                    args=[
                        self.sell_values_gpu,
                        self.sell_colinds_gpu,
                        np.int64(self.total_storage),
                        col_sums
                    ]
                )

                # -----------------------------
                # Row sums (on sorted/permuted rows)
                # -----------------------------
                blocks_row = (NT + threads - 1) // threads
                kernel_row = self.sparse_mod.get_function(
                    "accumulate_abs_rows__SELL__COMPLEX"
                    if self.isComplexSMatrix
                    else "accumulate_abs_rows__SELL__REAL"
                )
                kernel_row(
                    grid=(blocks_row, 1, 1),
                    block=(threads, 1, 1),
                    args=[
                        self.sell_values_gpu,
                        self.slice_ptr_gpu,
                        self.slice_len_gpu,
                        row_sums_sorted,
                        np.int32(NT),
                        np.int32(self.slice_height)
                    ]
                )

                cp.cuda.Stream.null.synchronize()

                # CRITICAL FIX: Map sorted row sums back to physical row order using inv_row_perm
                row_sums = row_sums_sorted[self.inv_row_perm_gpu]

                return row_sums, col_sums

        # ==========================================================
        # CPU fallback
        # ==========================================================
        row_sums_sorted = np.zeros(NT, dtype=np.float32)
        col_sums = np.zeros(ZX, dtype=np.float32)

        for sorted_row in range(NT):
            slice_id = sorted_row // self.slice_height
            row_in_slice = sorted_row % self.slice_height
            base = int(self.slice_ptr[slice_id])
            length = int(self.slice_len[slice_id])
            pos = base + row_in_slice

            s = 0.0
            for j in range(length):
                idx = pos + j * self.slice_height
                if idx >= self.total_storage:
                    continue
                value = np.abs(self.sell_values[idx])
                if value == 0:
                    continue
                col = int(self.sell_colinds[idx])
                s += value
                col_sums[col] += value

            row_sums_sorted[sorted_row] = s

        # Map back to physical row order on CPU
        row_sums = row_sums_sorted[self.inv_row_perm]

        return row_sums, col_sums