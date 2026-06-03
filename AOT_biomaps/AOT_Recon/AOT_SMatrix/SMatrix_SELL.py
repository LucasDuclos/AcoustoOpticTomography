"""
SMatrix_SELL.py

SELL-C-sigma sparse matrix construction and operations.
Supports both CPU (NumPy) and GPU (CuPy) implementations.
"""

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
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class SMatrix_SELL(SMatrix):
    """
    Sparse matrix in SELL-C-sigma format for efficient GPU operations.

    Usage:
        S = SMatrix_SELL(experiment, device='gpu')
        S.allocate()
    """

    def __init__(self, block_rows: int = 64, relative_threshold: float = 0.01, slice_height: int = 64, **kwargs):
        """
        Initialize SELL sparse matrix.
        Args:
            **kwargs: Arguments passed to base SMatrix class
                - experiment: Experiment object containing AcousticFields
                - device: 'cpu' or 'gpu:{gpu_id}' (optional, defaults to GPU if available)
            block_rows: Number of rows to process per block when building on GPU (default: 64)
            relative_threshold: Relative threshold for sparsity (default: 0.01)
            slice_height: Number of rows per slice in SELL format (default: 64)
        """
        super().__init__(**kwargs)
        self.matrix_type = SMatrixType.SELL
        
        # Hyperparameters
        self.block_rows = block_rows
        self.relative_threshold = relative_threshold
        self.slice_height = slice_height

        # Attributes specific to SELL
        self.sell_values = None
        self.sell_colinds = None
        self.slice_ptr = None
        self.slice_len = None
        self.total_storage = 0

        self.sell_values_gpu = None
        self.sell_colinds_gpu = None
        self.slice_ptr_gpu = None
        self.slice_len_gpu = None

    def _allocate_gpu(self):
        """Allocate and fill the SELL matrix on GPU using custom kernels."""
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        C = int(self.slice_height)
        br = int(self.block_rows)
        dense_host = np.empty((br, num_cols), dtype=np.float32)

        # 1) Count NNZ per row
        row_nnz = np.zeros(num_rows, dtype=np.int32)
        block_size = 128
        count_kernel = self.sparse_mod.get_function("count_nnz_rows_kernel")

        for b in trange(0, num_rows, br, desc="Count NNZ per row (GPU)"):
            R = min(br, num_rows - b)
            dense_host.fill(0.0)
            for i in range(R):
                rg = b + i
                n_idx = rg // self.T
                t_idx = rg % self.T
                dense_host[i, :] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

            dense_gpu = cp.asarray(dense_host)
            row_nnz_gpu_block = cp.zeros(R, dtype=np.int32)

            grid = ((R + block_size - 1) // block_size, 1, 1)
            count_kernel(
                grid=grid, block=(block_size, 1, 1),
                args=[dense_gpu, row_nnz_gpu_block, np.int32(R), np.int32(num_cols),
                      np.float32(self.relative_threshold)]
            )
            cp.cuda.Stream.null.synchronize()
            row_nnz[b:b+R] = cp.asnumpy(row_nnz_gpu_block)

        # 2) Compute per-slice maxlen and slice_ptr
        num_slices = (num_rows + C - 1) // C
        self.slice_len = np.zeros(num_slices, dtype=np.int32)

        for s in range(num_slices):
            r0 = s * C
            r1 = min(num_rows, r0 + C)
            self.slice_len[s] = int(np.max(row_nnz[r0:r1])) if (r1 > r0) else 0

        if np.all(self.slice_len == 0):
            raise ValueError("slice_len contains only zeros. Check row_nnz.")

        self.slice_ptr = np.zeros(num_slices + 1, dtype=np.int64)
        for s in range(num_slices):
            self.slice_ptr[s+1] = self.slice_ptr[s] + (self.slice_len[s] * C)

        self.total_storage = int(self.slice_ptr[-1])

        # Allocate device arrays
        self.sell_values_gpu = cp.zeros(self.total_storage, dtype=np.float32)
        self.sell_colinds_gpu = cp.zeros(self.total_storage, dtype=np.uint32)
        self.slice_ptr_gpu = cp.asarray(self.slice_ptr)
        self.slice_len_gpu = cp.asarray(self.slice_len)

        # 3) Fill SELL arrays
        fill_kernel = self.sparse_mod.get_function("fill_kernel__SELL")

        for b in trange(0, num_rows, br, desc="Fill SELL (GPU)"):
            R = min(br, num_rows - b)
            dense_host.fill(0.0)
            for i in range(R):
                rg = b + i
                n_idx = rg // self.T
                t_idx = rg % self.T
                dense_host[i, :] = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()

            dense_gpu = cp.asarray(dense_host)
            row_nnz_host_gpu = cp.asarray(row_nnz[b:b+R])

            grid = ((R + block_size - 1) // block_size, 1, 1)
            fill_kernel(
                grid=grid, block=(block_size, 1, 1),
                args=[dense_gpu, row_nnz_host_gpu, self.slice_ptr_gpu, self.slice_len_gpu,
                      self.sell_colinds_gpu, self.sell_values_gpu, np.int32(R), np.int32(num_cols),
                      np.int32(b), np.int32(C), np.float32(self.relative_threshold)]
            )
            cp.cuda.Stream.null.synchronize()

        self.sell_values = cp.asnumpy(self.sell_values_gpu)
        self.sell_colinds = cp.asnumpy(self.sell_colinds_gpu)

        self.compute_norm_factor()

    def _allocate_cpu(self):
        """Allocate and fill the SELL matrix on CPU."""
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        C = int(self.slice_height)

        # 1) Count NNZ per row
        row_nnz = np.zeros(num_rows, dtype=np.int32)
        for global_row in trange(num_rows, desc="Count NNZ per row (CPU)"):
            n_idx = global_row // self.T
            t_idx = global_row % self.T
            row = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()
            row_max = np.max(np.abs(row))
            thr = row_max * self.relative_threshold
            row_nnz[global_row] = int(np.count_nonzero(np.abs(row) > thr))

        # 2) Compute per-slice maxlen and slice_ptr
        num_slices = (num_rows + C - 1) // C
        self.slice_len = np.zeros(num_slices, dtype=np.int32)

        for s in range(num_slices):
            r0 = s * C
            r1 = min(num_rows, r0 + C)
            self.slice_len[s] = int(np.max(row_nnz[r0:r1])) if (r1 > r0) else 0

        if np.all(self.slice_len == 0):
            raise ValueError("slice_len contains only zeros. Check row_nnz.")

        self.slice_ptr = np.zeros(num_slices + 1, dtype=np.int64)
        for s in range(num_slices):
            self.slice_ptr[s+1] = self.slice_ptr[s] + (self.slice_len[s] * C)

        self.total_storage = int(self.slice_ptr[-1])
        self.sell_values = np.zeros(self.total_storage, dtype=np.float32)
        self.sell_colinds = np.zeros(self.total_storage, dtype=np.uint32)

        # 3) Fill SELL arrays
        for global_row in trange(num_rows, desc="Fill SELL (CPU)"):
            n_idx = global_row // self.T
            t_idx = global_row % self.T
            row = self.experiment.AcousticFields[n_idx].field[t_idx].flatten()
            row_max = np.max(np.abs(row))
            thr = row_max * self.relative_threshold

            slice_id = global_row // C
            row_in_slice = global_row % C
            base = int(self.slice_ptr[slice_id])
            len_slice = int(self.slice_len[slice_id])

            k = 0
            for col in range(num_cols):
                if np.abs(row[col]) > thr:
                    pos = base + row_in_slice + k * C
                    if pos < self.total_storage:
                        self.sell_values[pos] = row[col]
                        self.sell_colinds[pos] = col
                    k += 1

            for k_pad in range(k, len_slice):
                pos = base + row_in_slice + k_pad * C
                if pos < self.total_storage:
                    self.sell_values[pos] = 0.0
                    self.sell_colinds[pos] = 0

        self.compute_norm_factor()

    def forward_projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Perform forward projection: q = A * theta"""
        if check_gpu_available(self):
            theta_gpu = cp.asarray(theta) if not isinstance(theta, cp.ndarray) else theta
            q_gpu = cp.zeros(self.N * self.T, dtype=np.float32)

            proj_kernel = self.sparse_mod.get_function('forward_projection_kernel__SELL')
            threads = 256
            blocks = (self.N * self.T + threads - 1) // threads

            proj_kernel(
                grid=(blocks, 1), block=(threads, 1, 1),
                args=[q_gpu.data.ptr, self.sell_values_gpu, self.sell_colinds_gpu,
                      self.slice_ptr_gpu, self.slice_len_gpu, theta_gpu.data.ptr,
                      np.int32(self.N * self.T), np.int32(self.slice_height)]
            )
            cp.cuda.Stream.null.synchronize()
            return q_gpu
        else:
            theta_cpu = np.asarray(theta) if not isinstance(theta, np.ndarray) else theta
            if isinstance(theta_cpu, cp.ndarray):
                theta_cpu = cp.asnumpy(theta_cpu)

            q = np.zeros(self.N * self.T, dtype=np.float32)
            for row in range(self.N * self.T):
                slice_id = row // self.slice_height
                row_in_slice = row % self.slice_height
                base = int(self.slice_ptr[slice_id])
                len_slice = int(self.slice_len[slice_id])

                acc = 0.0
                for j in range(len_slice):
                    pos = base + row_in_slice + j * self.slice_height
                    if pos < self.total_storage:
                        val = self.sell_values[pos]
                        if val != 0.0:
                            col = int(self.sell_colinds[pos])
                            acc += val * theta_cpu[col]
                q[row] = acc

            return q

    def backward_projection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Perform backprojection: c = A^T * e"""
        if check_gpu_available(self):
            e_gpu = cp.asarray(e) if not isinstance(e, cp.ndarray) else e
            c_gpu = cp.zeros(self.Z * self.X, dtype=np.float32)

            bp_kernel = self.sparse_mod.get_function('backward_projection_kernel__SELL')
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
            e_cpu = np.asarray(e) if not isinstance(e, np.ndarray) else e
            if isinstance(e_cpu, cp.ndarray):
                e_cpu = cp.asnumpy(e_cpu)

            c = np.zeros(self.Z * self.X, dtype=np.float32)
            for row in range(self.N * self.T):
                e_val = e_cpu[row]
                if e_val == 0.0:
                    continue

                slice_id = row // self.slice_height
                row_in_slice = row % self.slice_height
                base = int(self.slice_ptr[slice_id])
                len_slice = int(self.slice_len[slice_id])

                for j in range(len_slice):
                    pos = base + row_in_slice + j * self.slice_height
                    if pos < self.total_storage:
                        val = self.sell_values[pos]
                        if val != 0.0:
                            col = int(self.sell_colinds[pos])
                            c[col] += val * e_val

            return c

    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        """Apply apodization window to the matrix values."""
        if check_gpu_available(self):
            window_gpu = cp.asarray(window_vector) if not isinstance(window_vector, cp.ndarray) else window_vector
            apodize_kernel = self.sparse_mod.get_function("apply_apodization_kernel__SELL")
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

            for i in range(self.total_storage):
                col = int(self.sell_colinds[i])
                if col < len(window_cpu):
                    self.sell_values[i] *= window_cpu[col]

    def flip_probe(self):
        """Permute the columns of the SELL-C-sigma matrix corresponding to opposite angles."""
        if self.N % 2 != 0:
            raise ValueError("Number of angles must be even to permute opposite angles.")

        ZX = self.Z * self.X
        angle_block_size = ZX // self.N

        if check_gpu_available(self) and self.sell_colinds_gpu is not None:
            sell_colinds_host = cp.asnumpy(self.sell_colinds_gpu)
        else:
            sell_colinds_host = self.sell_colinds.copy()

        for n in range(self.N // 2):
            n_opposite = n + self.N // 2
            block_n_start = n * angle_block_size
            block_n_end = (n + 1) * angle_block_size
            block_opposite_start = n_opposite * angle_block_size
            block_opposite_end = (n_opposite + 1) * angle_block_size

            mask_n = (sell_colinds_host >= block_n_start) & (sell_colinds_host < block_n_end)
            mask_opposite = (sell_colinds_host >= block_opposite_start) & (sell_colinds_host < block_opposite_end)

            sell_colinds_host[mask_n] = sell_colinds_host[mask_n] - block_n_start + block_opposite_start
            sell_colinds_host[mask_opposite] = sell_colinds_host[mask_opposite] - block_opposite_start + block_n_start

        self.sell_colinds = sell_colinds_host
        if CUPY_AVAILABLE:
            self.sell_colinds_gpu = cp.asarray(self.sell_colinds)

        self.compute_norm_factor()

    def compute_density(self) -> float:
        """Returns the density of the SELL-C-sigma matrix."""
        if self.slice_ptr is None:
            raise RuntimeError("The SELL-C-sigma matrix is not allocated.")
        total_elements = self.N * self.T * self.Z * self.X
        nnz_estimated = int(0.9 * self.total_storage)
        return nnz_estimated / total_elements

    def get_matrix_size(self) -> dict:
        """Returns the total size of the SELL-C-sigma matrix in GB."""
        if self.sell_values is None:
            return {"error": "The SELL-C-sigma matrix is not yet allocated."}

        total_bytes = 0
        if self.slice_ptr is not None: total_bytes += self.slice_ptr.nbytes
        if self.slice_len is not None: total_bytes += self.slice_len.nbytes
        if self.sell_values is not None: total_bytes += self.sell_values.nbytes
        if self.sell_colinds is not None: total_bytes += self.sell_colinds.nbytes
        if self.norm_factor_inv is not None: total_bytes += self.norm_factor_inv.nbytes

        if self.sell_values_gpu is not None: total_bytes += self.sell_values.nbytes
        if self.sell_colinds_gpu is not None: total_bytes += self.sell_colinds.nbytes
        if self.slice_ptr_gpu is not None: total_bytes += self.slice_ptr.nbytes
        if self.slice_len_gpu is not None: total_bytes += self.slice_len.nbytes
        if self.norm_factor_inv_gpu is not None: total_bytes += self.norm_factor_inv.nbytes

        return {"total_bytes": total_bytes, "total_gb": total_bytes / (1024 ** 3), "device": self.device}

    def _free_specific(self):
        """Free specific GPU memory allocated by SELL."""
        attrs = ["sell_values_gpu", "sell_colinds_gpu", "slice_ptr_gpu", "slice_len_gpu"]
        for a in attrs:
            gpu_mem = getattr(self, a, None)
            if gpu_mem is not None:
                try:
                    if hasattr(gpu_mem, 'free'): gpu_mem.free()
                    else: del gpu_mem
                except:
                    pass
            setattr(self, a, None)