import os
import warnings
import numpy as np
from tqdm import trange
from typing import Optional, Union
import contextlib

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

            sorted_keys = sorted(list(self.experiment.AcousticFields_demodulated.keys())) if self.isComplexSMatrix else None

            for b in trange(0, num_rows, br, desc=f'[AOT-biomaps] Filling CSR ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: {self.device.upper()}'):
                current_rows = min(br, num_rows - b)

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
        virt = self._is_virtual_truncated()
        ZX = (self._full_Z * self._full_X) if virt else self.Z * self.X

        if CUPY_AVAILABLE and check_gpu_available(self) and self.values_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                col_sum_gpu = cp.zeros(ZX, dtype=cp.float32)
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
            if self.h_values is None:
                raise RuntimeError("[AOT-biomaps] CSR matrix not allocated on CPU.")
            norm = np.bincount(self.h_col_ind.astype(np.int64),
                               weights=np.abs(self.h_values).astype(np.float64),
                               minlength=ZX).astype(np.float32)

        norm = np.maximum(norm.astype(np.float64), 1e-6)
        self.norm_factor_inv = (1.0 / norm).astype(np.float32)

        if CUPY_AVAILABLE and check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                self.norm_factor_inv_gpu = cp.asarray(self.norm_factor_inv)

    def _save_sparse_matrix(self, filePath):
        """Saves the complete CSR matrix to an uncompressed .npz file."""
        if self._is_virtual_truncated():
            raise RuntimeError("[AOT-biomaps] Call untruncate() before save_sparse_matrix().")

        if CUPY_AVAILABLE and check_gpu_available(self) and self.values_gpu is not None:
            values = cp.asnumpy(self.values_gpu)
            col_ind = cp.asnumpy(self.col_ind_gpu)
            row_ptr = cp.asnumpy(self.row_ptr_gpu)
            with cp.cuda.Device(self.gpu_index):
                self._release_pool()
        else:
            if self.h_values is None:
                raise RuntimeError("[AOT-biomaps] CSR matrix not allocated, cannot save.")
            values = self.h_values
            col_ind = self.h_col_ind
            row_ptr = self.row_ptr

        metadata = np.array([self.N, self.T, self.Z, self.X, self.total_nnz, int(self.isComplexSMatrix)])

        # Persist the physical spatial crop info (survives save/load cycles)
        pb = getattr(self, "_phys_box", None)
        if pb is not None and 'z' in pb:
            (zs, ze), (xs, xe) = pb['z'], pb['x']
            spatial_meta = np.array([1, zs, ze, xs, xe, pb['Zf'], pb['Xf']])
        elif pb is not None and 'dec' in pb:
            decZ = pb['dec'].get("Z", 1)
            decX = pb['dec'].get("X", 1)
            spatial_meta = np.array([2, float(decZ), float(decX), 0, 0, pb['Zf'], pb['Xf']])
        else:
            spatial_meta = np.zeros(7)

        metadata = np.concatenate([metadata, spatial_meta])

        norm_inv = self.norm_factor_inv if self.norm_factor_inv is not None else np.array([])

        np.savez(filePath, values=values, colinds=col_ind, row_ptr=row_ptr,
                 norm_factor_inv=norm_inv, metadata=metadata)
        print(f"[AOT-biomaps] CSR SMatrix successfully saved ({self.total_nnz} nnz) to: {filePath}")

    def _load_sparse_matrix_cpu(self, filePath):
        """Loads the CSR matrix from the .npz file into CPU RAM."""
        print(f"[AOT-biomaps] Loading CSR SMatrix from {filePath} into CPU RAM...")
        data = np.load(filePath)

        meta = data['metadata']
        self.N, self.T, self.Z, self.X, self.total_nnz = map(int, meta[:5])
        self.isComplexSMatrix = bool(meta[5])

        # Restore the physical spatial crop info if present (older files: len(meta)==6)
        self._phys_box = None
        if len(meta) >= 13 and int(meta[6]) == 1:
            self._phys_box = {'z': (int(meta[7]), int(meta[8])),
                              'x': (int(meta[9]), int(meta[10])),
                              'Zf': int(meta[11]), 'Xf': int(meta[12])}
        elif len(meta) >= 13 and int(meta[6]) == 2:
            self._phys_box = {'dec': {"Z": int(meta[7]), "X": int(meta[8])},
                              'Zf': int(meta[11]), 'Xf': int(meta[12])}

        self.row_ptr = data['row_ptr']
        self.h_col_ind = data['colinds']
        self.h_values = data['values']

        if self.isComplexSMatrix and self.h_values.dtype != np.complex64:
            self.h_values = self.h_values.astype(np.complex64)
        elif not self.isComplexSMatrix and self.h_values.dtype != np.float32:
            self.h_values = self.h_values.astype(np.float32)

        from scipy.sparse import csr_matrix
        self.scipy_csr = csr_matrix(
            (self.h_values, self.h_col_ind, self.row_ptr),
            shape=(int(self.N * self.T), int(self.Z * self.X))
        )

        if data['norm_factor_inv'].size > 0:
            self.norm_factor_inv = data['norm_factor_inv']

        data.close()
        self.device = 'cpu'
        print(f"[AOT-biomaps] CSR SMatrix loaded into CPU RAM (nnz={self.total_nnz}).")

    def _load_sparse_matrix_gpu(self, filePath):
        """Loads the arrays from the .npz file directly into GPU VRAM."""
        print(f"[AOT-biomaps] Direct-to-GPU loading of CSR SMatrix from {filePath}...")
        self.load_module()
        data = np.load(filePath)

        meta = data['metadata']
        self.N, self.T, self.Z, self.X, self.total_nnz = map(int, meta[:5])
        self.isComplexSMatrix = bool(meta[5])

        # Restore the physical spatial crop info if present (older files: len(meta)==6)
        self._phys_box = None
        if len(meta) >= 13 and int(meta[6]) == 1:
            self._phys_box = {'z': (int(meta[7]), int(meta[8])),
                              'x': (int(meta[9]), int(meta[10])),
                              'Zf': int(meta[11]), 'Xf': int(meta[12])}
        elif len(meta) >= 13 and int(meta[6]) == 2:
            self._phys_box = {'dec': {"Z": int(meta[7]), "X": int(meta[8])},
                              'Zf': int(meta[11]), 'Xf': int(meta[12])}

        cp_dtype = self._get_cp_dtype()

        with cp.cuda.Device(self.gpu_index):
            # Force the exact dtypes expected by the CUDA kernels
            self.values_gpu = cp.asarray(data['values']).astype(cp_dtype)
            self.col_ind_gpu = cp.asarray(data['colinds']).astype(cp.uint32)
            self.row_ptr_gpu = cp.asarray(data['row_ptr']).astype(cp.int64)

            # Vital CPU pointers for the class logic
            self.row_ptr = data['row_ptr']
            self.h_col_ind = None
            self.h_values = None

            if data['norm_factor_inv'].size > 0:
                self.norm_factor_inv_gpu = cp.asarray(data['norm_factor_inv'])
                self.norm_factor_inv = data['norm_factor_inv']

            self._release_pool()
        data.close()
        print(f"[AOT-biomaps] CSR SMatrix loaded into VRAM (nnz={self.total_nnz}).")

    def forward_projection(self, theta):
        """Forward projection: q = A * theta (supports virtual truncation)."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()
        virt = self._is_virtual_truncated()

        if CUPY_AVAILABLE and check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                full_NT = self._full_N * self._full_T if virt else self.N * self.T
                full_ZX = self._full_Z * self._full_X if virt else self.Z * self.X

                theta_gpu = cp.asarray(theta, dtype=cp_dtype) if not isinstance(theta, cp.ndarray) else theta
                if theta_gpu.dtype != cp_dtype:
                    theta_gpu = theta_gpu.astype(cp_dtype)

                # scatter: theta_eff -> theta_full (phi_s^T)
                if virt and self._vs_active_cols_gpu is not None:
                    theta_full = cp.zeros(full_ZX, dtype=cp_dtype)
                    theta_full[self._vs_active_cols_gpu] = theta_gpu
                    theta_gpu = theta_full

                q_gpu = cp.zeros(full_NT, dtype=cp_dtype)

                proj_kernel_name = "forward_projection_kernel__CSR__COMPLEX" if self.isComplexSMatrix else "forward_projection_kernel__CSR__REAL"
                proj_kernel = self.sparse_mod.get_function(proj_kernel_name)
                threads = 256
                blocks = (full_NT + threads - 1) // threads

                proj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[q_gpu.data.ptr, self.values_gpu, self.row_ptr_gpu, self.col_ind_gpu,
                        theta_gpu.data.ptr, np.int32(full_NT)]
                )
                cp.cuda.Stream.null.synchronize()

                # gather: q_full -> q_eff (phi_t)
                if virt:
                    return q_gpu[self._vt_active_rows_gpu]
                return q_gpu
        else:
            theta_cpu = np.asarray(theta, dtype=dtype) if not isinstance(theta, np.ndarray) else theta
            if virt:
                full_ZX = self._full_Z * self._full_X
                if self._vs_active_cols is not None:
                    theta_tmp = np.zeros(full_ZX, dtype=dtype)
                    theta_tmp[self._vs_active_cols] = theta_cpu
                    theta_cpu = theta_tmp
                q = self.scipy_csr.dot(theta_cpu)
                return q[self._vt_active_rows]
            return self.scipy_csr.dot(theta_cpu)

    def backward_projection(self, e):
        """Backward projection: c = A^T * e (supports virtual truncation)."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()
        virt = self._is_virtual_truncated()

        if CUPY_AVAILABLE and check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                full_NT = self._full_N * self._full_T if virt else self.N * self.T
                full_ZX = self._full_Z * self._full_X if virt else self.Z * self.X

                e_gpu = cp.asarray(e, dtype=cp_dtype) if not isinstance(e, cp.ndarray) else e
                if e_gpu.dtype != cp_dtype:
                    e_gpu = e_gpu.astype(cp_dtype)

                # scatter: e_eff -> e_full (phi_t^T)
                if virt:
                    e_tmp = cp.zeros(full_NT, dtype=cp_dtype)
                    e_tmp[self._vt_active_rows_gpu] = e_gpu
                    e_gpu = e_tmp

                c_gpu = cp.zeros(full_ZX, dtype=cp_dtype)

                backproj_kernel_name = "backward_projection_kernel__CSR__COMPLEX" if self.isComplexSMatrix else "backward_projection_kernel__CSR__REAL"
                backproj_kernel = self.sparse_mod.get_function(backproj_kernel_name)
                threads = 256
                blocks = (full_NT + threads - 1) // threads

                backproj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[c_gpu.data.ptr, self.values_gpu, self.row_ptr_gpu, self.col_ind_gpu,
                        e_gpu.data.ptr, np.int32(full_NT)]
                )
                cp.cuda.Stream.null.synchronize()

                # gather: c_full -> c_eff (phi_s)
                if virt and self._vs_active_cols_gpu is not None:
                    return c_gpu[self._vs_active_cols_gpu]
                return c_gpu
        else:
            e_cpu = np.asarray(e, dtype=dtype) if not isinstance(e, np.ndarray) else e
            if virt:
                full_NT = self._full_N * self._full_T
                full_ZX = self._full_Z * self._full_X
                e_tmp = np.zeros(full_NT, dtype=dtype)
                e_tmp[self._vt_active_rows] = e_cpu
                e_cpu = e_tmp
                c = self.scipy_csr.T.dot(e_cpu)
                if self._vs_active_cols is not None:
                    return c[self._vs_active_cols]
                return c
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
        if CUPY_AVAILABLE and check_gpu_available(self):
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

                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Stream.null.synchronize()
        
    def compute_hessian_diagonal(self):
        """
        diag(A_eff^H A_eff) at effective size (supports virtual truncation).
        """
        virt = self._is_virtual_truncated()
        full_ZX = self._full_Z * self._full_X if virt else self.Z * self.X

        if CUPY_AVAILABLE and check_gpu_available(self) and self.values_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                diag = cp.zeros(full_ZX, dtype=cp.float32)
                cupyx.scatter_add(diag, self.col_ind_gpu.astype(cp.int32), cp.abs(self.values_gpu) ** 2)
                if virt and self._vs_active_cols_gpu is not None:
                    diag = diag[self._vs_active_cols_gpu]
                return diag
        else:
            diag = np.zeros(full_ZX, dtype=np.float32)
            np.add.at(diag, self.h_col_ind.astype(np.int64), np.abs(self.h_values) ** 2)
            if virt and self._vs_active_cols is not None:
                diag = diag[self._vs_active_cols]
            return diag
        
    def normalize_matrix(self):
        """
        Normalizes the CSR matrix by its maximum absolute value.
        Restores the system conditioning for Primal-Dual solvers.
        """
        max_val = 0.0

        if CUPY_AVAILABLE and check_gpu_available(self) and self.values_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                max_val = float(cp.max(cp.abs(self.values_gpu)))
                if max_val > 0:
                    self.values_gpu /= max_val
                    # Check for the existence of the CPU cache (which is sometimes deleted in _allocate_gpu)
                    if getattr(self, 'h_values', None) is not None:
                        self.h_values /= max_val
                        if getattr(self, 'scipy_csr', None) is not None:
                            self.scipy_csr.data = self.h_values
        elif getattr(self, 'h_values', None) is not None:
            max_val = float(np.max(np.abs(self.h_values)))
            if max_val > 0:
                self.h_values /= max_val
                if getattr(self, 'scipy_csr', None) is not None:
                    self.scipy_csr.data = self.h_values
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
        Supports virtual truncation (A_eff = phi_t . A . phi_s^T).
        """
        virt = self._is_virtual_truncated()
        if virt:
            full_NT = self._full_N * self._full_T
            full_ZX = self._full_Z * self._full_X
        else:
            full_NT = self.N * self.T
            full_ZX = self.Z * self.X

        is_gpu = (CUPY_AVAILABLE and check_gpu_available(self)
                  and self.values_gpu is not None)

        if is_gpu:
            with cp.cuda.Device(self.gpu_index):
                abs_vals = cp.abs(self.values_gpu).astype(cp.float32)
                colinds = self.col_ind_gpu.astype(cp.int64)
                row_ptr = self.row_ptr_gpu

                # Column sums (|A|^T * 1)
                col_sums = cp.bincount(colinds, weights=abs_vals,
                                       minlength=full_ZX).astype(cp.float32)

                # Row sums (|A| * 1): expand row_ptr -> per-entry row index
                row_counts = (row_ptr[1:] - row_ptr[:-1]).astype(cp.int64)
                row_idx = cp.repeat(cp.arange(full_NT, dtype=cp.int64), row_counts)
                row_sums = cp.bincount(row_idx, weights=abs_vals,
                                       minlength=full_NT).astype(cp.float32)

                # gather: full -> effective (phi_t on rows, phi_s on cols)
                if virt:
                    row_sums = row_sums[self._vt_active_rows_gpu]
                    if self._vs_active_cols_gpu is not None:
                        col_sums = col_sums[self._vs_active_cols_gpu]

                return row_sums, col_sums
        else:
            if self.h_values is None:
                raise RuntimeError("[AOT-biomaps] CSR matrix not allocated on CPU.")

            abs_vals = np.abs(self.h_values).astype(np.float64)
            colinds = self.h_col_ind.astype(np.int64)
            row_counts = (self.row_ptr[1:] - self.row_ptr[:-1]).astype(np.int64)

            col_sums = np.bincount(colinds, weights=abs_vals,
                                   minlength=full_ZX).astype(np.float32)
            row_idx = np.repeat(np.arange(full_NT, dtype=np.int64), row_counts)
            row_sums = np.bincount(row_idx, weights=abs_vals,
                                   minlength=full_NT).astype(np.float32)

            if virt:
                row_sums = row_sums[self._vt_active_rows]
                if self._vs_active_cols is not None:
                    col_sums = col_sums[self._vs_active_cols]

            return row_sums, col_sums
        
    def physical_truncate(self, time_range=None, time_decimate=1, space_range=None, space_decimate=None, recompute_norm=True, verbose=True):
        """
        Physically truncates the CSR matrix in time (T) and space (X, Z).
        Fully vectorized, runs on GPU (CuPy) or CPU (NumPy).
        """
        if self.row_ptr is None and self.row_ptr_gpu is None:
            raise ValueError("[AOT-biomaps] CSR matrix not loaded.")
        if self._is_virtual_truncated():
            raise RuntimeError("[AOT-biomaps] Call untruncate() before physical_truncate().")

        # Invalidate the saved full dims: the matrix physically changed
        for a in ("_full_N", "_full_T", "_full_Z", "_full_X"):
            if hasattr(self, a):
                delattr(self, a)

        # Remember the physical crop window (needed to crop ground truths)
        self._phys_box = None
        if space_range is not None:
            zs, ze = space_range.get("Z", (0, self.Z))
            xs, xe = space_range.get("X", (0, self.X))
            self._phys_box = {'z': (zs, ze), 'x': (xs, xe), 'Zf': self.Z, 'Xf': self.X}
        elif space_decimate:
            self._phys_box = {'dec': dict(space_decimate), 'Zf': self.Z, 'Xf': self.X}

        on_gpu = (self.values_gpu is not None) and CUPY_AVAILABLE and check_gpu_available(self)
        xp = cp if on_gpu else np

        device_ctx = cp.cuda.Device(self.gpu_index) if on_gpu else contextlib.nullcontext()
        with device_ctx:
            if on_gpu:
                values, colinds = self.values_gpu, self.col_ind_gpu
                row_ptr = self.row_ptr_gpu
            else:
                values, colinds = self.h_values, self.h_col_ind
                row_ptr = self.row_ptr

            old_N, old_T, old_Z, old_X = self.N, self.T, self.Z, self.X
            old_NT = old_N * old_T
            old_ZX = old_Z * old_X

            # 1. Time mask (row = n*T + t)
            time_indices = xp.arange(old_T)
            time_mask = xp.ones(old_T, dtype=bool)
            if time_decimate > 1:
                time_mask &= (time_indices % time_decimate == 0)
            if time_range is not None:
                time_mask &= (time_indices >= time_range[0]) & (time_indices < time_range[1])
            new_T = int(time_mask.sum())
            if new_T == 0:
                raise ValueError("[AOT-biomaps] No time samples remaining.")

            # 2. Column remapping table
            old_to_new_col = xp.full(old_ZX, -1, dtype=xp.int32)
            new_Z, new_X = old_Z, old_X

            if space_decimate:
                dec_X = space_decimate.get("X", 1)
                dec_Z = space_decimate.get("Z", 1)
                new_X, new_Z = old_X // dec_X, old_Z // dec_Z
                Z_grid, X_grid = xp.meshgrid(xp.arange(new_Z), xp.arange(new_X), indexing='ij')
                old_cols = (Z_grid * dec_Z) * old_X + (X_grid * dec_X)
                new_cols = Z_grid * new_X + X_grid
                old_to_new_col[old_cols.ravel()] = new_cols.ravel()
            elif space_range:
                x_start, x_end = space_range.get("X", (0, old_X))
                z_start, z_end = space_range.get("Z", (0, old_Z))
                new_X, new_Z = x_end - x_start, z_end - z_start
                Z_grid, X_grid = xp.meshgrid(xp.arange(z_start, z_end), xp.arange(x_start, x_end), indexing='ij')
                old_cols = Z_grid * old_X + X_grid
                new_cols = (Z_grid - z_start) * new_X + (X_grid - x_start)
                old_to_new_col[old_cols.ravel()] = new_cols.ravel()
            else:
                old_to_new_col = xp.arange(old_ZX, dtype=xp.int32)

            # 3. Row remapping table
            time_mask_tiled = xp.tile(time_mask, old_N)
            new_NT = int(time_mask_tiled.sum())
            old_to_new_row = xp.full(old_NT, -1, dtype=xp.int32)
            old_to_new_row[time_mask_tiled] = xp.arange(new_NT, dtype=xp.int32)

            # 4. Vectorized COO reconstruction + filtering
            row_counts = row_ptr[1:] - row_ptr[:-1]
            row_idx = xp.repeat(xp.arange(old_NT, dtype=xp.int32), row_counts)

            keep = (old_to_new_row[row_idx] >= 0) & (old_to_new_col[colinds] >= 0)

            new_values = values[keep]
            new_row_idx = old_to_new_row[row_idx[keep]]
            new_colinds = old_to_new_col[colinds[keep]].astype(cp.uint32 if on_gpu else np.uint32)

            # 5. Rebuild row_ptr from the filtered row indices (vectorized)
            new_row_ptr = xp.zeros(new_NT + 1, dtype=xp.int64)
            counts = xp.bincount(new_row_idx, minlength=new_NT).astype(xp.int64)
            new_row_ptr[1:] = xp.cumsum(counts)

            new_total_nnz = int(new_row_ptr[-1])

            # 6. Swap the public dims and install the new arrays
            self.T, self.Z, self.X = new_T, new_Z, new_X
            self.total_nnz = new_total_nnz

            if on_gpu:
                self.values_gpu = cp.ascontiguousarray(new_values)
                self.col_ind_gpu = cp.ascontiguousarray(new_colinds)
                self.row_ptr_gpu = cp.ascontiguousarray(new_row_ptr)
                self.row_ptr = cp.asnumpy(new_row_ptr)
                self.h_values = None
                self.h_col_ind = None
                self.scipy_csr = None
                del values, colinds, row_ptr
                self._release_pool()
            else:
                self.h_values = np.ascontiguousarray(new_values)
                self.h_col_ind = np.ascontiguousarray(new_colinds)
                self.row_ptr = np.ascontiguousarray(new_row_ptr)
                self.values_gpu = None
                self.col_ind_gpu = None
                self.row_ptr_gpu = None
                from scipy.sparse import csr_matrix
                self.scipy_csr = csr_matrix(
                    (self.h_values, self.h_col_ind, self.row_ptr),
                    shape=(int(self.N * self.T), int(self.Z * self.X))
                )
                self.device = 'cpu'

            self.norm_factor_inv = None
            self.norm_factor_inv_gpu = None

        if recompute_norm:
            self.compute_norm_factor()
        if verbose:
            print(f"[AOT-biomaps] CSR physical truncation: T {old_T}->{self.T}, "
                f"Z {old_Z}->{self.Z}, X {old_X}->{self.X}, nnz {new_total_nnz}")

    def virtual_truncate(self, time_range=None, time_decimate=1, space_range=None, verbose=True):
        """
        Virtual truncation: A stays untouched in memory, only active indices are stored.
        A_eff = phi_t . A . phi_s^T is applied on the vectors inside the projections.
        """
        if self.row_ptr_gpu is None and self.row_ptr is None:
            raise ValueError("[AOT-biomaps] CSR matrix not loaded.")

        # Save the full dims once (idempotent over repeated calls)
        if not hasattr(self, "_full_T"):
            self._full_N, self._full_T = self.N, self.T
            self._full_Z, self._full_X = self.Z, self.X

        # Time mask
        time_mask = np.ones(self._full_T, dtype=bool)
        if time_decimate > 1:
            time_mask &= (np.arange(self._full_T) % time_decimate == 0)
        if time_range is not None:
            time_mask &= (np.arange(self._full_T) >= time_range[0]) & (np.arange(self._full_T) < time_range[1])
        if not time_mask.any():
            raise ValueError("[AOT-biomaps] No time samples remaining.")

        # Active physical rows (row = n*T + t), order preserved
        self._vt_time_mask = time_mask
        self._vt_active_rows = np.flatnonzero(np.tile(time_mask, self._full_N)).astype(np.int32)

        # Space mask
        if space_range is not None:
            xs, xe = space_range.get("X", (0, self._full_X))
            zs, ze = space_range.get("Z", (0, self._full_Z))
            Zg, Xg = np.meshgrid(np.arange(zs, ze), np.arange(xs, xe), indexing='ij')
            self._vs_active_cols = (Zg * self._full_X + Xg).ravel().astype(np.int32)
            self._vs_box = (zs, ze, xs, xe)
        else:
            self._vs_active_cols = None
            self._vs_box = None

        # GPU index arrays
        if CUPY_AVAILABLE and check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                self._vt_active_rows_gpu = cp.asarray(self._vt_active_rows)
                self._vs_active_cols_gpu = (cp.asarray(self._vs_active_cols)
                                            if self._vs_active_cols is not None else None)

        # Swap the public dims to effective values (optimizers read these)
        self.T = int(time_mask.sum())
        if self._vs_active_cols is not None:
            self.Z, self.X = ze - zs, xe - xs
        else:
            self.Z, self.X = self._full_Z, self._full_X

        if verbose:
            print(f"[AOT-biomaps] Virtual truncation active: T {self._full_T} -> {self.T}, "
                f"N {self.N}, Z {self.Z}, X {self.X} (matrix data untouched)")

    def untruncate(self, verbose=True):
        """Cancel the virtual truncation. A is already in memory, nothing to reload."""
        self._vt_time_mask = None
        self._vt_active_rows = None
        self._vt_active_rows_gpu = None
        self._vs_active_cols = None
        self._vs_active_cols_gpu = None
        if hasattr(self, "_full_T"):
            self.N, self.T = self._full_N, self._full_T
            self.Z, self.X = self._full_Z, self._full_X
        if verbose:
            print("[AOT-biomaps] Virtual truncation removed. Full matrix active.")

    def _is_virtual_truncated(self):
        return getattr(self, "_vt_time_mask", None) is not None