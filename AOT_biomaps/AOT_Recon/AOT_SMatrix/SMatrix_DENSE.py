import numpy as np
from tqdm import trange
from typing import Optional, Union
import contextlib
from AOT_biomaps.AOT_Recon.AOT_SMatrix._mainSMatrix import SMatrix
from AOT_biomaps.AOT_Recon.ReconEnums import SMatrixType, SMATRIX_FORMAT_TAG, SMATRIX_FORMAT_TAGS
from AOT_biomaps.AOT_Recon.ReconTools import check_gpu_available

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

class SMatrix_DENSE(SMatrix):
    """
    Construction of a DENSE matrix from a `experiment` object.
    Supports both REAL and COMPLEX fields via `isComplexSMatrix`.
    """
    _FORMAT_TAG = SMATRIX_FORMAT_TAG[SMatrixType.DENSE]

    def __init__(self, **kwargs):
        """
        Initialize DENSE matrix.
        Args:
            **kwargs: Arguments passed to base SMatrix class.
        """
        super().__init__(**kwargs)
        self.matrix_type = SMatrixType.DENSE

        # Attributes specific to DENSE
        self.dense_matrix = None
        self.dense_matrix_gpu = None

    def _allocate_gpu(self):
        """Allocate and fill the DENSE matrix on GPU using custom kernels."""
        with cp.cuda.Device(self.gpu_index):
            dtype = self._get_dtype()
            cp_dtype = self._get_cp_dtype()

            # Allocate dense matrix on GPU
            self.dense_matrix_gpu = cp.zeros((self.T, self.N, self.Z, self.X), dtype=cp_dtype)

            # Prepare host buffer for field data (1D array)
            field_host = np.empty((self.T * self.Z * self.X), dtype=dtype)

            # Sorted keys for consistent ordering with CPU allocation
            sorted_keys = sorted(list(self.experiment.AcousticFields_demodulated.keys())) if self.isComplexSMatrix else None

            # Fill dense matrix using CUDA kernel
            fill_kernel_name = "fill_kernel__DENSE__COMPLEX" if self.isComplexSMatrix else "fill_kernel__DENSE__REAL"
            fill_kernel = self.sparse_mod.get_function(fill_kernel_name)

            for n in trange(self.N, desc=f'[AOT-biomaps] Filling DENSE ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: {self.device.upper()}'):
                if self.isComplexSMatrix:
                    key = sorted_keys[n]
                    for t in range(self.T):
                        field_host[t * (self.Z * self.X) : (t + 1) * (self.Z * self.X)] = self.experiment.AcousticFields_demodulated[key][t].flatten()
                else:
                    for t in range(self.T):
                        field_host[t * (self.Z * self.X) : (t + 1) * (self.Z * self.X)] = self.experiment.AcousticFields[n].field[t].flatten().astype(dtype)

                field_gpu = cp.asarray(field_host, dtype=cp_dtype)

                threads = 256
                blocks = (self.T * self.Z * self.X + threads - 1) // threads

                fill_kernel(
                    grid=(blocks, 1, 1),
                    block=(threads, 1, 1),
                    args=[
                        self.dense_matrix_gpu.data.ptr,
                        field_gpu.data.ptr,
                        np.int32(self.T),
                        np.int32(self.N),
                        np.int32(self.Z),
                        np.int32(self.X),
                        np.int32(n)
                    ]
                )
                cp.cuda.Stream.null.synchronize()

            # Copy back to host for CPU access
            self.dense_matrix = cp.asnumpy(self.dense_matrix_gpu)

            # Compute normalization factor
            self.compute_norm_factor()

    def _allocate_cpu(self):
        """Allocate and fill the DENSE matrix on CPU with vectorized block processing."""
        dtype = self._get_dtype()
        self.dense_matrix = np.zeros((self.T, self.N, self.Z, self.X), dtype=dtype)
        num_cols = self.Z * self.X
        br = getattr(self, 'block_rows', 128)

        sorted_keys = sorted(list(self.experiment.AcousticFields_demodulated.keys())) if self.isComplexSMatrix else None

        for n in trange(0, self.N, br, desc=f'[AOT-biomaps] Filling DENSE ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: CPU'):
            current_n = min(br, self.N - n)
            for t in range(self.T):
                # Load a block of shape (current_n, num_cols)
                block = np.empty((current_n, num_cols), dtype=dtype)
                for i in range(current_n):
                    ni = n + i
                    if self.isComplexSMatrix:
                        key = sorted_keys[ni]
                        block[i] = self.experiment.AcousticFields_demodulated[key][t].flatten()
                    else:
                        block[i] = self.experiment.AcousticFields[ni].field[t].flatten()

                # Direct vectorized re-injection into the 4D tensor
                self.dense_matrix[t, n:n+current_n] = block.reshape((current_n, self.Z, self.X))

    def _save_sparse_matrix(self, filePath):
        """Saves the complete DENSE matrix to an uncompressed .npz file."""
        if self._is_virtual_truncated():
            raise RuntimeError("[AOT-biomaps] Call untruncate() before save_sparse_matrix().")

        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            dense = cp.asnumpy(self.dense_matrix_gpu)
            with cp.cuda.Device(self.gpu_index):
                self._release_pool()
        else:
            if self.dense_matrix is None:
                raise RuntimeError("[AOT-biomaps] DENSE matrix not allocated, cannot save.")
            dense = self.dense_matrix

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

        # Format tag (last field, checked at load time)
        metadata = np.concatenate([np.array([self.N, self.T, self.Z, self.X, int(self.isComplexSMatrix)]), spatial_meta, np.array([self._FORMAT_TAG])])
        
        np.savez(
            filePath,
            dense_matrix=dense,
            norm_factor_inv=self.norm_factor_inv if self.norm_factor_inv is not None else np.array([]),
            metadata=metadata,
            normalization_factor=np.float64(getattr(self, 'normalization_factor', 1.0))
        )
        print(f"[AOT-biomaps] DENSE SMatrix successfully saved to: {filePath}")

    def _check_format_tag(self, meta):
        """Ensure the npz was saved as a REAL/COMPLEX DENSE matrix (format tag + complex flag)."""
        # 1. Identify the file's format first (useful message before any length check)
        tag = int(meta[-1]) if len(meta) >= 1 else -1
        if tag != self._FORMAT_TAG and tag in SMATRIX_FORMAT_TAGS:
            raise ValueError(f"[AOT-biomaps] Wrong matrix format: file is {SMATRIX_FORMAT_TAGS[tag]} (tag {tag}), but this object is DENSE (tag {self._FORMAT_TAG}). Load it with the matching smatrixType or regenerate the SMatrix.")

        # 2. Unknown / legacy file
        if len(meta) < 13:
            raise ValueError(f"[AOT-biomaps] Incompatible npz: metadata too short ({len(meta)} fields, expected >= 13). Not a DENSE file from this AOT-biomaps version — regenerate the SMatrix.")

        # 3. REAL vs COMPLEX mismatch (isComplexSMatrix is meta[4] for DENSE)
        file_is_complex = bool(int(meta[4]))
        if file_is_complex != self.isComplexSMatrix:
            kind_file = "COMPLEX" if file_is_complex else "REAL"
            kind_self = "COMPLEX" if self.isComplexSMatrix else "REAL"
            raise ValueError(f"[AOT-biomaps] SMatrix type mismatch: file was saved as {kind_file} but this object is configured as {kind_self}. Recreate AlgebraicRecon with the matching isComplexRecon setting or regenerate the SMatrix.")

    def _load_sparse_matrix_cpu(self, filePath):
        """Loads the DENSE matrix from the .npz file into CPU RAM."""
        print(f"[AOT-biomaps] Loading DENSE SMatrix from {filePath} into CPU RAM...")
        data = np.load(filePath)
        if 'normalization_factor' in data:
            self.normalization_factor = float(data['normalization_factor'])
        else:
            raise ValueError("[AOT-biomaps] npz has no 'normalization_factor': the matrix scale is unknown (saved by an older version?). Regenerate and re-save the SMatrix with this version.")
        
        meta = data['metadata']
        self._check_format_tag(meta)
        self.N, self.T, self.Z, self.X = map(int, meta[:4])
        self.isComplexSMatrix = bool(meta[4])

        # Restore the physical spatial crop info if present
        self._phys_box = None
        if int(meta[5]) == 1:
            self._phys_box = {'z': (int(meta[6]), int(meta[7])), 'x': (int(meta[8]), int(meta[9])), 'Zf': int(meta[10]), 'Xf': int(meta[11])}
        elif int(meta[5]) == 2:
            self._phys_box = {'dec': {"Z": int(meta[6]), "X": int(meta[7])}, 'Zf': int(meta[10]), 'Xf': int(meta[11])}
        
        self.dense_matrix = data['dense_matrix']
        self.dense_matrix_gpu = None

        if data['norm_factor_inv'].size > 0:
            self.norm_factor_inv = data['norm_factor_inv']

        data.close()
        self.device = 'cpu'
        print(f"[AOT-biomaps] DENSE SMatrix loaded into CPU RAM {self.dense_matrix.shape}.")

    def _load_sparse_matrix_gpu(self, filePath):
        """Loads the DENSE matrix from the .npz file directly into GPU VRAM."""
        print(f"[AOT-biomaps] Direct-to-GPU loading of DENSE SMatrix from {filePath}...")
        self.load_module()
        data = np.load(filePath)
        if 'normalization_factor' in data:
            self.normalization_factor = float(data['normalization_factor'])
        else:
            raise ValueError("[AOT-biomaps] npz has no 'normalization_factor': the matrix scale is unknown (saved by an older version?). Regenerate and re-save the SMatrix with this version.")
        
        meta = data['metadata']
        self._check_format_tag(meta)
        self.N, self.T, self.Z, self.X = map(int, meta[:4])
        self.isComplexSMatrix = bool(meta[4])

        # Restore the physical spatial crop info if present
        self._phys_box = None
        if int(meta[5]) == 1:
            self._phys_box = {'z': (int(meta[6]), int(meta[7])), 'x': (int(meta[8]), int(meta[9])), 'Zf': int(meta[10]), 'Xf': int(meta[11])}
        elif int(meta[5]) == 2:
            self._phys_box = {'dec': {"Z": int(meta[6]), "X": int(meta[7])}, 'Zf': int(meta[10]), 'Xf': int(meta[11])}

        cp_dtype = self._get_cp_dtype()

        with cp.cuda.Device(self.gpu_index):
            self.dense_matrix_gpu = cp.asarray(data['dense_matrix']).astype(cp_dtype)
            self.dense_matrix = None  # no CPU copy on the compute node

            if data['norm_factor_inv'].size > 0:
                self.norm_factor_inv_gpu = cp.asarray(data['norm_factor_inv'])
                self.norm_factor_inv = data['norm_factor_inv']

            self._release_pool()
        data.close()
        print(f"[AOT-biomaps] DENSE SMatrix loaded into VRAM {self.dense_matrix_gpu.shape}.")

    def forward_projection(self, theta):
        """Forward projection: q = A * theta (supports virtual truncation)."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()
        virt = self._is_virtual_truncated()

        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
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

                dense_contiguous = cp.ascontiguousarray(self.dense_matrix_gpu)
                theta_contiguous = cp.ascontiguousarray(theta_gpu)

                proj_kernel_name = "forward_projection_kernel__DENSE__COMPLEX" if self.isComplexSMatrix else "forward_projection_kernel__DENSE__REAL"
                proj_kernel = self.sparse_mod.get_function(proj_kernel_name)
                threads = 256
                blocks = (full_NT + threads - 1) // threads

                proj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[q_gpu.data.ptr, dense_contiguous.data.ptr, theta_contiguous.data.ptr,
                        np.int32(self._full_T if virt else self.T), np.int32(self._full_N if virt else self.N),
                        np.int32(self._full_Z if virt else self.Z), np.int32(self._full_X if virt else self.X)]
                )
                cp.cuda.Stream.null.synchronize()

                # gather: q_full -> q_eff (phi_t)
                if virt:
                    return q_gpu[self._vt_active_rows_gpu]
                return q_gpu
        else:
            theta_cpu = np.asarray(theta, dtype=dtype) if not isinstance(theta, np.ndarray) else theta
            if theta_cpu.dtype != dtype:
                theta_cpu = theta_cpu.astype(dtype)

            dense = self.dense_matrix
            if virt:
                full_NT = self._full_N * self._full_T
                full_ZX = self._full_Z * self._full_X
                if self._vs_active_cols is not None:
                    theta_tmp = np.zeros(full_ZX, dtype=dtype)
                    theta_tmp[self._vs_active_cols] = theta_cpu
                    theta_cpu = theta_tmp
                dense_2d = dense.transpose(1, 0, 2, 3).reshape(full_NT, full_ZX)
                q = dense_2d @ theta_cpu
                return q[self._vt_active_rows]

            dense_2d = dense.transpose(1, 0, 2, 3).reshape(int(self.N * self.T), int(self.Z * self.X))
            return dense_2d @ theta_cpu

    def backward_projection(self, e):
        """Backward projection: c = A^T * e (supports virtual truncation)."""
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()
        virt = self._is_virtual_truncated()

        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
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

                dense_contiguous = cp.ascontiguousarray(self.dense_matrix_gpu)
                e_contiguous = cp.ascontiguousarray(e_gpu)

                bp_kernel_name = "backward_projection_kernel__DENSE__COMPLEX" if self.isComplexSMatrix else "backward_projection_kernel__DENSE__REAL"
                bp_kernel = self.sparse_mod.get_function(bp_kernel_name)
                threads = 256
                blocks = (full_ZX + threads - 1) // threads

                bp_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[c_gpu.data.ptr, dense_contiguous.data.ptr, e_contiguous.data.ptr,
                        np.int32(self._full_T if virt else self.T), np.int32(self._full_N if virt else self.N),
                        np.int32(self._full_Z if virt else self.Z), np.int32(self._full_X if virt else self.X)]
                )
                cp.cuda.Stream.null.synchronize()

                # gather: c_full -> c_eff (phi_s)
                if virt and self._vs_active_cols_gpu is not None:
                    return c_gpu[self._vs_active_cols_gpu]
                return c_gpu
        else:
            e_cpu = np.asarray(e, dtype=dtype) if not isinstance(e, np.ndarray) else e
            if e_cpu.dtype != dtype:
                e_cpu = e_cpu.astype(dtype)

            dense = self.dense_matrix
            if virt:
                full_NT = self._full_N * self._full_T
                full_ZX = self._full_Z * self._full_X
                e_tmp = np.zeros(full_NT, dtype=dtype)
                e_tmp[self._vt_active_rows] = e_cpu
                e_cpu = e_tmp
                dense_2d = dense.transpose(1, 0, 2, 3).reshape(full_NT, full_ZX)
                c = dense_2d.conj().T @ e_cpu
                if self._vs_active_cols is not None:
                    return c[self._vs_active_cols]
                return c

            dense_2d = dense.transpose(1, 0, 2, 3).reshape(int(self.N * self.T), int(self.Z * self.X))
            return dense_2d.conj().T @ e_cpu
    
    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        raise NotImplementedError("Apodization not implemented for DENSE matrix.")

    def get_matrix_size(self):
        """Get matrix size information."""
        if self.dense_matrix_gpu is not None:
            size_bytes = self.dense_matrix_gpu.nbytes
            return {
                'total_bytes': size_bytes,
                'total_gb': size_bytes / (1024 ** 3),
                'shape': tuple(self.dense_matrix_gpu.shape),
                'dtype': str(self.dense_matrix_gpu.dtype),
                'device': self.device
            }
        elif self.dense_matrix is not None:
            size_bytes = self.dense_matrix.nbytes
            return {
                'total_bytes': size_bytes,
                'total_gb': size_bytes / (1024 ** 3),
                'shape': self.dense_matrix.shape,
                'dtype': str(self.dense_matrix.dtype),
                'device': self.device
            }
        else:
            return {'error': '[AOT-biomaps] Matrix not allocated'}
    
    def _free_specific(self):
        """Free all GPU memory allocated by DENSE."""
        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                attrs = ['dense_matrix_gpu']
                for attr in attrs:
                    gpu_mem = getattr(self, attr, None)
                    if gpu_mem is not None:
                        try:
                            setattr(self, attr, None)
                            if hasattr(gpu_mem, 'free'):
                                gpu_mem.free()
                            del gpu_mem
                        except Exception as e:
                            print(f"[AOT-biomaps] Warning: Error freeing {attr}: {e}")

                if CUPY_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
    
    def compute_hessian_diagonal(self):
        """diag(A_eff^H A_eff) at effective size (supports virtual truncation)."""
        virt = self._is_virtual_truncated()

        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                diag = cp.sum(cp.abs(self.dense_matrix_gpu) ** 2, axis=(0, 1)).ravel().astype(cp.float32)
                # gather: full -> effective (phi_s)
                if virt and self._vs_active_cols_gpu is not None:
                    diag = diag[self._vs_active_cols_gpu]
                return diag
        else:
            diag = np.sum(np.abs(self.dense_matrix) ** 2, axis=(0, 1)).ravel().astype(np.float32)
            if virt and self._vs_active_cols is not None:
                diag = diag[self._vs_active_cols]
            return diag
        
    def normalize_matrix(self):
        """
        Normalizes the DENSE matrix by its maximum absolute value.
        Restores the system conditioning for Primal-Dual solvers.
        """
        max_val = 0.0
        
        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                max_val = float(cp.max(cp.abs(self.dense_matrix_gpu)))
                if max_val > 0:
                    self.dense_matrix_gpu /= max_val
                    if self.dense_matrix is not None:
                        self.dense_matrix /= max_val
        elif self.dense_matrix is not None:
            max_val = float(np.max(np.abs(self.dense_matrix)))
            if max_val > 0:
                self.dense_matrix /= max_val
        else:
            print(f"[AOT-biomaps] Warning: DENSE Matrix not allocated, normalization impossible.")
            return
        self.normalization_factor = max_val
        
        print(f"[AOT-biomaps] DENSE Matrix normalized (Original absolute max: {max_val:.2e})")
        
        # Critical update of the normalization factors (preconditioners)
        self.compute_norm_factor()
    
    def compute_absolute_row_col_sums(self):
        """
        Computes row and column sums of absolute values (|A| * 1 and |A|^T * 1)
        without phase cancellation for complex matrices in DENSE format.
        Supports virtual truncation.
        """
        virt = self._is_virtual_truncated()
        if virt:
            full_NT = self._full_N * self._full_T
            full_ZX = self._full_Z * self._full_X
        else:
            full_NT = self.N * self.T
            full_ZX = self.Z * self.X

        is_gpu = check_gpu_available(self) and self.dense_matrix_gpu is not None

        if is_gpu:
            with cp.cuda.Device(self.gpu_index):
                abs_dense = cp.abs(self.dense_matrix_gpu)
                # Layout shape is (T, N, Z, X). Reshape to (N * T, Z * X) at FULL size
                abs_2d = abs_dense.transpose(1, 0, 2, 3).reshape(full_NT, full_ZX)

                row_sums = cp.sum(abs_2d, axis=1)
                col_sums = cp.sum(abs_2d, axis=0)

                # gather: full -> effective (phi_t on rows, phi_s on cols)
                if virt:
                    row_sums = row_sums[self._vt_active_rows_gpu]
                    if self._vs_active_cols_gpu is not None:
                        col_sums = col_sums[self._vs_active_cols_gpu]

                return row_sums, col_sums
        else:
            if self.dense_matrix is None:
                raise RuntimeError("[AOT-biomaps] DENSE matrix not allocated on CPU.")

            abs_dense = np.abs(self.dense_matrix)
            abs_2d = abs_dense.transpose(1, 0, 2, 3).reshape(full_NT, full_ZX)

            row_sums = np.sum(abs_2d, axis=1).astype(np.float32)
            col_sums = np.sum(abs_2d, axis=0).astype(np.float32)

            if virt:
                row_sums = row_sums[self._vt_active_rows]
                if self._vs_active_cols is not None:
                    col_sums = col_sums[self._vs_active_cols]

            return row_sums, col_sums
        
    def physical_truncate(self, time_range=None, time_decimate=1, space_range=None, space_decimate=None, recompute_norm=True, verbose=True):
        """
        Physically truncates the DENSE matrix in time (T) and space (X, Z).
        Pure slicing, runs on GPU (CuPy) or CPU (NumPy). Layout: (T, N, Z, X).
        """
        if self.dense_matrix is None and self.dense_matrix_gpu is None:
            raise ValueError("[AOT-biomaps] DENSE matrix not loaded.")
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

        on_gpu = (self.dense_matrix_gpu is not None) and CUPY_AVAILABLE and check_gpu_available(self)
        xp = cp if on_gpu else np

        device_ctx = cp.cuda.Device(self.gpu_index) if on_gpu else contextlib.nullcontext()
        with device_ctx:
            dense = self.dense_matrix_gpu if on_gpu else self.dense_matrix
            old_T, old_N, old_Z, old_X = self.T, self.N, self.Z, self.X

            # 1. Time selection (axis 0)
            t_slice = slice(None)
            if time_decimate > 1 or time_range is not None:
                t_idx = xp.arange(old_T)
                mask = xp.ones(old_T, dtype=bool)
                if time_decimate > 1:
                    mask &= (t_idx % time_decimate == 0)
                if time_range is not None:
                    mask &= (t_idx >= time_range[0]) & (t_idx < time_range[1])
                if not mask.any():
                    raise ValueError("[AOT-biomaps] No time samples remaining.")
                t_slice = xp.flatnonzero(mask)
                new_T = int(mask.sum())
            else:
                new_T = old_T

            # 2. Space selection (axes 2, 3)
            z_slice, x_slice = slice(None), slice(None)
            if space_decimate:
                dec_Z = space_decimate.get("Z", 1)
                dec_X = space_decimate.get("X", 1)
                z_slice = slice(None, None, dec_Z)
                x_slice = slice(None, None, dec_X)
            elif space_range:
                z_slice = slice(space_range.get("Z", (0, old_Z))[0], space_range.get("Z", (0, old_Z))[1])
                x_slice = slice(space_range.get("X", (0, old_X))[0], space_range.get("X", (0, old_X))[1])

            new_dense = xp.ascontiguousarray(dense[t_slice, :, z_slice, x_slice])
            new_T_, _, new_Z, new_X = new_dense.shape

            # 3. Swap the public dims and install the new array
            self.T, self.Z, self.X = new_T_, new_Z, new_X

            if on_gpu:
                self.dense_matrix_gpu = new_dense
                self.dense_matrix = None
                del dense
                self._release_pool()
            else:
                self.dense_matrix = new_dense
                self.dense_matrix_gpu = None

            self.norm_factor_inv = None
            self.norm_factor_inv_gpu = None

        if recompute_norm:
            self.compute_norm_factor()
        if verbose:
            print(f"[AOT-biomaps] DENSE physical truncation: T {old_T}->{self.T}, "
                f"Z {old_Z}->{self.Z}, X {old_X}->{self.X}")
                
    def virtual_truncate(self, time_range=None, time_decimate=1, space_range=None, verbose=True):
        """
        Virtual truncation: A stays untouched in memory, only active indices are stored.
        A_eff = phi_t . A . phi_s^T is applied on the vectors inside the projections.
        """
        if self.dense_matrix_gpu is None and self.dense_matrix is None:
            raise ValueError("[AOT-biomaps] DENSE matrix not loaded.")

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
            for a in ("_full_N", "_full_T", "_full_Z", "_full_X"):  
                delattr(self, a) 
        if verbose:
            print("[AOT-biomaps] Virtual truncation removed. Full matrix active.")

    def _is_virtual_truncated(self):
        return getattr(self, "_vt_time_mask", None) is not None
    
    def to_gpu(self, gpu_index=0):
        """
        Transfer the DENSE matrix from CPU RAM to GPU VRAM (reversible with to_cpu).
        Refuses if the matrix currently lives on another GPU: call to_cpu() first (explicit two-step, avoids silent multi-GPU residency bugs).
        """
        if self.dense_matrix_gpu is not None:
            if self.gpu_index == gpu_index:
                return  # already on the requested GPU
            raise RuntimeError(f"[AOT-biomaps] Matrix currently lives on gpu:{self.gpu_index}. Call to_cpu() first, then to_gpu({gpu_index}) to move it explicitly.")

        if self.dense_matrix is None:
            raise RuntimeError("[AOT-biomaps] DENSE matrix not allocated on CPU, cannot transfer to GPU.")
        self.gpu_index = gpu_index
        self.load_module()
        with cp.cuda.Device(self.gpu_index):
            self.dense_matrix_gpu = cp.asarray(self.dense_matrix).astype(self._get_cp_dtype())
            if self.norm_factor_inv is not None:
                self.norm_factor_inv_gpu = cp.asarray(self.norm_factor_inv)
            self._release_pool()
        self.device = f'gpu:{self.gpu_index}'

    def to_cpu(self):
        """Transfer the DENSE matrix from GPU VRAM to CPU RAM (reversible with to_gpu)."""
        if self.dense_matrix_gpu is None:
            return  # already on CPU
        self.dense_matrix = cp.asnumpy(self.dense_matrix_gpu)
        self._free_specific()
        self.device = 'cpu'