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

class SMatrix_DENSE(SMatrix):
    """
    Construction of a DENSE matrix from a `experiment` object.
    Supports both REAL and COMPLEX fields via `isComplexSMatrix`.
    """

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

            # Fill dense matrix using CUDA kernel
            fill_kernel_name = "fill_kernel__DENSE__COMPLEX" if self.isComplexSMatrix else "fill_kernel__DENSE__REAL"
            fill_kernel = self.sparse_mod.get_function(fill_kernel_name)

            for n in trange(self.N, desc=f'[AOT-biomaps] Filling DENSE ({"Complex" if self.isComplexSMatrix else "Real"}) --- device: {self.device.upper()}'):
                if self.isComplexSMatrix:
                    # For complex: use demodulated_fields
                    key = list(self.experiment.AcousticFields_demodulated.keys())[n]
                    for t in range(self.T):
                        field_host[t * (self.Z * self.X) : (t + 1) * (self.Z * self.X)] = self.experiment.AcousticFields_demodulated[key][t].flatten()
                else:
                    # For real: use AcousticFields
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
                # Chargement par bloc de dimensions (current_n, num_cols)
                block = np.empty((current_n, num_cols), dtype=dtype)
                for i in range(current_n):
                    ni = n + i
                    if self.isComplexSMatrix:
                        key = sorted_keys[ni]
                        block[i] = self.experiment.AcousticFields_demodulated[key][t].flatten()
                    else:
                        block[i] = self.experiment.AcousticFields[ni].field[t].flatten()
                
                # Réinjection directe et vectorisée dans le tenseur 4D
                self.dense_matrix[t, n:n+current_n] = block.reshape((current_n, self.Z, self.X))

    def forward_projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform forward projection: q = A * theta
        """
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()

        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                theta_gpu = cp.asarray(theta, dtype=cp_dtype) if not isinstance(theta, cp.ndarray) else theta
                if theta_gpu.dtype != cp_dtype:
                    theta_gpu = theta_gpu.astype(cp_dtype)
                q_gpu = cp.zeros(self.N * self.T, dtype=cp_dtype)

                # Ensure contiguous memory layout for custom CUDA kernels
                dense_contiguous = cp.ascontiguousarray(self.dense_matrix_gpu)
                theta_contiguous = cp.ascontiguousarray(theta_gpu)

                proj_kernel_name = "forward_projection_kernel__DENSE__COMPLEX" if self.isComplexSMatrix else "forward_projection_kernel__DENSE__REAL"
                proj_kernel = self.sparse_mod.get_function(proj_kernel_name)
                threads = 256
                blocks = (self.N * self.T + threads - 1) // threads

                proj_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[q_gpu.data.ptr, dense_contiguous.data.ptr, theta_contiguous.data.ptr,
                        np.int32(self.T), np.int32(self.N), np.int32(self.Z), np.int32(self.X)]
                )
                cp.cuda.Stream.null.synchronize()
                return q_gpu
        else:
            theta_cpu = np.asarray(theta, dtype=dtype) if not isinstance(theta, np.ndarray) else theta
            if theta_cpu.dtype != dtype:
                theta_cpu = theta_cpu.astype(dtype)

            dense_2d = self.dense_matrix.transpose(1, 0, 2, 3).reshape(self.N * self.T, self.Z * self.X)
            return dense_2d @ theta_cpu

    def backward_projection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform backward projection: c = A^T * e
        """
        dtype = self._get_dtype()
        cp_dtype = self._get_cp_dtype()

        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            with cp.cuda.Device(self.gpu_index):
                e_gpu = cp.asarray(e, dtype=cp_dtype) if not isinstance(e, cp.ndarray) else e
                if e_gpu.dtype != cp_dtype:
                    e_gpu = e_gpu.astype(cp_dtype)
                c_gpu = cp.zeros(self.Z * self.X, dtype=cp_dtype)

                # Ensure contiguous memory layout for custom CUDA kernels
                dense_contiguous = cp.ascontiguousarray(self.dense_matrix_gpu)
                e_contiguous = cp.ascontiguousarray(e_gpu)

                bp_kernel_name = "backward_projection_kernel__DENSE__COMPLEX" if self.isComplexSMatrix else "backward_projection_kernel__DENSE__REAL"
                bp_kernel = self.sparse_mod.get_function(bp_kernel_name)
                threads = 256
                blocks = (self.Z * self.X + threads - 1) // threads

                bp_kernel(
                    grid=(blocks, 1), block=(threads, 1, 1),
                    args=[c_gpu.data.ptr, dense_contiguous.data.ptr, e_contiguous.data.ptr,
                        np.int32(self.T), np.int32(self.N), np.int32(self.Z), np.int32(self.X)]
                )
                cp.cuda.Stream.null.synchronize()
                return c_gpu
        else:
            e_cpu = np.asarray(e, dtype=dtype) if not isinstance(e, np.ndarray) else e
            if e_cpu.dtype != dtype:
                e_cpu = e_cpu.astype(dtype)

            dense_2d = self.dense_matrix.transpose(1, 0, 2, 3).reshape(self.N * self.T, self.Z * self.X)
            return dense_2d.T @ e_cpu

    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        raise NotImplementedError("Apodization not implemented for DENSE matrix.")

    def get_matrix_size(self):
        """Get matrix size information."""
        if self.dense_matrix is not None:
            size_bytes = self.dense_matrix.nbytes
            return {
                'total_bytes': size_bytes,
                'total_gb': size_bytes / (1024 ** 3),
                'shape': self.dense_matrix.shape,
                'dtype': str(self.dense_matrix.dtype)
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
        """
        Compute diag(A^H A).
        """
        if check_gpu_available(self):
            with cp.cuda.Device(self.gpu_index):
                return cp.sum(cp.abs(self.dense_matrix_gpu) ** 2, axis=(0,1)).ravel().astype(cp.float32)
        else:
            return np.sum(np.abs(self.dense_matrix) ** 2, axis=(0,1)).ravel().astype(np.float32)
        
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
        """
        is_gpu = check_gpu_available(self) and self.dense_matrix_gpu is not None
        
        if is_gpu:
            with cp.cuda.Device(self.gpu_index):
                abs_dense = cp.abs(self.dense_matrix_gpu)
                # Layout shape is (T, N, Z, X). Reshape to (N * T, Z * X)
                abs_2d = abs_dense.transpose(1, 0, 2, 3).reshape(int(self.N * self.T), int(self.Z * self.X))
                
                row_sums = cp.sum(abs_2d, axis=1)
                col_sums = cp.sum(abs_2d, axis=0)
                
                return row_sums, col_sums
        else:
            if self.dense_matrix is None:
                raise RuntimeError("[AOT-biomaps] DENSE matrix not allocated on CPU.")
            
            abs_dense = np.abs(self.dense_matrix)
            abs_2d = abs_dense.transpose(1, 0, 2, 3).reshape(int(self.N * self.T), int(self.Z * self.X))
            
            row_sums = np.sum(abs_2d, axis=1).astype(np.float32)
            col_sums = np.sum(abs_2d, axis=0).astype(np.float32)
            
            return row_sums, col_sums