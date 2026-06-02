"""
SMatrix_DENSE.py

Dense matrix construction and operations.
Supports both CPU (NumPy) and GPU (CuPy/CUDA) implementations.
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

class SMatrix_DENSE(SMatrix):
    """
    Construction of a DENSE matrix from a `experiment` object.

    Usage:
        S = SMatrix_DENSE(experiment, device='gpu')  # or 'cpu'
        S.allocate()
    """

    def __init__(self, **kwargs):
        """
        Initialize DENSE matrix.
        Args:
            **kwargs: Arguments passed to base SMatrix class
                - experiment: Experiment object containing AcousticFields
                - device: 'cpu' or 'gpu:{gpu_id}' (optional, defaults to GPU if available)
        """
        super().__init__(**kwargs)
        self.matrix_type = SMatrixType.DENSE

        # Attributes specific to DENSE
        self.dense_matrix = None
        self.dense_matrix_gpu = None

    def _allocate_gpu(self):
        """Allocate and fill the DENSE matrix on GPU using custom kernels."""
        # Allocate dense matrix on GPU
        self.dense_matrix_gpu = cp.zeros((self.T, self.N, self.Z, self.X), dtype=np.float32)

        # Prepare host buffer for field data (1D array)
        field_host = np.empty((self.T * self.Z * self.X), dtype=np.float32)

        # Fill dense matrix using CUDA kernel
        fill_kernel = self.sparse_mod.get_function("fill_dense_matrix_kernel")

        for n in trange(self.N, desc="Filling DENSE (GPU)"):
            # Copy field data for this angle to host buffer (flattened)
            for t in range(self.T):
                field_host[t * (self.Z * self.X) : (t + 1) * (self.Z * self.X)] = (
                    self.experiment.AcousticFields[n].field[t].flatten().astype(np.float32)
                )

            field_gpu = cp.asarray(field_host)

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
                    np.int32(n)  # Passer l'angle courant
                ]
            )
            cp.cuda.Stream.null.synchronize()

        # Copy back to host for CPU access
        self.dense_matrix = cp.asnumpy(self.dense_matrix_gpu)

        # Compute normalization factor
        self.compute_norm_factor()

    def _allocate_cpu(self):
        """Allocate and fill the DENSE matrix on CPU."""
        self.dense_matrix = np.zeros((self.T, self.N, self.Z, self.X), dtype=np.float32)

        for n in trange(self.N, desc="Filling DENSE (CPU)"):
            field = self.experiment.AcousticFields[n].field
            for t in range(self.T):
                self.dense_matrix[t, n] = field[t].astype(np.float32)

        self.compute_norm_factor()

    def forward_projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform forward projection: q = A * theta

        Args:
            theta: Input vector (Z*X,)

        Returns:
            Projection result (N*T,)
        """
        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            theta_gpu = cp.asarray(theta) if not isinstance(theta, cp.ndarray) else theta
            q_gpu = cp.zeros(self.N * self.T, dtype=np.float32)

            # Sécurisation des layouts mémoire bruts pour les kernels CUDA custom
            dense_contiguous = cp.ascontiguousarray(self.dense_matrix_gpu)
            theta_contiguous = cp.ascontiguousarray(theta_gpu)

            proj_kernel = self.sparse_mod.get_function('forward_projection_kernel__DENSE')
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
            theta_cpu = np.asarray(theta) if not isinstance(theta, np.ndarray) else theta
            if isinstance(theta_cpu, cp.ndarray):
                theta_cpu = cp.asnumpy(theta_cpu)

            q = np.zeros(self.N * self.T, dtype=np.float32)
            # Boucle CPU alignée sur le layout (N * T)
            for n in range(self.N):
                for t in range(self.T):
                    row_idx = n * self.T + t
                    acc = 0.0
                    for z in range(self.Z):
                        for x in range(self.X):
                            col_idx = z * self.X + x
                            acc += self.dense_matrix[t, n, z, x] * theta_cpu[col_idx]
                    q[row_idx] = acc
            return q

    def backward_projection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform backward projection: c = A^T * e

        Args:
            e: Input vector (N*T,)

        Returns:
            Backprojection result (Z*X,)
        """
        if check_gpu_available(self) and self.dense_matrix_gpu is not None:
            e_gpu = cp.asarray(e) if not isinstance(e, cp.ndarray) else e
            c_gpu = cp.zeros(self.Z * self.X, dtype=np.float32)

            # Sécurisation des layouts mémoire bruts pour les kernels CUDA custom
            dense_contiguous = cp.ascontiguousarray(self.dense_matrix_gpu)
            e_contiguous = cp.ascontiguousarray(e_gpu)

            bp_kernel = self.sparse_mod.get_function('backward_projection_kernel__DENSE')
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
            e_cpu = np.asarray(e) if not isinstance(e, np.ndarray) else e
            if isinstance(e_cpu, cp.ndarray):
                e_cpu = cp.asnumpy(e_cpu)

            c = np.zeros(self.Z * self.X, dtype=np.float32)
            # Boucle CPU alignée sur le layout (N * T)
            for n in range(self.N):
                for t in range(self.T):
                    row_idx = n * self.T + t
                    e_val = e_cpu[row_idx]
                    if e_val == 0.0:
                        continue
                    for z in range(self.Z):
                        for x in range(self.X):
                            col_idx = z * self.X + x
                            c[col_idx] += self.dense_matrix[t, n, z, x] * e_val
            return c    
    
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
            return {'error': 'Matrix not allocated'}

    def _free_specific(self):
        """Free specific GPU memory allocated by CSR."""
        for attr in ['col_ind_gpu', 'values_gpu', 'row_ptr_gpu']:
            gpu_mem = getattr(self, attr, None)
            if gpu_mem is not None:
                try:
                    if hasattr(gpu_mem, 'free'): gpu_mem.free()
                    else: del gpu_mem
                except:
                    pass
            setattr(self, attr, None)
