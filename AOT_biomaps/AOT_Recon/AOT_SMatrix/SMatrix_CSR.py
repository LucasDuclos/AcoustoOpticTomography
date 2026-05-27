"""
SMatrix_CSR.py

CSR (Compressed Sparse Row) sparse matrix construction and operations.
Supports both CPU (NumPy) and GPU (CuPy) implementations.
"""

import os
import warnings
import numpy as np
from tqdm import trange
from typing import Optional, Union, TYPE_CHECKING

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

if TYPE_CHECKING:
    import cupy as cp

# Import kernel utilities
try:
    from AOT_biomaps.AOT_Recon.AOT_Kernels import (
        check_cuda_available,
        check_pycuda_available,
        sparse_matrix_vector_product_csr
    )
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False


class SMatrix_CSR:
    """
    Construction of a CSR matrix from a `manip` object.
    
    Supports both CPU and GPU implementations:
    - On GPU: Uses CuPy for memory management and custom CUDA kernels
    - On CPU: Uses NumPy arrays and CPU implementations
    
    Usage:
        S = SMatrix_CSR(manip, device='gpu')  # or 'cpu'
        S.allocate()
    
    After allocate(), the following attributes are available:
    - row_ptr: Row pointer array (host)
    - col_ind: Column indices array (host)
    - values: Non-zero values array (host)
    - norm_factor_inv: Normalization factor (host)
    - For GPU: row_ptr_gpu, col_ind_gpu, values_gpu, norm_factor_inv_gpu
    """

    def __init__(self, manip, block_rows: int = 64, relative_threshold: float = 0.3, 
                 device: Optional[str] = None):
        """
        Initialize CSR matrix constructor.
        
        Args:
            manip: Manipulation object containing acoustic fields
            block_rows: Number of rows to process at once (for GPU)
            relative_threshold: Threshold for considering a value as non-zero
            device: 'cpu' or 'gpu' (auto-detected if None)
        """
        # Determine device
        if device is None:
            self.device = 'gpu' if CUPY_AVAILABLE else 'cpu'
        else:
            self.device = device
        
        self.manip = manip
        self.N = len(manip.AcousticFields)
        self.T = manip.AcousticFields[0].field.shape[0]
        self.Z = manip.AcousticFields[0].field.shape[1]
        self.X = manip.AcousticFields[0].field.shape[2]
        self.block_rows = block_rows
        self.relative_threshold = relative_threshold
        
        # Initialize attributes
        self.h_dense = None
        self.row_ptr = None
        self.h_col_ind = None
        self.h_values = None
        self.total_nnz = 0
        self.norm_factor_inv = None
        
        # GPU-specific attributes
        self.row_ptr_gpu = None
        self.col_ind_gpu = None
        self.values_gpu = None
        self.norm_factor_inv_gpu = None
        self.sparse_mod = None
        
        # Path to CUDA module
        cubin_parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.module_path = os.path.join(cubin_parent_dir, "AOT_biomaps_kernels.cubin")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.free()

    def _check_gpu_available(self):
        """Check if GPU operations are available."""
        if self.device != 'gpu':
            return False
        if not CUPY_AVAILABLE:
            warnings.warn("CuPy not available. Falling back to CPU.")
            self.device = 'cpu'
            return False
        return True

    def load_precompiled_module(self):
        """
        Load the pre-compiled CUDA module (.cubin) using CuPy.
        """
        if not self._check_gpu_available():
            return
            
        if not os.path.exists(self.module_path):
            warnings.warn(
                f"CUDA module {os.path.basename(self.module_path)} not found at: {self.module_path}. "
                "Falling back to CPU implementation."
            )
            self.device = 'cpu'
            return
            
        try:
            self.sparse_mod = cp.cuda.runtime.moduleFromFile(self.module_path)
        except Exception as e:
            warnings.warn(f"Failed to load CUDA module: {e}. Falling back to CPU.")
            self.device = 'cpu'

    def estimate_nnz_cpu(self) -> int:
        """
        Fast (non-exact) estimate of non-zero elements using CPU.
        
        Returns:
            Estimated number of non-zero elements
        """
        total = 0
        for n in range(self.N):
            field = self.manip.AcousticFields[n].field
            for t in range(self.T):
                row = field[t].flatten()
                row_max = np.max(np.abs(row))
                thr = row_max * self.relative_threshold
                total += np.count_nonzero(np.abs(row) > thr)
        return int(total)

    def count_nnz_cpu(self) -> int:
        """
        Exact count of non-zero elements using CPU.
        
        Returns:
            Exact number of non-zero elements
        """
        num_rows = self.N * self.T
        num_cols = self.Z * self.X
        
        # Initialize row pointer
        self.row_ptr = np.zeros(num_rows + 1, dtype=np.int64)
        
        # Count NNZ per row
        for global_row in trange(num_rows, desc='Counting NNZ (CPU)'):
            n_idx = global_row // self.T
            t_idx = global_row % self.T
            row = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            row_max = np.max(np.abs(row))
            thr = row_max * self.relative_threshold
            nnz = np.count_nonzero(np.abs(row) > thr)
            self.row_ptr[global_row + 1] = self.row_ptr[global_row] + nnz
        
        self.total_nnz = int(self.row_ptr[-1])
        return self.total_nnz

    def fill_csr_cpu(self):
        """
        Fill CSR matrix using CPU implementation.
        """
        num_rows = self.N * self.T
        num_cols = self.Z * self.X
        
        # Allocate arrays
        self.h_col_ind = np.zeros(self.total_nnz, dtype=np.uint32)
        self.h_values = np.zeros(self.total_nnz, dtype=np.float32)
        
        # Fill CSR
        ptr = 0
        for global_row in trange(num_rows, desc='Filling CSR (CPU)'):
            n_idx = global_row // self.T
            t_idx = global_row % self.T
            row = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            row_max = np.max(np.abs(row))
            thr = row_max * self.relative_threshold
            
            for col in range(num_cols):
                if np.abs(row[col]) > thr:
                    self.h_col_ind[ptr] = col
                    self.h_values[ptr] = row[col]
                    ptr += 1
        
        print('CSR generated (CPU) ✔')

    def allocate(self):
        """
        Allocate and fill the CSR matrix.
        Uses GPU if available, otherwise falls back to CPU.
        """
        num_rows = self.N * self.T
        num_cols = self.Z * self.X
        
        # Try GPU first
        if self._check_gpu_available():
            try:
                self.load_precompiled_module()
                if self.sparse_mod is not None:
                    self._allocate_gpu()
                    return
            except Exception as e:
                warnings.warn(f"GPU allocation failed: {e}. Falling back to CPU.")
        
        # Fall back to CPU
        self.device = 'cpu'
        self.count_nnz_cpu()
        self.fill_csr_cpu()
        self.compute_norm_factor_from_csr()

    def _allocate_gpu(self):
        """
        Allocate and fill the CSR matrix on GPU using custom kernels.
        """
        num_rows = self.N * self.T
        num_cols = self.Z * self.X
        
        # Allocate host row_ptr
        self.row_ptr = np.zeros(num_rows + 1, dtype=np.int64)
        
        # GPU temp buffers
        dense_block_host = np.empty((self.block_rows, num_cols), dtype=np.float32)
        dense_block_gpu = cp.cuda.alloc(dense_block_host.nbytes)
        row_nnz_gpu = cp.cuda.alloc(self.block_rows * np.dtype(np.int32).itemsize)
        
        block_size = 128
        
        # Count NNZ
        for b in trange(0, num_rows, self.block_rows, desc='Counting NNZ (GPU)'):
            current_rows = min(self.block_rows, num_rows - b)
            for r in range(current_rows):
                global_row = b + r
                n_idx = global_row // self.T
                t_idx = global_row % self.T
                dense_block_host[r, :] = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            cp.cuda.memcpy_htod(dense_block_gpu, dense_block_host)
            
            grid = ((current_rows + block_size - 1) // block_size, 1, 1)
            count_nnz_kernel = self.sparse_mod.get_function('count_nnz_rows_kernel')
            count_nnz_kernel(
                grid=grid, block=(block_size, 1, 1),
                args=[dense_block_gpu, row_nnz_gpu, np.int32(current_rows), np.int32(num_cols),
                      np.float32(self.relative_threshold)]
            )
            
            row_nnz_host = np.empty(current_rows, dtype=np.int32)
            cp.cuda.memcpy_dtoh(row_nnz_host, row_nnz_gpu)
            self.row_ptr[b + 1:b + current_rows + 1] = self.row_ptr[b] + np.cumsum(row_nnz_host, dtype=np.int64)
        
        # Total NNZ
        self.total_nnz = int(self.row_ptr[-1])
        print(f"Total NNZ: {self.total_nnz}")
        
        # Allocate final arrays
        self.h_col_ind = np.zeros(self.total_nnz, dtype=np.uint32)
        self.h_values = np.zeros(self.total_nnz, dtype=np.float32)
        
        # Copy row_ptr to device
        self.row_ptr_gpu = cp.cuda.alloc(self.row_ptr.nbytes)
        cp.cuda.memcpy_htod(self.row_ptr_gpu, self.row_ptr)
        
        # Allocate device arrays
        self.col_ind_gpu = cp.cuda.alloc(self.h_col_ind.nbytes)
        self.values_gpu = cp.cuda.alloc(self.h_values.nbytes)
        
        # Fill CSR
        for b in trange(0, num_rows, self.block_rows, desc='Filling CSR (GPU)'):
            current_rows = min(self.block_rows, num_rows - b)
            for r in range(current_rows):
                global_row = b + r
                n_idx = global_row // self.T
                t_idx = global_row % self.T
                dense_block_host[r, :] = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            cp.cuda.memcpy_htod(dense_block_gpu, dense_block_host)
            
            grid = ((current_rows + block_size - 1) // block_size, 1, 1)
            fill_csr_kernel = self.sparse_mod.get_function('fill_kernel__CSR')
            fill_csr_kernel(
                grid=grid, block=(block_size, 1, 1),
                args=[dense_block_gpu, self.row_ptr_gpu, self.col_ind_gpu, self.values_gpu,
                      np.int32(b), np.int32(current_rows), np.int32(num_cols),
                      np.float32(self.relative_threshold), np.int64(self.total_nnz)]
            )
            cp.cuda.Stream.null.synchronize()
        
        # Copy back to host
        cp.cuda.memcpy_dtoh(self.h_col_ind, self.col_ind_gpu)
        cp.cuda.memcpy_dtoh(self.h_values, self.values_gpu)
        print('CSR generated (GPU) ✔')
        
        # Compute normalization factor
        self.compute_norm_factor_from_csr()
        
        # Free temporaries
        dense_block_gpu.free()
        row_nnz_gpu.free()

    def compute_norm_factor_from_csr(self):
        """
        Compute normalization factor from CSR matrix (sum per column).
        """
        ZX = self.Z * self.X
        
        if self._check_gpu_available() and self.sparse_mod is not None:
            # GPU implementation
            col_sum_gpu = cp.cuda.alloc(ZX * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(col_sum_gpu, 0, ZX)
            
            acc_kernel = self.sparse_mod.get_function("accumulate_columns_atomic")
            threads = 256
            blocks = (self.total_nnz + threads - 1) // threads
            
            acc_kernel(
                grid=(blocks, 1), block=(threads, 1, 1),
                args=[self.values_gpu, self.col_ind_gpu, np.int64(self.total_nnz), col_sum_gpu]
            )
            cp.cuda.Stream.null.synchronize()
            
            norm = np.empty(ZX, dtype=np.float32)
            cp.cuda.memcpy_dtoh(norm, col_sum_gpu)
            col_sum_gpu.free()
        else:
            # CPU implementation
            norm = np.zeros(ZX, dtype=np.float32)
            for i in range(self.total_nnz):
                col = int(self.h_col_ind[i])
                norm[col] += np.abs(self.h_values[i])
        
        norm = np.maximum(norm.astype(np.float64), 1e-6)
        self.norm_factor_inv = (1.0 / norm).astype(np.float32)
        
        # Copy to GPU if available
        if self._check_gpu_available():
            self.norm_factor_inv_gpu = cp.cuda.alloc(self.norm_factor_inv.nbytes)
            cp.cuda.memcpy_htod(self.norm_factor_inv_gpu, self.norm_factor_inv)

    def forward_projection(self, theta: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Perform forward projection: q = A * theta
        
        Args:
            theta: Input vector (Z*X,)
            
        Returns:
            Projection result (N*T,)
        """
        if self.device == 'gpu' and CUPY_AVAILABLE and self.sparse_mod is not None:
            # GPU implementation using custom kernel
            theta_gpu = cp.asarray(theta) if not isinstance(theta, cp.ndarray) else theta
            q_gpu = cp.zeros(self.N * self.T, dtype=np.float32)
            
            proj_kernel = self.sparse_mod.get_function('projection_kernel__CSR')
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
            # CPU implementation
            theta_cpu = np.asarray(theta) if not isinstance(theta, np.ndarray) else theta
            if isinstance(theta_cpu, cp.ndarray):
                theta_cpu = cp.asnumpy(theta_cpu)
            
            q = np.zeros(self.N * self.T, dtype=np.float32)
            for i in range(self.N * self.T):
                start = int(self.row_ptr[i])
                end = int(self.row_ptr[i + 1])
                q[i] = np.sum(self.h_values[start:end] * theta_cpu[self.h_col_ind[start:end]])
            return q

    def backward_projection(self, e: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Perform backward projection: c = A^T * e
        
        Args:
            e: Input vector (N*T,)
            
        Returns:
            Backprojection result (Z*X,)
        """
        if self.device == 'gpu' and CUPY_AVAILABLE and self.sparse_mod is not None:
            # GPU implementation using custom kernel
            e_gpu = cp.asarray(e) if not isinstance(e, cp.ndarray) else e
            c_gpu = cp.zeros(self.Z * self.X, dtype=np.float32)
            
            backproj_kernel = self.sparse_mod.get_function('backprojection_kernel__CSR')
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
            # CPU implementation
            e_cpu = np.asarray(e) if not isinstance(e, np.ndarray) else e
            if isinstance(e_cpu, cp.ndarray):
                e_cpu = cp.asnumpy(e_cpu)
            
            c = np.zeros(self.Z * self.X, dtype=np.float32)
            for i in range(self.N * self.T):
                start = int(self.row_ptr[i])
                end = int(self.row_ptr[i + 1])
                for j in range(start, end):
                    col = int(self.h_col_ind[j])
                    c[col] += self.h_values[j] * e_cpu[i]
            return c

    def getMatrixSize(self) -> dict:
        """
        Returns the total size of the CSR matrix in GB.
        
        Returns:
            Dictionary with memory usage information
        """
        if self.row_ptr is None:
            return {"error": "Sparse matrix not allocated yet."}
        
        total_bytes = 0
        
        # Host memory
        if self.row_ptr is not None:
            total_bytes += self.row_ptr.nbytes
        if self.h_col_ind is not None:
            total_bytes += self.h_col_ind.nbytes
        if self.h_values is not None:
            total_bytes += self.h_values.nbytes
        if self.norm_factor_inv is not None:
            total_bytes += self.norm_factor_inv.nbytes
        
        # GPU memory
        if self.row_ptr_gpu is not None:
            total_bytes += self.row_ptr.nbytes  # Same size as host
        if self.col_ind_gpu is not None:
            total_bytes += self.h_col_ind.nbytes
        if self.values_gpu is not None:
            total_bytes += self.h_values.nbytes
        if self.norm_factor_inv_gpu is not None:
            total_bytes += self.norm_factor_inv.nbytes
        
        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024**3),
            "device": self.device
        }

    def free(self):
        """
        Free all GPU memory allocated by the CSR matrix.
        """
        try:
            for attr in ['col_ind_gpu', 'values_gpu', 'row_ptr_gpu', 'norm_factor_inv_gpu']:
                gpu_mem = getattr(self, attr, None)
                if gpu_mem is not None:
                    try:
                        gpu_mem.free()
                    except:
                        pass
                    setattr(self, attr, None)
        except Exception as e:
            warnings.warn(f"Error freeing GPU memory: {e}")

    def compute_density(self) -> float:
        """
        Returns the actual density of the CSR matrix = NNZ / (num_rows * num_cols).
        
        Returns:
            Density of the matrix
        """
        if self.row_ptr is None or self.h_values is None:
            raise RuntimeError("row_ptr and h_values required to compute density")
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        total_nnz = int(self.row_ptr[-1])
        density = total_nnz / (num_rows * num_cols)
        return density

    def flipAngle(self):
        """
        Permute the columns of the CSR matrix corresponding to opposite angles.
        Assumes N is even and angles are symmetrically organized.
        """
        if self.N % 2 != 0:
            raise ValueError("Number of angles must be even to permute opposite angles.")
        
        # Create new column indexing
        new_col_ind = self.h_col_ind.copy()
        ZX = self.Z * self.X
        angle_block_size = ZX // self.N
        
        # For each pair of opposite angles
        for n in range(self.N // 2):
            n_opposite = n + self.N // 2
            block_n_start = n * angle_block_size
            block_n_end = (n + 1) * angle_block_size
            block_opposite_start = n_opposite * angle_block_size
            block_opposite_end = (n_opposite + 1) * angle_block_size
            
            # Find all elements in col_ind pointing to these blocks
            mask_n = (self.h_col_ind >= block_n_start) & (self.h_col_ind < block_n_end)
            mask_opposite = (self.h_col_ind >= block_opposite_start) & (self.h_col_ind < block_opposite_end)
            
            # Permute indices
            new_col_ind[mask_n] = self.h_col_ind[mask_n] - block_n_start + block_opposite_start
            new_col_ind[mask_opposite] = self.h_col_ind[mask_opposite] - block_opposite_start + block_n_start
        
        # Update h_col_ind
        self.h_col_ind = new_col_ind
        
        # Recompute norm_factor_inv
        self.compute_norm_factor_from_csr()
        
        # Update GPU data
        if self._check_gpu_available() and self.col_ind_gpu is not None:
            cp.cuda.memcpy_htod(self.col_ind_gpu, self.h_col_ind)
