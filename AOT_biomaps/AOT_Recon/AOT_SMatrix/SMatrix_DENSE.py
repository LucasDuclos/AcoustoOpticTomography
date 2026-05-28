"""
SMatrix_DENSE.py

Dense matrix construction and operations.
Supports both CPU (NumPy) and GPU (CuPy/CUDA) implementations.
Follows the same pattern as SMatrix_SELL and SMatrix_CSR.
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


class SMatrix_DENSE:
    """
    Construction of a DENSE matrix from a `manip` object.
    
    Supports both CPU and GPU implementations:
    - On GPU: Uses CuPy for memory management and custom CUDA kernels (compiled from source)
    - On CPU: Uses NumPy arrays
    
    Usage:
        S = SMatrix_DENSE(manip, device='gpu')  # or 'cpu'
        S.allocate()
    
    After allocate(), the following attributes are available:
    - dense_matrix: Dense matrix array (host) with shape (T, N, Z, X)
    - norm_factor_inv: Normalization factor (host)
    - For GPU: dense_matrix_gpu, norm_factor_inv_gpu
    - matrix_type: 'DENSE'
    - device: 'cpu' or 'gpu'
    - Z, X: Image dimensions
    """
    
    # Class-level cache for compiled CUDA module
    _compiled_module = None

    def __init__(self, manip, device: Optional[str] = None):
        """
        Initialize DENSE matrix constructor.
        
        Args:
            manip: Manipulation object containing acoustic fields
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
        self.matrix_type = 'DENSE'
        
        # Initialize attributes
        self.dense_matrix = None
        self.norm_factor_inv = None
        
        # GPU-specific attributes
        self.dense_matrix_gpu = None
        self.norm_factor_inv_gpu = None
        self.sparse_mod = None
        
        # Preconditioner attributes
        self.preconditioner = None
        self.preconditioner_inv = None
        self.preconditioner_gpu = None
        self.preconditioner_inv_gpu = None
        
        # Path to CUDA source file
        cuda_parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.cuda_source_path = os.path.join(cuda_parent_dir, "AOT_biomaps_kernels.cu")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.free()

    def _check_gpu_available(self) -> bool:
        """Check if GPU operations are available."""
        if self.device != 'gpu':
            return False
        if not CUPY_AVAILABLE:
            warnings.warn("CuPy not available. Falling back to CPU.")
            self.device = 'cpu'
            return False
        return True

    def load_module(self):
        """Compile and load CUDA kernels from source using CuPy."""
        if not self._check_gpu_available():
            return
        
        # Use global cache to avoid recompiling multiple times
        if SMatrix_DENSE._compiled_module is None:
            if not os.path.exists(self.cuda_source_path):
                warnings.warn(
                    f"CUDA source file {os.path.basename(self.cuda_source_path)} not found at: {self.cuda_source_path}. "
                    "Falling back to CPU implementation."
                )
                self.device = 'cpu'
                return
            
            try:
                # Read CUDA source code
                with open(self.cuda_source_path, 'r', encoding='utf-8') as f:
                    cuda_source = f.read()
                
                # Compile with CuPy (uses cache automatically)
                SMatrix_DENSE._compiled_module = cp.cuda.compile_with_cache(cuda_source)
                
            except Exception as e:
                warnings.warn(f"Failed to compile CUDA module: {e}. Falling back to CPU.")
                self.device = 'cpu'
                return
        
        # Use the cached compiled module
        self.sparse_mod = SMatrix_DENSE._compiled_module

    def allocate(self):
        """
        Allocate and fill the DENSE matrix.
        Uses GPU if available, otherwise falls back to CPU.
        """
        # Try GPU first
        if self._check_gpu_available():
            try:
                self.load_module()
                if self.sparse_mod is not None:
                    self._allocate_gpu()
                    return
            except Exception as e:
                warnings.warn(f"GPU allocation failed: {e}. Falling back to CPU.")
        
        # Fall back to CPU
        self.device = 'cpu'
        self._allocate_cpu()

    def _allocate_gpu(self):
        """Allocate and fill the DENSE matrix on GPU using custom kernels."""
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        
        # Allocate dense matrix on GPU
        self.dense_matrix_gpu = cp.cuda.alloc(self.T * self.N * self.Z * self.X * np.dtype(np.float32).itemsize)
        
        # Prepare host buffer for field data
        field_host = np.empty((self.T, self.Z, self.X), dtype=np.float32)
        
        # Fill dense matrix using CUDA kernel
        fill_kernel = self.sparse_mod.get_function("fill_dense_matrix_kernel")
        
        # Process in blocks for better memory efficiency
        block_size = 256
        total_elements = self.T * self.N * self.Z * self.X
        
        for n in trange(self.N, desc="Filling DENSE (GPU)"):
            # Copy field data for this angle to host buffer
            for t in range(self.T):
                field_host[t] = self.manip.AcousticFields[n].field[t].astype(np.float32)
            
            # Allocate device memory for this field
            field_gpu = cp.cuda.alloc(field_host.nbytes)
            cp.cuda.memcpy_htod(field_gpu, field_host)
            
            # Launch kernel for this field
            threads = 256
            blocks = (self.T * self.Z * self.X + threads - 1) // threads
            
            fill_kernel(
                grid=(blocks, 1, 1), block=(threads, 1, 1),
                args=[self.dense_matrix_gpu, field_gpu, np.int32(self.T), np.int32(self.N), 
                      np.int32(self.Z), np.int32(self.X), np.int32(self.T * self.Z * self.X)]
            )
            cp.cuda.Stream.null.synchronize()
            
            field_gpu.free()
        
        # Copy back to host for CPU access
        self.dense_matrix = np.empty((self.T, self.N, self.Z, self.X), dtype=np.float32)
        cp.cuda.memcpy_dtoh(self.dense_matrix, self.dense_matrix_gpu)
        
        # Compute normalization factor
        self.compute_norm_factor()
        
        print(f"DENSE matrix allocated on GPU: shape=({self.T}, {self.N}, {self.Z}, {self.X})")

    def _allocate_cpu(self):
        """Allocate and fill the DENSE matrix on CPU."""
        # Build dense matrix from acoustic fields
        self.dense_matrix = np.zeros((self.T, self.N, self.Z, self.X), dtype=np.float32)
        
        for n in trange(self.N, desc="Filling DENSE (CPU)"):
            field = self.manip.AcousticFields[n].field
            for t in range(self.T):
                self.dense_matrix[t, n] = field[t].astype(np.float32)
        
        # Compute normalization factor
        self.compute_norm_factor()
        
        print(f"DENSE matrix allocated on CPU: shape=({self.T}, {self.N}, {self.Z}, {self.X})")

    def compute_norm_factor(self):
        """
        Compute the normalization factor: norm_factor_inv = 1 / (A^T * 1).
        Uses GPU if available, otherwise falls back to CPU.
        """
        ZX = int(self.Z * self.X)
        TN = int(self.T * self.N)
        
        if self._check_gpu_available() and self.sparse_mod is not None and self.dense_matrix_gpu is not None:
            # GPU implementation using CUDA kernel
            # First compute column sums (A^T * 1)
            ones_gpu = cp.cuda.alloc(TN * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(ones_gpu, 0x3f800000, TN)  # 1.0f bit pattern
            
            c_gpu = cp.cuda.alloc(ZX * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(c_gpu, 0, ZX)
            
            bp_kernel = self.sparse_mod.get_function("backprojection_kernel__DENSE")
            threads = 256
            blocks = (ZX + threads - 1) // threads
            
            bp_kernel(
                grid=(blocks, 1, 1), block=(threads, 1, 1),
                args=[c_gpu, self.dense_matrix_gpu, ones_gpu, np.int32(self.T), 
                      np.int32(self.N), np.int32(self.Z), np.int32(self.X)]
            )
            cp.cuda.Stream.null.synchronize()
            
            # Copy result to host
            c_host = np.empty(ZX, dtype=np.float32)
            cp.cuda.memcpy_dtoh(c_host, c_gpu)
            
            ones_gpu.free()
            c_gpu.free()
            
            # Compute normalization factor
            c_host = np.maximum(c_host, 1e-6)
            self.norm_factor_inv = (1.0 / c_host).astype(np.float32)
            
            # Copy to GPU
            self.norm_factor_inv_gpu = cp.cuda.alloc(self.norm_factor_inv.nbytes)
            cp.cuda.memcpy_htod(self.norm_factor_inv_gpu, self.norm_factor_inv)
        else:
            # CPU implementation
            ones = np.ones(TN, dtype=np.float32)
            c_host = self.backprojection(ones)
            
            c_host = np.maximum(c_host, 1e-6)
            self.norm_factor_inv = (1.0 / c_host).astype(np.float32)
            
            # Copy to GPU if available
            if self._check_gpu_available():
                self.norm_factor_inv_gpu = cp.array(self.norm_factor_inv, dtype=np.float32)

    def compute_preconditioner(self, preconditioner_type='diagonal'):
        """
        Compute preconditioner for the system matrix.
        
        Args:
            preconditioner_type: Type of preconditioner ('diagonal' or 'none')
            
        Returns:
            tuple: (preconditioner, preconditioner_inv) on appropriate device
        """
        from AOT_biomaps.AOT_Recon.ReconEnums import PreconditionerType
        
        if preconditioner_type == PreconditionerType.NONE or preconditioner_type == 'none':
            self.preconditioner = None
            self.preconditioner_inv = None
            self.preconditioner_gpu = None
            self.preconditioner_inv_gpu = None
            return None, None
        
        ZX = int(self.Z * self.X)
        TN = int(self.T * self.N)
        
        if self._check_gpu_available() and self.sparse_mod is not None and self.dense_matrix_gpu is not None:
            # GPU implementation
            ones_gpu = cp.cuda.alloc(TN * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(ones_gpu, 0x3f800000, TN)  # 1.0f bit pattern
            
            c_gpu = cp.cuda.alloc(ZX * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(c_gpu, 0, ZX)
            
            bp_kernel = self.sparse_mod.get_function("backprojection_kernel__DENSE")
            threads = 256
            blocks = (ZX + threads - 1) // threads
            
            bp_kernel(
                grid=(blocks, 1, 1), block=(threads, 1, 1),
                args=[c_gpu, self.dense_matrix_gpu, ones_gpu, np.int32(self.T), 
                      np.int32(self.N), np.int32(self.Z), np.int32(self.X)]
            )
            cp.cuda.Stream.null.synchronize()
            
            # Copy result to host
            c_host = np.empty(ZX, dtype=np.float32)
            cp.cuda.memcpy_dtoh(c_host, c_gpu)
            
            ones_gpu.free()
            c_gpu.free()
            
            # Store preconditioner
            self.preconditioner = c_host.copy()
            self.preconditioner = np.maximum(self.preconditioner, 1e-6)
            self.preconditioner_inv = (1.0 / self.preconditioner).astype(np.float32)
            
            # Copy to GPU
            self.preconditioner_gpu = cp.cuda.alloc(self.preconditioner.nbytes)
            self.preconditioner_inv_gpu = cp.cuda.alloc(self.preconditioner_inv.nbytes)
            cp.cuda.memcpy_htod(self.preconditioner_gpu, self.preconditioner)
            cp.cuda.memcpy_htod(self.preconditioner_inv_gpu, self.preconditioner_inv)
            
            return self.preconditioner_gpu, self.preconditioner_inv_gpu
        else:
            # CPU implementation
            ones = np.ones(TN, dtype=np.float32)
            c_host = self.backprojection(ones)
            
            # Store preconditioner
            self.preconditioner = c_host.copy()
            self.preconditioner = np.maximum(self.preconditioner, 1e-6)
            self.preconditioner_inv = (1.0 / self.preconditioner).astype(np.float32)
            
            return self.preconditioner, self.preconditioner_inv

    def apply_preconditioner(self, U):
        """
        Apply preconditioner to a vector.
        
        Args:
            U: Vector to precondition
            
        Returns:
            Preconditioned vector
        """
        if self._check_gpu_available() and self.preconditioner_inv_gpu is not None:
            U_gpu = cp.asarray(U) if not isinstance(U, cp.ndarray) else U
            return U_gpu * cp.asarray(self.preconditioner_inv_gpu)
        elif self.preconditioner_inv is not None:
            U_cpu = np.asarray(U) if not isinstance(U, np.ndarray) else U
            if isinstance(U_cpu, cp.ndarray):
                U_cpu = cp.asnumpy(U_cpu)
            return U_cpu * self.preconditioner_inv
        else:
            return U

    def get_matrix_size(self):
        """
        Get matrix size information.
        
        Returns:
            dict: Matrix size information
        """
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

    def forward_projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform forward projection: q = A * theta
        
        Args:
            theta: Input vector (Z*X,)
            
        Returns:
            Projection result (N*T,)
        """
        if self._check_gpu_available() and self.sparse_mod is not None and self.dense_matrix_gpu is not None:
            # GPU implementation
            theta_gpu = cp.asarray(theta) if not isinstance(theta, cp.ndarray) else theta
            q_gpu = cp.zeros(self.N * self.T, dtype=np.float32)
            
            proj_kernel = self.sparse_mod.get_function('projection_kernel__DENSE')
            threads = 256
            blocks = (self.N * self.T + threads - 1) // threads
            
            proj_kernel(
                grid=(blocks, 1), block=(threads, 1, 1),
                args=[q_gpu.data.ptr, self.dense_matrix_gpu, theta_gpu.data.ptr, 
                      np.int32(self.T), np.int32(self.N), np.int32(self.Z), np.int32(self.X)]
            )
            cp.cuda.Stream.null.synchronize()
            return q_gpu
        else:
            # CPU implementation
            theta_cpu = np.asarray(theta) if not isinstance(theta, np.ndarray) else theta
            if isinstance(theta_cpu, cp.ndarray):
                theta_cpu = cp.asnumpy(theta_cpu)
            
            q = np.zeros(self.N * self.T, dtype=np.float32)
            
            # Reshape dense_matrix for efficient computation
            # dense_matrix has shape (T, N, Z, X), we need to compute q[row] = sum_col(A[row, col] * theta[col])
            for t in range(self.T):
                for n in range(self.N):
                    row_idx = t * self.N + n
                    for z in range(self.Z):
                        for x in range(self.X):
                            col_idx = z * self.X + x
                            q[row_idx] += self.dense_matrix[t, n, z, x] * theta_cpu[col_idx]
            
            return q

    def backward_projection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform backward projection: c = A^T * e
        
        Args:
            e: Input vector (N*T,)
            
        Returns:
            Backprojection result (Z*X,)
        """
        if self._check_gpu_available() and self.sparse_mod is not None and self.dense_matrix_gpu is not None:
            # GPU implementation
            e_gpu = cp.asarray(e) if not isinstance(e, cp.ndarray) else e
            c_gpu = cp.zeros(self.Z * self.X, dtype=np.float32)
            
            bp_kernel = self.sparse_mod.get_function('backprojection_kernel__DENSE')
            threads = 256
            blocks = (self.Z * self.X + threads - 1) // threads
            
            bp_kernel(
                grid=(blocks, 1), block=(threads, 1, 1),
                args=[c_gpu.data.ptr, self.dense_matrix_gpu, e_gpu.data.ptr, 
                      np.int32(self.T), np.int32(self.N), np.int32(self.Z), np.int32(self.X)]
            )
            cp.cuda.Stream.null.synchronize()
            return c_gpu
        else:
            # CPU implementation
            e_cpu = np.asarray(e) if not isinstance(e, np.ndarray) else e
            if isinstance(e_cpu, cp.ndarray):
                e_cpu = cp.asnumpy(e_cpu)
            
            c = np.zeros(self.Z * self.X, dtype=np.float32)
            
            # Reshape dense_matrix for efficient computation
            for t in range(self.T):
                for n in range(self.N):
                    row_idx = t * self.N + n
                    e_val = e_cpu[row_idx]
                    if e_val == 0.0:
                        continue
                    for z in range(self.Z):
                        for x in range(self.X):
                            col_idx = z * self.X + x
                            c[col_idx] += self.dense_matrix[t, n, z, x] * e_val
            
            return c

    def free(self):
        """Free allocated memory."""
        self.dense_matrix = None
        self.dense_matrix_gpu = None
        self.norm_factor_inv = None
        self.norm_factor_inv_gpu = None
        self.sparse_mod = None

    def __repr__(self):
        return f"SMatrix_DENSE(device={self.device}, shape=({self.T}, {self.N}, {self.Z}, {self.X}))"
