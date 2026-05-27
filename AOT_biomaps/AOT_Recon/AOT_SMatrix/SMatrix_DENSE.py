"""
SMatrix_DENSE.py

Dense matrix construction and operations.
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


class SMatrix_DENSE:
    """
    Construction of a DENSE matrix from a `manip` object.
    
    Supports both CPU and GPU implementations:
    - On GPU: Uses CuPy for memory management and custom CUDA kernels
    - On CPU: Uses NumPy arrays
    
    Usage:
        S = SMatrix_DENSE(manip, device='gpu')  # or 'cpu'
        S.allocate()
    
    After allocate(), the following attributes are available:
    - dense_matrix: Dense matrix array (host)
    - For GPU: dense_matrix_gpu
    """

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
        
        # Initialize attributes
        self.dense_matrix = None
        self.norm_factor_inv = None
        
        # GPU-specific attributes
        self.dense_matrix_gpu = None
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

    def allocate(self, **kwargs):
        """
        Allocate memory for the dense matrix.
        
        For DENSE matrices, we store the full (T, N, Z, X) matrix.
        Uses GPU if available, otherwise falls back to CPU.
        """
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
        self._allocate_cpu()

    def _allocate_gpu(self):
        """Allocate and fill the dense matrix on GPU."""
        from AOT_Kernels import allocate_dense_matrix_gpu
        
        # Use the GPU kernel to allocate and fill the matrix
        self.dense_matrix_gpu, _ = allocate_dense_matrix_gpu(
            self.manip, device='gpu'
        )
        
        # Copy back to host for CPU access
        self.dense_matrix = cp.asnumpy(self.dense_matrix_gpu)
        
        # Compute normalization factor using GPU
        self.compute_norm_factor()
        
        print(f"DENSE matrix allocated on GPU: shape=({self.T}, {self.N}, {self.Z}, {self.X})")

    def _allocate_cpu(self):
        """Allocate and fill the dense matrix on CPU."""
        # Build dense matrix from acoustic fields
        self.dense_matrix = np.zeros((self.T, self.N, self.Z, self.X), dtype=np.float32)
        
        for n in trange(self.N, desc="Building dense matrix (CPU)"):
            field = self.manip.AcousticFields[n].field
            for t in range(self.T):
                self.dense_matrix[t, n] = field[t].astype(np.float32)
        
        # Compute normalization factor
        self.compute_norm_factor()
        
        print(f"DENSE matrix allocated on CPU: shape=({self.T}, {self.N}, {self.Z}, {self.X})")

    def compute_norm_factor(self):
        """
        Compute the normalization factor: norm_factor_inv = 1 / (sum(|A|) + eps).
        Uses GPU if available, otherwise falls back to CPU.
        """
        if self._check_gpu_available() and self.dense_matrix_gpu is not None:
            # GPU implementation
            from AOT_Kernels import compute_norm_factor_dense_gpu
            self.norm_factor_inv_gpu = compute_norm_factor_dense_gpu(
                self.dense_matrix_gpu, device='gpu'
            )
            self.norm_factor_inv = cp.asnumpy(self.norm_factor_inv_gpu)
        else:
            # CPU implementation
            sum_abs = np.sum(np.abs(self.dense_matrix))
            self.norm_factor_inv = 1.0 / (sum_abs + 1e-12)
            if self._check_gpu_available():
                self.norm_factor_inv_gpu = cp.array(self.norm_factor_inv, dtype=np.float32)

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

    def free(self):
        """Free allocated memory."""
        self.dense_matrix = None
        self.dense_matrix_gpu = None
        self.norm_factor_inv = None
        self.norm_factor_inv_gpu = None
        self.sparse_mod = None

    def __repr__(self):
        return f"SMatrix_DENSE(device={self.device}, shape=({self.T}, {self.N}, {self.Z}, {self.X}))"
