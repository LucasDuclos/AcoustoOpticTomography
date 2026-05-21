"""
SparseMatrixWrapper.py

Unified wrapper for sparse matrix operations (CSR and SELL-C-sigma).
Provides a consistent interface for both CPU and GPU implementations.
"""

import warnings
import numpy as np
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

if TYPE_CHECKING:
    import cupy as cp


class SparseMatrix:
    """
    Unified sparse matrix wrapper that supports both CSR and SELL-C-sigma formats.
    
    This wrapper provides a consistent interface for sparse matrix operations,
    automatically handling CPU/GPU implementations based on availability.
    
    Args:
        manip: Manipulation object containing acoustic fields
        matrix_type: 'CSR' or 'SELL' (default: 'CSR')
        device: 'cpu' or 'gpu' (auto-detected if None)
        **kwargs: Additional arguments passed to the specific matrix implementation
    """
    
    def __init__(self, manip, matrix_type: str = 'CSR', device: Optional[str] = None, **kwargs):
        self.matrix_type = matrix_type.upper()
        self.device = device
        self.manip = manip
        self._matrix = None
        
        # Import the appropriate matrix class
        if self.matrix_type == 'CSR':
            from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_CSR import SparseSMatrix_CSR
            self._matrix_class = SparseSMatrix_CSR
        elif self.matrix_type == 'SELL':
            from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL import SparseSMatrix_SELL
            self._matrix_class = SparseSMatrix_SELL
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}. Use 'CSR' or 'SELL'.")
        
        # Determine device if not specified
        if self.device is None:
            self.device = 'gpu' if CUPY_AVAILABLE else 'cpu'
        
        # Create the underlying matrix
        self._create_matrix(**kwargs)
    
    def _create_matrix(self, **kwargs):
        """Create the underlying sparse matrix implementation."""
        try:
            # Add device to kwargs if not already present
            if 'device' not in kwargs:
                kwargs['device'] = self.device
            self._matrix = self._matrix_class(self.manip, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to create {self.matrix_type} matrix: {e}. Falling back to CPU.")
            self.device = 'cpu'
            kwargs['device'] = 'cpu'
            self._matrix = self._matrix_class(self.manip, **kwargs)
    
    def allocate(self):
        """Allocate and build the sparse matrix."""
        self._matrix.allocate()
    
    def projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform forward projection: q = A * theta
        
        Args:
            theta: Input vector (Z*X,)
            
        Returns:
            Projection result (N*T,)
        """
        return self._matrix.projection(theta)
    
    def backprojection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform backprojection: c = A^T * e
        
        Args:
            e: Input vector (N*T,)
            
        Returns:
            Backprojection result (Z*X,)
        """
        return self._matrix.backprojection(e)
    
    def apply_normalization(self, x: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Apply normalization factor to the vector.
        
        Args:
            x: Input vector
            
        Returns:
            Normalized vector
        """
        if self._matrix.norm_factor_inv is not None:
            if self.device == 'gpu' and CUPY_AVAILABLE:
                import cupy as cp
                x_gpu = cp.asarray(x) if not isinstance(x, cp.ndarray) else x
                norm_gpu = cp.asarray(self._matrix.norm_factor_inv)
                return x_gpu * norm_gpu
            else:
                x_cpu = np.asarray(x) if not isinstance(x, np.ndarray) else x
                if isinstance(x_cpu, cp.ndarray):
                    x_cpu = cp.asnumpy(x_cpu)
                return x_cpu * self._matrix.norm_factor_inv
        return x
    
    def get_norm_factor_inv(self) -> Union[np.ndarray, 'cp.ndarray']:
        """Get the normalization factor inverse."""
        return self._matrix.norm_factor_inv
    
    def get_matrix_size(self) -> dict:
        """Get memory usage information."""
        return self._matrix.getMatrixSize()
    
    def compute_density(self) -> float:
        """Compute the density of the matrix."""
        return self._matrix.compute_density()
    
    def free(self):
        """Free all GPU memory."""
        self._matrix.free()
    
    def flipAngle(self):
        """Permute columns for opposite angles."""
        self._matrix.flipAngle()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.free()
    
    # Delegate other attributes to the underlying matrix
    @property
    def N(self) -> int:
        return self._matrix.N
    
    @property
    def T(self) -> int:
        return self._matrix.T
    
    @property
    def Z(self) -> int:
        return self._matrix.Z
    
    @property
    def X(self) -> int:
        return self._matrix.X
    
    @property
    def total_nnz(self) -> int:
        return self._matrix.total_nnz if hasattr(self._matrix, 'total_nnz') else 0
    
    @property
    def total_storage(self) -> int:
        return self._matrix.total_storage if hasattr(self._matrix, 'total_storage') else 0
    
    @property
    def row_ptr(self):
        return self._matrix.row_ptr
    
    @property
    def h_col_ind(self):
        return self._matrix.h_col_ind
    
    @property
    def h_values(self):
        return self._matrix.h_values
    
    @property
    def slice_ptr(self):
        return self._matrix.slice_ptr if hasattr(self._matrix, 'slice_ptr') else None
    
    @property
    def slice_len(self):
        return self._matrix.slice_len if hasattr(self._matrix, 'slice_len') else None


def create_sparse_matrix(manip, matrix_type: str = 'CSR', device: Optional[str] = None, **kwargs) -> SparseMatrix:
    """
    Factory function to create a sparse matrix.
    
    Args:
        manip: Manipulation object
        matrix_type: 'CSR' or 'SELL'
        device: 'cpu' or 'gpu'
        **kwargs: Additional arguments
        
    Returns:
        SparseMatrix wrapper
    """
    return SparseMatrix(manip, matrix_type=matrix_type, device=device, **kwargs)
