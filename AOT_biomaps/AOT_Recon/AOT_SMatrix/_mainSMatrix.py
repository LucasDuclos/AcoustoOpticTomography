"""
_mainSMatrix.py

Unified SMatrix interface for Acousto-Optic Tomography reconstruction.
Provides a device-agnostic (CPU/GPU) interface for sparse matrix operations.
"""

import numpy as np
from AOT_biomaps.AOT_Recon.ReconEnums import SMatrixType
# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SMatrix:
    """
    Unified sparse matrix interface for AOT reconstruction.
    
    This class provides a simple data container that delegates to specific
    implementations (CSR, SELL, DENSE) and automatically handles CPU/GPU
    operations based on the device parameter.
    
    Args:
        manip: Manipulation object containing matrix data
        matrix_type: 'CSR', 'SELL', or 'DENSE'
        device: 'cpu' or 'gpu'
        **kwargs: Additional arguments passed to the specific implementation
    """
    
    def __init__(self, manip, matrix_type=SMatrixType.CSR, device='cpu', **kwargs):
        if not isinstance(matrix_type, SMatrixType):
            # Try to convert string to enum
            try:
                matrix_type = SMatrixType[matrix_type.upper()]
            except KeyError:
                raise ValueError(f"Unknown matrix type: {matrix_type}. Use SMatrixType.CSR, SMatrixType.SELL, or SMatrixType.DENSE.")
        
        self.matrix_type = matrix_type
        self.device = device.lower()
        self.manip = manip
        self._data = None
        self.norm_factor_inv = None
        
        # Import the appropriate implementation
        if matrix_type == SMatrixType.CSR:
            from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_CSR import SparseSMatrix_CSR as ImplClass
        elif matrix_type == SMatrixType.SELL:
            from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_SELL import SparseSMatrix_SELL as ImplClass
        elif matrix_type == SMatrixType.DENSE:
            from AOT_biomaps.AOT_Recon.AOT_SMatrix.SMatrix_DENSE import SparseSMatrix_DENSE as ImplClass
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}. Use SMatrixType.CSR, SMatrixType.SELL, or SMatrixType.DENSE.")
        
        # Create the implementation instance
        self._data = ImplClass(manip, device=device, **kwargs)
        
        # Copy useful attributes from implementation
        self.Z = self._data.Z
        self.X = self._data.X
        self.T = self._data.T
        self.N = self._data.N
    
    def allocate(self, **kwargs):
        """Allocate matrix memory."""
        return self._data.allocate(**kwargs)
    
    def get_matrix_size(self):
        """Get matrix size information."""
        return self._data.get_matrix_size()
    
    @property
    def shape(self):
        """Matrix shape (TN, ZX)."""
        return (self.T * self.N, self.Z * self.X)
    
    def __repr__(self):
        return f"SMatrix(type={self.matrix_type}, device={self.device}, shape={self.shape})"
