"""
SMatrix package

Provides sparse matrix implementations (CSR, SELL, DENSE) with CPU/GPU support.
"""

from ._mainSMatrix import SMatrix
from .SMatrix_CSR import SparseSMatrix_CSR
from .SMatrix_SELL import SparseSMatrix_SELL
from .SMatrix_DENSE import SparseSMatrix_DENSE

__all__ = ['SMatrix', 'SparseSMatrix_CSR', 'SparseSMatrix_SELL', 'SparseSMatrix_DENSE']
