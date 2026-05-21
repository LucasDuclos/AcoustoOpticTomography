"""
AOT_SparseSMatrix package

Provides sparse matrix implementations (CSR and SELL-C-sigma) with CPU/GPU support.
"""

from .SparseSMatrix_CSR import SparseSMatrix_CSR
from .SparseSMatrix_SELL import SparseSMatrix_SELL

__all__ = ['SparseSMatrix_CSR', 'SparseSMatrix_SELL']
