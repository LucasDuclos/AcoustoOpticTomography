"""
SMatrix package

Provides sparse matrix implementations (CSR, SELL, DENSE) with CPU/GPU support.
"""

from ._mainSMatrix import SMatrix
from .SMatrix_CSR import SMatrix_CSR
from .SMatrix_SELL import SMatrix_SELL
from .SMatrix_DENSE import SMatrix_DENSE

