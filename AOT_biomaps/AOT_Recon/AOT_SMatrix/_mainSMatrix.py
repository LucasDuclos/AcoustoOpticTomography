"""
SMatrix.py

Abstract Base Class for sparse and dense matrix representations.
Handles common initialization, CUDA module compilation, preconditioners,
and normalization factors for all matrix types.
"""

import os
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union

from AOT_biomaps.AOT_Recon.ReconEnums import SMatrixType, PreconditionerType
from AOT_biomaps.AOT_Recon.ReconTools import check_gpu_available

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class SMatrix(ABC):
    """
    Abstract base class for system matrices (CSR, SELL, DENSE).
    Provides unified memory management, CUDA loading, and preconditioner computation.
    """

    # Class-level cache for compiled CUDA module shared across all matrix types
    _compiled_module = None

    def __init__(self, experiment, device: Optional[str] = None):
        """
        Initialize base matrix parameters.
        """
        # Determine device
        if device is None:
            self.device = 'gpu:0' if CUPY_AVAILABLE else 'cpu'
        else:
            self.device = device

        
        self.experiment = experiment
        # Standard dimensions from AcousticFields
        self.N = len(experiment.AcousticFields)
        self.T = experiment.AcousticFields[0].field.shape[0]
        self.Z = experiment.AcousticFields[0].field.shape[1]
        self.X = experiment.AcousticFields[0].field.shape[2]
        
        # Common attributes
        self.norm_factor_inv = None
        self.norm_factor_inv_gpu = None
        
        self.preconditioner = None
        self.preconditioner_inv = None
        self.preconditioner_gpu = None
        self.preconditioner_inv_gpu = None
        
        self.sparse_mod = None

        # Path to CUDA source file
        cuda_parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.cuda_source_path = os.path.join(cuda_parent_dir, "AOT_biomaps_kernels.cu")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.free()

    def load_module(self):
        """Compile and load CUDA kernels from source using CuPy."""
        if not check_gpu_available(self):
            return

        if SMatrix._compiled_module is None:
            if not os.path.exists(self.cuda_source_path):
                warnings.warn(
                    f"CUDA source file not found at: {self.cuda_source_path}. "
                    "Falling back to CPU implementation."
                )
                self.device = 'cpu'
                return
            try:
                with open(self.cuda_source_path, 'r') as f:
                    cuda_source = f.read()

                SMatrix._compiled_module = cp.RawModule(
                    code=cuda_source,
                    options=('--std=c++11', '-use_fast_math')
                )
            except Exception as e:
                warnings.warn(f"Failed to compile CUDA module: {e}. Falling back to CPU.")
                self.device = 'cpu'
                return
        self.sparse_mod = SMatrix._compiled_module

    def compute_norm_factor(self):
        """
        Compute the normalization factor: norm_factor_inv = 1 / (A^T * 1).
        By default, it uses standard backward_projection (algebraic sum).
        Can be overridden by child classes (like CSR) if absolute sum is needed.
        """
        TN = int(self.T * self.N)
        use_gpu = check_gpu_available(self)

        ones = cp.ones(TN, dtype=np.float32) if use_gpu else np.ones(TN, dtype=np.float32)
        
        c_device = self.backward_projection(ones) 
        c_host = cp.asnumpy(c_device) if use_gpu else c_device.copy()
        
        c_host = np.maximum(c_host, 1e-6)
        self.norm_factor_inv = (1.0 / c_host).astype(np.float32)

        if CUPY_AVAILABLE:
            self.norm_factor_inv_gpu = cp.asarray(self.norm_factor_inv)

    def free(self):
        """Free all common GPU memory, then call specific free method."""
        try:
            for attr in ['norm_factor_inv_gpu', 'preconditioner_gpu', 'preconditioner_inv_gpu']:
                gpu_mem = getattr(self, attr, None)
                if gpu_mem is not None:
                    try:
                        if hasattr(gpu_mem, 'free'):
                            gpu_mem.free()
                        else:
                            del gpu_mem
                    except:
                        pass
                setattr(self, attr, None)
            
            # Delegate specific array cleanup to child class
            self._free_specific()
            
        except Exception as e:
            warnings.warn(f"Error freeing GPU memory: {e}")

    def allocate(self):
        """
        Build the matrix.
        Attempts to use GPU if available, otherwise falls back gracefully to CPU.
        """
        if check_gpu_available(self):
            try:
                self.load_module()
                if self.sparse_mod is not None:
                    self._allocate_gpu()
                    return
            except Exception as e:
                warnings.warn(f"GPU allocation failed: {e}. Falling back to CPU.")

        # Fallback to CPU
        self.device = 'cpu'
        self._allocate_cpu()

    # --- Abstract Methods (Strict Contract) ---

    @abstractmethod
    def _allocate_gpu(self):
        """Allocate and fill the matrix on GPU."""
        pass

    @abstractmethod
    def _allocate_cpu(self):
        """Allocate and fill the matrix on CPU."""
        pass

    @abstractmethod
    def forward_projection(self, theta: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Perform forward projection: q = A * theta"""
        pass

    @abstractmethod
    def backward_projection(self, e: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """Perform backward projection: c = A^T * e"""
        pass

    @abstractmethod
    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        """Apply apodization window to the matrix."""
        pass

    @abstractmethod
    def flip_probe(self):
        """Flip the probe at 180 degrees."""
        pass

    @abstractmethod
    def _free_specific(self):
        """Free specific GPU memory allocated by the child class."""
        pass
    
    @abstractmethod
    def get_matrix_size(self) -> dict:
        """Returns the total size of the matrix in GB."""
        pass