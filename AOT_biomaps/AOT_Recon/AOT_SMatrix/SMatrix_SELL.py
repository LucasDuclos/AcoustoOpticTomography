"""
SMatrix_SELL.py

SELL-C-sigma sparse matrix construction and operations.
Supports both CPU (NumPy) and GPU (CuPy) implementations.

SELL-C-sigma is a sparse matrix format optimized for GPU operations.
It stores non-zero elements in a sliced ELL format with padding (sigma).
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


class SMatrix_SELL:
    """
    Sparse matrix in SELL-C-sigma format for efficient GPU operations.
    
    SELL-C-sigma format:
    - Divides rows into slices of height C (slice_height)
    - Each slice has a maximum of sigma non-zero elements per row (with padding)
    - Non-zero values and column indices are stored in flattened arrays
    
    Usage:
        S = SMatrix_SELL(manip, device='gpu')  # or 'cpu'
        S.allocate()
    
    After allocate(), the following attributes are available:
    - sell_values: Non-zero values array (host)
    - sell_colinds: Column indices array (host)
    - slice_ptr: Slice pointer array (host)
    - slice_len: Slice length array (host)
    - norm_factor_inv: Normalization factor (host)
    - For GPU: sell_values_gpu, sell_colinds_gpu, slice_ptr_gpu, slice_len_gpu, norm_factor_inv_gpu
    """

    def __init__(self, manip, block_rows: int = 64, relative_threshold: float = 0.3, 
                 device: Optional[str] = None, slice_height: int = 32):
        """
        Initialize SELL-C-sigma matrix constructor.
        
        Args:
            manip: Manipulation object containing acoustic fields
            block_rows: Number of rows to process at once
            relative_threshold: Threshold for considering a value as non-zero
            device: 'cpu' or 'gpu' (auto-detected if None)
            slice_height: Height of each slice (C parameter)
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
        self.slice_height = slice_height
        
        # Initialize attributes
        self.sell_values = None
        self.sell_colinds = None
        self.slice_ptr = None
        self.slice_len = None
        self.total_storage = 0
        self.norm_factor_inv = None
        
        # GPU-specific attributes
        self.sparse_mod = None
        self.sell_values_gpu = None
        self.sell_colinds_gpu = None
        self.slice_ptr_gpu = None
        self.slice_len_gpu = None
        self.norm_factor_inv_gpu = None
        
        # Path to CUDA module
        cubin_parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.module_path = os.path.join(cubin_parent_dir, "AOT_biomaps_kernels.cubin")

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
        """Load the pre-compiled CUDA module (.cubin) using CuPy."""
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

    def allocate(self):
        """
        Build SELL-C-sigma matrix from manip AcousticFields.
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
        """Allocate and fill the SELL matrix on GPU using custom kernels."""
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        C = int(self.slice_height)
        
        br = int(self.block_rows)
        dense_host = np.empty((br, num_cols), dtype=np.float32)
        
        # Allocate dense buffer on device
        dense_gpu = cp.cuda.alloc(dense_host.nbytes)
        
        # 1) Count NNZ per row
        row_nnz = np.zeros(num_rows, dtype=np.int32)
        row_nnz_gpu_block = cp.cuda.alloc(br * np.dtype(np.int32).itemsize)
        
        block_size = 128
        count_kernel = self.sparse_mod.get_function("count_nnz_rows_kernel")
        
        for b in trange(0, num_rows, br, desc="Count NNZ per row (GPU)"):
            R = min(br, num_rows - b)
            dense_host.fill(0.0)
            for i in range(R):
                rg = b + i
                n_idx = rg // self.T
                t_idx = rg % self.T
                dense_host[i, :] = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            cp.cuda.memcpy_htod(dense_gpu, dense_host)
            
            grid = ((R + block_size - 1) // block_size, 1, 1)
            count_kernel(
                grid=grid, block=(block_size, 1, 1),
                args=[dense_gpu, row_nnz_gpu_block, np.int32(R), np.int32(num_cols), 
                      np.float32(self.relative_threshold)]
            )
            cp.cuda.Stream.null.synchronize()
            
            tmp = np.empty(R, dtype=np.int32)
            cp.cuda.memcpy_dtoh(tmp, row_nnz_gpu_block)
            row_nnz[b:b+R] = tmp
        
        row_nnz_gpu_block.free()
        dense_gpu.free()
        
        # 2) Compute per-slice maxlen and slice_ptr
        num_slices = (num_rows + C - 1) // C
        self.slice_len = np.zeros(num_slices, dtype=np.int32)
        
        for s in range(num_slices):
            r0 = s * C
            r1 = min(num_rows, r0 + C)
            self.slice_len[s] = int(np.max(row_nnz[r0:r1])) if (r1 > r0) else 0
        
        if np.all(self.slice_len == 0):
            raise ValueError("slice_len contains only zeros. Check row_nnz.")
        
        self.slice_ptr = np.zeros(num_slices + 1, dtype=np.int64)
        for s in range(num_slices):
            self.slice_ptr[s+1] = self.slice_ptr[s] + (self.slice_len[s] * C)
        
        self.total_storage = int(self.slice_ptr[-1])
        
        # Allocate device SELL arrays
        self.sell_values_gpu = cp.cuda.alloc(self.total_storage * np.dtype(np.float32).itemsize)
        cp.cuda.memset_d32(self.sell_values_gpu, 0, self.total_storage)
        
        self.sell_colinds_gpu = cp.cuda.alloc(self.total_storage * np.dtype(np.uint32).itemsize)
        cp.cuda.memset_d32(self.sell_colinds_gpu, 0, self.total_storage)
        
        # Allocate slice metadata on device
        self.slice_ptr_gpu = cp.cuda.alloc(self.slice_ptr.nbytes)
        self.slice_len_gpu = cp.cuda.alloc(self.slice_len.nbytes)
        
        cp.cuda.memcpy_htod(self.slice_ptr_gpu, self.slice_ptr)
        cp.cuda.memcpy_htod(self.slice_len_gpu, self.slice_len)
        
        # 3) Fill SELL arrays
        dense_host = np.empty((br, num_cols), dtype=np.float32)
        dense_gpu = cp.cuda.alloc(dense_host.nbytes)
        row_nnz_host_gpu = cp.cuda.alloc(br * np.dtype(np.int32).itemsize)
        
        fill_kernel = self.sparse_mod.get_function("fill_kernel__SELL")
        
        for b in trange(0, num_rows, br, desc="Fill SELL (GPU)"):
            R = min(br, num_rows - b)
            dense_host.fill(0.0)
            for i in range(R):
                rg = b + i
                n_idx = rg // self.T
                t_idx = rg % self.T
                dense_host[i, :] = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            cp.cuda.memcpy_htod(dense_gpu, dense_host)
            cp.cuda.memcpy_htod(row_nnz_host_gpu, row_nnz[b:b+R])
            
            grid = ((R + block_size - 1) // block_size, 1, 1)
            fill_kernel(
                grid=grid, block=(block_size, 1, 1),
                args=[dense_gpu, row_nnz_host_gpu, self.slice_ptr_gpu, self.slice_len_gpu,
                      self.sell_colinds_gpu, self.sell_values_gpu, np.int32(R), np.int32(num_cols),
                      np.int32(b), np.int32(C), np.float32(self.relative_threshold)]
            )
            cp.cuda.Stream.null.synchronize()
        
        dense_gpu.free()
        row_nnz_host_gpu.free()
        
        # Copy back to host for CPU access
        self.sell_values = np.empty(self.total_storage, dtype=np.float32)
        self.sell_colinds = np.empty(self.total_storage, dtype=np.uint32)
        cp.cuda.memcpy_dtoh(self.sell_values, self.sell_values_gpu)
        cp.cuda.memcpy_dtoh(self.sell_colinds, self.sell_colinds_gpu)
        
        # Compute normalization factor
        self.compute_norm_factor()
        
        print(f"SELL-C-sigma matrix allocated on GPU: {num_rows} rows, {num_cols} cols, "
              f"{self.total_storage} storage, {self.compute_density():.4f} density")

    def _allocate_cpu(self):
        """Allocate and fill the SELL matrix on CPU."""
        num_rows = int(self.N * self.T)
        num_cols = int(self.Z * self.X)
        C = int(self.slice_height)
        
        # 1) Count NNZ per row
        row_nnz = np.zeros(num_rows, dtype=np.int32)
        
        for global_row in trange(num_rows, desc="Count NNZ per row (CPU)"):
            n_idx = global_row // self.T
            t_idx = global_row % self.T
            row = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            row_max = np.max(np.abs(row))
            thr = row_max * self.relative_threshold
            row_nnz[global_row] = int(np.count_nonzero(np.abs(row) > thr))
        
        # 2) Compute per-slice maxlen and slice_ptr
        num_slices = (num_rows + C - 1) // C
        self.slice_len = np.zeros(num_slices, dtype=np.int32)
        
        for s in range(num_slices):
            r0 = s * C
            r1 = min(num_rows, r0 + C)
            self.slice_len[s] = int(np.max(row_nnz[r0:r1])) if (r1 > r0) else 0
        
        if np.all(self.slice_len == 0):
            raise ValueError("slice_len contains only zeros. Check row_nnz.")
        
        self.slice_ptr = np.zeros(num_slices + 1, dtype=np.int64)
        for s in range(num_slices):
            self.slice_ptr[s+1] = self.slice_ptr[s] + (self.slice_len[s] * C)
        
        self.total_storage = int(self.slice_ptr[-1])
        
        # Allocate SELL arrays
        self.sell_values = np.zeros(self.total_storage, dtype=np.float32)
        self.sell_colinds = np.zeros(self.total_storage, dtype=np.uint32)
        
        # 3) Fill SELL arrays
        for global_row in trange(num_rows, desc="Fill SELL (CPU)"):
            n_idx = global_row // self.T
            t_idx = global_row % self.T
            row = self.manip.AcousticFields[n_idx].field[t_idx].flatten()
            row_max = np.max(np.abs(row))
            thr = row_max * self.relative_threshold
            
            slice_id = global_row // C
            row_in_slice = global_row % C
            base = int(self.slice_ptr[slice_id])
            len_slice = int(self.slice_len[slice_id])
            
            # Fill this row's entries
            k = 0
            for col in range(num_cols):
                if np.abs(row[col]) > thr:
                    pos = base + row_in_slice + k * C
                    if pos < self.total_storage:
                        self.sell_values[pos] = row[col]
                        self.sell_colinds[pos] = col
                    k += 1
            
            # Zero pad remaining
            for k_pad in range(k, len_slice):
                pos = base + row_in_slice + k_pad * C
                if pos < self.total_storage:
                    self.sell_values[pos] = 0.0
                    self.sell_colinds[pos] = 0
        
        # Compute normalization factor
        self.compute_norm_factor()
        
        print(f"SELL-C-sigma matrix allocated on CPU: {num_rows} rows, {num_cols} cols, "
              f"{self.total_storage} storage, {self.compute_density():.4f} density")

    def compute_norm_factor(self):
        """
        Compute the MLEM normalization factor: norm_factor_inv = 1 / (A^T * 1)
        """
        ZX = int(self.Z * self.X)
        TN = int(self.T * self.N)
        
        if self._check_gpu_available() and self.sparse_mod is not None:
            # GPU implementation
            ones_gpu = cp.cuda.alloc(TN * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(ones_gpu, 0x3f800000, TN)  # 1.0f bit pattern
            
            c_gpu = cp.cuda.alloc(ZX * np.dtype(np.float32).itemsize)
            cp.cuda.memset_d32(c_gpu, 0, ZX)
            
            bp_kernel = self.sparse_mod.get_function("backprojection_kernel__SELL")
            threads = 256
            blocks = (TN + threads - 1) // threads
            
            bp_kernel(
                grid=(blocks, 1, 1), block=(threads, 1, 1),
                args=[self.sell_values_gpu, self.sell_colinds_gpu, self.slice_ptr_gpu,
                      self.slice_len_gpu, ones_gpu, c_gpu, np.int32(TN), np.int32(self.slice_height)]
            )
            cp.cuda.Stream.null.synchronize()
            
            c_host = np.empty(ZX, dtype=np.float32)
            cp.cuda.memcpy_dtoh(c_host, c_gpu)
            ones_gpu.free()
            c_gpu.free()
        else:
            # CPU implementation
            ones = np.ones(TN, dtype=np.float32)
            c_host = self.backprojection(ones)
        
        c_host = np.maximum(c_host, 1e-6)
        self.norm_factor_inv = (1.0 / c_host).astype(np.float32)
        
        # Copy to GPU if available
        if self._check_gpu_available():
            self.norm_factor_inv_gpu = cp.cuda.alloc(self.norm_factor_inv.nbytes)
            cp.cuda.memcpy_htod(self.norm_factor_inv_gpu, self.norm_factor_inv)

    def forward_projection(self, theta: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform forward projection: q = A * theta
        
        Args:
            theta: Input vector (Z*X,)
            
        Returns:
            Projection result (N*T,)
        """
        if self._check_gpu_available() and self.sparse_mod is not None:
            # GPU implementation
            theta_gpu = cp.asarray(theta) if not isinstance(theta, cp.ndarray) else theta
            q_gpu = cp.zeros(self.N * self.T, dtype=np.float32)
            
            proj_kernel = self.sparse_mod.get_function('projection_kernel__SELL')
            threads = 256
            blocks = (self.N * self.T + threads - 1) // threads
            
            proj_kernel(
                grid=(blocks, 1), block=(threads, 1, 1),
                args=[q_gpu.data.ptr, self.sell_values_gpu, self.sell_colinds_gpu,
                      self.slice_ptr_gpu, self.slice_len_gpu, theta_gpu.data.ptr,
                      np.int32(self.N * self.T), np.int32(self.slice_height)]
            )
            cp.cuda.Stream.null.synchronize()
            return q_gpu
        else:
            # CPU implementation
            theta_cpu = np.asarray(theta) if not isinstance(theta, np.ndarray) else theta
            if isinstance(theta_cpu, cp.ndarray):
                theta_cpu = cp.asnumpy(theta_cpu)
            
            q = np.zeros(self.N * self.T, dtype=np.float32)
            
            for row in range(self.N * self.T):
                slice_id = row // self.slice_height
                row_in_slice = row % self.slice_height
                base = int(self.slice_ptr[slice_id])
                len_slice = int(self.slice_len[slice_id])
                
                acc = 0.0
                for j in range(len_slice):
                    pos = base + row_in_slice + j * self.slice_height
                    if pos < self.total_storage:
                        val = self.sell_values[pos]
                        if val != 0.0:
                            col = int(self.sell_colinds[pos])
                            acc += val * theta_cpu[col]
                q[row] = acc
            
            return q

    def backward_projection(self, e: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Perform backprojection: c = A^T * e
        
        Args:
            e: Input vector (N*T,)
            
        Returns:
            Backprojection result (Z*X,)
        """
        if self._check_gpu_available() and self.sparse_mod is not None:
            # GPU implementation
            e_gpu = cp.asarray(e) if not isinstance(e, cp.ndarray) else e
            c_gpu = cp.zeros(self.Z * self.X, dtype=np.float32)
            
            bp_kernel = self.sparse_mod.get_function('backprojection_kernel__SELL')
            threads = 256
            blocks = (self.N * self.T + threads - 1) // threads
            
            bp_kernel(
                grid=(blocks, 1), block=(threads, 1, 1),
                args=[self.sell_values_gpu, self.sell_colinds_gpu, self.slice_ptr_gpu,
                      self.slice_len_gpu, e_gpu.data.ptr, c_gpu.data.ptr,
                      np.int32(self.N * self.T), np.int32(self.slice_height)]
            )
            cp.cuda.Stream.null.synchronize()
            return c_gpu
        else:
            # CPU implementation
            e_cpu = np.asarray(e) if not isinstance(e, np.ndarray) else e
            if isinstance(e_cpu, cp.ndarray):
                e_cpu = cp.asnumpy(e_cpu)
            
            c = np.zeros(self.Z * self.X, dtype=np.float32)
            
            for row in range(self.N * self.T):
                e_val = e_cpu[row]
                if e_val == 0.0:
                    continue
                
                slice_id = row // self.slice_height
                row_in_slice = row % self.slice_height
                base = int(self.slice_ptr[slice_id])
                len_slice = int(self.slice_len[slice_id])
                
                for j in range(len_slice):
                    pos = base + row_in_slice + j * self.slice_height
                    if pos < self.total_storage:
                        val = self.sell_values[pos]
                        if val != 0.0:
                            col = int(self.sell_colinds[pos])
                            c[col] += val * e_val
            
            return c

    def apply_apodization(self, window_vector: Union[np.ndarray, 'cp.ndarray']):
        """
        Apply apodization window to the matrix values.
        
        Args:
            window_vector: Apodization window vector
        """
        if self._check_gpu_available() and self.sparse_mod is not None:
            # GPU implementation
            window_gpu = cp.asarray(window_vector) if not isinstance(window_vector, cp.ndarray) else window_vector
            
            apodize_kernel = self.sparse_mod.get_function("apply_apodisation_kernel__SELL")
            threads = 128
            blocks = (self.total_storage + threads - 1) // threads
            
            apodize_kernel(
                grid=(blocks, 1, 1), block=(threads, 1, 1),
                args=[self.sell_values_gpu, self.sell_colinds_gpu, window_gpu.data.ptr, 
                      np.int64(self.total_storage)]
            )
            cp.cuda.Stream.null.synchronize()
            
            # Update host copy
            cp.cuda.memcpy_dtoh(self.sell_values, self.sell_values_gpu)
        else:
            # CPU implementation
            window_cpu = np.asarray(window_vector) if not isinstance(window_vector, np.ndarray) else window_vector
            if isinstance(window_cpu, cp.ndarray):
                window_cpu = cp.asnumpy(window_cpu)
            
            for i in range(self.total_storage):
                col = int(self.sell_colinds[i])
                if col < len(window_cpu):
                    self.sell_values[i] *= window_cpu[col]

    def get_matrix_size(self) -> dict:
        """
        Returns the total size of the SELL-C-sigma matrix in GB.
        
        Returns:
            Dictionary with memory usage information
        """
        if self.sell_values is None:
            return {"error": "The SELL-C-sigma matrix is not yet allocated."}
        
        total_bytes = 0
        
        # Host-side arrays
        if self.slice_ptr is not None:
            total_bytes += self.slice_ptr.nbytes
        if self.slice_len is not None:
            total_bytes += self.slice_len.nbytes
        if self.sell_values is not None:
            total_bytes += self.sell_values.nbytes
        if self.sell_colinds is not None:
            total_bytes += self.sell_colinds.nbytes
        if self.norm_factor_inv is not None:
            total_bytes += self.norm_factor_inv.nbytes
        
        # GPU-side arrays
        if self.sell_values_gpu is not None:
            total_bytes += self.sell_values.nbytes
        if self.sell_colinds_gpu is not None:
            total_bytes += self.sell_colinds.nbytes
        if self.slice_ptr_gpu is not None:
            total_bytes += self.slice_ptr.nbytes
        if self.slice_len_gpu is not None:
            total_bytes += self.slice_len.nbytes
        if self.norm_factor_inv_gpu is not None:
            total_bytes += self.norm_factor_inv.nbytes
        
        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024 ** 3),
            "device": self.device
        }

    def compute_density(self) -> float:
        """
        Returns the density of the SELL-C-sigma matrix.
        
        Returns:
            Density (NNZ / total elements)
        """
        if self.slice_ptr is None:
            raise RuntimeError("The SELL-C-sigma matrix is not allocated.")
        
        num_rows = self.N * self.T
        num_cols = self.Z * self.X
        total_elements = num_rows * num_cols
        
        # Estimate NNZ (excluding padding)
        nnz_estimated = int(0.9 * self.total_storage)
        return nnz_estimated / total_elements

    def free(self):
        """Free all GPU memory allocated by the SELL matrix."""
        try:
            attrs = ["sell_values_gpu", "sell_colinds_gpu", "slice_ptr_gpu", "slice_len_gpu", "norm_factor_inv_gpu"]
            for a in attrs:
                gpu_mem = getattr(self, a, None)
                if gpu_mem is not None:
                    try:
                        gpu_mem.free()
                    except:
                        pass
                    setattr(self, a, None)
        except Exception as e:
            warnings.warn(f"Error freeing GPU memory: {e}")

    def flip_angle(self):
        """
        Permute the columns of the SELL-C-sigma matrix corresponding to opposite angles.
        Assumes N is even and angles are symmetrically organized.
        """
        if self.N % 2 != 0:
            raise ValueError("Number of angles must be even to permute opposite angles.")
        
        ZX = self.Z * self.X
        angle_block_size = ZX // self.N
        
        # Get current column indices
        if self._check_gpu_available() and self.sell_colinds_gpu is not None:
            sell_colinds_host = np.empty(self.total_storage, dtype=np.uint32)
            cp.cuda.memcpy_dtoh(sell_colinds_host, self.sell_colinds_gpu)
        else:
            sell_colinds_host = self.sell_colinds.copy()
        
        # For each pair of opposite angles
        for n in range(self.N // 2):
            n_opposite = n + self.N // 2
            block_n_start = n * angle_block_size
            block_n_end = (n + 1) * angle_block_size
            block_opposite_start = n_opposite * angle_block_size
            block_opposite_end = (n_opposite + 1) * angle_block_size
            
            mask_n = (sell_colinds_host >= block_n_start) & (sell_colinds_host < block_n_end)
            mask_opposite = (sell_colinds_host >= block_opposite_start) & (sell_colinds_host < block_opposite_end)
            
            sell_colinds_host[mask_n] = sell_colinds_host[mask_n] - block_n_start + block_opposite_start
            sell_colinds_host[mask_opposite] = sell_colinds_host[mask_opposite] - block_opposite_start + block_n_start
        
        # Update column indices
        self.sell_colinds = sell_colinds_host
        if self._check_gpu_available() and self.sell_colinds_gpu is not None:
            cp.cuda.memcpy_htod(self.sell_colinds_gpu, self.sell_colinds)
        
        # Recompute norm_factor_inv
        self.compute_norm_factor()
