/**
 * AOT_biomaps_kernels.cu
 * 
 * Centralized CUDA kernels for AOT_biomaps library
 * All custom CUDA kernels are organized in this file with:
 * - Clear naming convention: module_operation_purpose
 * - English documentation
 * - Consistent error handling
 * - Optimized for performance
 * 
 * Modules:
 * - SPARSE: Sparse matrix operations (CSR, SELL-C-sigma)
 * - MLEM: Maximum Likelihood Expectation Maximization
 * - LS: Least Squares reconstruction
 * - TV: Total Variation regularization
 * - PDHG: Primal-Dual Hybrid Gradient
 * - LBFGS: Limited-memory BFGS
 * - UTIL: Utility operations
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

// MEMORY MANAGEMENT UTILITIES
// ============================================================================

/**
 * Function: get_device_memory_info
 * Purpose: Get available and total GPU memory
 */
extern "C" cudaError_t get_device_memory_info(size_t* free, size_t* total) {
    return cudaMemGetInfo(free, total);
}

/**
 * Function: check_cuda_error
 * Purpose: Check CUDA error and print message
 */
extern "C" bool check_cuda_error(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR [%s]: %s\n", message, cudaGetErrorString(err));
        return true;
    }
    return false;
}
=======
// ============================================================================
// DENSE MATRIX KERNELS
// ============================================================================

/**
 * Kernel: fill_dense_matrix
 * Purpose: Fill dense matrix from acoustic fields on GPU
 * Used for: DENSE matrix construction
 */
extern "C" __global__ void fill_dense_matrix_kernel(
    float* __restrict__ dense_matrix,
    const float* __restrict__ field_data,
    int T,
    int N,
    int Z,
    int X,
    int field_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * N * Z * X) return;
    
    // Calculate position in output matrix
    int t = idx / (N * Z * X);
    int remaining = idx % (N * Z * X);
    int n = remaining / (Z * X);
    int zx = remaining % (Z * X);
    int z = zx / X;
    int x = zx % X;
    
    // Calculate position in input field data
    int field_idx = n * T * Z * X + t * Z * X + z * X + x;
    
    if (field_idx < T * N * Z * X) {
        dense_matrix[idx] = field_data[field_idx];
    }
}

/**
 * Kernel: compute_norm_factor_dense
 * Purpose: Compute normalization factor for dense matrix: 1 / (sum(|A|) + eps)
 * Used for: DENSE matrix normalization
 */
extern "C" __global__ void compute_norm_factor_dense_kernel(
    const float* __restrict__ dense_matrix,
    float* __restrict__ norm_factor_inv,
    int T,
    int N,
    int Z,
    int X
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Z * X) return;
    
    // Each thread computes sum of absolute values for one column
    float sum_abs = 0.0f;
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            int pos = t * N * Z * X + n * Z * X + idx;
            sum_abs += fabsf(dense_matrix[pos]);
        }
    }
    
    // Store sum for this column
    float* sum_buffer = norm_factor_inv; // Reuse buffer for sum
    sum_buffer[idx] = sum_abs;
    
    __syncthreads();
    
    // Reduction in shared memory (simplified - actual reduction would need more work)
    // For now, we'll do this in Python after kernel execution
}

/**
 * Kernel: projection_dense
 * Purpose: Forward projection using DENSE format: q = A * theta
 */
extern "C" __global__ void projection_kernel__DENSE(
    float* __restrict__ q_out,
    const float* __restrict__ dense_matrix,
    const float* __restrict__ theta,
    int T,
    int N,
    int Z,
    int X
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= T * N) return;
    
    float sum = 0.0f;
    for (int col = 0; col < Z * X; col++) {
        int pos = row * Z * X + col;
        sum += dense_matrix[pos] * theta[col];
    }
    q_out[row] = sum;
}

/**
 * Kernel: backprojection_dense
 * Purpose: Backprojection using DENSE format: c += A^T * e
 */
extern "C" __global__ void backprojection_kernel__DENSE(
    float* __restrict__ c_out,
    const float* __restrict__ dense_matrix,
    const float* __restrict__ e,
    int T,
    int N,
    int Z,
    int X
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= Z * X) return;
    
    float sum = 0.0f;
    for (int row = 0; row < T * N; row++) {
        int pos = row * Z * X + col;
        sum += dense_matrix[pos] * e[row];
    }
    c_out[col] = sum;
}

// ============================================================================
// MEMORY MANAGEMENT UTILITIES
// ============================================================================

/**
 * Function: get_device_memory_info
 * Purpose: Get available and total GPU memory
 */
extern "C" cudaError_t get_device_memory_info(size_t* free, size_t* total) {
    return cudaMemGetInfo(free, total);
}

/**
 * Function: check_cuda_error
 * Purpose: Check CUDA error and print message
 */
extern "C" bool check_cuda_error(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR [%s]: %s\n", message, cudaGetErrorString(err));
        return true;
    }
    return false;
}============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * Kernel: fill_array_value
 * Purpose: Fill an array with a constant value
 */
extern "C" __global__ void fill_array_value(float* ptr, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr[idx] = value;
}

/**
 * Kernel: fill_array_zero
 * Purpose: Fill an array with zeros
 */
extern "C" __global__ void fill_array_zero(float* ptr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) ptr[idx] = 0.0f;
}

/**
 * Kernel: array_copy
 * Purpose: Copy elements from source to destination
 */
extern "C" __global__ void array_copy(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] = src[idx];
}

/**
 * Kernel: clamp_positive
 * Purpose: Clamp all values to be non-negative (max with 0)
 */
extern "C" __global__ void clamp_positive_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = fmaxf(data[idx], 0.0f);
}

/**
 * Kernel: vector_axpby
 * Purpose: Compute z = alpha * x + beta * y (element-wise)
 */
extern "C" __global__ void vector_axpby_kernel(
    float* __restrict__ z_out,
    const float* __restrict__ x_in,
    const float* __restrict__ y_in,
    float alpha,
    float beta,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x_val = x_in[idx];
    float y_val = y_in[idx];
    float result = alpha * x_val + beta * y_val;
    if (!isfinite(result)) result = 0.0f;
    z_out[idx] = result;
}

/**
 * Kernel: vector_minus_axpy
 * Purpose: Compute r = r - alpha * z (in-place axpy)
 */
extern "C" __global__ void vector_minus_axpy_kernel(
    float* __restrict__ r_in_out,
    const float* __restrict__ z_in,
    float alpha,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float r_val = r_in_out[idx];
    float z_val = z_in[idx];
    float result = r_val - alpha * z_val;
    if (!isfinite(result)) result = 0.0f;
    r_in_out[idx] = result;
}

/**
 * Kernel: invert_vector
 * Purpose: Compute output = 1 / input with clipping to avoid division by zero
 */
extern "C" __global__ void invert_vector_kernel(
    float* __restrict__ vec_out,
    const float* __restrict__ vec_in,
    float clip_min,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = vec_in[idx];
    if (val < clip_min) val = clip_min;
    vec_out[idx] = 1.0f / val;
}

// ============================================================================
// SPARSE MATRIX KERNELS (CSR and SELL-C-sigma)
// ============================================================================

/**
 * Kernel: count_nnz_rows
 * Purpose: Count non-zero elements per row in a dense matrix block
 * Used for: CSR and SELL-C-sigma matrix construction
 */
extern "C" __global__ void count_nnz_rows_kernel(
    const float* __restrict__ dense,
    int* __restrict__ row_nnz,
    int rows_in_block,
    int cols,
    float thr_rel
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows_in_block) return;
    
    const float* row = dense + (long long)r * cols;
    float maxv = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = fabsf(row[c]);
        if (v > maxv) maxv = v;
    }
    float cut = maxv * thr_rel;
    int cnt = 0;
    for (int c = 0; c < cols; ++c) {
        if (fabsf(row[c]) > cut) ++cnt;
    }
    row_nnz[r] = cnt;
}

/**
 * Kernel: fill_sell
 * Purpose: Fill SELL-C-sigma format from dense matrix block
 * Used for: SELL-C-sigma sparse matrix construction
 */
extern "C" __global__ void fill_kernel__SELL(
    const float* __restrict__ dense,
    const int* __restrict__ row_nnz,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    unsigned int* __restrict__ col_ind,
    float* __restrict__ values_out,
    int rows_in_block,
    int cols,
    int rows_global_offset,
    int slice_height,
    float thr_rel
) {
    int r_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_local >= rows_in_block) return;
    int r_global = rows_global_offset + r_local;
    int slice_id = r_global / slice_height;
    int row_in_slice = r_global % slice_height;
    
    const float* row = dense + (long long)r_local * cols;
    float maxv = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = fabsf(row[c]);
        if (v > maxv) maxv = v;
    }
    float cut = maxv * thr_rel;
    
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    long long out_base = base + (long long)row_in_slice;
    
    int k = 0;
    for (int c = 0; c < cols; ++c) {
        float v = row[c];
        if (fabsf(v) > cut) {
            long long pos = out_base + (long long)k * slice_height;
            values_out[pos] = v;
            col_ind[pos] = (unsigned int)c;
            ++k;
        }
    }
    for (; k < len; ++k) {
        long long pos = out_base + (long long)k * slice_height;
        values_out[pos] = 0.0f;
        col_ind[pos] = 0u;
    }
}

/**
 * Kernel: fill_csr
 * Purpose: Fill CSR format from dense matrix block
 * Used for: CSR sparse matrix construction
 */
extern "C" __global__ void fill_kernel__CSR(
    const float* __restrict__ dense_block,
    const long long* __restrict__ row_ptr,
    unsigned int* __restrict__ col_ind,
    float* __restrict__ values,
    int block_start_row,
    int current_rows,
    int num_cols,
    float relative_threshold,
    long long total_nnz
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= current_rows) return;
    int global_row = block_start_row + row;
    
    const float* row_ptr_local = dense_block + (long long)row * num_cols;
    
    float row_max = 0.f;
    for (int c = 0; c < num_cols; ++c) {
        float v = fabsf(row_ptr_local[c]);
        if (v > row_max) row_max = v;
    }
    float thr = row_max * relative_threshold;
    
    long long base = row_ptr[global_row];
    int nnz = 0;
    for (int c = 0; c < num_cols; ++c) {
        float v = row_ptr_local[c];
        if (fabsf(v) > thr) {
            long long pos = base + nnz;
            if (pos < total_nnz) {
                col_ind[pos] = (unsigned int)c;
                values[pos] = v;
            }
            nnz++;
        }
    }
}

/**
 * Kernel: accumulate_columns_atomic
 * Purpose: Accumulate column sums from CSR matrix using atomic operations
 * Used for: Matrix analysis, normalization
 */
extern "C" __global__ void accumulate_columns_atomic(
    const float* __restrict__ values,
    const unsigned int* __restrict__ col_ind,
    long long total_nnz,
    float* __restrict__ col_sum
) {
    const unsigned full_mask = 0xffffffffu;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    int lane = threadIdx.x & 31;
    
    for (long long idx = gid; idx < total_nnz; idx += stride) {
        unsigned int col = col_ind[idx];
        float v = values[idx];
        if (v == 0.0f) continue;
        
        float sum = v;
        for (int offset = 1; offset <= 16; offset <<= 1) {
            unsigned int col_down = __shfl_down_sync(full_mask, col, offset);
            float sum_down = __shfl_down_sync(full_mask, sum, offset);
            if (col_down == col) {
                sum += sum_down;
            }
        }
        
        unsigned int col_up = __shfl_up_sync(full_mask, col, 1);
        bool is_head = (lane == 0) || (col_up != col);
        
        if (is_head) {
            atomicAdd(&col_sum[col], sum);
        }
    }
}

/**
 * Kernel: apply_apodization_sell
 * Purpose: Apply apodization window to SELL matrix values
 * Used for: Acoustic field correction
 */
extern "C" __global__ void apply_apodisation_kernel__SELL(
    float* sell_values,
    const unsigned int* sell_colinds,
    const float* window_vector,
    long long num_elements
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        unsigned int pixel_index = sell_colinds[i];
        if (pixel_index < ZX) {
            sell_values[i] *= window_vector[pixel_index];
        }
    }
}

// ============================================================================
// SPARSE MATRIX-VECTOR OPERATIONS KERNELS
// ============================================================================

/**
 * Kernel: sparse_matrix_vector_product_csr
 * Purpose: Compute y = A * x for CSR sparse matrix
 */
extern "C" __global__ void sparse_matrix_vector_product_csr(
    float* y,
    const float* data,
    const int* indices,
    const int* indptr,
    const float* x,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int start = indptr[row];
        int end = indptr[row + 1];
        for (int i = start; i < end; i++) {
            sum += data[i] * x[indices[i]];
        }
        y[row] = sum;
    }
}

/**
 * Kernel: projection_sell
 * Purpose: Forward projection using SELL format: q = A * theta
 */
extern "C" __global__ void projection_kernel__SELL(
    float* __restrict__ q_out,
    const float* __restrict__ sell_values,
    const unsigned int* __restrict__ sell_colinds,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    const float* __restrict__ theta,
    int num_rows,
    int slice_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    int slice_id = row / slice_height;
    int row_in_slice = row % slice_height;
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    
    float acc = 0.0f;
    long long pos = base + (long long)row_in_slice;
    for (int j = 0; j < len; ++j) {
        float v = sell_values[pos];
        if (v != 0.0f) {
            unsigned int col = sell_colinds[pos];
            float t = __ldg(&theta[col]);
            acc += v * t;
        }
        pos += (long long)slice_height;
    }
    q_out[row] = acc;
}

/**
 * Kernel: backprojection_sell
 * Purpose: Backprojection using SELL format: c += A^T * e
 */
extern "C" __global__ void backprojection_kernel__SELL(
    const float* __restrict__ sell_values,
    const unsigned int* __restrict__ sell_colinds,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    const float* __restrict__ e_flat,
    float* __restrict__ c_flat,
    int num_rows,
    int slice_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    float e = e_flat[row];
    if (e == 0.0f) return;
    
    int slice_id = row / slice_height;
    int row_in_slice = row % slice_height;
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    
    long long pos = base + (long long)row_in_slice;
    for (int j = 0; j < len; ++j) {
        float v = sell_values[pos];
        if (v != 0.0f) {
            unsigned int col = sell_colinds[pos];
            float contrib = v * e;
            atomicAdd(&c_flat[col], contrib);
        }
        pos += (long long)slice_height;
    }
}

/**
 * Kernel: projection_csr
 * Purpose: Forward projection using CSR format: q = A * theta
 */
extern "C" __global__ void projection_kernel__CSR(
    float* __restrict__ q_flat,
    const float* __restrict__ values,
    const long long* __restrict__ row_ptr,
    const unsigned int* __restrict__ col_ind,
    const float* __restrict__ theta_flat,
    int TN
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= TN) return;
    
    long long start = row_ptr[row];
    long long end = row_ptr[row + 1];
    
    float sum = 0.f;
    for (long long i = start; i < end; ++i) {
        sum += values[i] * theta_flat[col_ind[i]];
    }
    q_flat[row] = sum;
}

/**
 * Kernel: backprojection_csr
 * Purpose: Backprojection using CSR format: c += A^T * e
 */
extern "C" __global__ void backprojection_kernel__CSR(
    float* __restrict__ c_flat,
    const float* __restrict__ values,
    const long long* __restrict__ row_ptr,
    const unsigned int* __restrict__ col_ind,
    const float* __restrict__ e_flat,
    int TN
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= TN) return;
    
    float e = e_flat[row];
    long long start = row_ptr[row];
    long long end = row_ptr[row + 1];
    
    for (long long i = start; i < end; ++i) {
        unsigned int col = col_ind[i];
        float contrib = values[i] * e;
        atomicAdd(&c_flat[col], contrib);
    }
}

// ============================================================================
// MLEM KERNELS
// ============================================================================

/**
 * Kernel: ratio_kernel
 * Purpose: Compute element-wise ratio e = y / max(q, threshold) for MLEM
 */
extern "C" __global__ void ratio_kernel(
    float* __restrict__ e_out,
    const float* __restrict__ y_in,
    const float* __restrict__ q_in,
    float threshold,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float denom = q_in[idx];
    if (!(denom > threshold)) denom = threshold;
    float r = y_in[idx] / denom;
    if (!isfinite(r)) r = 0.0f;
    e_out[idx] = r;
}

/**
 * Kernel: update_theta_kernel
 * Purpose: Update theta values in MLEM: theta *= norm_inv * c_flat
 */
extern "C" __global__ void update_theta_kernel(
    float* __restrict__ theta_flat,
    const float* __restrict__ c_flat,
    const float* __restrict__ norm_factor_inv,
    int ZX
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ZX) return;
    float v = theta_flat[idx] * (norm_factor_inv[idx] * c_flat[idx]);
    if (!isfinite(v)) v = 0.0f;
    theta_flat[idx] = fmaxf(v, 0.0f);
}

// ============================================================================
// TOTAL VARIATION (TV) KERNELS
// ============================================================================

/**
 * Kernel: gradient_2d
 * Purpose: Compute 2D gradient (forward differences) for TV regularization
 * Output: p[0:N] = gradient in x direction, p[N:2N] = gradient in z direction
 */
extern "C" __global__ void gradient_kernel(
    float* __restrict__ p_out,
    const float* __restrict__ x_in,
    int Z,
    int X,
    int ZX
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ZX) return;
    
    int z_idx = idx / X;
    int x_idx = idx % X;
    
    float v_curr = x_in[idx];
    float dx = 0.0f;
    float dz = 0.0f;
    
    if (x_idx < X - 1) dx = x_in[idx + 1] - v_curr;
    if (z_idx < Z - 1) dz = x_in[idx + X] - v_curr;
    
    p_out[idx] = dx;
    p_out[ZX + idx] = dz;
}

/**
 * Kernel: divergence_2d
 * Purpose: Compute 2D divergence (adjoint of gradient) for TV regularization
 */
extern "C" __global__ void divergence_kernel(
    float* __restrict__ div_out,
    const float* __restrict__ p_in,
    int Z,
    int X,
    int ZX
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ZX) return;
    
    int z_idx = idx / X;
    int x_idx = idx % X;
    
    float val_x = 0.0f;
    float val_z = 0.0f;
    
    if (x_idx < X - 1) val_x -= p_in[idx];
    if (x_idx > 0) val_x += p_in[idx - 1];
    if (z_idx < Z - 1) val_z -= p_in[ZX + idx];
    if (z_idx > 0 && idx >= X) val_z += p_in[ZX + idx - X];
    
    div_out[idx] = val_x + val_z;
}

/**
 * Kernel: proj_tv
 * Purpose: Project onto TV constraint set (L2 ball with radius alpha)
 */
extern "C" __global__ void proj_tv_kernel(
    float* __restrict__ p_in_out,
    float alpha,
    int ZX
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ZX) return;
    
    float px = p_in_out[idx];
    float pz = p_in_out[ZX + idx];
    float norm_p = sqrtf(px * px + pz * pz);
    
    float scale_factor = 1.0f;
    if (alpha > 1e-8f) {
        float ratio = norm_p / alpha;
        if (ratio > 1.0f) scale_factor = ratio;
    }
    
    float inv_scale = 1.0f / scale_factor;
    p_in_out[idx] *= inv_scale;
    p_in_out[ZX + idx] *= inv_scale;
}

/**
 * Kernel: laplacian_2d
 * Purpose: Compute 2D Laplacian with Neumann boundary conditions
 */
extern "C" __global__ void laplacian_kernel(
    float* out,
    const float* in,
    int Z,
    int X,
    int ZX
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= X || y >= Z) return;
    
    int idx = y * X + x;
    float val = 0.0f;
    float center = in[idx];
    
    val += (x > 0) ? in[idx - 1] : center;
    val += (x < X - 1) ? in[idx + 1] : center;
    val += (y > 0) ? in[idx - X] : center;
    val += (y < Z - 1) ? in[idx + X] : center;
    val -= 4.0f * center;
    
    out[idx] = val;
}

// ============================================================================
// PDHG KERNELS (Primal-Dual Hybrid Gradient)
// ============================================================================

/**
 * Kernel: pdhg_primal_update
 * Purpose: Update primal variable with positivity constraint
 */
extern "C" __global__ void dpdhg_primal_update_kernel(
    float* x,
    const float* delta_z,
    const float* tau,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] = fmaxf(0.0f, x[i] - tau[i] * delta_z[i]);
    }
}

/**
 * Kernel: pdhg_extrapolation
 * Purpose: Extrapolation step in PDHG
 */
extern "C" __global__ void dpdhg_extrapolation_kernel(
    float* z_bar,
    const float* x_new,
    const float* x_old,
    float theta,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z_bar[i] = x_new[i] + theta * (x_new[i] - x_old[i]);
    }
}

/**
 * Kernel: pdhg_gradient
 * Purpose: Compute gradient for PDHG TV regularization
 */
extern "C" __global__ void dpdhg_gradient_kernel(
    float* grad,
    const float* x,
    int Nz,
    int Nx,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int z = i / Nx;
    int x_id = i % Nx;
    int right = (x_id < Nx - 1) ? i + 1 : i;
    int down = (z < Nz - 1) ? i + Nx : i;
    
    grad[i] = x[right] - x[i];
    grad[i + N] = x[down] - x[i];
}

/**
 * Kernel: pdhg_divergence
 * Purpose: Compute divergence for PDHG
 */
extern "C" __global__ void dpdhg_divergence_kernel(
    float* div,
    const float* p,
    int Nz,
    int Nx,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int z = i / Nx;
    int x_id = i % Nx;
    
    float val_x = 0.0f;
    float val_z = 0.0f;
    
    if (x_id < Nx - 1) val_x -= p[i];
    if (x_id > 0) val_x += p[i - 1];
    if (z < Nz - 1) val_z -= p[N + i];
    if (z > 0 && i >= Nx) val_z += p[N + i - Nx];
    
    div[i] = val_x + val_z;
}

/**
 * Kernel: pdhg_prox_tv
 * Purpose: Proximal operator for TV in PDHG
 */
extern "C" __global__ void dpdhg_prox_tv_kernel(
    float* p,
    const float* grad,
    const float* sigma,
    float lambda,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float vx = p[i] + sigma[i] * grad[i];
    float vz = p[i + N] + sigma[i + N] * grad[i + N];
    
    float norm = sqrtf(vx * vx + vz * vz + 1e-12f);
    float factor = fminf(1.0f, lambda / norm);
    
    p[i] = vx * factor;
    p[i + N] = vz * factor;
}

/**
 * Kernel: pdhg_prox_data
 * Purpose: Proximal operator for data fidelity (L2) in PDHG
 */
extern "C" __global__ void dpdhg_prox_data_kernel(
    float* q,
    const float* Ax,
    const float* y,
    const float* sigma,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        q[i] = (q[i] + sigma[i] * (Ax[i] - y[i])) / (1.0f + sigma[i]);
    }
}

/**
 * Kernel: pdhg_backprojection_sell
 * Purpose: Backprojection for PDHG using SELL format
 */
extern "C" __global__ void dpdhg_backprojection_kernel(
    const float* __restrict__ sell_values,
    const unsigned int* __restrict__ sell_colinds,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    const float* __restrict__ q,
    float* __restrict__ delta_z,
    int TN,
    int slice_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= TN) return;
    
    float q_val = q[row];
    if (q_val == 0.0f) return;
    
    int slice_id = row / slice_height;
    int row_in_slice = row % slice_height;
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    
    long long pos = base + (long long)row_in_slice;
    for (int j = 0; j < len; ++j) {
        float v = sell_values[pos];
        if (v != 0.0f) {
            unsigned int col = sell_colinds[pos];
            atomicAdd(&delta_z[col], v * q_val);
        }
        pos += (long long)slice_height;
    }
}

/**
 * Kernel: pdhg_sell_sums
 * Purpose: Compute row and column sums for PDHG preconditioning
 */
extern "C" __global__ void dpdhg_sell_sums_kernel(
    float* s_row,
    float* s_col,
    const float* sell_values,
    const unsigned int* sell_colinds,
    const long long* slice_ptr,
    const int* slice_len,
    int num_rows,
    int slice_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    int slice_idx = row / slice_height;
    int row_in_slice = row % slice_height;
    long long offset = slice_ptr[slice_idx];
    int length = slice_len[slice_idx];
    
    float sum_r = 0.0f;
    for (int k = 0; k < length; ++k) {
        long long data_idx = offset + (long long)k * slice_height + row_in_slice;
        float val = fabsf(sell_values[data_idx]);
        unsigned int col = sell_colinds[data_idx];
        
        if (val != 0.0f) {
            sum_r += val;
            atomicAdd(&s_col[col], val);
        }
    }
    s_row[row] = sum_r;
}

// ============================================================================
// LBFGS KERNELS
// ============================================================================

/**
 * Kernel: lbfgs_calc_q
 * Purpose: Compute residual q = Ax - y for LBFGS
 */
extern "C" __global__ void lbfgs_calc_q_kernel(
    float* q,
    const float* Ax,
    const float* y,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) q[i] = Ax[i] - y[i];
}

/**
 * Kernel: lbfgs_backprojection_sell
 * Purpose: Backprojection for LBFGS using SELL format
 */
extern "C" __global__ void lbfgs_backprojection_kernel(
    const float* __restrict__ sell_values,
    const unsigned int* __restrict__ sell_colinds,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    const float* __restrict__ q,
    float* __restrict__ grad_data,
    int TN,
    int slice_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= TN) return;
    
    float q_val = q[row];
    if (q_val == 0.0f) return;
    
    int slice_id = row / slice_height;
    int row_in_slice = row % slice_height;
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    
    long long pos = base + (long long)row_in_slice;
    for (int j = 0; j < len; ++j) {
        float v = sell_values[pos];
        if (v != 0.0f) {
            unsigned int col = sell_colinds[pos];
            atomicAdd(&grad_data[col], v * q_val);
        }
        pos += (long long)slice_height;
    }
}

/**
 * Kernel: lbfgs_aniso_tv_eval
 * Purpose: Evaluate anisotropic TV for LBFGS
 */
extern "C" __global__ void lbfgs_aniso_tv_eval_kernel(
    float* p,
    float* cost_val,
    const float* x,
    float alpha_x,
    float alpha_z,
    float eps,
    int Nz,
    int Nx,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int z = i / Nx;
    int x_id = i % Nx;
    
    float val = x[i];
    float dx = (x_id < Nx - 1) ? (x[i + 1] - val) : 0.0f;
    float dz = (z < Nz - 1) ? (x[i + Nx] - val) : 0.0f;
    
    float norm_x = sqrtf(dx * dx + eps * eps);
    float norm_z = sqrtf(dz * dz + eps * eps);
    
    p[i] = alpha_x * (dx / norm_x);
    p[i + N] = alpha_z * (dz / norm_z);
    
    cost_val[i] = alpha_x * norm_x + alpha_z * norm_z;
}

/**
 * Kernel: lbfgs_divergence
 * Purpose: Compute divergence for LBFGS TV
 */
extern "C" __global__ void lbfgs_divergence_kernel(
    float* grad_reg,
    const float* p,
    int Nz,
    int Nx,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int z = i / Nx;
    int x_id = i % Nx;
    
    float val_x = 0.0f;
    float val_z = 0.0f;
    
    if (x_id < Nx - 1) val_x -= p[i];
    if (x_id > 0) val_x += p[i - 1];
    if (z < Nz - 1) val_z -= p[N + i];
    if (z > 0 && i >= Nx) val_z += p[N + i - Nx];
    
    grad_reg[i] = val_x + val_z;
}

// ============================================================================
// PRECONDITIONING KERNELS (Pock & Chambolle)
// ============================================================================

/**
 * Kernel: update_dual_data_precond
 * Purpose: Update dual variable with vector preconditioning for data term
 */
extern "C" __global__ void update_dual_data_precond_kernel(
    float* __restrict__ q_out,
    const float* __restrict__ Ax,
    const float* __restrict__ y,
    const float* __restrict__ sigma_vec,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float sig = sigma_vec[idx];
    float ax_val = Ax[idx];
    float y_val = y[idx];
    float q_val = q_out[idx];
    
    float num = q_val + sig * (ax_val - y_val);
    float denom = 1.0f + sig;
    
    q_out[idx] = num / denom;
}

/**
 * Kernel: update_primal_precond
 * Purpose: Update primal variable with vector preconditioning
 */
extern "C" __global__ void update_primal_precond_kernel(
    float* __restrict__ x_out,
    const float* __restrict__ gradient_combined,
    const float* __restrict__ tau_vec,
    int ZX
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ZX) return;
    
    float t = tau_vec[idx];
    float grad = gradient_combined[idx];
    float x_val = x_out[idx];
    
    x_out[idx] = x_val - t * grad;
}

// ============================================================================
// SUBSET OPERATIONS KERNELS (for Stochastic Methods)
// ============================================================================

/**
 * Kernel: proj_tv_inplace_and_diff
 * Purpose: TV projection with in-place difference computation
 */
extern "C" __global__ void proj_tv_inplace_and_diff_kernel(
    float* p,
    float* grad,
    const float* sigma,
    float lambda,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float p_old_x = p[i];
    float p_old_z = p[i + N];
    
    float val_x = p_old_x + sigma[i] * grad[i];
    float val_z = p_old_z + sigma[i + N] * grad[i + N];
    
    float norm = sqrtf(val_x * val_x + val_z * val_z + 1e-12f);
    float factor = fminf(1.0f, lambda / norm);
    
    float p_new_x = val_x * factor;
    float p_new_z = val_z * factor;
    
    p[i] = p_new_x;
    p[i + N] = p_new_z;
    
    grad[i] = p_new_x - p_old_x;
    grad[i + N] = p_new_z - p_old_z;
}

/**
 * Kernel: prox_and_diff_subset
 * Purpose: Proximal operator with in-place difference for subset
 */
extern "C" __global__ void prox_and_diff_subset_kernel(
    float* q,
    float* Ax,
    const float* y,
    const float* sigma,
    int start_idx,
    int end_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = start_idx + idx;
    if (i >= end_idx) return;
    
    float q_old = q[i];
    float q_new = (q_old + sigma[i] * (Ax[i] - y[i])) / (1.0f + sigma[i]);
    
    q[i] = q_new;
    Ax[i] = q_new - q_old;
}

/**
 * Kernel: projection_subset_sell
 * Purpose: Forward projection on subset using SELL format
 */
extern "C" __global__ void projection_subset_kernel(
    float* __restrict__ q_out,
    const float* __restrict__ sell_values,
    const unsigned int* __restrict__ sell_colinds,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    const float* __restrict__ theta,
    int start_row,
    int end_row,
    int slice_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = start_row + idx;
    if (row >= end_row) return;
    
    int slice_id = row / slice_height;
    int row_in_slice = row % slice_height;
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    
    float acc = 0.0f;
    long long pos = base + (long long)row_in_slice;
    for (int j = 0; j < len; ++j) {
        float v = sell_values[pos];
        if (v != 0.0f) {
            unsigned int col = sell_colinds[pos];
            float t = __ldg(&theta[col]);
            acc += v * t;
        }
        pos += (long long)slice_height;
    }
    q_out[row] = acc;
}

/**
 * Kernel: backprojection_subset_sell
 * Purpose: Backprojection on subset using SELL format
 */
extern "C" __global__ void backprojection_subset_kernel(
    const float* __restrict__ sell_values,
    const unsigned int* __restrict__ sell_colinds,
    const long long* __restrict__ slice_ptr,
    const int* __restrict__ slice_len,
    const float* __restrict__ e_flat,
    float* __restrict__ c_flat,
    int start_row,
    int end_row,
    int slice_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = start_row + idx;
    if (row >= end_row) return;
    
    float e = e_flat[row];
    if (e == 0.0f) return;
    
    int slice_id = row / slice_height;
    int row_in_slice = row % slice_height;
    long long base = slice_ptr[slice_id];
    int len = slice_len[slice_id];
    
    long long pos = base + (long long)row_in_slice;
    for (int j = 0; j < len; ++j) {
        float v = sell_values[pos];
        if (v != 0.0f) {
            unsigned int col = sell_colinds[pos];
            float contrib = v * e;
            atomicAdd(&c_flat[col], contrib);
        }
        pos += (long long)slice_height;
    }
}

/**
 * Kernel: sell_sums
 * Purpose: Compute row and column sums for SELL matrix
 */
extern "C" __global__ void sell_sums_kernel(
    float* s_row,
    float* s_col,
    const float* sell_values,
    const unsigned int* sell_colinds,
    const long long* slice_ptr,
    const int* slice_len,
    int num_rows,
    int slice_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    int slice_idx = row / slice_height;
    int row_in_slice = row % slice_height;
    long long offset = slice_ptr[slice_idx];
    int length = slice_len[slice_idx];
    
    float sum_r = 0.0f;
    for (int k = 0; k < length; ++k) {
        long long data_idx = offset + (long long)k * slice_height + row_in_slice;
        float val = sell_values[data_idx];
        unsigned int col = sell_colinds[data_idx];
        
        if (val != 0.0f) {
            sum_r += val;
            atomicAdd(&s_col[col], val);
        }
    }
    s_row[row] = sum_r;
}

// ============================================================================
// MEMORY MANAGEMENT UTILITIES
// ============================================================================

/**
 * Function: get_device_memory_info
 * Purpose: Get available and total GPU memory
 */
extern "C" cudaError_t get_device_memory_info(size_t* free, size_t* total) {
    return cudaMemGetInfo(free, total);
}

/**
 * Function: check_cuda_error
 * Purpose: Check CUDA error and print message
 */
extern "C" bool check_cuda_error(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR [%s]: %s\n", message, cudaGetErrorString(err));
        return true;
    }
    return false;
}
