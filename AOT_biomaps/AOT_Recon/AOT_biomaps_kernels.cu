/**
 * AOT_biomaps_kernels.cu
 * 
 * Centralized CUDA kernels for sparse matrix operations on GPU.
 * All custom CUDA kernels are organized in this file with:
 * - Clear naming convention: module_operation_purpose
 * - English documentation
 * - Consistent error handling
 * - Optimized for performance
 */

extern "C"{

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    /**
     * Function: complex_multiply
     * Purpose: Multiply two complex numbers (a + bi) * (c + di)
     * Input: a, b - first complex number (as float2)
     *        c, d - second complex number (as float2)
     * Output: Result as float2 (real, imag)
     */
    __device__ __forceinline__ float2 complex_multiply(float2 a, float2 b) {
        return make_float2(
            a.x * b.x - a.y * b.y,  // Real part: (a.x * b.x - a.y * b.y)
            a.x * b.y + a.y * b.x   // Imag part: (a.x * b.y + a.y * b.x)
        );
    }

    /**
     * Function: complex_abs
     * Purpose: Compute the absolute value (norm) of a complex number
     * Input: a - complex number (as float2)
     * Output: Norm as float
     */
    __device__ __forceinline__ float complex_abs(float2 a) {
        return sqrtf(a.x * a.x + a.y * a.y);
    }

    /**
    * Kernel: count_nnz_rows_kernel__REAL
    * Purpose: Count non-zero elements per row in a dense matrix block for real values, based on a relative threshold.
    * Used for: CSR and SELL-C-sigma matrix construction
    */
    __global__ void count_nnz_rows_kernel__REAL(
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
     * Kernel: count_nnz_rows_kernel__COMPLEX
     * Purpose: Count non-zero elements per row in a dense complex matrix block
     */
    __global__ void count_nnz_rows_kernel__COMPLEX(
        const float2* __restrict__ dense,
        int* __restrict__ row_nnz,
        int rows_in_block,
        int cols,
        float thr_rel
    ) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows_in_block) return;

        const float2* row = dense + (long long)r * cols;
        float maxv = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float norm = complex_abs(row[c]);
            if (norm > maxv) maxv = norm;
        }
        float cut = maxv * thr_rel;
        int cnt = 0;
        for (int c = 0; c < cols; ++c) {
            if (complex_abs(row[c]) > cut) ++cnt;
        }
        row_nnz[r] = cnt;
    }

    // ============================================================================
    // DENSE MATRIX KERNELS
    // ============================================================================

    /**
    * Kernel: fill_kernel__DENSE__REAL
    * Purpose: Fill dense real matrix from acoustic fields on GPU
    * Used for: Basic real DENSE matrix construction
    */
    __global__ void fill_kernel__DENSE__REAL(
        float* __restrict__ dense_matrix,
        const float* __restrict__ field_data,
        int T,
        int N,
        int Z,
        int X,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= T * Z * X) return;

        int t = idx / (Z * X);
        int zx = idx % (Z * X);
        int z = zx / X;
        int x = zx % X;

        // Position in dense_matrix: [t, n, z, x]
        int dense_idx = t * (N * Z * X) + n * (Z * X) + z * X + x;

        // Position in field_data: [t, z, x] (1D array of size T * Z * X)
        int field_idx = t * (Z * X) + z * X + x;

        if (dense_idx < T * N * Z * X && field_idx < T * Z * X) {
            dense_matrix[dense_idx] = field_data[field_idx];
        }
    }

    /**
     * Kernel: fill_kernel__DENSE__COMPLEX
     * Purpose: Fill dense complex matrix from acoustic fields on GPU
     */
    __global__ void fill_kernel__DENSE__COMPLEX(
        float2* __restrict__ dense_matrix,
        const float2* __restrict__ field_data,
        int T,
        int N,
        int Z,
        int X,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= T * Z * X) return;

        int t = idx / (Z * X);
        int zx = idx % (Z * X);
        int z = zx / X;
        int x = zx % X;

        int dense_idx = t * (N * Z * X) + n * (Z * X) + z * X + x;
        int field_idx = t * (Z * X) + z * X + x;

        if (dense_idx < T * N * Z * X && field_idx < T * Z * X) {
            dense_matrix[dense_idx] = field_data[field_idx];
        }
    }

    /**
    * Kernel: forward_projection_kernel__DENSE
    * Purpose: Forward projection using DENSE format for real values: q = A * theta
    * Layout expectation: row = n * T + t
    */
    __global__ void forward_projection_kernel__DENSE__REAL(
        float* __restrict__ q_out,
        const float* __restrict__ dense_matrix,
        const float* __restrict__ theta,
        int T,
        int N,
        int Z,
        int X
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= N * T) return;

        int n = row / T;
        int t = row % T;

        float sum = 0.0f;
        for (int z = 0; z < Z; z++) {
            for (int x = 0; x < X; x++) {
                long long pos = (((long long)t * N + n) * Z + z) * X + x;
                long long theta_idx = (long long)z * X + x;
                
                sum += dense_matrix[pos] * theta[theta_idx];
            }
        }
        q_out[row] = sum;
    }

    /**
     * Kernel: forward_projection_kernel__DENSE__COMPLEX
     * Purpose: Forward projection using DENSE format for complex values: q = A * theta
     */
    __global__ void forward_projection_kernel__DENSE__COMPLEX(
        float2* __restrict__ q_out,
        const float2* __restrict__ dense_matrix,
        const float2* __restrict__ theta,
        int T, int N, int Z, int X
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= N * T) return;

        int n = row / T;
        int t = row % T;

        float2 sum = make_float2(0.0f, 0.0f);
        for (int z = 0; z < Z; z++) {
            for (int x = 0; x < X; x++) {
                long long pos = (((long long)t * N + n) * Z + z) * X + x;
                long long theta_idx = (long long)z * X + x;
    
                float2 prod = complex_multiply(dense_matrix[pos], theta[theta_idx]);
                sum.x += prod.x;
                sum.y += prod.y;
            }
        }
        q_out[row] = sum;
    }

    /**
    * Kernel: backward_projection_kernel__DENSE__REAL
    * Purpose: Backward projection using DENSE format for real values: c = A^T * e
    * Layout expectation: col = z * X + x
    */
    __global__ void backward_projection_kernel__DENSE__REAL(
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

        int z = col / X;
        int x = col % X;

        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < T; t++) {
                long long e_idx = (long long)n * T + t;
                long long pos = (((long long)t * N + n) * Z + z) * X + x;
                
                sum += dense_matrix[pos] * e[e_idx];
            }
        }
        c_out[col] = sum;
    }

    /**
     * Kernel: backward_projection_kernel__DENSE__COMPLEX
     * Purpose: Backward projection using DENSE format for complex values: c = A^H * e
     */
    __global__ void backward_projection_kernel__DENSE__COMPLEX(
        float2* __restrict__ c_out,
        const float2* __restrict__ dense_matrix,
        const float2* __restrict__ e,
        int T, int N, int Z, int X
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= Z * X) return;

        int z = col / X;
        int x = col % X;

        float2 sum = make_float2(0.0f, 0.0f);
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < T; t++) {
                long long e_idx = (long long)n * T + t;
                long long pos = (((long long)t * N + n) * Z + z) * X + x;
                
                float2 dense_val = dense_matrix[pos];
                float2 dense_conj = make_float2(dense_val.x, -dense_val.y); 
                
                float2 prod = complex_multiply(dense_conj, e[e_idx]);
                sum.x += prod.x;
                sum.y += prod.y;
            }
        }
        c_out[col] = sum;
    }

    // ============================================================================
    // SELL MATRIX KERNELS
    // ============================================================================  

    /**
    * Kernel: fill_kernel__SELL__REAL
    * Purpose: Fill real SELL-C-sigma format from dense matrix block
    * Used for: Basic real SELL-C-sigma sparse matrix construction using block streaming
    */
    __global__ void fill_kernel__SELL__REAL(
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
        
        // Find max absolute value in the row to compute local threshold
        for (int c = 0; c < cols; ++c) {
            float v = fabsf(row[c]);
            if (v > maxv) maxv = v;
        }
        float cut = maxv * thr_rel;
        
        long long base = slice_ptr[slice_id];
        int len = slice_len[slice_id];
        long long out_base = base + (long long)row_in_slice;
        
        int k = 0;
        // Populate non-zero values above the threshold
        for (int c = 0; c < cols; ++c) {
            float v = row[c];
            if (fabsf(v) > cut) {
                long long pos = out_base + (long long)k * slice_height;
                values_out[pos] = v;
                col_ind[pos] = (unsigned int)c;
                ++k;
            }
        }
        
        // Explicitly pad the remaining elements in the slice with zeros
        for (; k < len; ++k) {
            long long pos = out_base + (long long)k * slice_height;
            values_out[pos] = 0.0f;
            col_ind[pos] = 0u;
        }
    }

    /**
     * Kernel: fill_kernel__SELL__COMPLEX
     * Purpose: Fill SELL-C-sigma format from dense complex matrix block
     */
    __global__ void fill_kernel__SELL__COMPLEX(
        const float2* __restrict__ dense,
        const int* __restrict__ row_nnz,
        const long long* __restrict__ slice_ptr,
        const int* __restrict__ slice_len,
        unsigned int* __restrict__ col_ind,
        float2* __restrict__ values_out,
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

        const float2* row = dense + (long long)r_local * cols;
        float maxv = 0.0f;

        // Find max norm in the row to compute local threshold
        for (int c = 0; c < cols; ++c) {
            float norm = complex_abs(row[c]);
            if (norm > maxv) maxv = norm;
        }
        float cut = maxv * thr_rel;

        long long base = slice_ptr[slice_id];
        int len = slice_len[slice_id];
        long long out_base = base + (long long)row_in_slice;

        int k = 0;
        for (int c = 0; c < cols; ++c) {
            float2 v = row[c];
            if (complex_abs(v) > cut) {
                long long pos = out_base + (long long)k * slice_height;
                values_out[pos] = v;
                col_ind[pos] = (unsigned int)c;
                ++k;
            }
        }

        // Pad remaining elements with zeros
        for (; k < len; ++k) {
            long long pos = out_base + (long long)k * slice_height;
            values_out[pos] = make_float2(0.0f, 0.0f);
            col_ind[pos] = 0u;
        }
    }

    /**
    * Kernel: forward_projection_kernel__SELL__REAL
    * Purpose: Forward projection using SELL format for real values: q = A * theta
    * Optimized for coalesced memory access (1 thread = 1 row)
    */
    __global__ void forward_projection_kernel__SELL__REAL(
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

        // Sequential loop for the thread ensures coalesced memory access for the warp
        for (int j = 0; j < len; ++j) {
            float v = sell_values[pos + (long long)j * slice_height];
            if (v != 0.0f) {
                unsigned int col = sell_colinds[pos + (long long)j * slice_height];
                // __ldg is highly recommended for read-only data caching
                acc += v * __ldg(&theta[col]);
            }
        }
        
        // Direct write without warp reduction
        q_out[row] = acc;
    }

    /**
     * Kernel: forward_projection_kernel__SELL__COMPLEX
     * Purpose: Forward projection using SELL format for complex values: q = A * theta
     */
    __global__ void forward_projection_kernel__SELL__COMPLEX(
        float2* __restrict__ q_out,
        const float2* __restrict__ sell_values,
        const unsigned int* __restrict__ sell_colinds,
        const long long* __restrict__ slice_ptr,
        const int* __restrict__ slice_len,
        const float2* __restrict__ theta,
        int num_rows, int slice_height
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows) return;

        int slice_id = row / slice_height;
        int row_in_slice = row % slice_height;
        long long base = slice_ptr[slice_id];
        int len = slice_len[slice_id];

        float2 acc = make_float2(0.0f, 0.0f);
        long long pos = base + (long long)row_in_slice;

        for (int j = 0; j < len; ++j) {
            float2 v = sell_values[pos + (long long)j * slice_height];
            if (v.x != 0.0f || v.y != 0.0f) {
                unsigned int col = sell_colinds[pos + (long long)j * slice_height];
                float2 theta_val = __ldg(&theta[col]);
                
                float2 prod = complex_multiply(v, theta_val);
                acc.x += prod.x;
                acc.y += prod.y;
            }
        }
        q_out[row] = acc;
    }

    /**
    * Kernel: backward_projection_kernel__SELL__REAL
    * Purpose: Backward projection using SELL format for real values: c += A^T * e
    * Optimized: 1 thread reads 1 error value and scatters it to the volume
    */
    __global__ void backward_projection_kernel__SELL__REAL(
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
        if (e == 0.0f) return; // Skip empty error contributions

        int slice_id = row / slice_height;
        int row_in_slice = row % slice_height;
        long long base = slice_ptr[slice_id];
        int len = slice_len[slice_id];
        
        long long pos = base + (long long)row_in_slice;

        for (int j = 0; j < len; ++j) {
            float v = sell_values[pos + (long long)j * slice_height];
            if (v != 0.0f) {
                unsigned int col = sell_colinds[pos + (long long)j * slice_height];
                // Scatter operation: each element contributes to its specific voxel
                atomicAdd(&c_flat[col], v * e);
            }
        }
    }

    /**
     * Kernel: backward_projection_kernel__SELL__COMPLEX
     * Purpose: Backward projection using SELL format for complex values: 
     * c += Re(A^H * e). Result is written to a REAL array.
     */
    __global__ void backward_projection_kernel__SELL__COMPLEX(
        const float2* __restrict__ sell_values,
        const unsigned int* __restrict__ sell_colinds,
        const long long* __restrict__ slice_ptr,
        const int* __restrict__ slice_len,
        const float2* __restrict__ e_flat,
        float2* __restrict__ c_flat, 
        int num_rows, int slice_height
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows) return;

        float2 e = e_flat[row];
        if (e.x == 0.0f && e.y == 0.0f) return;

        int slice_id = row / slice_height;
        int row_in_slice = row % slice_height;
        long long base = slice_ptr[slice_id];
        int len = slice_len[slice_id];
        long long pos = base + (long long)row_in_slice;

        for (int j = 0; j < len; ++j) {
            float2 v = sell_values[pos + (long long)j * slice_height];
            if (v.x != 0.0f || v.y != 0.0f) {
                unsigned int col = sell_colinds[pos + (long long)j * slice_height];
                float2 v_conj = make_float2(v.x, -v.y);  
                float2 prod = complex_multiply(v_conj, e); 
                atomicAdd(&c_flat[col].x, prod.x);  
                atomicAdd(&c_flat[col].y, prod.y); 
            }
        }
    }

    /**
    * Kernel: apply_apodization_kernel__SELL__REAL
    * Purpose: Apply apodization window to SELL matrix real values
    * Used for: Acoustic field correction
    */
    __global__ void apply_apodization_kernel__SELL__REAL(
        float* sell_values,
        const unsigned int* sell_colinds,
        const float* window_vector,
        long long num_elements,
        unsigned int ZX
    ) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_elements) {
            unsigned int pixel_index = sell_colinds[i];
            if (pixel_index < ZX) {
                sell_values[i] *= window_vector[pixel_index];
            }
        }
    }

    /**
     * Kernel: apply_apodization_kernel__SELL__COMPLEX
     * Purpose: Apply apodization window to SELL matrix values (complex)
     */
    __global__ void apply_apodization_kernel__SELL__COMPLEX(
        float2* sell_values,
        const unsigned int* sell_colinds,
        const float* window_vector,
        long long num_elements,
        unsigned int ZX
    ) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_elements) {
            unsigned int pixel_index = sell_colinds[i];
            if (pixel_index < ZX) {
                float window_val = window_vector[pixel_index];
                sell_values[i].x *= window_val;
                sell_values[i].y *= window_val;
            }
        }
    }

    // ============================================================================
    // CSR MATRIX KERNELS
    // ============================================================================     

    /**
    * Kernel: fill_kernel__CSR__REAL
    * Purpose: Fill local CSR arrays from a dense matrix block (1-Pass Algorithm)
    */
    __global__ void fill_kernel__CSR__REAL(
        const float* __restrict__ dense_block,
        const long long* __restrict__ local_row_ptr,
        unsigned int* __restrict__ col_ind,
        float* __restrict__ values,
        int current_rows,
        int num_cols,
        float relative_threshold,
        long long local_total_nnz
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= current_rows) return;
        
        const float* row_dense = dense_block + (long long)row * num_cols;
        
        float row_max = 0.f;
        for (int c = 0; c < num_cols; ++c) {
            float v = fabsf(row_dense[c]);
            if (v > row_max) row_max = v;
        }
        float thr = row_max * relative_threshold;
        
        long long base = local_row_ptr[row];
        int nnz = 0;
        for (int c = 0; c < num_cols; ++c) {
            float v = row_dense[c];
            if (fabsf(v) > thr) {
                long long pos = base + nnz;
                if (pos < local_total_nnz) {
                    col_ind[pos] = (unsigned int)c;
                    values[pos] = v;
                }
                nnz++;
            }
        }
    }

    /**
     * Kernel: fill_kernel__CSR__COMPLEX
     * Purpose: Fill local CSR arrays from a dense complex matrix block
     */
    __global__ void fill_kernel__CSR__COMPLEX(
        const float2* __restrict__ dense_block,
        const long long* __restrict__ local_row_ptr,
        unsigned int* __restrict__ col_ind,
        float2* __restrict__ values,
        int current_rows,
        int num_cols,
        float relative_threshold,
        long long local_total_nnz
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= current_rows) return;

        const float2* row_dense = dense_block + (long long)row * num_cols;

        float row_max = 0.f;
        for (int c = 0; c < num_cols; ++c) {
            float norm = complex_abs(row_dense[c]);
            if (norm > row_max) row_max = norm;
        }
        float thr = row_max * relative_threshold;

        long long base = local_row_ptr[row];
        int nnz = 0;
        for (int c = 0; c < num_cols; ++c) {
            float2 v = row_dense[c];
            if (complex_abs(v) > thr) {
                long long pos = base + nnz;
                if (pos < local_total_nnz) {
                    col_ind[pos] = (unsigned int)c;
                    values[pos] = v;
                }
                nnz++;
            }
        }
    }

    /**
    * Kernel: forward_projection_kernel__CSR__REAL
    * Purpose: Forward projection using CSR format for real values: q = A * theta
    */
    __global__ void forward_projection_kernel__CSR__REAL(
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
     * Kernel: forward_projection_kernel__CSR__COMPLEX
     * Purpose: Forward projection using CSR format for complex values: q = A * theta
     */
    __global__ void forward_projection_kernel__CSR__COMPLEX(
        float2* __restrict__ q_flat,
        const float2* __restrict__ values,
        const long long* __restrict__ row_ptr,
        const unsigned int* __restrict__ col_ind,
        const float2* __restrict__ theta_flat,
        int TN
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= TN) return;

        long long start = row_ptr[row];
        long long end = row_ptr[row + 1];

        float2 sum = make_float2(0.0f, 0.0f);
        for (long long i = start; i < end; ++i) {
            float2 v = values[i];
            float2 theta_val = theta_flat[col_ind[i]];
            
            float2 prod = complex_multiply(v, theta_val);
            sum.x += prod.x;
            sum.y += prod.y;
        }
        q_flat[row] = sum;
    }

    /**
    * Kernel: backward_projection_kernel__CSR__REAL
    * Purpose: backward projection using CSR format for real values: c += A^T * e
    */
    __global__ void backward_projection_kernel__CSR__REAL(
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

    /**
     * Kernel: backward_projection_kernel__CSR__COMPLEX
     * Purpose: Backward projection using CSR format for complex values: c += A^H * e
     */
    __global__ void backward_projection_kernel__CSR__COMPLEX(
        float2* __restrict__ c_flat,
        const float2* __restrict__ values,
        const long long* __restrict__ row_ptr,
        const unsigned int* __restrict__ col_ind,
        const float2* __restrict__ e_flat,
        int TN
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= TN) return;

        float2 e = e_flat[row];
        long long start = row_ptr[row];
        long long end = row_ptr[row + 1];

        for (long long i = start; i < end; ++i) {
            unsigned int col = col_ind[i];
            float2 v = values[i];
            
            float2 v_conj = make_float2(v.x, -v.y);
            float2 prod = complex_multiply(v_conj, e);
            
            atomicAdd(&c_flat[col].x, prod.x);
            atomicAdd(&c_flat[col].y, prod.y);
        }
    }

    /**
    * Kernel: accumulate_columns_atomic__REAL
    * Purpose: Accumulate column sums from CSR matrix using atomic operations for real values.
    * Optimized: Uses warp-level reduction with __shfl_down_sync for efficiency.
    */
    __global__ void accumulate_columns_atomic__REAL(
        const float* __restrict__ values,
        const unsigned int* __restrict__ col_ind,
        long long total_nnz,
        float* __restrict__ col_sum
    ) {
        const unsigned full_mask = 0xffffffffu;
        long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long stride = (long long)blockDim.x * gridDim.x;
        int lane = threadIdx.x & 31;  // Lane ID within the warp (0-31)

        for (long long idx = gid; idx < total_nnz; idx += stride) {
            unsigned int col = col_ind[idx];
            float v = values[idx];
            if (v == 0.0f) continue;

            // Warp-level reduction for the same column
            float sum = v;
            for (int offset = 1; offset <= 16; offset <<= 1) {
                unsigned int col_down = __shfl_down_sync(full_mask, col, offset);
                float v_down = __shfl_down_sync(full_mask, v, offset);
                // Only add if the column is the same
                if (col_down == col) {
                    sum += v_down;
                }
            }

            // Check if this thread is the "head" of the column group
            unsigned int col_up = __shfl_up_sync(full_mask, col, 1);
            bool is_head = (lane == 0) || (col != col_up);

            if (is_head) {
                atomicAdd(&col_sum[col], sum);
            }
        }
    }

    /**
    * Kernel: accumulate_columns_atomic__COMPLEX
    * Purpose: Accumulate column sums from CSR matrix using atomic operations for complex values.
    * Optimized: Uses warp-level reduction with separate real/imaginary parts.
    */
    __global__ void accumulate_columns_atomic__COMPLEX(
        const float2* __restrict__ values,
        const unsigned int* __restrict__ col_ind,
        long long total_nnz,
        float2* __restrict__ col_sum
    ) {
        const unsigned full_mask = 0xffffffffu;
        long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long stride = (long long)blockDim.x * gridDim.x;
        int lane = threadIdx.x & 31;  // Lane ID within the warp (0-31)

        for (long long idx = gid; idx < total_nnz; idx += stride) {
            unsigned int col = col_ind[idx];
            float2 v = values[idx];
            // Skip if both real and imaginary parts are zero
            if (v.x == 0.0f && v.y == 0.0f) continue;

            // Separate real and imaginary parts for warp reduction
            float sum_real = v.x;
            float sum_imag = v.y;

            for (int offset = 1; offset <= 16; offset <<= 1) {
                unsigned int col_down = __shfl_down_sync(full_mask, col, offset);
                float v_real_down = __shfl_down_sync(full_mask, sum_real, offset);
                float v_imag_down = __shfl_down_sync(full_mask, sum_imag, offset);
                if (col_down == col) {
                    sum_real += v_real_down;
                    sum_imag += v_imag_down;
                }
            }

            // Check if this thread is the "head" of the column group
            unsigned int col_up = __shfl_up_sync(full_mask, col, 1);
            bool is_head = (lane == 0) || (col != col_up);

            if (is_head) {
                atomicAdd(&col_sum[col].x, sum_real);
                atomicAdd(&col_sum[col].y, sum_imag);
            }
        }
    }

    /**
    * Kernel: accumulate_abs_columns_atomic__REAL
    *
    * Purpose:
    * Compute the column-wise absolute sums
    *
    *      c_j = sum_i |A_ij|
    *
    * from the sparse matrix coefficients.
    *
    * This quantity is required for the diagonal PDHG preconditioner of Ehrhardt et al. (2019, Theorem 2).
    *
    * Notes:
    * - Operates directly on the sparse coefficient arrays (values, col_ind), independently of the sparse storage format (CSR, SELL, ...).
    * - Each thread processes one non-zero coefficient.
    * - Atomic additions ensure correct accumulation into the column sums.
    */
    __global__ void accumulate_abs_columns_atomic__REAL(
        const float* __restrict__ values,
        const unsigned int* __restrict__ col_ind,
        long long total_nnz,
        float* __restrict__ col_sum
    ) {
            long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= total_nnz) return;

            float v = fabsf(values[idx]);

            if(v==0.0f) return;

            atomicAdd(&col_sum[col_ind[idx]], v);
    }

    /**
    * Kernel: accumulate_abs_columns_atomic__COMPLEX
    *
    * Purpose:
    * Compute the column-wise sums of coefficient magnitudes
    *
    *      c_j = sum_i |A_ij|
    *
    * where |A_ij| denotes the complex modulus.
    *
    * This kernel is used to construct the diagonal primal step sizes of the
    * Ehrhardt PDHG preconditioner.
    */
    __global__ void accumulate_abs_rows__SELL__COMPLEX(
        const float2* __restrict__ sell_values,
        const long long* __restrict__ slice_ptr,
        const int* __restrict__ slice_len,
        float* __restrict__ row_sum,
        int num_rows,
        int slice_height
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row>=num_rows) return;

        int slice=row/slice_height;
        int row_in_slice=row%slice_height;

        long long base=slice_ptr[slice];
        int len=slice_len[slice];

        float s=0.f;

        long long pos=base+row_in_slice;

        for(int j=0;j<len;j++)
        {
            float2 v=sell_values[pos+(long long)j*slice_height];
            s+=hypotf(v.x,v.y);
        }

        row_sum[row]=s;
    }

    /**
    * Kernel: accumulate_abs_columns_atomic__REAL
    *
    * Purpose:
    * Compute the column-wise absolute sums
    *
    *      c_j = sum_i |A_ij|
    *
    * from the sparse matrix coefficients.
    *
    * This quantity is required to build the diagonal primal preconditioner of Ehrhardt et al. (2019, Theorem 2).
    *
    * Notes:
    * - One thread processes one non-zero coefficient.
    * - The matrix storage format is irrelevant since only the value array and column indices are accessed.
    * - Atomic additions guarantee correct accumulation when multiple coefficients contribute to the same column.
    */
    __global__
    void accumulate_abs_rows__SELL__REAL(
        const float* __restrict__ sell_values,
        const long long* __restrict__ slice_ptr,
        const int* __restrict__ slice_len,
        float* __restrict__ row_sum,
        int num_rows,
        int slice_height)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= num_rows)
            return;

        int slice = row / slice_height;
        int row_in_slice = row % slice_height;

        long long base = slice_ptr[slice];
        int len = slice_len[slice];

        float s = 0.f;

        long long pos = base + row_in_slice;

        for(int j=0;j<len;j++)
            s += fabsf(sell_values[pos + (long long)j*slice_height]);

        row_sum[row] = s;
    }

    /**
    * Kernel: accumulate_abs_columns_atomic__COMPLEX
    *
    * Purpose:
    * Compute the column-wise sums of coefficient magnitudes
    *
    *      c_j = sum_i |A_ij|
    *
    * where |A_ij| denotes the complex modulus.
    *
    * This quantity is required to build the diagonal primal preconditioner of Ehrhardt et al. (2019, Theorem 2).
    *
    * Notes:
    * - One thread processes one complex non-zero coefficient.
    * - The complex modulus is computed as hypotf(real, imag).
    * - Atomic additions guarantee correct accumulation into the column sums.
    */
    __global__
    void accumulate_abs_columns_atomic__COMPLEX(
        const float2* __restrict__ values,
        const unsigned int* __restrict__ col_ind,
        long long total_nnz,
        float* __restrict__ col_sum)
    {
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >= total_nnz)
            return;

        float v = hypotf(values[idx].x, values[idx].y);

        if(v == 0.f)
            return;

        atomicAdd(&col_sum[col_ind[idx]], v);
    }


}