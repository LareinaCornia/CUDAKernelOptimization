#ifndef KERNELS
#define KERNELS

#include <stdio.h>

// Kernel implementations of SGEMM (Single precision GEneral Matrix Multiply)
// Performs the operation C = α*(A@B)+β*C for matrices A, B, C and scalars α, β
// where A is MxK, B is KxN, and C is MxN

// -------------------------------------------------------------------------------------
// Sequential implementations

void sgemm_sequential(int M, int K, int N, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int r = 0; r < K; r++) {
                sum += A[i * K + r] * B[r * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// -------------------------------------------------------------------------------------
// Kernel 1 implementation

__global__ void sgemm_naive(int M, int K, int N, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t c = (size_t)blockIdx.y * blockDim.y + threadIdx.y;

    if (r < static_cast<size_t>(M) && c < static_cast<size_t>(N)) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = alpha * sum + beta * C[r * N + c];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 2 implementation

__global__ void sgemm_global_coalescing(int M, int K, int N, float alpha, const float *A,
                                       const float *B, float beta, float *C) {
    size_t r = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

     if (r < static_cast<size_t>(M) && c < static_cast<size_t>(N)) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = alpha * sum + beta * C[r * N + c];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 3 implementation

template <const int SMCACHEDIM>
__global__ void sgemm_shared_mem_cache(int M, int K, int N, float alpha, const float *A,
                                       const float *B, float beta, float *C) {
    __shared__ float blockA[SMCACHEDIM * SMCACHEDIM];
    __shared__ float blockB[SMCACHEDIM * SMCACHEDIM];

    size_t r = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;

    for (int k = 0; k < K; k += SMCACHEDIM) {
        // HBM -> shared memory with divisible check and zero padding
        int A_r = r;
        int A_c = k + threadIdx.x;
        int B_r = k + threadIdx.y;
        int B_c = c;
        int localIdx = threadIdx.y * SMCACHEDIM + threadIdx.x;

        blockA[localIdx] = A_r < M && A_c < K ? A[A_r * K + A_c] : 0.0f;
        blockB[localIdx] = B_r < K && B_c < N ? B[B_r * N + B_c] : 0.0f;

        __syncthreads();

        for (int i = 0; i < SMCACHEDIM; i++) {
            sum += blockA[threadIdx.y * SMCACHEDIM + i] * blockB[i * SMCACHEDIM + threadIdx.x];
        }
        __syncthreads();
    }

    if (r < static_cast<size_t>(M) && c < static_cast<size_t>(N)) {
        C[r * N + c] = alpha * sum + beta * C[r * N + c];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 4 implementation

template <const int BM, const int BN, const int BK, const int TM>
__global__ void __launch_bounds__(BM * BN / TM, 1)
sgemm_1D_thread_tiling(int M, int N, int K, float alpha, const float *A, 
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    float regB[BK];

    float sums[TM];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        sums[i] = 0.0f;
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int thr_num = BM * BN / TM;

    #pragma unroll
    for (int base_k = 0; base_k < K; base_k += BK) {
        // block-level tiling: HBM -> shared memory
        #pragma unroll
        for (int localA_idx = tid; localA_idx < BM * BK; localA_idx += thr_num) {
            int localA_r = localA_idx / BK;
            int localA_c = localA_idx % BK;

            int globalA_r = blockIdx.y * BM + localA_r;
            int globalA_c = base_k + localA_c;
            int globalA_idx = globalA_r * K + globalA_c;

            // Divisible check with zero padding
            blockA[localA_idx] = globalA_r < M && globalA_c < K ? A[globalA_idx] : 0.0f;
        }
        #pragma unroll
        for (int localB_idx = tid; localB_idx < BK * BN; localB_idx += thr_num) {
            int localB_r = localB_idx / BN;
            int localB_c = localB_idx % BN;

            int globalB_r = base_k + localB_r;
            int globalB_c = blockIdx.x * BN + localB_c;
            int globalB_idx = globalB_r * N + globalB_c;

            // Divisible check with zero padding
            blockB[localB_idx] = globalB_r < K && globalB_c < N ? B[globalB_idx] : 0.0f;
        }
        __syncthreads();

        // thread-level 1D tiling: shared memory -> registers
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            regB[i] = blockB[i * BN + threadIdx.x];
        }

        // SGEMM calculation
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < BK; j++) {
                sums[i] += blockA[(threadIdx.y * TM + i) * BK + j] * regB[j];
            }
        }
        __syncthreads();
    }

    // write output back to HBM
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int globalC_r = blockIdx.y * BM + threadIdx.y * TM + i;
        int globalC_c = blockIdx.x * BN + threadIdx.x;
        
        // Divisible check
        if (globalC_r < M && globalC_c < N) {
            int globalC_idx = globalC_r * N + globalC_c;
            C[globalC_idx] = alpha * sums[i] + beta * C[globalC_idx];
        }
    }
}

// -------------------------------------------------------------------------------------
// Kernel 5 implementation

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(BM * BN / (TM * TN), 1)
sgemm_2D_thread_tiling(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    float regA[TM];
    float regB[TN];

    float sums[TM * TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            sums[i * TN + j] = 0.0f;
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int thr_num = BM * BN / (TM * TN);

    #pragma unroll
    for (int base_k = 0; base_k < K; base_k += BK) {
        // block-level tiling: HBM -> shared memory
        #pragma unroll
        for (int localA_idx = tid; localA_idx < BM * BK; localA_idx += thr_num) {
            int localA_r = localA_idx / BK;
            int localA_c = localA_idx % BK;

            int globalA_r = blockIdx.y * BM + localA_r;
            int globalA_c = base_k + localA_c;
            int globalA_idx = globalA_r * K + globalA_c;

            // Divisible check with zero padding
            blockA[localA_idx] = globalA_r < M && globalA_c < K ? A[globalA_idx] : 0.0f;
        }
        #pragma unroll
        for (int localB_idx = tid; localB_idx < BK * BN; localB_idx += thr_num) {
            int localB_r = localB_idx / BN;
            int localB_c = localB_idx % BN;

            int globalB_r = base_k + localB_r;
            int globalB_c = blockIdx.x * BN + localB_c;
            int globalB_idx = globalB_r * N + globalB_c;

            // Divisible check with zero padding
            blockB[localB_idx] = globalB_r < K && globalB_c < N ? B[globalB_idx] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // thread-level 2D titling: shared memory -> registers
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = blockA[(threadIdx.y * TM + i) * BK + k];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = blockB[k * BN + threadIdx.x * TN + j];
            }

            // SGEMM calculation
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    sums[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
         __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalC_r = blockIdx.y * BM + threadIdx.y * TM + i;
            int globalC_c = blockIdx.x * BN + threadIdx.x * TN + j;
            
            // Divisible check
            if (globalC_r < M && globalC_c < N) {
                int globalC_idx = globalC_r * N + globalC_c;
                C[globalC_idx] = alpha * sums[i * TN + j] + beta * C[globalC_idx];
            }
        }
    }
}

// -------------------------------------------------------------------------------------
// Kernel launcher

inline void launch_kernel(long kernel_num, int M, int K, int N, float alpha, 
                          const float *d_A, const float *d_B, float beta, float *d_C) {
    if(kernel_num == 1) {
        // Set execution configuration parameters
        const int BLOCK_SIZE = 32;
        dim3 thr_per_blk(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blk_in_grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        sgemm_naive<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 2) {
        // Set execution configuration parameters
        const int BLOCK_SIZE = 32;
        dim3 thr_per_blk(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blk_in_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        sgemm_global_coalescing<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 3) {
        const uint SMCACHEDIM = 32;

        // Set execution configuration parameters
        dim3 thr_per_blk(SMCACHEDIM, SMCACHEDIM);
        dim3 blk_in_grid((N + SMCACHEDIM - 1) / SMCACHEDIM, (M + SMCACHEDIM - 1) / SMCACHEDIM);

        sgemm_shared_mem_cache<SMCACHEDIM><<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 4) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 8;
        const uint TM = 8;

        // Set execution configuration parameters
        dim3 blk_in_grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 thr_per_blk(BN, BM / TM);

        sgemm_1D_thread_tiling<BM, BN, BK, TM><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 5) {
        const uint BM = 128;
        const uint BN = 128;
        const uint BK = 8;
        const uint TM = 8;
        const uint TN = 8;

        // Set execution configuration parameters
        dim3 blk_in_grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 thr_per_blk(BN / TN, BM / TM);

        sgemm_2D_thread_tiling<BM, BN, BK, TM, TN><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else {
        fprintf(stderr, "Error: %ld is not a valid kernel number.\n", kernel_num);
        exit(1);
    }
}

#endif // KERNELS
