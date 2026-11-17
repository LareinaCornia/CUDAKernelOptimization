#pragma once

#include <cuda_runtime.h>
#include "cuda_error.cuh"

const int MAX_K = 5;
const int TILE_SIZE = 256;

__global__ void conv1d_kernel(const float* u, const float* filter,
                              const float* bias, float* out,
                              int L, int D, int K) {
    __shared__ float sharedInput[TILE_SIZE + MAX_K - 1];
    __shared__ float sharedFilter[MAX_K];

    int tileStart = blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    if (localIndex < K) {
        sharedFilter[localIndex] = filter[blockIdx.y * K + localIndex];
    }

    for (int s = localIndex; s < TILE_SIZE + K - 1; s += TILE_SIZE) {
        int pos = tileStart - K / 2 + s;

        if (pos >= 0 && pos < L) {
            sharedInput[s] = u[(blockIdx.z * D + blockIdx.y) * L + pos];
        } else {
            sharedInput[s] = 0.0f;
        }
    }

    __syncthreads();

    if (tileStart + localIndex < L) {
        float sum = bias[blockIdx.y];

        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += sharedInput[localIndex + k] * sharedFilter[k];
        }

        out[(blockIdx.z * D + blockIdx.y) * L + tileStart + localIndex] = sum;
    }
}

inline void launch_kernel(const float* u, const float* filter,
                          const float* bias, float* out,
                          uint B, uint L, uint D, uint K) {

    dim3 blockDims(TILE_SIZE);
    dim3 gridDims((L + TILE_SIZE - 1) / TILE_SIZE, D, B);

    cudaCheckError(cudaFuncSetCacheConfig(conv1d_kernel, cudaFuncCachePreferL1));
    conv1d_kernel<<<gridDims, blockDims>>>(u, filter, bias, out, L, D, K);
    cudaCheckError(cudaGetLastError());
}
