#pragma once
#include <cuda_runtime.h>

// Shared memory tiling: load tiles of A and B into SMEM, compute partial sums.
template <int kBlockSize>
__global__ void matmul_v2_tiled(float* A, float* B, float* C, int N) {
  __shared__ float As[kBlockSize][kBlockSize];
  __shared__ float Bs[kBlockSize][kBlockSize];
  float sum = 0;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = block_row * kBlockSize + ty;
  int col = block_col * kBlockSize + tx;
  for (int b = 0; b < N / kBlockSize; b++) {
    As[ty][tx] = A[row * N + (b * kBlockSize + tx)];
    Bs[ty][tx] = B[(b * kBlockSize + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < kBlockSize; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  C[row * N + col] = sum;
}
