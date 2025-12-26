#pragma once
#include <cuda_runtime.h>

// Bank conflict avoidance using XOR swizzling on B_smem.
// Note: This swizzles B but bank conflicts are actually in A (column access).
template <int kBlockSize>
__global__ void matmul_v3_noconflict(float* A, float* B, float* C, int N) {
  __shared__ float A_smem[kBlockSize][kBlockSize];
  __shared__ float B_smem[kBlockSize][kBlockSize];
  float sum = 0;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = block_row * kBlockSize + ty;
  int col = block_col * kBlockSize + tx;
  for (int b_i = 0; b_i < N / kBlockSize; b_i++) {
    A_smem[ty][tx] = A[row * N + (b_i * kBlockSize + tx)];
    B_smem[ty][tx ^ ty] = B[(b_i * kBlockSize + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < kBlockSize; k++) {
      sum += A_smem[ty][k] * B_smem[k][tx ^ k];
    }
    __syncthreads();
  }
  C[row * N + col] = sum;
}
