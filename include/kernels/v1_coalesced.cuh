#pragma once
#include <cuda_runtime.h>

// GMEM coalescing: threads in a warp access adjacent memory addresses.
// Thread mapping ensures threads with consecutive IDs access consecutive columns.
template <int kBlockSize>
__global__ void matmul_v1_coalesced(float* A, float* B, float* C, int N) {
  int tid = threadIdx.x;
  int row = blockIdx.y * kBlockSize + (tid / kBlockSize);
  int col = blockIdx.x * kBlockSize + (tid % kBlockSize);
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
