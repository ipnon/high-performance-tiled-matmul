#pragma once
#include <cuda_runtime.h>

template <int kBlockSize>
__global__ void matmul_v0_naive(float* A, float* B, float* C, int N) {
  int row = blockIdx.y * kBlockSize + threadIdx.y;
  int col = blockIdx.x * kBlockSize + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
