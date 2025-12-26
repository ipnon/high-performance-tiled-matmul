#pragma once
#include <cuda_runtime.h>

// 1D blocktiling: each thread computes TM elements (a column of output).
template <int BM, int BN, int BK, int TM>
__global__ void matmul_v4_blocktile_1d(float* A, float* B, float* C, int N) {
  __shared__ float A_tile[BM][BK];
  __shared__ float B_tile[BK][BN];
  float acc[TM] = {0.0f};
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  constexpr int NUM_THREADS = (BM / TM) * BN;
  int tid = threadIdx.x;
  int thread_col = tid % BN;
  int thread_row = tid / BN;
  int global_row = block_row * BM + thread_row * TM;
  int global_col = block_col * BN + thread_col;
  for (int tile_k = 0; tile_k < N / BK; tile_k++) {
    for (int i = tid; i < BM * BK; i += NUM_THREADS) {
      int load_row = i / BK;
      int load_col = i % BK;
      A_tile[load_row][load_col] =
          A[(block_row * BM + load_row) * N + (tile_k * BK + load_col)];
    }
    for (int i = tid; i < BK * BN; i += NUM_THREADS) {
      int load_row = i / BN;
      int load_col = i % BN;
      B_tile[load_row][load_col] =
          B[(tile_k * BK + load_row) * N + (block_col * BN + load_col)];
    }
    __syncthreads();
    for (int k = 0; k < BK; k++) {
      float b_val = B_tile[k][thread_col];
      for (int m = 0; m < TM; m++) {
        acc[m] += A_tile[thread_row * TM + m][k] * b_val;
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; m++) {
    C[(global_row + m) * N + global_col] = acc[m];
  }
}
