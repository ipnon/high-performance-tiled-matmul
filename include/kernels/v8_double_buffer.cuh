#pragma once
#include <cuda_runtime.h>

template <int kBlockM, int kBlockK, int kThreadCount>
__device__ void load_tile_A(float* A, float smem_A[kBlockM][kBlockK + 1],
                            int bm, int k_tile, int N, int tid) {
  constexpr int kLoadsPerThread = (kBlockM * kBlockK) / (kThreadCount * 4);
  for (int i = 0; i < kLoadsPerThread; i++) {
    int float4_idx = tid + i * kThreadCount;
    int float_idx = float4_idx * 4;
    int row = float_idx / kBlockK;
    int col = float_idx % kBlockK;
    float4 tmp = reinterpret_cast<float4*>(
        &A[(bm * kBlockM + row) * N + (k_tile * kBlockK + col)])[0];
    smem_A[row][col + 0] = tmp.x;
    smem_A[row][col + 1] = tmp.y;
    smem_A[row][col + 2] = tmp.z;
    smem_A[row][col + 3] = tmp.w;
  }
}

template <int kBlockK, int kBlockN, int kThreadCount>
__device__ void load_tile_B(float* B, float smem_B[kBlockK][kBlockN + 1],
                            int bn, int k_tile, int N, int tid) {
  constexpr int kLoadsPerThread = (kBlockK * kBlockN) / (kThreadCount * 4);
  for (int i = 0; i < kLoadsPerThread; i++) {
    int float4_idx = tid + i * kThreadCount;
    int float_idx = float4_idx * 4;
    int row = float_idx / kBlockN;
    int col = float_idx % kBlockN;
    float4 tmp = reinterpret_cast<float4*>(
        &B[(k_tile * kBlockK + row) * N + (bn * kBlockN + col)])[0];
    smem_B[row][col + 0] = tmp.x;
    smem_B[row][col + 1] = tmp.y;
    smem_B[row][col + 2] = tmp.z;
    smem_B[row][col + 3] = tmp.w;
  }
}

template <int kBlockM, int kBlockN, int kBlockK, int kThreadM, int kThreadN>
__device__ void compute_tile(float smem_A[kBlockM][kBlockK + 1],
                             float smem_B[kBlockK][kBlockN + 1], float* accum,
                             int thread_idx_m, int thread_idx_n) {
  for (int k = 0; k < kBlockK; k++) {
    for (int m = 0; m < kThreadM; m++) {
      float a_val = smem_A[thread_idx_m * kThreadM + m][k];
      for (int n = 0; n < kThreadN; n++) {
        float b_val = smem_B[k][thread_idx_n * kThreadN + n];
        accum[m * kThreadN + n] += a_val * b_val;
      }
    }
  }
}

template <int kBlockM, int kBlockN, int kBlockK, int kThreadM, int kThreadN>
__global__ void matmul_v8_double_buffer(float* A, float* B, float* C, int N) {
  __shared__ float smem_A[2][kBlockM][kBlockK + 1];
  __shared__ float smem_B[2][kBlockK][kBlockN + 1];
  float accum[kThreadM * kThreadN] = {0.0f};

  int bm = blockIdx.y;
  int bn = blockIdx.x;
  int tid = threadIdx.x;
  int thread_idx_n = tid % (kBlockN / kThreadN);
  int thread_idx_m = tid / (kBlockN / kThreadN);
  int gm = bm * kBlockM + thread_idx_m * kThreadM;
  int gn = bn * kBlockN + thread_idx_n * kThreadN;

  constexpr int kThreadCount = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  int num_k_tiles = N / kBlockK;

  // Prologue is load first tile into buffer 0
  load_tile_A<kBlockM, kBlockK, kThreadCount>(A, smem_A[0], bm, 0, N, tid);
  load_tile_B<kBlockK, kBlockN, kThreadCount>(B, smem_B[0], bn, 0, N, tid);
  __syncthreads();

  // Main loop is overlap load[i+1] with compute [i]
  for (int k_tile = 0; k_tile < num_k_tiles - 1; k_tile++) {
    int curr = k_tile % 2;
    int next = (k_tile + 1) % 2;
    load_tile_A<kBlockM, kBlockK, kThreadCount>(A, smem_A[next], bm, k_tile + 1,
                                                N, tid);
    load_tile_B<kBlockK, kBlockN, kThreadCount>(B, smem_B[next], bn, k_tile + 1,
                                                N, tid);
    compute_tile<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>(
        smem_A[curr], smem_B[curr], accum, thread_idx_m, thread_idx_n);
    __syncthreads();
  }

  // Epilogue is compute final tile
  int last = (num_k_tiles - 1) % 2;
  compute_tile<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>(
      smem_A[last], smem_B[last], accum, thread_idx_m, thread_idx_n);
  for (int m = 0; m < kThreadM; m++) {
    for (int n = 0; n < kThreadN; n++) {
      C[(gm + m) * N + (gn + n)] = accum[m * kThreadN + n];
    }
  }
}
