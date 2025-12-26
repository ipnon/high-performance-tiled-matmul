#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================================
// Kernel implementations (copied inline to avoid header complexity)
// ============================================================================

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
    for (int k = 0; k < kBlockSize; k++) sum += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = sum;
}

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

template <int kBlockM, int kBlockN, int kBlockK, int kThreadM, int kThreadN>
__global__ void matmul_v5_blocktile_2d(float* A, float* B, float* C, int N) {
  __shared__ float smem_A[kBlockM][kBlockK];
  __shared__ float smem_B[kBlockK][kBlockN];
  float accum[kThreadM * kThreadN] = {0.0f};
  constexpr int kThreadCount = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  int bm = blockIdx.y;
  int bn = blockIdx.x;
  int tid = threadIdx.x;
  int thread_idx_n = tid % (kBlockN / kThreadN);
  int thread_idx_m = tid / (kBlockN / kThreadN);
  int gm = bm * kBlockM + thread_idx_m * kThreadM;
  int gn = bn * kBlockN + thread_idx_n * kThreadN;
  for (int k_tile = 0; k_tile < N / kBlockK; k_tile++) {
    for (int i = tid; i < kBlockM * kBlockK; i += kThreadCount) {
      int load_row = i / kBlockK;
      int load_col = i % kBlockK;
      smem_A[load_row][load_col] =
          A[(bm * kBlockM + load_row) * N + (k_tile * kBlockK + load_col)];
    }
    for (int i = tid; i < kBlockK * kBlockN; i += kThreadCount) {
      int load_row = i / kBlockN;
      int load_col = i % kBlockN;
      smem_B[load_row][load_col] =
          B[(k_tile * kBlockK + load_row) * N + (bn * kBlockN + load_col)];
    }
    __syncthreads();
    for (int k = 0; k < kBlockK; k++) {
      for (int m = 0; m < kThreadM; m++) {
        float a_val = smem_A[thread_idx_m * kThreadM + m][k];
        for (int n = 0; n < kThreadN; n++) {
          float b_val = smem_B[k][thread_idx_n * kThreadN + n];
          accum[m * kThreadN + n] += a_val * b_val;
        }
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < kThreadM; m++) {
    for (int n = 0; n < kThreadN; n++) {
      C[(gm + m) * N + (gn + n)] = accum[m * kThreadN + n];
    }
  }
}

template <int kBlockM, int kBlockN, int kBlockK, int kThreadM, int kThreadN>
__global__ void matmul_v6_vectorized(float* A, float* B, float* C, int N) {
  __shared__ float smem_A[kBlockM][kBlockK + 1];
  __shared__ float smem_B[kBlockK][kBlockN + 1];
  float accum[kThreadM * kThreadN] = {0.0f};
  int bm = blockIdx.y;
  int bn = blockIdx.x;
  int tid = threadIdx.x;
  int thread_idx_n = tid % (kBlockN / kThreadN);
  int thread_idx_m = tid / (kBlockN / kThreadN);
  int gm = bm * kBlockM + thread_idx_m * kThreadM;
  int gn = bn * kBlockN + thread_idx_n * kThreadN;
  constexpr int kThreadCount = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  for (int k_tile = 0; k_tile < N / kBlockK; k_tile++) {
    constexpr int kALoadsPerThread = (kBlockM * kBlockK) / (kThreadCount * 4);
    for (int i = 0; i < kALoadsPerThread; i++) {
      int float4_idx = tid + i * kThreadCount;
      int float_idx = float4_idx * 4;
      int load_row = float_idx / kBlockK;
      int load_col = float_idx % kBlockK;
      float4 tmp = reinterpret_cast<float4*>(
          &A[(bm * kBlockM + load_row) * N + (k_tile * kBlockK + load_col)])[0];
      smem_A[load_row][load_col + 0] = tmp.x;
      smem_A[load_row][load_col + 1] = tmp.y;
      smem_A[load_row][load_col + 2] = tmp.z;
      smem_A[load_row][load_col + 3] = tmp.w;
    }
    constexpr int kBLoadsPerThread = (kBlockK * kBlockN) / (kThreadCount * 4);
    for (int i = 0; i < kBLoadsPerThread; i++) {
      int float4_idx = tid + i * kThreadCount;
      int float_idx = float4_idx * 4;
      int load_row = float_idx / kBlockN;
      int load_col = float_idx % kBlockN;
      float4 tmp = reinterpret_cast<float4*>(
          &B[(k_tile * kBlockK + load_row) * N + (bn * kBlockN + load_col)])[0];
      smem_B[load_row][load_col + 0] = tmp.x;
      smem_B[load_row][load_col + 1] = tmp.y;
      smem_B[load_row][load_col + 2] = tmp.z;
      smem_B[load_row][load_col + 3] = tmp.w;
    }
    __syncthreads();
    for (int k = 0; k < kBlockK; k++) {
      for (int m = 0; m < kThreadM; m++) {
        float a_val = smem_A[thread_idx_m * kThreadM + m][k];
        for (int n = 0; n < kThreadN; n++) {
          float b_val = smem_B[k][thread_idx_n * kThreadN + n];
          accum[m * kThreadN + n] += a_val * b_val;
        }
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < kThreadM; m++) {
    for (int n = 0; n < kThreadN; n++) {
      C[(gm + m) * N + (gn + n)] = accum[m * kThreadN + n];
    }
  }
}

// ============================================================================
// Benchmarking infrastructure
// ============================================================================

float benchmark_cublas(float* A, float* B, float* C, int n,
                       int warmup = 3, int iters = 10) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  for (int i = 0; i < warmup; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n,
                &beta, C, n);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n,
                &beta, C, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * n * n * n * iters / (ms * 1e6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  return gflops;
}

template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, float* A, float* B, float* C,
                       int n, dim3 threads, dim3 blocks,
                       int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; i++) {
    kernel<<<blocks, threads>>>(A, B, C, n);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    kernel<<<blocks, threads>>>(A, B, C, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * n * n * n * iters / (ms * 1e6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return gflops;
}

// ============================================================================
// Main benchmark suite
// ============================================================================

int main() {
  std::vector<int> sizes = {512, 1024, 2048, 4096};

  printf("CUDA SGEMM Benchmark Suite\n");
  printf("==========================\n\n");

  // Get GPU info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  printf("Theoretical FP32 Peak: %.1f TFLOPS\n\n",
         prop.clockRate * 1e-6 * prop.multiProcessorCount * 128 * 2 / 1000.0);

  printf("%-6s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-10s\n",
         "N", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "cuBLAS");
  printf("-------|----------|----------|----------|----------|----------|----------|----------|------------\n");

  for (int n : sizes) {
    const int n_squared = n * n;
    const int bytes = n_squared * sizeof(float);

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, A_d, n_squared);
    curandGenerateUniform(gen, B_d, n_squared);
    curandDestroyGenerator(gen);

    // cuBLAS baseline
    float cublas = benchmark_cublas(A_d, B_d, C_d, n);

    // V0: Naive
    constexpr int BS = 16;
    dim3 threads0(BS, BS);
    dim3 blocks0((n + BS - 1) / BS, (n + BS - 1) / BS);
    float v0 = benchmark_kernel(matmul_v0_naive<BS>, A_d, B_d, C_d, n, threads0, blocks0);

    // V1: Coalesced
    float v1 = benchmark_kernel(matmul_v1_coalesced<BS>, A_d, B_d, C_d, n, threads0, blocks0);

    // V2: Tiled
    float v2 = benchmark_kernel(matmul_v2_tiled<BS>, A_d, B_d, C_d, n, threads0, blocks0);

    // V3: No conflict
    float v3 = benchmark_kernel(matmul_v3_noconflict<BS>, A_d, B_d, C_d, n, threads0, blocks0);

    // V4: 1D Blocktile
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 threads4((BM / TM) * BN);
    dim3 blocks4(n / BN, n / BM);
    float v4 = benchmark_kernel(matmul_v4_blocktile_1d<BM, BN, BK, TM>,
                                 A_d, B_d, C_d, n, threads4, blocks4);

    // V5: 2D Blocktile
    constexpr int TN = 8;
    dim3 threads5((BM / TM) * (BN / TN));
    dim3 blocks5(n / BN, n / BM);
    float v5 = benchmark_kernel(matmul_v5_blocktile_2d<BM, BN, BK, TM, TN>,
                                 A_d, B_d, C_d, n, threads5, blocks5);

    // V6: Vectorized
    float v6 = benchmark_kernel(matmul_v6_vectorized<BM, BN, BK, TM, TN>,
                                 A_d, B_d, C_d, n, threads5, blocks5);

    printf("%-6d | %7.0f | %7.0f | %7.0f | %7.0f | %7.0f | %7.0f | %7.0f | %9.0f\n",
           n, v0, v1, v2, v3, v4, v5, v6, cublas);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
  }

  printf("\n");
  printf("All values in GFLOPS. V6 target: >=70%% of cuBLAS across all sizes.\n");

  return 0;
}
