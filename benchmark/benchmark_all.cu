#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <kernels/v0_naive.cuh>
#include <kernels/v1_coalesced.cuh>
#include <kernels/v2_tiled.cuh>
#include <kernels/v3_noconflict.cuh>
#include <kernels/v4_blocktile_1d.cuh>
#include <kernels/v5_blocktile_2d.cuh>
#include <kernels/v6_vectorized.cuh>

// Tile parameters
constexpr int kBlockSize = 16;
constexpr int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;

// Benchmark infrastructure
float benchmark_cublas(float* A, float* B, float* C, int n,
                       int warmup = 3, int iters = 10) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  for (int i = 0; i < warmup; i++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  return 2.0 * n * n * n * iters / (ms * 1e6);
}

template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, float* A, float* B, float* C,
                       int n, dim3 threads, dim3 blocks,
                       int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; i++)
    kernel<<<blocks, threads>>>(A, B, C, n);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    kernel<<<blocks, threads>>>(A, B, C, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 2.0 * n * n * n * iters / (ms * 1e6);
}

struct KernelConfig {
  std::string name;
  std::function<float(float*, float*, float*, int)> run;
};

std::vector<KernelConfig> make_kernels() {
  return {
    {"V0", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v0_naive<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V1", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v1_coalesced<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V2", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v2_tiled<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V3", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v3_noconflict<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V4", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * BN);
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v4_blocktile_1d<BM, BN, BK, TM>, A, B, C, n, threads, blocks);
    }},
    {"V5", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v5_blocktile_2d<BM, BN, BK, TM, TN>, A, B, C, n, threads, blocks);
    }},
    {"V6", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v6_vectorized<BM, BN, BK, TM, TN>, A, B, C, n, threads, blocks);
    }},
  };
}

int main() {
  std::array<int, 4> sizes = {512, 1024, 2048, 4096};
  auto kernels = make_kernels();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  float peak_tflops = prop.clockRate * 1e-6f * prop.multiProcessorCount * 128 * 2 / 1000.0f;

  printf("\nCUDA SGEMM Benchmark Suite\n");
  printf("==========================\n");
  printf("GPU: %s (%.1f TFLOPS FP32 peak)\n\n", prop.name, peak_tflops);

  // Header
  printf("%6s", "N");
  for (const auto& k : kernels) printf(" | %7s", k.name.c_str());
  printf(" | %9s\n", "cuBLAS");

  printf("------");
  for (size_t i = 0; i < kernels.size(); i++) printf("-|--------");
  printf("-|----------\n");

  // Run benchmarks
  for (int n : sizes) {
    float *A, *B, *C;
    cudaMalloc(&A, n * n * sizeof(float));
    cudaMalloc(&B, n * n * sizeof(float));
    cudaMalloc(&C, n * n * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, A, n * n);
    curandGenerateUniform(gen, B, n * n);
    curandDestroyGenerator(gen);

    float cublas = benchmark_cublas(A, B, C, n);

    printf("%6d", n);
    for (const auto& k : kernels) {
      float gflops = k.run(A, B, C, n);
      printf(" | %7.0f", gflops);
    }
    printf(" | %9.0f\n", cublas);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }

  // Summary with percentages
  printf("\n%% of cuBLAS:\n");
  printf("%6s", "N");
  for (const auto& k : kernels) printf(" | %7s", k.name.c_str());
  printf("\n");

  printf("------");
  for (size_t i = 0; i < kernels.size(); i++) printf("-|--------");
  printf("\n");

  for (int n : sizes) {
    float *A, *B, *C;
    cudaMalloc(&A, n * n * sizeof(float));
    cudaMalloc(&B, n * n * sizeof(float));
    cudaMalloc(&C, n * n * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, A, n * n);
    curandGenerateUniform(gen, B, n * n);
    curandDestroyGenerator(gen);

    float cublas = benchmark_cublas(A, B, C, n);

    printf("%6d", n);
    for (const auto& k : kernels) {
      float gflops = k.run(A, B, C, n);
      printf(" | %6.1f%%", 100.0f * gflops / cublas);
    }
    printf("\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }

  printf("\nTarget: V6 >= 70%% of cuBLAS across all sizes\n");
  return 0;
}
