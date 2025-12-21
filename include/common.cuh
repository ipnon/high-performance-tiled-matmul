#pragma once
#include <cuda_runtime.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>

template <int kBlockSize, typename KernelFunc>
inline void run_matmul_test(KernelFunc kernel, size_t n) {
  const int n_squared = n * n;
  const int bytes = n_squared * sizeof(float);

  float* A_h = static_cast<float*>(malloc(bytes));
  float* B_h = static_cast<float*>(malloc(bytes));
  float* C_h = static_cast<float*>(malloc(bytes));

  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, bytes);
  cudaMalloc(&B_d, bytes);
  cudaMalloc(&C_d, bytes);

  // Fill with random [0, 1]
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandGenerateUniform(gen, A_d, n_squared);
  curandGenerateUniform(gen, B_d, n_squared);

  cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

  dim3 threads(kBlockSize, kBlockSize);
  dim3 blocks((n + kBlockSize - 1) / kBlockSize,
              (n + kBlockSize - 1) / kBlockSize);

  kernel<<<blocks, threads>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

  int errors = 0;
  for (size_t i = 0; i < n * n; i++) {
    if (C_h[i] != static_cast<float>(n)) errors++;
  }
  printf("Errors: %d\n", errors);

  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
