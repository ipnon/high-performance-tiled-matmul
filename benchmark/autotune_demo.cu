#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cstdio>
#include <array>

#include <autotune.cuh>

float benchmark_cublas(float* A, float* B, float* C, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 3; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 10; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return 2.0f * n * n * n * 10 / (ms * 1e6f);
}

int main() {
    std::array<int, 4> sizes = {1024, 2048, 4096, 8192};

    printf("\n=== SGEMM Runtime Autotuner Demo ===\n\n");

    // Phase 1: Autotune for each size
    Autotuner tuner;

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

        tuner.tune(n, A, B, C, true);

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    // Phase 2: Run with autotuned configs and compare to cuBLAS
    printf("=== Results with Autotuned Configs ===\n\n");
    printf("%-6s | %-14s | %10s | %10s | %s\n",
           "Size", "Config", "GFLOPS", "cuBLAS", "% cuBLAS");
    printf("-------|----------------|------------|------------|----------\n");

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

        const auto& cfg = tuner.get_best(n);
        float our_gflops = benchmark_config(cfg, A, B, C, n, 3, 10);
        float cublas_gflops = benchmark_cublas(A, B, C, n);
        float pct = 100.0f * our_gflops / cublas_gflops;

        printf("%-6d | %-14s | %10.0f | %10.0f | %6.1f%%\n",
               n, cfg.name.c_str(), our_gflops, cublas_gflops, pct);

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    printf("\n");
    return 0;
}
