# High-Performance Tiled Matrix Multiplication

A deep dive into GPU optimization: 10 progressively optimized CUDA kernels, from naive (~1% of cuBLAS) to warptiled (~94% of cuBLAS).

Based on [Simon Boehm's excellent worklog](https://siboehm.com/articles/22/CUDA-MMM).

## Goal

Achieve **90%+ of cuBLAS SGEMM performance** through iterative optimization.

## Kernel Progression

| Version | Optimization | Key Concept | Target |
|---------|--------------|-------------|--------|
| V0 | Naive | Baseline, each thread computes one element | ~1% |
| V1 | GMEM Coalescing | Reorder thread indexing for coalesced access | ~8% |
| V2 | Shared Memory Tiling | Cache tiles in SMEM, reduce GMEM traffic | ~13% |
| V3 | 1D Blocktiling | Each thread computes TM elements (column) | ~37% |
| V4 | 2D Blocktiling | Each thread computes TM×TN elements | ~69% |
| V5 | Vectorized Loads | float4 for 128-bit memory transactions | ~78% |
| V6 | Bank Conflict Avoidance | SMEM padding/swizzling | ~80% |
| V7 | Double Buffering | Overlap compute with memory loads | ~82% |
| V8 | Autotuning | Parameter search (BM, BN, BK, TM, TN) | ~85% |
| V9 | Warptiling | Warp-level scheduling, register cache locality | ~94% |

## Concepts Learned

- **V0-V1**: Memory coalescing, warp-level access patterns
- **V2**: Shared memory as programmer-managed cache, `__syncthreads()`
- **V3-V4**: Arithmetic intensity, register reuse, thread-level blocking
- **V5**: Vectorized memory access, `float4`, `reinterpret_cast`
- **V6**: Shared memory banks, conflict avoidance via padding
- **V7**: Latency hiding, software pipelining
- **V8**: Empirical tuning, parameter search space
- **V9**: Warp scheduling, ILP, register cache locality

## Project Structure

```
high-performance-tiled-matmul/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── v0_naive.cu
│   ├── v1_coalesced.cu
│   ├── v2_tiled.cu
│   ├── v3_blocktile_1d.cu
│   ├── v4_blocktile_2d.cu
│   ├── v5_vectorized.cu
│   ├── v6_noconflict.cu
│   ├── v7_double_buffer.cu
│   ├── v8_autotuned.cu
│   └── v9_warptiled.cu
└── benchmark/
    └── benchmark_all.cu
```

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Running

```bash
# Individual kernels
./v0_naive
./v1_coalesced
# ...

# Full benchmark
./benchmark_all
```

## Profiling

```bash
# Occupancy and throughput
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second ./v2_tiled

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./v6_noconflict

# Full analysis
ncu --set full -o profile ./benchmark_all
```

## References

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS GEMM Tutorial](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md)
- Programming Massively Parallel Processors (Hwu, Kirk & Wen)

## Hardware Target

- NVIDIA A10 (Ampere, SM 8.6)
- 24 GB GDDR6, ~600 GB/s bandwidth
- 31.2 TFLOPS FP32 (CUDA cores)
