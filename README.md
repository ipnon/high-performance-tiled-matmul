### Month 3: CUDA Optimization & Shared Memory

#### Objectives

- Learn shared memory tiling, bank conflicts, and warp divergence solutions.

#### Readings

- CUDA Programming Guide: shared memory sections.
- NVIDIA Occupancy Calculator.
- Nsight Compute advanced usage.

#### Project: High-Performance Tiled Matmul ⭐ PORTFOLIO PIECE #1

##### Requirements

- Implement shared-memory tiled GEMM.
- Add register tiling (thread-level blocking).
- Achieve at least 50% of cuBLAS SGEMM performance on your GPU.
- Provide Nsight Compute screenshots.
- Explain:
  - Global load efficiency (>90%)
  - Shared memory bank conflict avoidance
  - Warp occupancy analysis

##### Why This Matters

- This is the single strongest kernel project you can show recruiters.
- Demonstrates real-world GPU optimization ability.
