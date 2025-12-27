#include <common.cuh>
#include <kernels/v8_double_buffer.cuh>

int main() {
  constexpr int kBlockM = 64, kBlockN = 64, kBlockK = 8;
  constexpr int kThreadM = 8, kThreadN = 8;
  run_matmul_benchmark_blocktiled_2d<kBlockM, kBlockN, kBlockK, kThreadM,
                                     kThreadN>(
      matmul_v8_double_buffer<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>,
      1024);
}
