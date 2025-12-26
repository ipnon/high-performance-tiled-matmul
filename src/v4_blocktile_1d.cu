#include <common.cuh>
#include <kernels/v4_blocktile_1d.cuh>

int main() {
  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
  run_matmul_benchmark_blocktiled<BM, BN, BK, TM>(
      matmul_v4_blocktile_1d<BM, BN, BK, TM>, 1024);
}
