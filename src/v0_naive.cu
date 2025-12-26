#include <common.cuh>
#include <kernels/v0_naive.cuh>

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v0_naive<kBlockSize>, 1024);
}
