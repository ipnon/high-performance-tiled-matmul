#include <common.cuh>
#include <kernels/v1_coalesced.cuh>

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v1_coalesced<kBlockSize>, 1024);
}
