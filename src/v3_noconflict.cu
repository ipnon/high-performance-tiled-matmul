#include <common.cuh>
#include <kernels/v3_noconflict.cuh>

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v3_noconflict<kBlockSize>, 1024);
}
