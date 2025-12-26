#include <common.cuh>
#include <kernels/v2_tiled.cuh>

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v2_tiled<kBlockSize>, 1024);
}
