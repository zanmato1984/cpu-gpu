#include <cstring>

#include "cpu.h"
#include "gpu.cuh"

constexpr size_t default_max_n = 32;
constexpr size_t default_max_size = 256 * 1024 * 1024;

template <size_t n, typename T>
inline void cpu_gpu_core(const T *input, size_t size) {
  {
    auto *cpu_input = new T[size];
    auto *cpu_output = new T[size];

    memcpy(cpu_input, input, size * sizeof(T));
    cpu<n>(cpu_input, cpu_output, size);

    delete[] cpu_input;
    delete[] cpu_output;
  }

  {
    auto *gpu_input = new T[size];
    auto *gpu_output = new T[size];

    memcpy(gpu_input, input, size * sizeof(T));
    gpu<n>(gpu_input, gpu_output, size);

    delete[] gpu_input;
    delete[] gpu_output;
  }
}

template <size_t n, typename T>
inline void cpu_gpu(const T *input, size_t size) {
  cpu_gpu_core<n>(input, size);
  if constexpr (n < default_max_n)
    cpu_gpu<n * 2>(input, size);
}

template <typename T> inline void cpu_gpu(size_t size) {
  auto *input = new T[size];

  cpu_gpu<1>(input, size);

  delete[] input;
}

int main(int argc, char *argv[]) {
  size_t size = default_max_size;

  if (argc == 2)
    size = std::atol(argv[0]) * 1024 * 1024;

  cpu_gpu<int8_t>(size);
  cpu_gpu<int16_t>(size);
  cpu_gpu<int32_t>(size);
  cpu_gpu<int64_t>(size);
  cpu_gpu<float>(size);
  cpu_gpu<double>(size);

  return 0;
}