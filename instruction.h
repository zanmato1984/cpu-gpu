#pragma once

template <size_t n, typename T>
inline __host__ __device__ T instruction(T arg) {
  if constexpr (n == 0)
    return 1;
  else
    return arg * instruction<n - 1>(arg);

  return 0;
}
