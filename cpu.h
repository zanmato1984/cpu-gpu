#pragma once

#include "instruction.h"
#include "util.h"

template <size_t n, typename T>
void cpu(const T *input, T *output, size_t size) {
  GpuTimer timer(std::string("CPU " + get_profile_name<n, T>(size)));

  for (size_t i = 0; i < size; i++)
    output[i] = instruction<n>(input[i]);
}
