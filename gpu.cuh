#pragma once

#include "instruction.h"
#include "util.h"

template <size_t n, typename T>
__global__ void kernel(const T *input, T *output, size_t size) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < size) {
    output[i] = instruction<n>(input[i]);
  }
}

template <size_t n, typename T>
void gpu(const T *input, T *output, size_t size) {
  GpuTimer gpu_timer(std::string("GPU " + get_profile_name<n, T>(size)));

  T *d_input = nullptr, *d_output = nullptr;
  HANDLE_GPU_ERROR(cudaMalloc((void **)&d_input, size * sizeof(T)),
                   "allocate device input");
  HANDLE_GPU_ERROR(cudaMalloc((void **)&d_output, size * sizeof(T)),
                   "allocate device output");
  HANDLE_GPU_ERROR(
      cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice),
      "copy to device");

  size_t threads_per_block = 256;
  size_t blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
  kernel<n><<<blocks_per_grid, threads_per_block>>>(d_input, d_output, size);
  HANDLE_GPU_ERROR(cudaGetLastError(), "kernel launch");

  HANDLE_GPU_ERROR(
      cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost),
      "copy to host");
  HANDLE_GPU_ERROR(cudaFree(d_input), "free device input");
  HANDLE_GPU_ERROR(cudaFree(d_output), "free device output");
}
