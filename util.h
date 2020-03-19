#pragma once

#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>

#define HANDLE_GPU_ERROR(call, msg)                                            \
  do {                                                                         \
    cudaError_t const err = (call);                                            \
    if (cudaSuccess != err) {                                                  \
      throw std::runtime_error(std::string("Failed to ") + msg + ": " +        \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

template <size_t n, typename T> std::string get_profile_name(size_t size) {
  return std::string("(type ") + typeid(T).name() + ", " + std::to_string(n) +
         " instructions, " + std::to_string(size * sizeof(T) / 1024 / 1024) +
         " mb)";
}

struct CpuTimer {
  CpuTimer(std::string &&msg_)
      : start(std::chrono::high_resolution_clock::now()), msg(std::move(msg_)) {
  }

  ~CpuTimer() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << msg << " costs: " << duration.count() << " ms" << std::endl;
  }

private:
  std::chrono::high_resolution_clock::time_point start;
  std::string msg;
};

struct GpuTimer {
  GpuTimer(std::string &&msg_) : msg(std::move(msg_)) {
    HANDLE_GPU_ERROR(cudaEventCreate(&start), "create start event");
    HANDLE_GPU_ERROR(cudaEventCreate(&stop), "create stop event");

    HANDLE_GPU_ERROR(cudaEventRecord(start, 0), "record start event");
  }

  ~GpuTimer() {
    float elapsed = 0;
    HANDLE_GPU_ERROR(cudaEventRecord(stop, 0), "record stop event");
    HANDLE_GPU_ERROR(cudaEventSynchronize(stop), "sync stop event");
    HANDLE_GPU_ERROR(cudaEventElapsedTime(&elapsed, start, stop),
                     "calculate elapsed");

    std::cout << msg << " costs: " << elapsed << " ms" << std::endl;

    HANDLE_GPU_ERROR(cudaEventDestroy(start), "destroy start event");
    HANDLE_GPU_ERROR(cudaEventDestroy(stop), "destroy stop event");
  }

private:
  cudaEvent_t start, stop;
  std::string msg;
};
