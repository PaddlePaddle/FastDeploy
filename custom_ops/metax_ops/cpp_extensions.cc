#include "paddle/extension.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

// 自定义异常类，用于处理CUDA错误
class CudaError : public std::exception {
 public:
  explicit CudaError(cudaError_t error) : error_(error) {}

  const char* what() const noexcept override {
    return cudaGetErrorString(error_);
  }

 private:
  cudaError_t error_;
};

// 检查CUDA错误并抛出异常
void check_cuda_error(cudaError_t error) {
  if (error != cudaSuccess) {
    throw CudaError(error);
  }
}

// 封装cudaHostAlloc的Python函数
uintptr_t cuda_host_alloc(size_t size,
                          unsigned int flags = cudaHostAllocDefault) {
  void* ptr = nullptr;
  check_cuda_error(cudaHostAlloc(&ptr, size, flags));
  return reinterpret_cast<uintptr_t>(ptr);
}

// 封装cudaFreeHost的Python函数
void cuda_host_free(uintptr_t ptr) {
  check_cuda_error(cudaFreeHost(reinterpret_cast<void*>(ptr)));
}

PYBIND11_MODULE(fastdeploy_ops, m) {
  /**
   * alloc_cache_pinned.cc
   * cuda_host_alloc
   * cuda_host_free
   */
  m.def("cuda_host_alloc",
        &cuda_host_alloc,
        "Allocate pinned memory",
        py::arg("size"),
        py::arg("flags") = cudaHostAllocDefault);
  m.def(
      "cuda_host_free", &cuda_host_free, "Free pinned memory", py::arg("ptr"));
  py::register_exception<CudaError>(m, "CudaError");
}
