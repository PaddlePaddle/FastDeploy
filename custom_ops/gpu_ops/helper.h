// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_fp8.h>

#ifndef PADDLE_WITH_COREX
#include "glog/logging.h"
#endif
#include <fcntl.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <cstdlib>
#include <cstring>

#ifdef PADDLE_WITH_HIP
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
#ifndef PADDLE_WITH_COREX
#include "nlohmann/json.hpp"
#endif
#include <fstream>
#include <iostream>

#include "env.h"
#include "paddle/extension.h"
#include "paddle/phi/core/allocator.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/custom/custom_context.h"
#else
#include "paddle/phi/core/cuda_stream.h"
#endif
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"

#ifdef PADDLE_WITH_COREX
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#ifndef PADDLE_WITH_COREX
using json = nlohmann::json;
#endif

#define CUDA_CHECK(call)                           \
  do {                                             \
    const cudaError_t error_code = call;           \
    if (error_code != cudaSuccess) {               \
      std::printf("at %s:%d - %s.\n",              \
                  __FILE__,                        \
                  __LINE__,                        \
                  cudaGetErrorString(error_code)); \
      exit(1);                                     \
    }                                              \
  } while (0)

#ifdef PADDLE_WITH_HIP
template <size_t kBlockSize = 256, size_t kNumWaves = 16>
inline hipError_t GetNumBlocks(int64_t n, int *num_blocks) {
  int dev;
  {
    hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    hipError_t err = hipDeviceGetAttribute(
        &sm_count, hipDeviceAttributeMultiprocessorCount, dev);
    if (err != hipSuccess) {
      return err;
    }
  }
  int tpm;
  {
    hipError_t err = hipDeviceGetAttribute(
        &tpm, hipDeviceAttributeMaxThreadsPerMultiProcessor, dev);
    if (err != hipSuccess) {
      return err;
    }
  }
  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * tpm / kBlockSize * kNumWaves));
  return hipSuccess;
}
#else
template <size_t kBlockSize = 256, size_t kNumWaves = 16>
inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(
        &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

inline int GetGPUComputeCapability(int id) {
  int major, minor;
  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);
  return major * 10 + minor;
}

#endif

#ifndef FP8_E4M3_MAX
#define FP8_E4M3_MAX 448.0
#endif

#ifndef DISPATCH_FLOAT_FP6_DTYPE
#define DISPATCH_FLOAT_FP6_DTYPE(pd_dtype, c_type, ...)                     \
  switch (pd_dtype) {                                                       \
    case phi::DataType::FLOAT32: {                                          \
      using c_type = float;                                                 \
      __VA_ARGS__                                                           \
      break;                                                                \
    }                                                                       \
    case phi::DataType::BFLOAT16: {                                         \
      using c_type = phi::dtype::bfloat16;                                  \
      __VA_ARGS__                                                           \
      break;                                                                \
    }                                                                       \
    case phi::DataType::FLOAT16: {                                          \
      using c_type = phi::dtype::float16;                                   \
      __VA_ARGS__                                                           \
      break;                                                                \
    }                                                                       \
    default: {                                                              \
      PD_THROW("Only supported attr of input type in [fp32, fp16, bf16]."); \
    }                                                                       \
  }
#endif

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
 public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
 public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
 public:
#ifdef PADDLE_WITH_HIP
  typedef hip_bfloat16 DataType;
#else
  typedef __nv_bfloat16 DataType;
#endif
  typedef paddle::bfloat16 data_t;
};

template <>
class PDTraits<paddle::DataType::INT8> {
 public:
  typedef int8_t DataType;
  typedef int8_t data_t;
};

template <>
class PDTraits<paddle::DataType::UINT8> {
 public:
  typedef uint8_t DataType;
  typedef uint8_t data_t;
};

#ifndef PADDLE_WITH_COREX
template <>
class PDTraits<paddle::DataType::FLOAT8_E4M3FN> {
 public:
  typedef __nv_fp8_e4m3 DataType;
  typedef paddle::float8_e4m3fn data_t;
};
#endif

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T &operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T &operator[](int i) { return val[i]; }
};

template <typename T, int Size>
HOSTDEVICE inline void Load(const T *addr, AlignedVector<T, Size> *vec) {
  const AlignedVector<T, Size> *addr_vec =
      reinterpret_cast<const AlignedVector<T, Size> *>(addr);
  *vec = *addr_vec;
}

template <typename T, int Size>
HOSTDEVICE inline void Store(const AlignedVector<T, Size> &vec, T *addr) {
  AlignedVector<T, Size> *addr_vec =
      reinterpret_cast<AlignedVector<T, Size> *>(addr);
  *addr_vec = vec;
}

#ifdef PADDLE_WITH_HIP
template <int Size>
HOSTDEVICE inline void Store(const AlignedVector<hip_bfloat16, Size> &vec,
                             int8_t *addr) {
  printf("Error: Store hip_bfloat16 to int8_t is not supported!");
}
#else
template <int Size>
HOSTDEVICE inline void Store(const AlignedVector<__nv_bfloat16, Size> &vec,
                             int8_t *addr) {
  printf("Error: Store __nv_bfloat16 to int8_t is not supported!");
}
#endif

template <int Size>
HOSTDEVICE inline void Store(const AlignedVector<half, Size> &vec,
                             int8_t *addr) {
  printf("Error: Store half to int8_t is not supported!");
}

constexpr int VEC_16B = 16;

template <typename T>
__device__ T max_func(const T a, const T b) {
  return a > b ? a : b;
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return max_func(a, b);
  }
};

inline int GetBlockSize(int vocab_size) {
  if (vocab_size > 512) {
    return 1024;
  } else if (vocab_size > 256) {
    return 512;
  } else if (vocab_size > 128) {
    return 256;
  } else if (vocab_size > 64) {
    return 128;
  } else {
    return 64;
  }
}

#ifndef PADDLE_WITH_COREX
inline json readJsonFromFile(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + filePath);
  }

  json j;
  file >> j;
  return j;
}
#endif

#define cudaCheckError()                                                \
  {                                                                     \
    cudaError_t e = cudaGetLastError();                                 \
    if (e != cudaSuccess) {                                             \
      std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(e) << std::endl;                  \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// place must be an existing place object and cannot use paddle::CPUPlace() or
// paddle::GPUPlace()

#ifdef PADDLE_DEV
inline paddle::Tensor GetEmptyTensor(const common::DDim &dims,
                                     const paddle::DataType &dtype,
                                     const paddle::Place &place) {
  auto *allocator = paddle::GetAllocator(place);
  phi::DenseTensor dense_tensor;
  dense_tensor.Resize(dims);
  dense_tensor.AllocateFrom(
      allocator, dtype, dense_tensor.numel() * phi::SizeOf(dtype));
  return paddle::Tensor(std::make_shared<phi::DenseTensor>(dense_tensor));
}

inline paddle::Tensor GetEmptyTensor(const common::DDim &dims,
                                     const common::DDim &strides,
                                     const paddle::DataType &dtype,
                                     const paddle::Place &place) {
  auto *allocator = paddle::GetAllocator(place);
  phi::DenseTensor dense_tensor;
  dense_tensor.Resize(dims);
  dense_tensor.AllocateFrom(
      allocator, dtype, dense_tensor.numel() * phi::SizeOf(dtype));
  dense_tensor.set_strides(strides);
  return paddle::Tensor(std::make_shared<phi::DenseTensor>(dense_tensor));
}
#endif

__global__ void free_and_dispatch_block(bool *stop_flags,
                                        int *seq_lens_this_time,
                                        int *seq_lens_decoder,
                                        int *block_tables,
                                        int *encoder_block_lens,
                                        bool *is_block_step,
                                        int *step_block_list,  // [bsz]
                                        int *step_len,
                                        int *recover_block_list,
                                        int *recover_len,
                                        int *need_block_list,
                                        int *need_block_len,
                                        int *used_list_len,
                                        int *free_list,
                                        int *free_list_len,
                                        int64_t *first_token_ids,
                                        const int bsz,
                                        const int block_size,
                                        const int block_num_per_seq,
                                        const int max_decoder_block_num);

__global__ void speculate_free_and_dispatch_block(
    bool *stop_flags,
    int *seq_lens_this_time,
    int *seq_lens_decoder,
    int *block_tables,
    int *encoder_block_lens,
    bool *is_block_step,
    int *step_block_list,  // [bsz]
    int *step_len,
    int *recover_block_list,
    int *recover_len,
    int *need_block_list,
    int *need_block_len,
    int *used_list_len,
    int *free_list,
    int *free_list_len,
    int64_t *first_token_ids,
    int *accept_num,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_decoder_block_num,
    const int max_draft_tokens);

__device__ bool speculate_free_and_dispatch_block(const int &qid,
                                                  int *need_block_list,
                                                  const int &need_block_len);

static std::string global_base64_chars =  // NOLINT
    "Tokp9lA/BjimRVKx32edMPFftOzsbNQ8C15Xn+YUEGc4WD0uLIq7hyJ6vZaHSwrg";

// Base64 编码函数
inline std::string base64_encode(const std::string &input) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  for (const auto &c : input) {
    char_array_3[i++] = c;
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; i < 4; i++) {
        ret += global_base64_chars[char_array_4[i]];
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++) {
      char_array_3[j] = '\0';
    }

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; j < i + 1; j++) {
      ret += global_base64_chars[char_array_4[j]];
    }

    while (i++ < 3) {
      ret += '=';
    }
  }

  return ret;
}

// Base64 解码函数
inline std::string base64_decode(const std::string &encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && (encoded_string[in_] != '=') &&
         (isalnum(encoded_string[in_]) || (encoded_string[in_] == '+') ||
          (encoded_string[in_] == '/'))) {
    char_array_4[i++] = encoded_string[in_];
    in_++;
    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = global_base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] =
          (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] =
          ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; i < 3; i++) {
        ret += char_array_3[i];
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 4; j++) {
      char_array_4[j] = 0;
    }

    for (j = 0; j < 4; j++) {
      char_array_4[j] = global_base64_chars.find(char_array_4[j]);
    }

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] =
        ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; j < i - 1; j++) {
      ret += char_array_3[j];
    }
  }

  return ret;
}

#ifndef PADDLE_WITH_COREX
template <typename T>
inline T get_relative_best(nlohmann::json *json_data,
                           const std::string &target_key,
                           const T &default_value) {
  if (json_data->contains(target_key)) {
    return json_data->at(target_key);
  } else {
    // std::cerr << "The key " << target_key << " is not found in the JSON
    // data." << std::endl;
    return default_value;
  }
}
#endif

__device__ inline bool is_in_end(const int64_t id,
                                 const int64_t *end_ids,
                                 int length) {
  bool flag = false;
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return flag;
}

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
__device__ __inline__ T ClipFunc(const T v, const T min, const T max) {
  if (v > max) return max;
  if (v < min) return min;
  return v;
}

template <typename T>
static void PrintMatrix3(const T *mat_d, int num, std::string name) {
  std::vector<T> tmp(num);
#ifdef PADDLE_WITH_HIP
  hipMemcpy(tmp.data(), mat_d, sizeof(T) * num, hipMemcpyDeviceToHost);
#else
  cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);
#endif

  std::ofstream outfile;
  outfile.open(name + ".txt", std::ios::out);
  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
      ss << static_cast<int>(tmp[i]) << std::endl;
    } else {
      ss << std::setprecision(8) << (float)(tmp[i]) << std::endl;  // NOLINT
    }
  }
  outfile << ss.str();
  outfile.close();
}

#ifndef PADDLE_WITH_HIP
#ifndef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
__forceinline__ __device__ uint32_t ld_flag_acquire(uint32_t *flag_addr,
                                                    int mode = 0) {
  uint32_t flag;
  if (mode == 0) {
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(flag_addr));
  } else if (mode == 1) {
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(flag_addr));
  } else {
    asm volatile("ld.acquire.cta.global.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(flag_addr));
  }
  return flag;
}

__forceinline__ __device__ void st_flag_release(uint32_t *flag_addr,
                                                uint32_t flag,
                                                int mode = 0) {
  if (mode == 0) {
    asm volatile("st.release.sys.global.b32 [%1], %0;" ::"r"(flag),
                 "l"(flag_addr));
  } else if (mode == 1) {
    asm volatile("st.release.gpu.global.b32 [%1], %0;" ::"r"(flag),
                 "l"(flag_addr));
  } else {
    asm volatile("st.release.cta.global.b32 [%1], %0;" ::"r"(flag),
                 "l"(flag_addr));
  }
}
#endif
inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device);
  return max_shared_mem_per_block_opt_in;
}
#endif

inline int GetSMVersion() {
#ifdef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
  return 80;
#else
  static int sm_version = phi::backends::gpu::GetGPUComputeCapability(
      phi::backends::gpu::GetCurrentDeviceId());
  return sm_version;
#endif
}

inline bool GetMlaUseTensorcore() {
  static const bool flags_mla_use_tensorcore = get_mla_use_tensorcore();
  static const bool enable_mla_tensorcore = GetSMVersion() >= 90 ? true : false;
  const bool mla_use_tensorcore =
      flags_mla_use_tensorcore && enable_mla_tensorcore;
  return mla_use_tensorcore;
}

inline const char *getEnvVar(const char *varName) {
  return std::getenv(varName);
}

inline bool checkAttentionBackend() {
  const char *backend = getEnvVar("FD_ATTENTION_BACKEND");
  if (backend && std::strcmp(backend, "MLA_ATTN") == 0) {
    return true;
  }
  return false;
}

#ifndef GPU_MEMORY_CHECKER_H
#define GPU_MEMORY_CHECKER_H
class GPUMemoryChecker {
 public:
  static GPUMemoryChecker *getInstance() {
    static GPUMemoryChecker instance;
    return &instance;
  }

  void addCheckPoint(const char *call_file, int call_line);
  unsigned int getGPUCount() const { return deviceCount_; }
  void getCUDAVisibleDevice();

  GPUMemoryChecker(const GPUMemoryChecker &) = delete;
  void operator=(const GPUMemoryChecker &) = delete;

 private:
  GPUMemoryChecker();
  ~GPUMemoryChecker();

  unsigned int deviceCount_;
  std::vector<unsigned int> visible_device_;
  std::vector<unsigned int> visible_device_mem_usage_;
};

#endif  // GPU_MEMORY_CHECKER_H
__device__ __forceinline__ float warpReduceMax(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 16));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 8));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
  return value;
}

__device__ __forceinline__ float blockReduceMax(float value) {
  static __shared__ float warpLevelMaxs[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  value = warpReduceMax(value);

  if (laneId == 0) warpLevelMaxs[warpId] = value;
  __syncthreads();

  value = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0) value = warpReduceMax(value);

  return value;
}
inline bool getBoolEnv(char const *name) {
  char const *env = std::getenv(name);
  return env && env[0] == '1' && env[1] == '\0';
}

bool getEnvEnablePDL();

#ifndef PADDLE_WITH_COREX
template <typename KernelFn, typename... Args>
inline void launchWithPdlWhenEnabled(KernelFn kernelFn,
                                     dim3 grid,
                                     dim3 block,
                                     size_t dynamicShmSize,
                                     cudaStream_t stream,
                                     Args &&...args) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
  (*kernelFn)<<<grid, block, dynamicShmSize, stream>>>(
      std::forward<Args>(args)...);
#else
  cudaLaunchConfig_t kernelConfig;
  kernelConfig.gridDim = grid;
  kernelConfig.blockDim = block;
  kernelConfig.dynamicSmemBytes = dynamicShmSize;
  kernelConfig.stream = stream;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = getEnvEnablePDL();
  kernelConfig.attrs = attrs;
  kernelConfig.numAttrs = 1;

  cudaLaunchKernelEx(&kernelConfig, kernelFn, std::forward<Args>(args)...);
#endif
}
#endif
