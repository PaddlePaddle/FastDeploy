// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda/barrier>
#include <stdexcept>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

template <typename T>
struct cu_tensor_map_type_traits {
  static const CUtensorMapDataType type =
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};

template <>
struct cu_tensor_map_type_traits<phi::dtype::bfloat16> {
  static const CUtensorMapDataType type =
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};

template <>
struct cu_tensor_map_type_traits<phi::dtype::float16> {
  static const CUtensorMapDataType type =
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
};

template <>
struct cu_tensor_map_type_traits<uint8_t> {
  static const CUtensorMapDataType type =
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
};

template <>
struct cu_tensor_map_type_traits<phi::dtype::float8_e4m3fn> {
  static const CUtensorMapDataType type =
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
};

template <typename T>
CUtensorMap makeTensorMapForKVCache(T const* addr,
                                    uint32_t block_num,
                                    uint32_t kv_num_head,
                                    uint32_t second_size,
                                    uint32_t last_size) {
  CUtensorMap tensorMap{};

  uint32_t elem_bytes = sizeof(T);

  uint32_t const last_size_bytes = elem_bytes * last_size;
  // VLLM Layout
  CUtensorMapDataType data_dtype = cu_tensor_map_type_traits<T>::type;
  constexpr uint32_t rank = 4;
  uint64_t global_dims[] = {last_size, second_size, kv_num_head, block_num};
  uint64_t global_strides[] = {last_size_bytes,
                               second_size * last_size_bytes,
                               kv_num_head * second_size * last_size_bytes};

  uint32_t box_dims[] = {last_size, second_size, 1, 1};
  uint32_t elem_strides[] = {1, 1, 1, 1};

  auto const swizzle = [&] {
    switch (last_size_bytes) {
      case 128:
        return CU_TENSOR_MAP_SWIZZLE_128B;
      case 64:
        return CU_TENSOR_MAP_SWIZZLE_64B;
      default:
        throw std::runtime_error("unsupported cache last_size");
    }
  }();
  CUresult res = cuTensorMapEncodeTiled(
      &tensorMap,
      data_dtype,
      rank,
      reinterpret_cast<void*>(const_cast<T*>(addr)),
      global_dims,
      global_strides,
      box_dims,
      elem_strides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  switch (res) {
    case CUDA_SUCCESS:
      printf("CUDA_SUCCESS!\n");
      break;
    case CUDA_ERROR_INVALID_VALUE:
      printf("CUDA_ERROR_INVALID_VALUE\n");
      break;
    case CUDA_ERROR_OUT_OF_MEMORY:
      printf("CUDA_ERROR_OUT_OF_MEMORY\n");
      break;
    case CUDA_ERROR_NOT_INITIALIZED:
      printf("CUDA_ERROR_NOT_INITIALIZED\n");
      break;
    case CUDA_ERROR_DEINITIALIZED:
      printf("CUDA_ERROR_DEINITIALIZED\n");
      break;
    case CUDA_ERROR_PROFILER_DISABLED:
      printf("CUDA_ERROR_PROFILER_DISABLED\n");
      break;
    default:
      throw std::runtime_error("unsupported res!");
  }

  return tensorMap;
}
