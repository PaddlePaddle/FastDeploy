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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xpu/runtime.h>
#include "paddle/extension.h"

template <typename T>
void SwapCacheImplAllLayers(
    const std::vector<paddle::Tensor>& cache_xpu_tensors,  // xpu
    const std::vector<int64_t>& cache_cpu_ptrs,            // cpu
    const int64_t& max_block_num_cpu,
    const std::vector<int64_t>& swap_block_ids_xpu,
    const std::vector<int64_t>& swap_block_ids_cpu,
    int mode) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  for (int layer_idx = 0; layer_idx < cache_xpu_tensors.size(); layer_idx++) {
    const paddle::Tensor& cache_xpu = cache_xpu_tensors[layer_idx];
    const int64_t& cache_cpu_pointer = cache_cpu_ptrs[layer_idx];
    // XPUType* cache_xpu_ptr =
    // reinterpret_cast<XPUType*>(const_cast<T*>(cache_xpu.data<T>()));
    T* cache_xpu_ptr = const_cast<T*>(cache_xpu.data<T>());
    auto* cache_cpu_ptr = reinterpret_cast<T*>(cache_cpu_pointer);
    auto cache_shape = cache_xpu.shape();
    const int64_t max_block_num_xpu = cache_shape[0];
    const int64_t num_heads = cache_shape[1];
    const int64_t block_size = cache_shape[2];
    const int64_t head_dim = cache_shape[3];
    const int64_t cache_stride = num_heads * block_size * head_dim;

    if (swap_block_ids_xpu.size() == 0) {
      return;
    }
    int i = 0;
    int64_t consecutive_block_count = 1;
    int64_t last_xpu_block_id = swap_block_ids_xpu[i];
    int64_t last_cpu_block_id = swap_block_ids_cpu[i];
    int64_t first_xpu_block_id =
        last_xpu_block_id;  // first block id in a consecutive block ids
    int64_t first_cpu_block_id = last_cpu_block_id;
    i += 1;
    while (true) {
      if (i >= swap_block_ids_xpu.size()) {
        break;
      }
      int64_t xpu_block_id = swap_block_ids_xpu[i];
      int64_t cpu_block_id = swap_block_ids_cpu[i];
      PD_CHECK(xpu_block_id >= 0 && xpu_block_id < max_block_num_xpu);
      PD_CHECK(cpu_block_id >= 0 && cpu_block_id < max_block_num_cpu);
      if (xpu_block_id == last_xpu_block_id + 1 &&
          cpu_block_id == last_cpu_block_id + 1) {  // consecutive
        consecutive_block_count += 1;
        last_xpu_block_id = xpu_block_id;
        last_cpu_block_id = cpu_block_id;
      } else {
        // end of a consecutive block ids
        auto* cache_xpu_ptr_now =
            cache_xpu_ptr + first_xpu_block_id * cache_stride;
        auto* cache_cpu_ptr_now =
            cache_cpu_ptr + first_cpu_block_id * cache_stride;
        if (mode == 0) {  // copy from device to host
          xpu_memcpy(cache_cpu_ptr_now,
                     cache_xpu_ptr_now,
                     cache_stride * sizeof(XPUType) * consecutive_block_count,
                     XPU_DEVICE_TO_HOST);
        } else {  // copy from host to device
          xpu_memcpy(cache_xpu_ptr_now,
                     cache_cpu_ptr_now,
                     cache_stride * sizeof(XPUType) * consecutive_block_count,
                     XPU_HOST_TO_DEVICE);
        }
        first_xpu_block_id = xpu_block_id;
        first_cpu_block_id = cpu_block_id;
        last_xpu_block_id = xpu_block_id;
        last_cpu_block_id = cpu_block_id;
        consecutive_block_count = 1;
      }
      i += 1;
    }
    // last batch
    auto* cache_xpu_ptr_now = cache_xpu_ptr + first_xpu_block_id * cache_stride;
    auto* cache_cpu_ptr_now = cache_cpu_ptr + first_cpu_block_id * cache_stride;
    if (mode == 0) {  // copy from device to host
      xpu_memcpy(cache_cpu_ptr_now,
                 cache_xpu_ptr_now,
                 cache_stride * sizeof(XPUType) * consecutive_block_count,
                 XPU_DEVICE_TO_HOST);
    } else {  // copy from host to device
      xpu_memcpy(cache_xpu_ptr_now,
                 cache_cpu_ptr_now,
                 cache_stride * sizeof(XPUType) * consecutive_block_count,
                 XPU_HOST_TO_DEVICE);
    }
  }
}

void SwapCacheAllLayers(
    const std::vector<paddle::Tensor>& cache_xpu_tensors,  // xpu
    const std::vector<int64_t>& cache_cpu_ptrs,            // cpu memory pointer
    int64_t max_block_num_cpu,                             // cpu max block num
    const std::vector<int64_t>& swap_block_ids_xpu,
    const std::vector<int64_t>& swap_block_ids_cpu,
    int rank,
    int mode) {
  xpu_set_device(rank);  // used for distributed launch
  PD_CHECK(cache_xpu_tensors.size() > 0 &&
           cache_xpu_tensors.size() == cache_cpu_ptrs.size());
  switch (cache_xpu_tensors[0].dtype()) {
    case paddle::DataType::FLOAT16:
      return SwapCacheImplAllLayers<paddle::float16>(cache_xpu_tensors,
                                                     cache_cpu_ptrs,
                                                     max_block_num_cpu,
                                                     swap_block_ids_xpu,
                                                     swap_block_ids_cpu,
                                                     mode);
    case paddle::DataType::UINT8:
      return SwapCacheImplAllLayers<uint8_t>(cache_xpu_tensors,
                                             cache_cpu_ptrs,
                                             max_block_num_cpu,
                                             swap_block_ids_xpu,
                                             swap_block_ids_cpu,
                                             mode);
    case paddle::DataType::INT8:
      return SwapCacheImplAllLayers<int8_t>(cache_xpu_tensors,
                                            cache_cpu_ptrs,
                                            max_block_num_cpu,
                                            swap_block_ids_xpu,
                                            swap_block_ids_cpu,
                                            mode);
    case paddle::DataType::BFLOAT16:
      return SwapCacheImplAllLayers<paddle::bfloat16>(cache_xpu_tensors,
                                                      cache_cpu_ptrs,
                                                      max_block_num_cpu,
                                                      swap_block_ids_xpu,
                                                      swap_block_ids_cpu,
                                                      mode);
    default:
      PD_THROW("Unsupported data type.");
  }
}

PD_BUILD_OP(swap_cache_all_layers)
    .Inputs({paddle::Vec("cache_xpu_tensors")})
    .Attrs({
        "cache_cpu_ptrs: std::vector<int64_t>",
        "max_block_num_cpu: int64_t",
        "swap_block_ids_xpu: std::vector<int64_t>",
        "swap_block_ids_cpu: std::vector<int64_t>",
        "rank: int",
        "mode: int",
    })
    .Outputs({paddle::Vec("cache_dst_outs")})
    .SetInplaceMap({{paddle::Vec("cache_xpu_tensors"),
                     paddle::Vec("cache_dst_outs")}})
    .SetKernelFn(PD_KERNEL(SwapCacheAllLayers));
