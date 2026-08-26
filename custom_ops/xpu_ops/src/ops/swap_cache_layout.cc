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

#include <xpu/runtime.h>
#include "paddle/extension.h"

/*
 * XPU KV cache layout  : layer_num * [block_num, head_num, block_size,
 * head_dim] CPU pinned buf layout: [block_num, layer_num, head_num, block_size,
 * head_dim]
 *
 * mode 0 : XPU  -> CPU
 * mode 1 : CPU  -> XPU
 *
 * 地址计算
 *   cache_block_stride = head_num * block_size * head_dim   (=
 * cache_shape[1]*[2]*[3]) XPU ptr  = tensor[layer_idx].data() + xpu_block_id *
 * cache_block_stride CPU ptr  = cpu_base_ptr
 *              + cpu_block_id * cache_block_stride * layer_number   // block
 * 维度
 *              + layer_idx    * cache_block_stride                  // layer
 * 维度
 */

template <typename T>
void SwapCacheImpLayout(const std::vector<paddle::Tensor>& cache_xpu_tensors,
                        const int64_t& cache_cpu_pointer,
                        const std::vector<int64_t>& cache_shape,
                        const std::vector<int64_t>& xpu_block_ids,
                        const std::vector<int64_t>& cpu_block_ids,
                        int mode) {
  const int64_t layer_number = static_cast<int64_t>(cache_xpu_tensors.size());

  // cache_block_stride = product(cache_shape[1:])
  int64_t cache_block_stride = 1;
  for (int i = 1; i < static_cast<int>(cache_shape.size()); i++) {
    cache_block_stride *= cache_shape[i];
  }

  const XPUMemcpyKind copy_kind =
      (mode == 0) ? XPU_DEVICE_TO_HOST : XPU_HOST_TO_DEVICE;

  for (int layer_idx = 0; layer_idx < layer_number; layer_idx++) {
    const paddle::Tensor& cache_xpu = cache_xpu_tensors[layer_idx];
    T* cache_xpu_ptr = const_cast<T*>(cache_xpu.data<T>());
    auto* cache_cpu_ptr = reinterpret_cast<T*>(cache_cpu_pointer);

    for (int block_idx = 0; block_idx < static_cast<int>(xpu_block_ids.size());
         block_idx++) {
      auto cur_xpu_block_id = xpu_block_ids[block_idx];
      auto cur_cpu_block_id = cpu_block_ids[block_idx];

      auto* xpu_ptr_now = cache_xpu_ptr + cur_xpu_block_id * cache_block_stride;
      auto* cpu_ptr_now = cache_cpu_ptr +
                          cur_cpu_block_id * cache_block_stride * layer_number +
                          layer_idx * cache_block_stride;

      void* dst = (mode == 0) ? static_cast<void*>(cpu_ptr_now)
                              : static_cast<void*>(xpu_ptr_now);
      void* src = (mode == 0) ? static_cast<void*>(xpu_ptr_now)
                              : static_cast<void*>(cpu_ptr_now);

      int ret = xpu_memcpy(dst, src, cache_block_stride * sizeof(T), copy_kind);
      PD_CHECK(
          ret == XPU_SUCCESS, "xpu_memcpy failed with error code: %d", ret);
    }
  }
}

void SwapCacheLayout(const std::vector<paddle::Tensor>& cache_xpu_tensors,
                     const int64_t& cache_cpu_ptrs,
                     const std::vector<int64_t>& cache_shape,
                     const std::vector<int64_t>&
                         gpu_block_ids,  // XPU 侧 block ids（复用 gpu_block_ids
                                         // 参数名与 GPU 版接口一致）
                     const std::vector<int64_t>& cpu_block_ids,
                     int rank,
                     int mode) {
  xpu_set_device(rank);  // used for distributed launch
  PD_CHECK(cache_xpu_tensors.size() > 0, "cache_xpu_tensors must not be empty");

  switch (cache_xpu_tensors[0].dtype()) {
    case paddle::DataType::FLOAT16:
      return SwapCacheImpLayout<paddle::float16>(cache_xpu_tensors,
                                                 cache_cpu_ptrs,
                                                 cache_shape,
                                                 gpu_block_ids,
                                                 cpu_block_ids,
                                                 mode);
    case paddle::DataType::BFLOAT16:
      return SwapCacheImpLayout<paddle::bfloat16>(cache_xpu_tensors,
                                                  cache_cpu_ptrs,
                                                  cache_shape,
                                                  gpu_block_ids,
                                                  cpu_block_ids,
                                                  mode);
    case paddle::DataType::UINT8:
      return SwapCacheImpLayout<uint8_t>(cache_xpu_tensors,
                                         cache_cpu_ptrs,
                                         cache_shape,
                                         gpu_block_ids,
                                         cpu_block_ids,
                                         mode);
    case paddle::DataType::INT8:
      return SwapCacheImpLayout<int8_t>(cache_xpu_tensors,
                                        cache_cpu_ptrs,
                                        cache_shape,
                                        gpu_block_ids,
                                        cpu_block_ids,
                                        mode);
    default:
      PD_THROW("Unsupported data type.");
  }
}

PD_BUILD_OP(swap_cache_layout)
    .Inputs({paddle::Vec("cache_xpu_tensors")})
    .Attrs({
        "cache_cpu_ptrs: int64_t",
        "cache_shape: std::vector<int64_t>",
        "gpu_block_ids: std::vector<int64_t>",
        "cpu_block_ids: std::vector<int64_t>",
        "rank: int",
        "mode: int",
    })
    .Outputs({paddle::Vec("cache_dst_outs")})
    .SetInplaceMap({{paddle::Vec("cache_xpu_tensors"),
                     paddle::Vec("cache_dst_outs")}})
    .SetKernelFn(PD_KERNEL(SwapCacheLayout));
