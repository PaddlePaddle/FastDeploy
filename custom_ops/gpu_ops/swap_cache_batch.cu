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

#include "helper.h"
#include "paddle/extension.h"

template <paddle::DataType D>
void SwapCacheImplAllLayers(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,  // gpu
    const std::vector<int64_t>& cache_cpu_ptrs,            // cpu
    const int64_t& max_block_num_cpu,
    const std::vector<int64_t>& swap_block_ids_gpu,
    const std::vector<int64_t>& swap_block_ids_cpu,
    int mode) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto stream = cache_gpu_tensors[0].stream();
  for (int layer_idx = 0; layer_idx < cache_gpu_tensors.size(); layer_idx++) {
    const paddle::Tensor& cache_gpu = cache_gpu_tensors[layer_idx];
    const int64_t& cache_cpu_pointer = cache_cpu_ptrs[layer_idx];
    data_t* cache_gpu_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
    auto* cache_cpu_ptr = reinterpret_cast<data_t*>(cache_cpu_pointer);
    auto cache_shape = cache_gpu.shape();
    const int64_t max_block_num_gpu = cache_shape[0];
    const int64_t num_heads = cache_shape[1];
    const int64_t block_size = cache_shape[2];
    int64_t head_dim = 1;
    if (cache_shape.size() == 4) {
      head_dim = cache_shape[3];
    }
    const int64_t cache_stride = num_heads * block_size * head_dim;

    auto stream = cache_gpu.stream();
    if (swap_block_ids_gpu.size() == 0) {
      return;
    }
    int i = 0;
    int64_t consecutive_block_count = 1;
    int64_t last_gpu_block_id = swap_block_ids_gpu[i];
    int64_t last_cpu_block_id = swap_block_ids_cpu[i];
    int64_t first_gpu_block_id =
        last_gpu_block_id;  // first block id in a consecutive block ids
    int64_t first_cpu_block_id = last_cpu_block_id;
    i += 1;
    while (true) {
      if (i >= swap_block_ids_gpu.size()) {
        break;
      }
      int64_t gpu_block_id = swap_block_ids_gpu[i];
      int64_t cpu_block_id = swap_block_ids_cpu[i];
      assert(gpu_block_id >= 0 && gpu_block_id < max_block_num_gpu);
      assert(cpu_block_id >= 0 && cpu_block_id < max_block_num_cpu);
      if (gpu_block_id == last_gpu_block_id + 1 &&
          cpu_block_id == last_cpu_block_id + 1) {  // consecutive
        consecutive_block_count += 1;
        last_gpu_block_id = gpu_block_id;
        last_cpu_block_id = cpu_block_id;
      } else {
        // end of a consecutive block ids
        auto* cache_gpu_ptr_now =
            cache_gpu_ptr + first_gpu_block_id * cache_stride;
        auto* cache_cpu_ptr_now =
            cache_cpu_ptr + first_cpu_block_id * cache_stride;
        if (mode == 0) {  // copy from device to host
          cudaMemcpyAsync(
              cache_cpu_ptr_now,
              cache_gpu_ptr_now,
              cache_stride * sizeof(DataType_) * consecutive_block_count,
              cudaMemcpyDeviceToHost,
              stream);
        } else {  // copy from host to device
          cudaMemcpyAsync(
              cache_gpu_ptr_now,
              cache_cpu_ptr_now,
              cache_stride * sizeof(DataType_) * consecutive_block_count,
              cudaMemcpyHostToDevice,
              stream);
        }
        first_gpu_block_id = gpu_block_id;
        first_cpu_block_id = cpu_block_id;
        last_gpu_block_id = gpu_block_id;
        last_cpu_block_id = cpu_block_id;
        consecutive_block_count = 1;
      }
      i += 1;
    }
    // last batch
    auto* cache_gpu_ptr_now = cache_gpu_ptr + first_gpu_block_id * cache_stride;
    auto* cache_cpu_ptr_now = cache_cpu_ptr + first_cpu_block_id * cache_stride;
    if (mode == 0) {  // copy from device to host
      cudaMemcpyAsync(
          cache_cpu_ptr_now,
          cache_gpu_ptr_now,
          cache_stride * sizeof(DataType_) * consecutive_block_count,
          cudaMemcpyDeviceToHost,
          stream);
    } else {  // copy from host to device
      cudaMemcpyAsync(
          cache_gpu_ptr_now,
          cache_cpu_ptr_now,
          cache_stride * sizeof(DataType_) * consecutive_block_count,
          cudaMemcpyHostToDevice,
          stream);
    }
  }
  cudaStreamSynchronize(stream);
}

void SwapCacheAllLayers(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,  // gpu
    const std::vector<int64_t>& cache_cpu_ptrs,            // cpu memory pointer
    int64_t max_block_num_cpu,                             // cpu max block num
    const std::vector<int64_t>& swap_block_ids_gpu,
    const std::vector<int64_t>& swap_block_ids_cpu,
    int rank,
    int mode) {
  cudaSetDevice(rank);  // used for distributed launch
  assert(cache_gpu_tensors.size() > 0 &&
         cache_gpu_tensors.size() == cache_cpu_ptrs.size());
  switch (cache_gpu_tensors[0].dtype()) {
    case paddle::DataType::BFLOAT16:
      return SwapCacheImplAllLayers<paddle::DataType::BFLOAT16>(
          cache_gpu_tensors,
          cache_cpu_ptrs,
          max_block_num_cpu,
          swap_block_ids_gpu,
          swap_block_ids_cpu,
          mode);
    case paddle::DataType::FLOAT16:
      return SwapCacheImplAllLayers<paddle::DataType::FLOAT16>(
          cache_gpu_tensors,
          cache_cpu_ptrs,
          max_block_num_cpu,
          swap_block_ids_gpu,
          swap_block_ids_cpu,
          mode);
    case paddle::DataType::UINT8:
      return SwapCacheImplAllLayers<paddle::DataType::UINT8>(cache_gpu_tensors,
                                                             cache_cpu_ptrs,
                                                             max_block_num_cpu,
                                                             swap_block_ids_gpu,
                                                             swap_block_ids_cpu,
                                                             mode);
    default:
      PD_THROW("Unsupported data type.");
  }
}

PD_BUILD_STATIC_OP(swap_cache_all_layers)
    .Inputs({paddle::Vec("cache_gpu_tensors")})
    .Attrs({
        "cache_cpu_ptrs: std::vector<int64_t>",
        "max_block_num_cpu: int64_t",
        "swap_block_ids_gpu: std::vector<int64_t>",
        "swap_block_ids_cpu: std::vector<int64_t>",
        "rank: int",
        "mode: int",
    })
    .Outputs({paddle::Vec("cache_dst_outs")})
    .SetInplaceMap({{paddle::Vec("cache_gpu_tensors"),
                     paddle::Vec("cache_dst_outs")}})
    .SetKernelFn(PD_KERNEL(SwapCacheAllLayers));
