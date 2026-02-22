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

// #define SWAP_DEBUG

template <paddle::DataType D>
void SwapCacheImpLayout(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,  // gpu
    const int64_t& cache_cpu_pointer,                      // cpu
    const std::vector<int64_t>& cache_shape,
    const std::vector<int64_t>& gpu_block_ids,
    const std::vector<int64_t>& cpu_block_ids,
    int mode) {
  /*
  mode is 0: gpu to cpu; 1: cpu to gpu

  cache layout: layer_num * [block_num, head_num, block_size, head_dim]
  scale layout: layer_num * [block_num, head_num, block_size]
  cache buffer layout: [block_num, layer_num, head_num, block_size, head_dim]
  scale buffer layout: [block_num, layer_num, head_num, block_size]
  */
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int64_t layer_number = cache_gpu_tensors.size();
  int64_t cache_block_stride = 1;
  for (int i = 1; i < cache_shape.size(); i++) {
    cache_block_stride *= cache_shape[i];
  }

  auto stream = cache_gpu_tensors[0].stream();
  const cudaMemcpyKind copy_kind =
      (mode == 0) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

  for (int layer_idx = 0; layer_idx < cache_gpu_tensors.size(); layer_idx++) {
    const paddle::Tensor& cache_gpu = cache_gpu_tensors[layer_idx];
    data_t* cache_gpu_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
    auto* cache_cpu_ptr = reinterpret_cast<data_t*>(cache_cpu_pointer);

    for (int block_idx = 0; block_idx < gpu_block_ids.size(); block_idx++) {
      auto cur_gpu_block_id = gpu_block_ids[block_idx];
      auto cur_cpu_block_id = cpu_block_ids[block_idx];
      auto* cache_gpu_ptr_now =
          cache_gpu_ptr + cur_gpu_block_id * cache_block_stride;
      auto* cache_cpu_ptr_now =
          cache_cpu_ptr + cur_cpu_block_id * cache_block_stride * layer_number +
          layer_idx * cache_block_stride;

      cudaError_t status = cudaMemcpyAsync(
          (copy_kind == cudaMemcpyDeviceToHost) ? cache_cpu_ptr_now
                                                : cache_gpu_ptr_now,
          (copy_kind == cudaMemcpyDeviceToHost) ? cache_gpu_ptr_now
                                                : cache_cpu_ptr_now,
          cache_block_stride * sizeof(DataType_),
          copy_kind,
          stream);

      PADDLE_ENFORCE_EQ(status,
                        cudaSuccess,
                        phi::errors::External("cudaMemcpyAsync failed: %s",
                                              cudaGetErrorString(status)));

#ifdef SWAP_DEBUG
      cudaStreamSynchronize(stream);
      std::cout << "mode:" << mode << ", layer_idx:" << layer_idx
                << ", block_idx:" << block_idx << ", cache_cpu_ptr_now data:"
                << static_cast<float>(*cache_cpu_ptr_now) << std::endl;
#endif
    }
  }
  cudaError_t sync_status = cudaStreamSynchronize(stream);
  PADDLE_ENFORCE_EQ(sync_status,
                    cudaSuccess,
                    phi::errors::External("cudaStreamSynchronize failed: %s",
                                          cudaGetErrorString(sync_status)));
}

void SwapCacheLayout(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,  // gpu
    const int64_t& cache_cpu_ptrs,                         // cpu memory pointer
    const std::vector<int64_t>& cache_shape,
    const std::vector<int64_t>& gpu_block_ids,
    const std::vector<int64_t>& cpu_block_ids,
    int rank,
    int mode) {
  cudaSetDevice(rank);  // used for distributed launch
  assert(cache_gpu_tensors.size() > 0);
  switch (cache_gpu_tensors[0].dtype()) {
    case paddle::DataType::BFLOAT16:
      return SwapCacheImpLayout<paddle::DataType::BFLOAT16>(cache_gpu_tensors,
                                                            cache_cpu_ptrs,
                                                            cache_shape,
                                                            gpu_block_ids,
                                                            cpu_block_ids,
                                                            mode);
    case paddle::DataType::FLOAT16:
      return SwapCacheImpLayout<paddle::DataType::FLOAT16>(cache_gpu_tensors,
                                                           cache_cpu_ptrs,
                                                           cache_shape,
                                                           gpu_block_ids,
                                                           cpu_block_ids,
                                                           mode);
    case paddle::DataType::UINT8:
      return SwapCacheImpLayout<paddle::DataType::UINT8>(cache_gpu_tensors,
                                                         cache_cpu_ptrs,
                                                         cache_shape,
                                                         gpu_block_ids,
                                                         cpu_block_ids,
                                                         mode);
    default:
      PD_THROW("Unsupported data type.");
  }
}

PD_BUILD_STATIC_OP(swap_cache_layout)
    .Inputs({paddle::Vec("cache_gpu_tensors")})
    .Attrs({
        "cache_cpu_ptrs: int64_t",
        "cache_shape: std::vector<int64_t>",
        "gpu_block_ids: std::vector<int64_t>",
        "cpu_block_ids: std::vector<int64_t>",
        "rank: int",
        "mode: int",
    })
    .SetKernelFn(PD_KERNEL(SwapCacheLayout));
