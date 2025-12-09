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
    const std::vector<int64_t>& swap_block_ids_gpu,
    int mode) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  const int64_t layer_number = cache_gpu_tensors.size();
#ifdef SWAP_DEBUG
  std::cout << "layer_number " << layer_number << std::endl;
  std::cout << "cache_shape size: test" << std::endl;
  std::cout << "cache_shape:" << cache_shape[0] << ", " << cache_shape[1]
            << ", " << cache_shape[2] << ", " << cache_shape[3] << std::endl;
#endif

  const int64_t max_block_num_gpu = cache_shape[0];
  const int64_t num_heads = cache_shape[1];
  const int64_t block_size = cache_shape[2];
  const int64_t head_dim = cache_shape[3];
  const int64_t cache_stride = num_heads * block_size * head_dim;
#ifdef SWAP_DEBUG
  std::cout << "cache_stride " << cache_stride << std::endl;
#endif

  auto stream = cache_gpu_tensors[0].stream();
  const cudaMemcpyKind copy_kind =
      (mode == 0) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
  // 【layer， block， block size，head num， head dim】
  // 【block， layer， block size，head num， head dim】
  auto* cache_cpu_ptr = reinterpret_cast<data_t*>(cache_cpu_pointer);
#ifdef SWAP_DEBUG
  std::cout << "cache_cpu_ptr: " << cache_cpu_ptr << std::endl;
#endif

  for (int layer_idx = 0; layer_idx < cache_gpu_tensors.size(); layer_idx++) {
    const paddle::Tensor& cache_gpu = cache_gpu_tensors[layer_idx];
    data_t* cache_gpu_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
    auto stream = cache_gpu.stream();
    for (int id = 0; id < swap_block_ids_gpu.size(); id++) {
#ifdef SWAP_DEBUG
      std::cout << "current block " << swap_block_ids_gpu[id] << std::endl;
#endif

      auto* cache_gpu_ptr_now =
          cache_gpu_ptr + swap_block_ids_gpu[id] * cache_stride;
      auto* cache_cpu_ptr_now = cache_cpu_ptr +
                                id * cache_stride * layer_number +
                                layer_idx * cache_stride;

#ifdef SWAP_DEBUG
      std::cout << "current data" << *cache_cpu_ptr_now << std::endl;
#endif
      cudaError_t status = cudaMemcpyAsync(
          (copy_kind == cudaMemcpyDeviceToHost) ? cache_cpu_ptr_now
                                                : cache_gpu_ptr_now,
          (copy_kind == cudaMemcpyDeviceToHost) ? cache_gpu_ptr_now
                                                : cache_cpu_ptr_now,
          cache_stride * sizeof(DataType_),
          copy_kind,
          stream);
#ifdef SWAP_DEBUG
      std::cout << "current data11 " << *cache_cpu_ptr_now << std::endl;
#endif
    }
  }
  cudaStreamSynchronize(stream);
#ifdef SWAP_DEBUG
  std::cout << "finished " << std::endl;
#endif
}

void SwapCacheLayout(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,  // gpu
    const int64_t& cache_cpu_ptrs,                         // cpu memory pointer
    const std::vector<int64_t>& cache_shape,
    const std::vector<int64_t>& swap_block_ids_gpu,
    int rank,
    int mode) {
  cudaSetDevice(rank);  // used for distributed launch
  assert(cache_gpu_tensors.size() > 0);
  switch (cache_gpu_tensors[0].dtype()) {
    case paddle::DataType::BFLOAT16:
      return SwapCacheImpLayout<paddle::DataType::BFLOAT16>(cache_gpu_tensors,
                                                            cache_cpu_ptrs,
                                                            cache_shape,
                                                            swap_block_ids_gpu,
                                                            mode);
    case paddle::DataType::FLOAT16:
      return SwapCacheImpLayout<paddle::DataType::FLOAT16>(cache_gpu_tensors,
                                                           cache_cpu_ptrs,
                                                           cache_shape,
                                                           swap_block_ids_gpu,
                                                           mode);
    case paddle::DataType::UINT8:
      return SwapCacheImpLayout<paddle::DataType::UINT8>(cache_gpu_tensors,
                                                         cache_cpu_ptrs,
                                                         cache_shape,
                                                         swap_block_ids_gpu,
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
        "swap_block_ids_gpu: std::vector<int64_t>",
        "rank: int",
        "mode: int",
    })
    .SetKernelFn(PD_KERNEL(SwapCacheLayout));
