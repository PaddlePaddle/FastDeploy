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
void SwapCacheImpl(const paddle::Tensor& cache_gpu, // gpu
                   const int64_t& cache_cpu_pointer, // cpu
                   const int64_t& max_block_num_cpu,
                   const std::vector<int64_t>& swap_block_ids_gpu,
                   const std::vector<int64_t>& swap_block_ids_cpu,
                //    const paddle::Tensor& swap_block_ids_dst, // cpu
                //    const paddle::Tensor& swap_block_ids_src, // cpu
                   int mode) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    data_t* cache_gpu_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
    auto* cache_cpu_ptr = reinterpret_cast<data_t*>(cache_cpu_pointer);
    auto cache_shape = cache_gpu.shape();
    // auto* swap_block_ids_dst_ptr = swap_block_ids_dst.data<int32_t>();
    // auto* swap_block_ids_src_ptr = swap_block_ids_src.data<int32_t>();
    // const int swap_block_length = swap_block_ids_dst.shape()[0];
    const int64_t max_block_num_gpu = cache_shape[0];
    const int num_heads = cache_shape[1];
    const int block_size = cache_shape[2];
    const int head_dim = cache_shape[3];
    const int64_t cache_stride = num_heads * block_size * head_dim;
    auto stream = cache_gpu.stream();
    for (int i = 0; i < swap_block_ids_gpu.size(); ++i) {
        int64_t gpu_block_id = swap_block_ids_gpu[i];
        int64_t cpu_block_id = swap_block_ids_cpu[i];
        assert(gpu_block_id >= 0 && gpu_block_id < max_block_num_gpu);
        assert(cpu_block_id >= 0 && cpu_block_id < max_block_num_cpu);
        auto *cache_gpu_ptr_now = cache_gpu_ptr + gpu_block_id * cache_stride;
        auto *cache_cpu_ptr_now = cache_cpu_ptr + cpu_block_id * cache_stride;
        if (mode == 0) { // copy from device to host
            cudaMemcpyAsync(cache_cpu_ptr_now, cache_gpu_ptr_now, cache_stride * sizeof(DataType_), cudaMemcpyDeviceToHost, stream);
            // cudaMemcpy(cache_dst_ptr_now, cache_src_ptr_now, cache_stride * sizeof(DataType_), cudaMemcpyDeviceToHost);
        } else { // copy from host to device
            cudaMemcpyAsync(cache_gpu_ptr_now, cache_cpu_ptr_now, cache_stride * sizeof(DataType_), cudaMemcpyHostToDevice, stream);
            // cudaMemcpy(cache_dst_ptr_now, cache_src_ptr_now, cache_stride * sizeof(DataType_), cudaMemcpyHostToDevice);
        }
    }
    cudaStreamSynchronize(stream);
}

void SwapCache(const paddle::Tensor& cache_gpu, // gpu
               int64_t cache_cpu_ptr, // cpu memory pointer
               int64_t max_block_num_cpu, // cpu max block num
               const std::vector<int64_t>& swap_block_ids_gpu,
               const std::vector<int64_t>& swap_block_ids_cpu,
               int rank,
               int mode) {
    cudaSetDevice(rank); // used for distributed launch
    switch (cache_gpu.dtype()) {
        case paddle::DataType::BFLOAT16:
            return SwapCacheImpl<paddle::DataType::BFLOAT16>(
                        cache_gpu,
                        cache_cpu_ptr,
                        max_block_num_cpu,
                        swap_block_ids_gpu,
                        swap_block_ids_cpu,
                        mode);
        case paddle::DataType::FLOAT16:
            return SwapCacheImpl<paddle::DataType::FLOAT16>(
                        cache_gpu,
                        cache_cpu_ptr,
                        max_block_num_cpu,
                        swap_block_ids_gpu,
                        swap_block_ids_cpu,
                        mode);
        case paddle::DataType::UINT8:
            return SwapCacheImpl<paddle::DataType::UINT8>(
                        cache_gpu,
                        cache_cpu_ptr,
                        max_block_num_cpu,
                        swap_block_ids_gpu,
                        swap_block_ids_cpu,
                        mode);
        default:
            PD_THROW("Unsupported data type.");
    }
}

PD_BUILD_STATIC_OP(swap_cache)
    .Inputs({"cache_gpu",})
    .Attrs({"cache_cpu_ptr: int64_t",
            "max_block_num_cpu: int64_t",
            "swap_block_ids_gpu: std::vector<int64_t>",
            "swap_block_ids_cpu: std::vector<int64_t>",
            "rank: int",
            "mode: int",})
    .Outputs({"cache_dst_out"})
    .SetInplaceMap({{"cache_gpu", "cache_dst_out"}})
    .SetKernelFn(PD_KERNEL(SwapCache));
