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

// D2H: Each thread block handles ALL layers for one swap block.
// This produces perfectly contiguous host writes (1 block × all layers),
// maximizing write-combining efficiency.
template <typename T>
__global__ void swap_d2h_kernel(T** __restrict__ layer_ptrs,
                                T* __restrict__ cpu_buffer,
                                const int64_t* __restrict__ gpu_block_ids,
                                int n_blocks,
                                int layer_num,
                                int64_t block_stride) {
  int block_idx = blockIdx.x;
  if (block_idx >= n_blocks) return;

  int64_t gpu_block = gpu_block_ids[block_idx];
  int64_t num_vec_per_layer = (block_stride * sizeof(T)) / sizeof(float4);

  T* dst_base = cpu_buffer + (int64_t)block_idx * layer_num * block_stride;

  for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
    const T* src = layer_ptrs[layer_idx] + gpu_block * block_stride;
    float4* dst4 =
        reinterpret_cast<float4*>(dst_base + layer_idx * block_stride);
    const float4* src4 = reinterpret_cast<const float4*>(src);

    for (int64_t i = threadIdx.x; i < num_vec_per_layer; i += blockDim.x) {
      dst4[i] = src4[i];
    }
  }
}

// H2D: scatter from contiguous staging buffer to scattered GPU layer tensors
template <typename T>
__global__ void scatter_blocks_kernel(T** __restrict__ layer_ptrs,
                                      const T* __restrict__ staging,
                                      const int64_t* __restrict__ gpu_block_ids,
                                      int n_blocks,
                                      int layer_num,
                                      int64_t block_stride) {
  int pair_idx = blockIdx.x;
  int block_idx = pair_idx / layer_num;
  int layer_idx = pair_idx % layer_num;

  if (block_idx >= n_blocks) return;

  int64_t gpu_block = gpu_block_ids[block_idx];
  const T* src = staging + (int64_t)block_idx * layer_num * block_stride +
                 layer_idx * block_stride;
  T* dst = layer_ptrs[layer_idx] + gpu_block * block_stride;

  int64_t num_vec = (block_stride * sizeof(T)) / sizeof(float4);
  const float4* src4 = reinterpret_cast<const float4*>(src);
  float4* dst4 = reinterpret_cast<float4*>(dst);

  for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
    dst4[i] = src4[i];
  }
}

static void* g_staging_buffer = nullptr;
static size_t g_staging_buffer_size = 0;
static void* g_device_block_ids = nullptr;
static size_t g_device_block_ids_size = 0;
static void* g_device_layer_ptrs = nullptr;
static size_t g_device_layer_ptrs_size = 0;

static void ensure_staging_buffer(size_t required_size) {
  if (g_staging_buffer_size < required_size) {
    if (g_staging_buffer) cudaFree(g_staging_buffer);
    cudaError_t err = cudaMalloc(&g_staging_buffer, required_size);
    PADDLE_ENFORCE_EQ(
        err,
        cudaSuccess,
        phi::errors::External("cudaMalloc staging buffer failed: %s",
                              cudaGetErrorString(err)));
    g_staging_buffer_size = required_size;
  }
}

static void ensure_device_block_ids(size_t required_size) {
  if (g_device_block_ids_size < required_size) {
    if (g_device_block_ids) cudaFree(g_device_block_ids);
    cudaError_t err = cudaMalloc(&g_device_block_ids, required_size);
    PADDLE_ENFORCE_EQ(
        err,
        cudaSuccess,
        phi::errors::External("cudaMalloc device block_ids failed: %s",
                              cudaGetErrorString(err)));
    g_device_block_ids_size = required_size;
  }
}

static void ensure_device_layer_ptrs(size_t required_size) {
  if (g_device_layer_ptrs_size < required_size) {
    if (g_device_layer_ptrs) cudaFree(g_device_layer_ptrs);
    cudaError_t err = cudaMalloc(&g_device_layer_ptrs, required_size);
    PADDLE_ENFORCE_EQ(
        err,
        cudaSuccess,
        phi::errors::External("cudaMalloc device layer_ptrs failed: %s",
                              cudaGetErrorString(err)));
    g_device_layer_ptrs_size = required_size;
  }
}

static bool is_cpu_block_ids_sequential(
    const std::vector<int64_t>& cpu_block_ids) {
  if (cpu_block_ids.empty()) return true;
  int64_t start = cpu_block_ids[0];
  for (size_t i = 1; i < cpu_block_ids.size(); i++) {
    if (cpu_block_ids[i] != start + static_cast<int64_t>(i)) return false;
  }
  return true;
}

template <paddle::DataType D>
void SwapCacheImpLayout(const std::vector<paddle::Tensor>& cache_gpu_tensors,
                        const int64_t& cache_cpu_pointer,
                        const std::vector<int64_t>& cache_shape,
                        const std::vector<int64_t>& gpu_block_ids,
                        const std::vector<int64_t>& cpu_block_ids,
                        int mode) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int64_t layer_number = cache_gpu_tensors.size();
  int64_t cache_block_stride = 1;
  for (size_t i = 1; i < cache_shape.size(); i++) {
    cache_block_stride *= cache_shape[i];
  }

  const int n_blocks = gpu_block_ids.size();
  if (n_blocks == 0) return;

  auto stream = cache_gpu_tensors[0].stream();
  const size_t block_bytes = cache_block_stride * sizeof(DataType_);
  const size_t total_bytes = (size_t)n_blocks * layer_number * block_bytes;

  bool use_optimized = is_cpu_block_ids_sequential(cpu_block_ids);

  // float4 vectorized kernels require block_bytes to be 16-byte aligned
  // and cache_cpu_base to be 16-byte aligned for correct float4 access.
  if (use_optimized && (block_bytes % sizeof(float4) != 0)) {
    use_optimized = false;
  }
  if (use_optimized) {
    int64_t cpu_start_block = cpu_block_ids[0];
    uintptr_t cpu_base_addr =
        static_cast<uintptr_t>(cache_cpu_pointer) +
        cpu_start_block * layer_number * cache_block_stride * sizeof(DataType_);
    if (cpu_base_addr % sizeof(float4) != 0) {
      use_optimized = false;
    }
  }

  if (use_optimized) {
    ensure_device_block_ids(n_blocks * sizeof(int64_t));
    ensure_device_layer_ptrs(layer_number * sizeof(DataType_*));

    cudaError_t status = cudaMemcpyAsync(g_device_block_ids,
                                         gpu_block_ids.data(),
                                         n_blocks * sizeof(int64_t),
                                         cudaMemcpyHostToDevice,
                                         stream);
    PADDLE_ENFORCE_EQ(
        status,
        cudaSuccess,
        phi::errors::External("cudaMemcpyAsync block_ids H2D failed: %s",
                              cudaGetErrorString(status)));

    std::vector<DataType_*> h_layer_ptrs(layer_number);
    for (int64_t i = 0; i < layer_number; i++) {
      h_layer_ptrs[i] = reinterpret_cast<DataType_*>(
          const_cast<data_t*>(cache_gpu_tensors[i].data<data_t>()));
    }
    status = cudaMemcpyAsync(g_device_layer_ptrs,
                             h_layer_ptrs.data(),
                             layer_number * sizeof(DataType_*),
                             cudaMemcpyHostToDevice,
                             stream);
    PADDLE_ENFORCE_EQ(
        status,
        cudaSuccess,
        phi::errors::External("cudaMemcpyAsync layer_ptrs H2D failed: %s",
                              cudaGetErrorString(status)));

    int64_t cpu_start_block = cpu_block_ids[0];
    auto* cache_cpu_base = reinterpret_cast<DataType_*>(cache_cpu_pointer) +
                           cpu_start_block * layer_number * cache_block_stride;

    int grid_size = n_blocks * layer_number;

    if (mode == 0) {
      // GPU→CPU: direct kernel write to pinned host memory
      // Multi-layer kernel: each block handles all layers for one swap block
      swap_d2h_kernel<DataType_><<<n_blocks, 512, 0, stream>>>(
          reinterpret_cast<DataType_**>(g_device_layer_ptrs),
          cache_cpu_base,
          reinterpret_cast<int64_t*>(g_device_block_ids),
          n_blocks,
          layer_number,
          cache_block_stride);
    } else {
      // CPU→GPU: DMA memcpy to staging then scatter kernel
      ensure_staging_buffer(total_bytes);

      status = cudaMemcpyAsync(g_staging_buffer,
                               cache_cpu_base,
                               total_bytes,
                               cudaMemcpyHostToDevice,
                               stream);
      PADDLE_ENFORCE_EQ(status,
                        cudaSuccess,
                        phi::errors::External("cudaMemcpyAsync H2D failed: %s",
                                              cudaGetErrorString(status)));

      scatter_blocks_kernel<DataType_><<<grid_size, 256, 0, stream>>>(
          reinterpret_cast<DataType_**>(g_device_layer_ptrs),
          reinterpret_cast<const DataType_*>(g_staging_buffer),
          reinterpret_cast<int64_t*>(g_device_block_ids),
          n_blocks,
          layer_number,
          cache_block_stride);
    }
  } else {
    const cudaMemcpyKind copy_kind =
        (mode == 0) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    for (int64_t layer_idx = 0; layer_idx < layer_number; layer_idx++) {
      const paddle::Tensor& cache_gpu = cache_gpu_tensors[layer_idx];
      data_t* cache_gpu_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
      auto* cache_cpu_ptr = reinterpret_cast<data_t*>(cache_cpu_pointer);

      for (int block_idx = 0; block_idx < n_blocks; block_idx++) {
        auto cur_gpu_block_id = gpu_block_ids[block_idx];
        auto cur_cpu_block_id = cpu_block_ids[block_idx];
        auto* cache_gpu_ptr_now =
            cache_gpu_ptr + cur_gpu_block_id * cache_block_stride;
        auto* cache_cpu_ptr_now =
            cache_cpu_ptr +
            cur_cpu_block_id * cache_block_stride * layer_number +
            layer_idx * cache_block_stride;

        cudaError_t status = cudaMemcpyAsync(
            (copy_kind == cudaMemcpyDeviceToHost) ? cache_cpu_ptr_now
                                                  : cache_gpu_ptr_now,
            (copy_kind == cudaMemcpyDeviceToHost) ? cache_gpu_ptr_now
                                                  : cache_cpu_ptr_now,
            block_bytes,
            copy_kind,
            stream);
        PADDLE_ENFORCE_EQ(status,
                          cudaSuccess,
                          phi::errors::External("cudaMemcpyAsync failed: %s",
                                                cudaGetErrorString(status)));
      }
    }
  }

  cudaError_t sync_status = cudaStreamSynchronize(stream);
  PADDLE_ENFORCE_EQ(sync_status,
                    cudaSuccess,
                    phi::errors::External("cudaStreamSynchronize failed: %s",
                                          cudaGetErrorString(sync_status)));
}

void SwapCacheLayout(const std::vector<paddle::Tensor>& cache_gpu_tensors,
                     const int64_t& cache_cpu_ptrs,
                     const std::vector<int64_t>& cache_shape,
                     const std::vector<int64_t>& gpu_block_ids,
                     const std::vector<int64_t>& cpu_block_ids,
                     int rank,
                     int mode) {
  cudaSetDevice(rank);
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
