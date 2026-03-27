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

/**
 * @file swap_cache_optimized.cu
 * @brief Optimized KV cache swap operators using warp-level parallelism.
 *
 * This file implements two high-performance operators for KV cache transfer
 * between GPU and CPU pinned memory:
 *
 * 1. swap_cache_per_layer: Single-layer transfer with warp-level parallelism
 * 2. swap_cache_all_layers_batch: Multi-layer batch transfer with single kernel
 * launch
 *
 * Key optimizations (inspired by sglang):
 * - Warp-level parallel data transfer using 32 threads per warp
 * - PTX inline assembly for non-cacheable loads and cache-globing stores
 * - Single kernel launch for all blocks (reduces launch overhead)
 * - Layer base table for non-contiguous layer memory
 */

#include "cuda_multiprocess.h"
#include "helper.h"
#include "paddle/extension.h"

#include <cstdint>

// ============================================================================
// Device Functions: Warp-Level Parallel Transfer
// ============================================================================

/**
 * @brief Warp-level parallel data transfer function.
 *
 * Uses PTX inline assembly for optimized memory access:
 * - ld.global.nc.b64: Non-cacheable load (avoids L2 cache pollution)
 * - st.global.cg.b64: Cache-globing store (optimizes write performance)
 *
 * @param lane_id Thread lane ID within the warp (0-31)
 * @param src_addr Source memory address
 * @param dst_addr Destination memory address
 * @param item_size_bytes Size of the item to transfer in bytes (must be 8-byte
 * aligned)
 */
__device__ __forceinline__ void transfer_item_warp(int32_t lane_id,
                                                   const void* src_addr,
                                                   void* dst_addr,
                                                   int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
#ifdef PADDLE_WITH_HIP
    // ROCm/HIP path using built-in nontemporal operations
    tmp = __builtin_nontemporal_load(src + j);
    __builtin_nontemporal_store(tmp, dst + j);
#else
    // NVIDIA CUDA path using PTX inline assembly
    asm volatile("ld.global.nc.b64 %0,[%1];"
                 : "=l"(tmp)
                 : "l"(src + j)
                 : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp)
                 : "memory");
#endif
  }
}

// ============================================================================
// Kernel: Single Layer Transfer
// ============================================================================

/**
 * @brief CUDA kernel for single-layer KV cache transfer.
 *
 * Each warp processes one block, transferring the entire block data
 * using warp-level parallel loads and stores.
 *
 * @tparam D2H Transfer direction: true for Device->Host, false for Host->Device
 * @param src_ptr Source memory base pointer (GPU or CPU)
 * @param dst_ptr Destination memory base pointer (GPU or CPU)
 * @param src_block_ids Array of source block IDs
 * @param dst_block_ids Array of destination block IDs
 * @param num_blocks Number of blocks to transfer
 * @param item_size_bytes Size of each block in bytes
 */
template <bool D2H>
__global__ void swap_cache_per_layer_kernel(
    const void* __restrict__ src_ptr,
    void* __restrict__ dst_ptr,
    const int64_t* __restrict__ src_block_ids,
    const int64_t* __restrict__ dst_block_ids,
    int64_t num_blocks,
    int64_t item_size_bytes) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  // Each warp processes one block
  if (warp_id >= num_blocks) return;

  int64_t src_block_id = src_block_ids[warp_id];
  int64_t dst_block_id = dst_block_ids[warp_id];

  const char* src_now =
      static_cast<const char*>(src_ptr) + src_block_id * item_size_bytes;
  char* dst_now = static_cast<char*>(dst_ptr) + dst_block_id * item_size_bytes;

  transfer_item_warp(lane_id, src_now, dst_now, item_size_bytes);
}

// ============================================================================
// Kernel: Multi-Layer Batch Transfer
// ============================================================================

/**
 * @brief CUDA kernel for multi-layer batch KV cache transfer.
 *
 * Uses layer base table to support non-contiguous layer memory.
 * Single kernel launch processes all layers and all blocks.
 *
 * @tparam D2H Transfer direction: true for Device->Host, false for Host->Device
 * @param src_layer_tbl Layer base table for source memory (array of pointers)
 * @param dst_layer_tbl Layer base table for destination memory (array of
 * pointers)
 * @param src_block_ids Array of source block IDs
 * @param dst_block_ids Array of destination block IDs
 * @param num_layers Number of layers to transfer
 * @param num_blocks Number of blocks to transfer per layer
 * @param items_per_warp Number of blocks each warp processes
 * @param item_size_bytes Size of each block in bytes
 */
template <bool D2H>
__global__ void swap_cache_all_layers_batch_kernel(
    const uintptr_t* __restrict__ src_layer_tbl,
    const uintptr_t* __restrict__ dst_layer_tbl,
    const int64_t* __restrict__ src_block_ids,
    const int64_t* __restrict__ dst_block_ids,
    int64_t num_layers,
    int64_t num_blocks,
    int64_t items_per_warp,
    int64_t item_size_bytes) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  for (int64_t i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_blocks) break;

    int64_t src_block_id = src_block_ids[item_id];
    int64_t dst_block_id = dst_block_ids[item_id];

    // Process all layers for this block
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      const char* src_ptr =
          reinterpret_cast<const char*>(src_layer_tbl[layer_id]) +
          src_block_id * item_size_bytes;
      char* dst_ptr = reinterpret_cast<char*>(dst_layer_tbl[layer_id]) +
                      dst_block_id * item_size_bytes;

      transfer_item_warp(lane_id, src_ptr, dst_ptr, item_size_bytes);
    }
  }
}

// ============================================================================
// Implementation Functions
// ============================================================================

/**
 * @brief Implementation for single-layer KV cache transfer.
 */
template <paddle::DataType D, bool D2H>
void SwapCachePerLayerImpl(const paddle::Tensor& cache_gpu,
                           int64_t cache_cpu_ptr,
                           int64_t max_block_num_cpu,
                           const std::vector<int64_t>& swap_block_ids_gpu,
                           const std::vector<int64_t>& swap_block_ids_cpu,
                           cudaStream_t stream) {
  typedef typename PDTraits<D>::DataType DataType_;
  typedef typename PDTraits<D>::data_t data_t;

  auto cache_shape = cache_gpu.shape();
  const int64_t max_block_num_gpu = cache_shape[0];
  const int64_t num_heads = cache_shape[1];
  const int64_t block_size = cache_shape[2];
  const int64_t head_dim = cache_shape.size() == 4 ? cache_shape[3] : 1;
  const int64_t item_size_bytes =
      num_heads * block_size * head_dim * sizeof(DataType_);

  const int64_t num_blocks = swap_block_ids_gpu.size();
  if (num_blocks == 0) return;

  // Validate block IDs - always check in both debug and release
  for (size_t i = 0; i < swap_block_ids_gpu.size(); ++i) {
    if (swap_block_ids_gpu[i] < 0 ||
        swap_block_ids_gpu[i] >= max_block_num_gpu) {
      PD_THROW("Invalid swap_block_ids_gpu at index " + std::to_string(i) +
               ": " + std::to_string(swap_block_ids_gpu[i]) +
               " out of range [0, " + std::to_string(max_block_num_gpu) + ")");
    }
    if (swap_block_ids_cpu[i] < 0 ||
        swap_block_ids_cpu[i] >= max_block_num_cpu) {
      PD_THROW("Invalid swap_block_ids_cpu at index " + std::to_string(i) +
               ": " + std::to_string(swap_block_ids_cpu[i]) +
               " out of range [0, " + std::to_string(max_block_num_cpu) + ")");
    }
  }

  // Allocate and copy block IDs to GPU
  int64_t *d_src_block_ids, *d_dst_block_ids;
  checkCudaErrors(
      cudaMallocAsync(&d_src_block_ids, num_blocks * sizeof(int64_t), stream));
  checkCudaErrors(
      cudaMallocAsync(&d_dst_block_ids, num_blocks * sizeof(int64_t), stream));
  checkCudaErrors(cudaMemcpyAsync(d_src_block_ids,
                                  swap_block_ids_gpu.data(),
                                  num_blocks * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
  checkCudaErrors(cudaMemcpyAsync(d_dst_block_ids,
                                  swap_block_ids_cpu.data(),
                                  num_blocks * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

  // Configure kernel launch
  constexpr int kWarpsPerBlock = 4;
  const int threads_per_block = kWarpsPerBlock * WARP_SIZE;
  const int num_blocks_grid =
      (num_blocks + kWarpsPerBlock - 1) / kWarpsPerBlock;

  // Set up source and destination pointers based on transfer direction
  const void* src_ptr;
  void* dst_ptr;

  if (D2H) {
    src_ptr = cache_gpu.data<data_t>();
    dst_ptr = reinterpret_cast<void*>(cache_cpu_ptr);
  } else {
    src_ptr = reinterpret_cast<const void*>(cache_cpu_ptr);
    dst_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
  }

  // Launch kernel
  swap_cache_per_layer_kernel<D2H>
      <<<num_blocks_grid, threads_per_block, 0, stream>>>(src_ptr,
                                                          dst_ptr,
                                                          d_src_block_ids,
                                                          d_dst_block_ids,
                                                          num_blocks,
                                                          item_size_bytes);

  // Clean up
  checkCudaErrors(cudaFreeAsync(d_src_block_ids, stream));
  checkCudaErrors(cudaFreeAsync(d_dst_block_ids, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

/**
 * @brief Implementation for multi-layer batch KV cache transfer.
 */
template <paddle::DataType D, bool D2H>
void SwapCacheAllLayersBatchImpl(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,
    const std::vector<int64_t>& cache_cpu_ptrs,
    int64_t max_block_num_cpu,
    const std::vector<int64_t>& swap_block_ids_gpu,
    const std::vector<int64_t>& swap_block_ids_cpu,
    cudaStream_t stream) {
  typedef typename PDTraits<D>::DataType DataType_;
  typedef typename PDTraits<D>::data_t data_t;

  const int64_t num_layers = cache_gpu_tensors.size();
  if (num_layers == 0) return;

  auto cache_shape = cache_gpu_tensors[0].shape();
  const int64_t max_block_num_gpu = cache_shape[0];
  const int64_t num_heads = cache_shape[1];
  const int64_t block_size = cache_shape[2];
  const int64_t head_dim = cache_shape.size() == 4 ? cache_shape[3] : 1;
  const int64_t item_size_bytes =
      num_heads * block_size * head_dim * sizeof(DataType_);

  const int64_t num_blocks = swap_block_ids_gpu.size();
  if (num_blocks == 0) return;

  // Validate - always check in both debug and release
  if (cache_gpu_tensors.size() != static_cast<size_t>(cache_cpu_ptrs.size())) {
    PD_THROW("Cache tensors and CPU pointers size mismatch: " +
             std::to_string(cache_gpu_tensors.size()) + " vs " +
             std::to_string(cache_cpu_ptrs.size()));
  }
  for (size_t i = 0; i < swap_block_ids_gpu.size(); ++i) {
    if (swap_block_ids_gpu[i] < 0 ||
        swap_block_ids_gpu[i] >= max_block_num_gpu) {
      PD_THROW("Invalid swap_block_ids_gpu at index " + std::to_string(i) +
               ": " + std::to_string(swap_block_ids_gpu[i]) +
               " out of range [0, " + std::to_string(max_block_num_gpu) + ")");
    }
    if (swap_block_ids_cpu[i] < 0 ||
        swap_block_ids_cpu[i] >= max_block_num_cpu) {
      PD_THROW("Invalid swap_block_ids_cpu at index " + std::to_string(i) +
               ": " + std::to_string(swap_block_ids_cpu[i]) +
               " out of range [0, " + std::to_string(max_block_num_cpu) + ")");
    }
  }

  // Build layer base tables
  std::vector<uintptr_t> h_src_layer_tbl(num_layers);
  std::vector<uintptr_t> h_dst_layer_tbl(num_layers);

  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    if (D2H) {
      h_src_layer_tbl[layer_id] = reinterpret_cast<uintptr_t>(
          cache_gpu_tensors[layer_id].data<data_t>());
      h_dst_layer_tbl[layer_id] =
          static_cast<uintptr_t>(cache_cpu_ptrs[layer_id]);
    } else {
      h_src_layer_tbl[layer_id] =
          static_cast<uintptr_t>(cache_cpu_ptrs[layer_id]);
      h_dst_layer_tbl[layer_id] = reinterpret_cast<uintptr_t>(
          cache_gpu_tensors[layer_id].data<data_t>());
    }
  }

  // Allocate and copy to GPU
  uintptr_t *d_src_layer_tbl, *d_dst_layer_tbl;
  int64_t *d_src_block_ids, *d_dst_block_ids;

  checkCudaErrors(cudaMallocAsync(
      &d_src_layer_tbl, num_layers * sizeof(uintptr_t), stream));
  checkCudaErrors(cudaMallocAsync(
      &d_dst_layer_tbl, num_layers * sizeof(uintptr_t), stream));
  checkCudaErrors(cudaMemcpyAsync(d_src_layer_tbl,
                                  h_src_layer_tbl.data(),
                                  num_layers * sizeof(uintptr_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
  checkCudaErrors(cudaMemcpyAsync(d_dst_layer_tbl,
                                  h_dst_layer_tbl.data(),
                                  num_layers * sizeof(uintptr_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

  checkCudaErrors(
      cudaMallocAsync(&d_src_block_ids, num_blocks * sizeof(int64_t), stream));
  checkCudaErrors(
      cudaMallocAsync(&d_dst_block_ids, num_blocks * sizeof(int64_t), stream));
  checkCudaErrors(cudaMemcpyAsync(d_src_block_ids,
                                  swap_block_ids_gpu.data(),
                                  num_blocks * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
  checkCudaErrors(cudaMemcpyAsync(d_dst_block_ids,
                                  swap_block_ids_cpu.data(),
                                  num_blocks * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

  // Configure kernel launch
  constexpr int kWarpsPerBlock = 4;
  const int threads_per_block = kWarpsPerBlock * WARP_SIZE;
  constexpr int kBlockQuota = 16;

  const int64_t items_per_warp =
      (num_blocks + kBlockQuota * kWarpsPerBlock - 1) /
      (kBlockQuota * kWarpsPerBlock);
  const int num_blocks_grid =
      (num_blocks + items_per_warp * kWarpsPerBlock - 1) /
      (items_per_warp * kWarpsPerBlock);

  // Launch kernel
  swap_cache_all_layers_batch_kernel<D2H>
      <<<num_blocks_grid, threads_per_block, 0, stream>>>(d_src_layer_tbl,
                                                          d_dst_layer_tbl,
                                                          d_src_block_ids,
                                                          d_dst_block_ids,
                                                          num_layers,
                                                          num_blocks,
                                                          items_per_warp,
                                                          item_size_bytes);

  // Clean up
  checkCudaErrors(cudaFreeAsync(d_src_layer_tbl, stream));
  checkCudaErrors(cudaFreeAsync(d_dst_layer_tbl, stream));
  checkCudaErrors(cudaFreeAsync(d_src_block_ids, stream));
  checkCudaErrors(cudaFreeAsync(d_dst_block_ids, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

// ============================================================================
// Operator Entry Points
// ============================================================================

/**
 * @brief Single-layer KV cache swap operator.
 *
 * @param cache_gpu GPU tensor for the cache (single layer)
 * @param cache_cpu_ptr CPU pinned memory pointer (int64_t address)
 * @param max_block_num_cpu Maximum number of blocks in CPU memory
 * @param swap_block_ids_gpu Block IDs on GPU to swap
 * @param swap_block_ids_cpu Corresponding block IDs on CPU
 * @param rank GPU device rank
 * @param mode Transfer mode: 0 = Device->Host (evict), 1 = Host->Device (load)
 */
void SwapCachePerLayer(const paddle::Tensor& cache_gpu,
                       int64_t cache_cpu_ptr,
                       int64_t max_block_num_cpu,
                       const std::vector<int64_t>& swap_block_ids_gpu,
                       const std::vector<int64_t>& swap_block_ids_cpu,
                       int rank,
                       int mode) {
  checkCudaErrors(cudaSetDevice(rank));
  auto stream = cache_gpu.stream();

  switch (cache_gpu.dtype()) {
    case paddle::DataType::BFLOAT16:
      if (mode == 0) {
        SwapCachePerLayerImpl<paddle::DataType::BFLOAT16, true>(
            cache_gpu,
            cache_cpu_ptr,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      } else {
        SwapCachePerLayerImpl<paddle::DataType::BFLOAT16, false>(
            cache_gpu,
            cache_cpu_ptr,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      }
      break;
    case paddle::DataType::FLOAT16:
      if (mode == 0) {
        SwapCachePerLayerImpl<paddle::DataType::FLOAT16, true>(
            cache_gpu,
            cache_cpu_ptr,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      } else {
        SwapCachePerLayerImpl<paddle::DataType::FLOAT16, false>(
            cache_gpu,
            cache_cpu_ptr,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      }
      break;
    case paddle::DataType::UINT8:
      if (mode == 0) {
        SwapCachePerLayerImpl<paddle::DataType::UINT8, true>(cache_gpu,
                                                             cache_cpu_ptr,
                                                             max_block_num_cpu,
                                                             swap_block_ids_gpu,
                                                             swap_block_ids_cpu,
                                                             stream);
      } else {
        SwapCachePerLayerImpl<paddle::DataType::UINT8, false>(
            cache_gpu,
            cache_cpu_ptr,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      }
      break;
    default:
      PD_THROW("Unsupported data type for swap_cache_per_layer.");
  }
}

/**
 * @brief Multi-layer batch KV cache swap operator.
 *
 * @param cache_gpu_tensors Vector of GPU tensors (one per layer)
 * @param cache_cpu_ptrs Vector of CPU pinned memory pointers (one per layer)
 * @param max_block_num_cpu Maximum number of blocks in CPU memory
 * @param swap_block_ids_gpu Block IDs on GPU to swap
 * @param swap_block_ids_cpu Corresponding block IDs on CPU
 * @param rank GPU device rank
 * @param mode Transfer mode: 0 = Device->Host (evict), 1 = Host->Device (load)
 */
void SwapCacheAllLayersBatch(
    const std::vector<paddle::Tensor>& cache_gpu_tensors,
    const std::vector<int64_t>& cache_cpu_ptrs,
    int64_t max_block_num_cpu,
    const std::vector<int64_t>& swap_block_ids_gpu,
    const std::vector<int64_t>& swap_block_ids_cpu,
    int rank,
    int mode) {
  if (cache_gpu_tensors.empty()) return;

  checkCudaErrors(cudaSetDevice(rank));
  auto stream = cache_gpu_tensors[0].stream();

  switch (cache_gpu_tensors[0].dtype()) {
    case paddle::DataType::BFLOAT16:
      if (mode == 0) {
        SwapCacheAllLayersBatchImpl<paddle::DataType::BFLOAT16, true>(
            cache_gpu_tensors,
            cache_cpu_ptrs,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      } else {
        SwapCacheAllLayersBatchImpl<paddle::DataType::BFLOAT16, false>(
            cache_gpu_tensors,
            cache_cpu_ptrs,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      }
      break;
    case paddle::DataType::FLOAT16:
      if (mode == 0) {
        SwapCacheAllLayersBatchImpl<paddle::DataType::FLOAT16, true>(
            cache_gpu_tensors,
            cache_cpu_ptrs,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      } else {
        SwapCacheAllLayersBatchImpl<paddle::DataType::FLOAT16, false>(
            cache_gpu_tensors,
            cache_cpu_ptrs,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      }
      break;
    case paddle::DataType::UINT8:
      if (mode == 0) {
        SwapCacheAllLayersBatchImpl<paddle::DataType::UINT8, true>(
            cache_gpu_tensors,
            cache_cpu_ptrs,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      } else {
        SwapCacheAllLayersBatchImpl<paddle::DataType::UINT8, false>(
            cache_gpu_tensors,
            cache_cpu_ptrs,
            max_block_num_cpu,
            swap_block_ids_gpu,
            swap_block_ids_cpu,
            stream);
      }
      break;
    default:
      PD_THROW("Unsupported data type for swap_cache_all_layers_batch.");
  }
}

// ============================================================================
// Operator Registration
// ============================================================================

PD_BUILD_STATIC_OP(swap_cache_per_layer)
    .Inputs({"cache_gpu"})
    .Attrs({
        "cache_cpu_ptr: int64_t",
        "max_block_num_cpu: int64_t",
        "swap_block_ids_gpu: std::vector<int64_t>",
        "swap_block_ids_cpu: std::vector<int64_t>",
        "rank: int",
        "mode: int",
    })
    .Outputs({"cache_dst_out"})
    .SetInplaceMap({{"cache_gpu", "cache_dst_out"}})
    .SetKernelFn(PD_KERNEL(SwapCachePerLayer));

PD_BUILD_STATIC_OP(swap_cache_all_layers_batch)
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
    .SetKernelFn(PD_KERNEL(SwapCacheAllLayersBatch));
