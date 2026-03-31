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
 * This file implements high-performance operators for KV cache transfer
 * between GPU and CPU pinned memory:
 *
 * swap_cache_per_layer:       Single-layer transfer (sync, backward compatible)
 * swap_cache_per_layer_async: Single-layer transfer (async, no cudaStreamSync)
 *
 * Key optimizations vs original:
 * 1. Consecutive block fast path: detects consecutive block ID runs and uses
 *    cudaMemcpyAsync instead of warp kernel (avoids kernel launch overhead).
 * 2. Async variant: swap_cache_per_layer_async omits cudaStreamSynchronize,
 *    enabling true async pipelining when called on a dedicated cupy stream.
 * 3. Warp-level PTX: non-temporal load/store for non-consecutive blocks to
 *    avoid L2 cache pollution.
 */

#include "cuda_multiprocess.h"
#include "helper.h"
#include "paddle/extension.h"

#include <cstdint>
#include <vector>

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
 * @param lane_id Thread lane ID within the warp (0-WARP_SIZE-1)
 * @param src_addr Source memory address
 * @param dst_addr Destination memory address
 * @param item_size_bytes Size of the item in bytes (must be 8-byte aligned)
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
// Kernels
// ============================================================================

/**
 * @brief CUDA kernel for single-layer KV cache transfer (non-consecutive path).
 *
 * Each warp processes one block using warp-level parallel PTX loads/stores.
 * Used only when block IDs are non-consecutive; consecutive runs are handled
 * by cudaMemcpyAsync in the host-side fast path.
 *
 * @tparam D2H true = Device->Host (evict), false = Host->Device (load)
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

  if (warp_id >= num_blocks) return;

  int64_t src_block_id = src_block_ids[warp_id];
  int64_t dst_block_id = dst_block_ids[warp_id];

  const char* src_now =
      static_cast<const char*>(src_ptr) + src_block_id * item_size_bytes;
  char* dst_now = static_cast<char*>(dst_ptr) + dst_block_id * item_size_bytes;

  transfer_item_warp(lane_id, src_now, dst_now, item_size_bytes);
}

// ============================================================================
// Helper: Consecutive Block Fast Path
// ============================================================================

/**
 * @brief Transfer a single layer using consecutive-block detection.
 *
 * Scans src/dst block ID pairs for consecutive runs. For each run, issues
 * a single cudaMemcpyAsync (like swap_cache_all_layers). Non-consecutive
 * blocks are batched and handled by the warp kernel.
 *
 * @tparam D2H true = Device->Host, false = Host->Device
 * @param src_ptr     Source base pointer (GPU or CPU depending on D2H)
 * @param dst_ptr     Destination base pointer
 * @param src_block_ids Host vector of source block IDs
 * @param dst_block_ids Host vector of destination block IDs
 * @param num_blocks  Number of blocks to transfer
 * @param item_size_bytes Bytes per block
 * @param stream      CUDA stream
 */
template <bool D2H>
void TransferSingleLayerWithFastPath(const void* src_ptr,
                                     void* dst_ptr,
                                     const std::vector<int64_t>& src_block_ids,
                                     const std::vector<int64_t>& dst_block_ids,
                                     int64_t num_blocks,
                                     int64_t item_size_bytes,
                                     cudaStream_t stream) {
  // --- Pass 1: handle consecutive runs with cudaMemcpyAsync ---
  // Collect indices of non-consecutive blocks for the kernel fallback.
  std::vector<int64_t> nc_src, nc_dst;
  const cudaMemcpyKind kind =
      D2H ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

  int64_t run_start = 0;
  for (int64_t i = 1; i <= num_blocks; ++i) {
    bool end_of_run = (i == num_blocks) ||
                      (src_block_ids[i] != src_block_ids[i - 1] + 1) ||
                      (dst_block_ids[i] != dst_block_ids[i - 1] + 1);
    if (!end_of_run) continue;

    int64_t run_len = i - run_start;
    if (run_len > 1) {
      // Consecutive run: merge into a single cudaMemcpyAsync
      const char* src_run = static_cast<const char*>(src_ptr) +
                            src_block_ids[run_start] * item_size_bytes;
      char* dst_run = static_cast<char*>(dst_ptr) +
                      dst_block_ids[run_start] * item_size_bytes;
      checkCudaErrors(cudaMemcpyAsync(
          dst_run, src_run, run_len * item_size_bytes, kind, stream));
    } else {
      // Single non-consecutive block: defer to warp kernel
      nc_src.push_back(src_block_ids[run_start]);
      nc_dst.push_back(dst_block_ids[run_start]);
    }
    run_start = i;
  }

  // --- Pass 2: warp kernel for remaining non-consecutive blocks ---
  if (!nc_src.empty()) {
    int64_t nc_count = static_cast<int64_t>(nc_src.size());
    int64_t *d_src, *d_dst;
    checkCudaErrors(
        cudaMallocAsync(&d_src, nc_count * sizeof(int64_t), stream));
    checkCudaErrors(
        cudaMallocAsync(&d_dst, nc_count * sizeof(int64_t), stream));
    checkCudaErrors(cudaMemcpyAsync(d_src,
                                    nc_src.data(),
                                    nc_count * sizeof(int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(d_dst,
                                    nc_dst.data(),
                                    nc_count * sizeof(int64_t),
                                    cudaMemcpyHostToDevice,
                                    stream));

    constexpr int kWarpsPerBlock = 4;
    const int threads_per_block = kWarpsPerBlock * WARP_SIZE;
    const int grid =
        (static_cast<int>(nc_count) + kWarpsPerBlock - 1) / kWarpsPerBlock;

    swap_cache_per_layer_kernel<D2H><<<grid, threads_per_block, 0, stream>>>(
        src_ptr, dst_ptr, d_src, d_dst, nc_count, item_size_bytes);

    checkCudaErrors(cudaFreeAsync(d_src, stream));
    checkCudaErrors(cudaFreeAsync(d_dst, stream));
  }
}

// ============================================================================
// Implementation: Single Layer
// ============================================================================

/**
 * @brief Core implementation for single-layer KV cache transfer.
 *
 * @param do_sync  If true, calls cudaStreamSynchronize at end (sync op).
 *                 Set to false for the async variant.
 */
template <paddle::DataType D, bool D2H>
void SwapCachePerLayerImpl(const paddle::Tensor& cache_gpu,
                           int64_t cache_cpu_ptr,
                           int64_t max_block_num_cpu,
                           const std::vector<int64_t>& swap_block_ids_gpu,
                           const std::vector<int64_t>& swap_block_ids_cpu,
                           cudaStream_t stream,
                           bool do_sync) {
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

  // Validate block IDs
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

  // D2H: src=GPU, dst=CPU; H2D: src=CPU, dst=GPU
  const auto& src_block_ids = D2H ? swap_block_ids_gpu : swap_block_ids_cpu;
  const auto& dst_block_ids = D2H ? swap_block_ids_cpu : swap_block_ids_gpu;

  const void* src_ptr;
  void* dst_ptr;
  if (D2H) {
    src_ptr = cache_gpu.data<data_t>();
    dst_ptr = reinterpret_cast<void*>(cache_cpu_ptr);
  } else {
    src_ptr = reinterpret_cast<const void*>(cache_cpu_ptr);
    dst_ptr = const_cast<data_t*>(cache_gpu.data<data_t>());
  }

  TransferSingleLayerWithFastPath<D2H>(src_ptr,
                                       dst_ptr,
                                       src_block_ids,
                                       dst_block_ids,
                                       num_blocks,
                                       item_size_bytes,
                                       stream);

  if (do_sync) {
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
}

// ============================================================================
// Operator Registration
// ============================================================================
// Operator Entry Points
// ============================================================================

// Helper macro to dispatch dtype and direction for SwapCachePerLayerImpl
#define DISPATCH_PER_LAYER(DTYPE, MODE, DO_SYNC, ...)                         \
  switch (DTYPE) {                                                            \
    case paddle::DataType::BFLOAT16:                                          \
      if ((MODE) == 0)                                                        \
        SwapCachePerLayerImpl<paddle::DataType::BFLOAT16, true>(__VA_ARGS__,  \
                                                                DO_SYNC);     \
      else                                                                    \
        SwapCachePerLayerImpl<paddle::DataType::BFLOAT16, false>(__VA_ARGS__, \
                                                                 DO_SYNC);    \
      break;                                                                  \
    case paddle::DataType::FLOAT16:                                           \
      if ((MODE) == 0)                                                        \
        SwapCachePerLayerImpl<paddle::DataType::FLOAT16, true>(__VA_ARGS__,   \
                                                               DO_SYNC);      \
      else                                                                    \
        SwapCachePerLayerImpl<paddle::DataType::FLOAT16, false>(__VA_ARGS__,  \
                                                                DO_SYNC);     \
      break;                                                                  \
    case paddle::DataType::UINT8:                                             \
      if ((MODE) == 0)                                                        \
        SwapCachePerLayerImpl<paddle::DataType::UINT8, true>(__VA_ARGS__,     \
                                                             DO_SYNC);        \
      else                                                                    \
        SwapCachePerLayerImpl<paddle::DataType::UINT8, false>(__VA_ARGS__,    \
                                                              DO_SYNC);       \
      break;                                                                  \
    default:                                                                  \
      PD_THROW("Unsupported data type for swap_cache_per_layer.");            \
  }

/**
 * @brief Single-layer KV cache swap (synchronous, backward compatible).
 */
void SwapCachePerLayer(const paddle::Tensor& cache_gpu,
                       int64_t cache_cpu_ptr,
                       int64_t max_block_num_cpu,
                       const std::vector<int64_t>& swap_block_ids_gpu,
                       const std::vector<int64_t>& swap_block_ids_cpu,
                       int rank,
                       int mode) {
  auto stream = cache_gpu.stream();
  DISPATCH_PER_LAYER(cache_gpu.dtype(),
                     mode,
                     /*do_sync=*/true,
                     cache_gpu,
                     cache_cpu_ptr,
                     max_block_num_cpu,
                     swap_block_ids_gpu,
                     swap_block_ids_cpu,
                     stream);
}

/**
 * @brief Single-layer KV cache swap (async, no cudaStreamSynchronize).
 *
 * Designed for use inside a cupy stream context. Completion is tracked
 * by the caller via CUDA events (record_input_stream_event).
 */
void SwapCachePerLayerAsync(const paddle::Tensor& cache_gpu,
                            int64_t cache_cpu_ptr,
                            int64_t max_block_num_cpu,
                            const std::vector<int64_t>& swap_block_ids_gpu,
                            const std::vector<int64_t>& swap_block_ids_cpu,
                            int rank,
                            int mode) {
  auto stream = cache_gpu.stream();
  DISPATCH_PER_LAYER(cache_gpu.dtype(),
                     mode,
                     /*do_sync=*/false,
                     cache_gpu,
                     cache_cpu_ptr,
                     max_block_num_cpu,
                     swap_block_ids_gpu,
                     swap_block_ids_cpu,
                     stream);
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

PD_BUILD_STATIC_OP(swap_cache_per_layer_async)
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
    .SetKernelFn(PD_KERNEL(SwapCachePerLayerAsync));
