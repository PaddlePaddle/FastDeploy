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
#include "cute/tensor.hpp"
#include "helper.h"
#include "paddle/extension.h"
#ifndef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/core/memory/memcpy.h"
#endif
#include "utils.cuh"

template <int THREADBLOCK_SIZE>
__global__ void GetMaxLenKernel(const int* seq_lens_decoder,
                                const int* seq_lens_this_time,
                                const int* seq_lens_encoder,
                                int* max_lens,
                                const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_time_this_thread = 0;
  int max_len_encoder_this_thread = 0;
  int max_len_decoder_this_thread = 0;
  int max_len_this_thread = 0;
  int max_just_dec_len_this_thread = 0;
  int max_len_kv_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    const int seq_len_this_time = seq_lens_this_time[i];
    const int seq_len_decoder = seq_lens_decoder[i];
    max_len_this_time_this_thread =
        max(seq_len_this_time, max_len_this_time_this_thread);
    max_len_encoder_this_thread =
        max(seq_lens_encoder[i], max_len_encoder_this_thread);
    max_len_decoder_this_thread =
        max(seq_len_decoder, max_len_decoder_this_thread);
    if (seq_len_this_time <= 0) continue;
    const int max_just_dec_len_now =
        seq_lens_encoder[i] > 0 ? 0 : seq_len_decoder;
    max_len_this_thread =
        max(seq_len_decoder + seq_len_this_time, max_len_this_thread);
    max_just_dec_len_this_thread =
        max(max_just_dec_len_this_thread, max_just_dec_len_now);

    if (seq_len_decoder == 0) continue;
    max_len_kv_this_thread =
        max(seq_len_this_time + seq_len_decoder, max_len_kv_this_thread);
  }
  int total_max_len_this_time =
      BlockReduce(temp_storage)
          .Reduce(max_len_this_time_this_thread, MaxOp<int>());
  int total_max_len_encoder =
      BlockReduce(temp_storage)
          .Reduce(max_len_encoder_this_thread, MaxOp<int>());
  int total_max_len_decoder =
      BlockReduce(temp_storage)
          .Reduce(max_len_decoder_this_thread, MaxOp<int>());
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  int total_just_dec = BlockReduce(temp_storage)
                           .Reduce(max_just_dec_len_this_thread, MaxOp<int>());
  int total_max_len_kv =
      BlockReduce(temp_storage).Reduce(max_len_kv_this_thread, MaxOp<int>());
  if (tid == 0) {
    max_lens[0] = total_max_len_this_time;
    max_lens[1] = total_max_len_encoder;
    max_lens[2] = total_max_len_decoder;
    max_lens[3] = total;
    max_lens[4] = total_just_dec;
    max_lens[5] = total_max_len_kv;
  }
}

template <int min_chunk_size,
          int chunk_step,
          uint32_t block_size,
          int max_chunk_size>
__global__ void config_decode_attn(const int* __restrict__ seq_lens_this_time,
                                   const int* __restrict__ seq_lens_encoder,
                                   const int* __restrict__ seq_lens_decoder,
                                   int4* __restrict__ block_indices,
                                   int* __restrict__ num_blocks,
                                   int* __restrict__ chunk_size,
                                   const int bsz,
                                   const int group_size,
                                   const int kv_num_heads,
                                   const int q_tile_size,
                                   const int max_tokens_per_batch,
                                   const int config_gridx) {
  const int tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t warp_size = blockDim.x;
  __shared__ int num_block_all_shared[block_size];
  __shared__ int chunk_size_res[1];

  const int lane_id = tid + wid * warp_size;

  // Merged Step 1+2: single bsz loop computing both Scheme E metrics and
  // split-KV block counts per lane. Avoids redundant seq_lens reads and
  // shared intermediate values (token_num, kv_len, q_tile_num).
  const int target_blocks = config_gridx / 3;  // sm_count * 3
  // Search chunk_size from 512 with step 128: {512, 640, 768, ...}

  const int cur_chunk_size =
      min(min_chunk_size + lane_id * chunk_step, max_chunk_size);
  int num_block_no_chunk = 0;
  int max_kv_len_no_chunk = 0;
  int num_block_all = 0;
  for (int bid = 0; bid < bsz; bid++) {
    if (seq_lens_this_time[bid] <= 0 || seq_lens_encoder[bid] > 0) {
      continue;
    }
    const int token_num_cur_batch = seq_lens_this_time[bid];
    const int kv_len_cur_batch = seq_lens_decoder[bid] + token_num_cur_batch;
    const int q_tile_num =
        div_up(token_num_cur_batch * group_size, q_tile_size);
    num_block_no_chunk += q_tile_num * kv_num_heads;
    max_kv_len_no_chunk = max(max_kv_len_no_chunk, kv_len_cur_batch);
    const int kv_chunk_num = div_up(kv_len_cur_batch, cur_chunk_size);
    num_block_all += q_tile_num * kv_chunk_num * kv_num_heads;
  }
  num_block_all_shared[lane_id] = num_block_all;
  __syncthreads();

  // Step 3: find best chunk_size, then decide Scheme E vs split-KV
  if (tid == 0 && wid == 0) {
    // Strategy:
    //   1. Must fill target_blocks (2*sm_count) to maintain SM concurrency
    //   2. Among valid choices, prefer minimum per-SM max KV traffic
    //      (= waves * chunk_size, since kernel time = slowest SM)
    //   3. Within 5% of minimum KV traffic, prefer larger chunk_size
    int chunk_size_best = min_chunk_size;
    int num_block_all_best = num_block_all_shared[0];
    // Step 1: find minimum kv_traffic among chunk_sizes that fill SMs
    int64_t kv_traffic_min = INT64_MAX;
    for (int i = 0; i < static_cast<int>(block_size); i++) {
      const int nb = num_block_all_shared[i];
      if (nb < target_blocks) continue;
      const int cs = min(min_chunk_size + i * chunk_step, max_chunk_size);
      const int w = div_up(nb, target_blocks);
      const int64_t kv_traffic = static_cast<int64_t>(w) * cs;
      if (kv_traffic < kv_traffic_min) {
        kv_traffic_min = kv_traffic;
      }
    }
    // Step 2: if no chunk_size fills SMs, fall back to smallest
    if (kv_traffic_min == INT64_MAX) {
      chunk_size_best = min_chunk_size;
      num_block_all_best = num_block_all_shared[0];
    } else {
      // Step 3: scan from largest chunk_size downward; accept the first
      // one that fills SMs AND has kv_traffic within 20% of minimum
      for (int i = block_size - 1; i >= 0; i--) {
        const int nb = num_block_all_shared[i];
        if (nb < target_blocks) continue;
        const int cs = min(min_chunk_size + i * chunk_step, max_chunk_size);
        const int w = div_up(nb, target_blocks);
        const int64_t kv_traffic = static_cast<int64_t>(w) * cs;
        if (kv_traffic <= kv_traffic_min + kv_traffic_min / 4) {
          chunk_size_best = cs;
          num_block_all_best = nb;
          break;
        }
      }
    }

    // Decide Scheme E: prefer when blocks fill SMs AND estimated latency
    // is no worse than split-KV.
    //   Scheme E: waves_E * max_kv_len (few heavy blocks)
    //   Split-KV: waves_split * chunk_size_best (many light blocks)
    // When no splitting is needed (num_block_all_best == num_block_no_chunk),
    // Scheme E is strictly better (saves merge overhead).
    bool use_scheme_e = false;
    if (num_block_no_chunk >= target_blocks) {
      if (num_block_all_best == num_block_no_chunk) {
        use_scheme_e = true;
      } else {
        // target_blocks = sm_count * 3 ≈ CTAs per wave (sm_count × occupancy).
        // Using target_blocks as denominator correctly accounts for occupancy
        // in wave count estimation.
        const int waves_e = div_up(num_block_no_chunk, target_blocks);
        const int waves_split = div_up(num_block_all_best, target_blocks);
        use_scheme_e = (static_cast<int64_t>(waves_e) * max_kv_len_no_chunk <=
                        static_cast<int64_t>(waves_split) * chunk_size_best);
      }
    }

    if (use_scheme_e) {
      num_blocks[0] = num_block_no_chunk;
      chunk_size[0] = INT_MAX;
      chunk_size_res[0] = INT_MAX;
    } else {
      num_blocks[0] = num_block_all_best;
      chunk_size[0] = chunk_size_best;
      chunk_size_res[0] = chunk_size_best;
    }
  }

  __syncthreads();
  if (wid == 0) {
    const int chunk_size_final = chunk_size_res[0];

    int prev_offset = 0;
    for (int base = 0; base < bsz; base += warp_size) {
      const int bid = base + tid;
      int num_block_cur = 0;
      int q_tile_num = 0;
      int kv_chunk_num = 0;

      if (bid < bsz) {
        int token_num_cur_batch = seq_lens_this_time[bid];
        if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
          token_num_cur_batch = 0;
        }
        q_tile_num = div_up(token_num_cur_batch * group_size, q_tile_size);
        const int kv_len_cur_batch =
            seq_lens_decoder[bid] + token_num_cur_batch;
        kv_chunk_num = div_up(kv_len_cur_batch, chunk_size_final);
        num_block_cur = q_tile_num * kv_chunk_num * kv_num_heads;
      }

      // inclusive prefix sum
      int x = num_block_cur;
      for (int offset = 1; offset < warp_size; offset <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, offset);
        if (tid >= offset) x += y;
      }
      int bid_offset = x - num_block_cur;
      int tile_sum = __shfl_sync(0xffffffff, x, warp_size - 1);

      // Write block_indices using int4 vectorized stores.
      // Each entry is exactly 4 ints (bid, kv_head_id, kv_chunk_id, q_tile_id),
      // matching int4 layout. This reduces 4 scalar stores to 1 vector store.
      if (bid < bsz && num_block_cur > 0) {
        int4* write_ptr = block_indices + prev_offset + bid_offset;
        int flat_idx = 0;
        const int kv_chunk_num_x_q_tile_num = kv_chunk_num * q_tile_num;
#pragma unroll 2
        for (int kv_head_id = 0; kv_head_id < kv_num_heads; kv_head_id++) {
          const int head_base = kv_head_id * kv_chunk_num_x_q_tile_num;
#pragma unroll 2
          for (int kv_chunk_id = 0; kv_chunk_id < kv_chunk_num; kv_chunk_id++) {
            const int chunk_base = head_base + kv_chunk_id * q_tile_num;
#pragma unroll
            for (int q_tile_id = 0; q_tile_id < q_tile_num; q_tile_id++) {
              write_ptr[flat_idx] =
                  make_int4(bid, kv_head_id, kv_chunk_id, q_tile_id);
              flat_idx++;
            }
          }
        }
      }
      prev_offset += tile_sum;
    }
  }
}

void ConfigForAttention(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    paddle::Tensor& block_indices,  // Inplace, shape:[block_num,4], block's
                                    // indices with 4 dimension[batch_idx,
                                    // kv_head_idx, kv_chunk_idx, q_tile_idx]
    paddle::Tensor& num_blocks,     // Inplace
    paddle::Tensor& chunk_size,     // Inplace
    paddle::Tensor& max_len_tensor_cpu,  // Inplace, CPU
    const std::string cache_quant_type,
    const int group_size,
    const int kv_num_heads,
    const int max_tokens_per_batch) {
  auto stream = seq_lens_encoder.stream();
  int bsz = seq_lens_this_time.shape()[0];

  paddle::Tensor max_len_tensor_gpu =
      GetEmptyTensor({max_len_tensor_cpu.shape()[0]},
                     paddle::DataType::INT32,
                     seq_lens_this_time.place());

  GetMaxLenKernel<1024><<<1, 1024, 0, stream>>>(seq_lens_decoder.data<int>(),
                                                seq_lens_this_time.data<int>(),
                                                seq_lens_encoder.data<int>(),
                                                max_len_tensor_gpu.data<int>(),
                                                bsz);
#ifndef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
  if (!phi::backends::gpu::IsCUDAGraphCapturing())
#endif
    max_len_tensor_cpu.copy_(
        max_len_tensor_gpu, max_len_tensor_cpu.place(), false);
  auto max_len_cpu_ptr = max_len_tensor_cpu.data<int>();
  int max_just_dec_len_this_time = max_len_cpu_ptr[4];

  const uint32_t block_indices_ele_num = block_indices.size();

  // decoder
  if (max_just_dec_len_this_time > 0) {
    CUDA_CHECK(cudaMemsetAsync(block_indices.data<int>(),
                               0,
                               block_indices_ele_num * sizeof(int32_t),
                               stream));
    CUDA_CHECK(
        cudaMemsetAsync(num_blocks.data<int>(), 0, sizeof(int32_t), stream));
    CUDA_CHECK(
        cudaMemsetAsync(chunk_size.data<int>(), 0, sizeof(int32_t), stream));

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int sm_cout;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_cout, cudaDevAttrMultiProcessorCount, device));
    const int config_gridx = sm_cout * 6;

    const int q_tile_size = 16;
    dim3 blocks(32, 4);
    // Cast block_indices to int4* for vectorized stores.
    // Each block_indices entry is 4 ints = 16 bytes = sizeof(int4),
    // and block_num * 4 ints = block_num int4s, so the reinterpret is valid.
    int4* block_indices_i4 = reinterpret_cast<int4*>(block_indices.data<int>());
    if (cache_quant_type == "cache_int4_zp") {
      config_decode_attn<512, 256, 128, 32768>
          <<<1, blocks, 0, stream>>>(seq_lens_this_time.data<int>(),
                                     seq_lens_encoder.data<int>(),
                                     seq_lens_decoder.data<int>(),
                                     block_indices_i4,
                                     num_blocks.data<int>(),
                                     chunk_size.data<int>(),
                                     bsz,
                                     group_size,
                                     kv_num_heads,
                                     q_tile_size,
                                     max_tokens_per_batch,
                                     config_gridx);
    } else {
      config_decode_attn<512, 128, 128, 16384>
          <<<1, blocks, 0, stream>>>(seq_lens_this_time.data<int>(),
                                     seq_lens_encoder.data<int>(),
                                     seq_lens_decoder.data<int>(),
                                     block_indices_i4,
                                     num_blocks.data<int>(),
                                     chunk_size.data<int>(),
                                     bsz,
                                     group_size,
                                     kv_num_heads,
                                     q_tile_size,
                                     max_tokens_per_batch,
                                     config_gridx);
    }
  }
}

std::vector<std::vector<int64_t>> ConfigForAttentionInferShape(
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& num_blocks_shape,
    const std::vector<int64_t>& chunk_size_shape,
    const std::vector<int64_t>& max_len_tensor_cpu_shape,
    const std::string cache_quant_type,
    const int group_size,
    const int kv_num_heads,
    const int max_tokens_per_batch) {
  return {};
}

std::vector<paddle::DataType> ConfigForAttentionInferDtype(
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& num_blocks_dtype,
    const paddle::DataType& chunk_size_dtype,
    const paddle::DataType& max_len_tensor_cpu_dtype,
    const std::string cache_quant_type,
    const int group_size,
    const int kv_num_heads,
    const int max_tokens_per_batch) {
  return {};
}

PD_BUILD_STATIC_OP(config_for_attention)
    .Inputs({
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "block_indices",
        "num_blocks",
        "chunk_size",
        "max_len_tensor_cpu",
    })
    .Outputs({

    })
    .Attrs({"cache_quant_type: std::string",
            "group_size: int",
            "kv_num_heads: int",
            "max_tokens_per_batch: int"})
    .SetKernelFn(PD_KERNEL(ConfigForAttention))
    .SetInferShapeFn(PD_INFER_SHAPE(ConfigForAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ConfigForAttentionInferDtype));
