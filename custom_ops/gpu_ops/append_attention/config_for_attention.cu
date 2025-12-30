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
__global__ void GetMaxLenKernel(const int *seq_lens_decoder,
                                const int *seq_lens_this_time,
                                const int *seq_lens_encoder,
                                int *max_lens,
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

template <int min_chunk_size, uint32_t block_size>
__global__ void config_decode_attn(const int *__restrict__ seq_lens_this_time,
                                   const int *__restrict__ seq_lens_encoder,
                                   const int *__restrict__ seq_lens_decoder,
                                   int *__restrict__ block_indices,
                                   int *__restrict__ num_blocks,
                                   int *__restrict__ chunk_size,
                                   const int bsz,
                                   const int group_size,
                                   const int kv_num_heads,
                                   const int q_tile_size,
                                   const int max_tokens_per_batch,
                                   const int config_gridx) {
  // one block one warp
  const int tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t warp_size = blockDim.x;
  __shared__ int num_block_all_shared[block_size];

  const int lane_id = tid + wid * warp_size;
  int cur_chunk_size = min_chunk_size * (lane_id + 1);

  // calculate num_block_all
  int num_block_all = 0;
  for (int bid = 0; bid < bsz; bid++) {
    if (seq_lens_this_time[bid] <= 0 || seq_lens_encoder[bid] > 0) {
      continue;
    }

    int token_num_cur_batch = seq_lens_this_time[bid];
    int kv_len_cur_batch = seq_lens_decoder[bid] + token_num_cur_batch;
    int q_tile_num = div_up(token_num_cur_batch * group_size, q_tile_size);
    int kv_chunk_num = div_up(kv_len_cur_batch, cur_chunk_size);
    num_block_all += q_tile_num * kv_chunk_num * kv_num_heads;
  }
  num_block_all_shared[lane_id] = num_block_all;
  __syncthreads();

  // search optimal chunk_size
  int chunk_size_best;
  int num_block_all_best;
  if (tid == 0 && wid == 0) {
    if (num_block_all_shared[0] <= config_gridx) {
      chunk_size_best = min_chunk_size;
      num_block_all_best = num_block_all_shared[0];
    } else if (num_block_all_shared[block_size - 1] >= config_gridx) {
      chunk_size_best = min_chunk_size * block_size;
      num_block_all_best = num_block_all_shared[block_size - 1];
      for (int i = block_size - 1; i >= 0; i--) {
        if (num_block_all_shared[i] > num_block_all_best) {
          break;
        }
        chunk_size_best = min_chunk_size * (i + 1);
      }
    } else {
      chunk_size_best = min_chunk_size;
      num_block_all_best = num_block_all_shared[0];
      for (int i = block_size - 1; i >= 0; i--) {
        if (num_block_all_shared[i] > config_gridx) {
          break;
        }
        if (num_block_all_shared[i] > num_block_all_best) {
          num_block_all_best = num_block_all_shared[i];
          chunk_size_best = min_chunk_size * (i + 1);
        }
      }
    }
    num_blocks[0] = num_block_all_best;
    chunk_size[0] = chunk_size_best;
  }

  __syncthreads();
  if (wid == 0) {
    chunk_size_best = __shfl_sync(0xffffffff, chunk_size_best, 0);

    // one block one warp
    int prev_offset = 0;
    // loop on warp tile：[base, base+32)
    for (int base = 0; base < bsz; base += warp_size) {
      const int bid = base + tid;
      int q_tile_num = 0;
      int kv_chunk_num = 0;

      // calculate loop_times for bid
      int num_block_all = 0;
      if (bid < bsz) {
        int token_num_cur_batch = seq_lens_this_time[bid];
        if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
          token_num_cur_batch = 0;
        }
        int kv_len_cur_batch = seq_lens_decoder[bid] + token_num_cur_batch;
        q_tile_num = div_up(token_num_cur_batch * group_size, q_tile_size);
        kv_chunk_num = div_up(kv_len_cur_batch, chunk_size_best);
        num_block_all += q_tile_num * kv_chunk_num * kv_num_heads;
      }

      // prefix sum for each lane, get the start offset in this tile
      // inclusive scan
      int x = num_block_all;
      for (int offset = 1; offset < warp_size; offset <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, offset);
        if (tid >= offset) x += y;
      }
      // exclusive prefix sum
      int bid_offset = x - num_block_all;
      int tile_sum = __shfl_sync(0xffffffff, x, warp_size - 1);

      // write batch_ids and tile_ids_per_batch
      if (bid < bsz && num_block_all > 0) {
        int write_base = prev_offset + bid_offset;
        for (int kv_head_id = 0; kv_head_id < kv_num_heads; kv_head_id++) {
          for (int kv_chunk_id = 0; kv_chunk_id < kv_chunk_num; kv_chunk_id++) {
            for (int q_tile_id = 0; q_tile_id < q_tile_num; q_tile_id++) {
              int idx =
                  write_base * 4 +
                  ((kv_head_id * kv_chunk_num + kv_chunk_id) * q_tile_num +
                   q_tile_id) *
                      4;
              block_indices[idx] = bid;
              block_indices[idx + 1] = kv_head_id;
              block_indices[idx + 2] = kv_chunk_id;
              block_indices[idx + 3] = q_tile_id;
            }
          }
        }
      }
      // for next warp tile
      prev_offset += tile_sum;
    }
  }
}

void ConfigForAttention(
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_this_time,
    paddle::Tensor &block_indices,  // Inplace, shape:[block_num,4], block's
                                    // indices with 4 dimension[batch_idx,
                                    // kv_head_idx, kv_chunk_idx, q_tile_idx]
    paddle::Tensor &num_blocks,     // Inplace
    paddle::Tensor &chunk_size,     // Inplace
    paddle::Tensor &max_len_tensor_cpu,  // Inplace, CPU
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
  // Note (sunxin): Skip capturing the DtoH copy (it's time-consuming); CPU data
  // is only for branching in attention.
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
    const int config_gridx = sm_cout * 2;

    // 选择最优的q_tile_size
    int q_tile_size = 32;
    if (group_size * max_tokens_per_batch <= 16) {
      q_tile_size = 16;
    }
    dim3 blocks(32, 4);
    if (cache_quant_type == "cache_int4_zp") {
      config_decode_attn<256, 128>
          <<<1, blocks, 0, stream>>>(seq_lens_this_time.data<int>(),
                                     seq_lens_encoder.data<int>(),
                                     seq_lens_decoder.data<int>(),
                                     block_indices.data<int>(),
                                     num_blocks.data<int>(),
                                     chunk_size.data<int>(),
                                     bsz,
                                     group_size,
                                     kv_num_heads,
                                     q_tile_size,
                                     max_tokens_per_batch,
                                     config_gridx);
    } else {
      config_decode_attn<128, 128>
          <<<1, blocks, 0, stream>>>(seq_lens_this_time.data<int>(),
                                     seq_lens_encoder.data<int>(),
                                     seq_lens_decoder.data<int>(),
                                     block_indices.data<int>(),
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
    const std::vector<int64_t> &seq_lens_encoder_shape,
    const std::vector<int64_t> &seq_lens_decoder_shape,
    const std::vector<int64_t> &seq_lens_this_time_shape,
    const std::vector<int64_t> &num_blocks_shape,
    const std::vector<int64_t> &chunk_size_shape,
    const std::vector<int64_t> &max_len_tensor_cpu_shape,
    const std::string cache_quant_type,
    const int group_size,
    const int kv_num_heads,
    const int max_tokens_per_batch) {
  return {};
}

std::vector<paddle::DataType> ConfigForAttentionInferDtype(
    const paddle::DataType &seq_lens_encoder_dtype,
    const paddle::DataType &seq_lens_decoder_dtype,
    const paddle::DataType &seq_lens_this_time_dtype,
    const paddle::DataType &num_blocks_dtype,
    const paddle::DataType &chunk_size_dtype,
    const paddle::DataType &max_len_tensor_cpu_dtype,
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
