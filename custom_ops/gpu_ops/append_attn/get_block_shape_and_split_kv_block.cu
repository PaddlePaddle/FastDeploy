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

template <uint32_t config_size>
__global__ void search_chunk_size_for_mla(
    const int *__restrict__ seq_lens_q,
    const int *__restrict__ seq_lens_encoder,
    const int *__restrict__ seq_lens_decoder,
    int *__restrict__ num_blocks_x,
    int *__restrict__ res_chunk_size,
    const int bsz,
    const int set_chunk_size,
    const int block_size,
    const int sm_cout) {
  const uint32_t conf_id = threadIdx.x;
  int gridx = 0;
  if (set_chunk_size > 0 && conf_id == 0) {
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      int seq_len_encoder = seq_lens_encoder[bid];
      int seq_len_decoder = seq_lens_decoder[bid] + seq_len;
      if (seq_len == 0 || seq_len_encoder > 0) continue;

      int loop_times;
      loop_times = cute::ceil_div(seq_len_decoder, set_chunk_size);
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
    *res_chunk_size = set_chunk_size;
  } else if (conf_id < config_size) {
    __shared__ int gridx_shared[config_size];
    // chunk_size is a multiple of 64
    const int chunk_size = block_size << conf_id;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      int seq_len_encoder = seq_lens_encoder[bid];
      int seq_len_decoder = seq_lens_decoder[bid] + seq_len;
      if (seq_len == 0 || seq_len_encoder > 0) continue;

      int loop_times;
      loop_times = cute::ceil_div(seq_len_decoder, chunk_size);
      gridx += loop_times;
    }
    gridx_shared[conf_id] = gridx;
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t res_id = 0;
      uint32_t max_last_wave_block = 0;
      for (uint32_t i = 1; i < config_size; i++) {
        uint32_t last_wave_block = gridx_shared[i] % sm_cout;
        if (last_wave_block >= max_last_wave_block) {
          res_id = i;
          max_last_wave_block = last_wave_block;
        }
      }
      *num_blocks_x = gridx_shared[res_id];
      *res_chunk_size = block_size << res_id;
    }
  }
}

__global__ void split_block_for_mla(const int *__restrict__ seq_lens_q,
                                    const int *__restrict__ seq_lens_encoder,
                                    const int *__restrict__ seq_lens_decoder,
                                    int *__restrict__ batch_ids,
                                    int *__restrict__ tile_ids_per_batch,
                                    const int bsz,
                                    const int chunk_size) {
  if (threadIdx.x == 0) {
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      int seq_len_encoder = seq_lens_encoder[bid];
      int seq_len_decoder = seq_lens_decoder[bid] + seq_len;

      if (seq_len == 0) continue;

      int loop_times;
      loop_times = cute::ceil_div(seq_len_decoder, chunk_size);
      if (seq_len_encoder > 0) {
        loop_times = 0;
      }
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
    }
  }
}

__global__ void split_q_block(const int *__restrict__ seq_lens_q,
                              const int *__restrict__ seq_lens_encoder,
                              int *__restrict__ batch_ids,
                              int *__restrict__ tile_ids_per_batch,
                              int *__restrict__ num_blocks_x,
                              const int bsz,
                              const int num_rows_per_block,
                              const int group_size) {
  // one block one warp
  const int lane_id = threadIdx.x % warpSize;
  int prev_offset = 0;

  // loop on warp tile：[base, base+32)
  for (int base = 0; base < bsz; base += warpSize) {
    const int bid = base + lane_id;

    // calculate loop_times for bid
    int loop_times = 0;
    if (bid < bsz) {
      int seq_len = seq_lens_q[bid];
      if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
        seq_len = 0;
      }
      loop_times = div_up(seq_len * group_size, num_rows_per_block);
    }

    // prefix sum for each lane, get the start offset in this tile
    // inclusive scan
    int x = loop_times;
    for (int offset = 1; offset < warpSize; offset <<= 1) {
      int y = __shfl_up_sync(0xffffffff, x, offset);
      if (lane_id >= offset) x += y;
    }
    // exclusive prefix sum
    int bid_offset = x - loop_times;
    int tile_sum = __shfl_sync(0xffffffff, x, warpSize - 1);

    // write batch_ids and tile_ids_per_batch
    if (bid < bsz && loop_times > 0) {
      int write_base = prev_offset + bid_offset;
      for (int t = 0; t < loop_times; ++t) {
        int pos = write_base + t;
        batch_ids[pos] = bid;
        tile_ids_per_batch[pos] = t;
      }
    }

    // for next warp tile
    prev_offset += tile_sum;
  }

  if (threadIdx.x == 0) {
    *num_blocks_x = prev_offset;
  }
}

__global__ void split_kv_block(const int *__restrict__ seq_lens_decoder,
                               const int *__restrict__ seq_lens_encoder,
                               int *__restrict__ batch_ids,
                               int *__restrict__ tile_ids_per_batch,
                               int *__restrict__ num_blocks_x,
                               const int bsz,
                               const int pad_len,
                               const int num_row_per_block) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      const int start_len = seq_lens_decoder[bid];
      int seq_len = seq_lens_encoder[bid] + start_len % pad_len;
      if (seq_lens_encoder[bid] == 0) {
        seq_len = 0;
      }
      const int loop_times = div_up(seq_len, num_row_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

void GetBlockShapeAndSplitKVBlock(
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_this_time,
    paddle::Tensor &decoder_batch_ids,           // Inplace
    paddle::Tensor &decoder_tile_ids_per_batch,  // Inplace
    paddle::Tensor &decoder_num_blocks_cpu,      // Inplace, Pinned Memory
    paddle::Tensor &decoder_num_blocks_device,   // Inplace
    paddle::Tensor &decoder_chunk_size_device,   // Inplace
    paddle::Tensor &max_len_tensor_cpu,          // Inplace, CPU
    paddle::Tensor &encoder_batch_ids,           // Inplace
    paddle::Tensor &encoder_tile_ids_per_batch,  // Inplace
    paddle::Tensor &encoder_num_blocks_x_cpu,    // Inplace, CPU
    paddle::Tensor &kv_batch_ids,                // Inplace
    paddle::Tensor &kv_tile_ids_per_batch,       // Inplace
    paddle::Tensor &kv_num_blocks_x_cpu,         // Inplace, CPU
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int group_size,
    const int block_size) {
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
  int max_len_this_time = max_len_cpu_ptr[0];
  int max_enc_len_this_time = max_len_cpu_ptr[1];
  int max_dec_len_this_time = max_len_cpu_ptr[2];
  int max_enc_dec_len_this_time = max_len_cpu_ptr[3];
  int max_just_dec_len_this_time = max_len_cpu_ptr[4];
  int max_kv_len_this_time = max_len_cpu_ptr[5];

  const uint32_t decoder_batch_ele_num = decoder_batch_ids.shape()[0];

  // decoder
  if (max_dec_len_this_time > 0) {
    const bool mla_backend = checkAttentionBackend();
    if (mla_backend && group_size <= 64) {
      const int set_chunk_size = get_mla_dec_chunk_size(bsz);

      CUDA_CHECK(cudaMemsetAsync(
          decoder_chunk_size_device.data<int>(), 64, sizeof(int32_t), stream));

      CUDA_CHECK(cudaMemsetAsync(
          decoder_num_blocks_device.data<int>(), 0, sizeof(int32_t), stream));

      int device;
      CUDA_CHECK(cudaGetDevice(&device));
      int sm_cout;
      CUDA_CHECK(cudaDeviceGetAttribute(
          &sm_cout, cudaDevAttrMultiProcessorCount, device));
      constexpr int config_size =
          12;  // search space for chunk size:[64, 128, 256, ... 131072]

      search_chunk_size_for_mla<config_size>
          <<<1, 32, 0, stream>>>(seq_lens_this_time.data<int>(),
                                 seq_lens_encoder.data<int>(),
                                 seq_lens_decoder.data<int>(),
                                 decoder_num_blocks_device.data<int>(),
                                 decoder_chunk_size_device.data<int>(),
                                 bsz,
                                 set_chunk_size,
                                 block_size,
                                 sm_cout);

      decoder_num_blocks_cpu.copy_(
          decoder_num_blocks_device, decoder_num_blocks_cpu.place(), false);
      auto decoder_chunk_size_cpu =
          decoder_chunk_size_device.copy_to(paddle::CPUPlace(), false);
      const int chunk_size = decoder_chunk_size_cpu.data<int>()[0];

      CUDA_CHECK(cudaMemsetAsync(decoder_batch_ids.data<int>(),
                                 0,
                                 decoder_batch_ele_num * sizeof(int32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(decoder_tile_ids_per_batch.data<int>(),
                                 0,
                                 decoder_batch_ele_num * sizeof(int32_t),
                                 stream));

      split_block_for_mla<<<1, 32, 0, stream>>>(
          seq_lens_this_time.data<int>(),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          decoder_batch_ids.data<int>(),
          decoder_tile_ids_per_batch.data<int>(),
          bsz,
          chunk_size);

    } else {
      CUDA_CHECK(cudaMemsetAsync(decoder_batch_ids.data<int>(),
                                 0,
                                 decoder_batch_ele_num * sizeof(int32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(decoder_tile_ids_per_batch.data<int>(),
                                 0,
                                 decoder_batch_ele_num * sizeof(int32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(
          decoder_num_blocks_device.data<int>(), 0, sizeof(int32_t), stream));

      split_q_block<<<1, 32, 0, stream>>>(
          seq_lens_this_time.data<int>(),
          seq_lens_encoder.data<int>(),
          decoder_batch_ids.data<int>(),
          decoder_tile_ids_per_batch.data<int>(),
          decoder_num_blocks_device.data<int>(),
          bsz,
          decoder_block_shape_q,
          group_size);
      // Note (sunxin): Skip capturing the DtoH copy (it's time-consuming); CPU
      // data is only for branching in attention.
#ifndef PADDLE_WITH_CUSTOM_DEVICE_METAX_GPU
      if (!phi::backends::gpu::IsCUDAGraphCapturing())
#endif
        decoder_num_blocks_cpu.copy_(
            decoder_num_blocks_device, decoder_num_blocks_cpu.place(), false);
    }
  }

  // encoder
  if (max_enc_len_this_time > 0) {
    const uint32_t max_tile_size_per_bs_kv =
        div_up(max_enc_dec_len_this_time, block_size);
    const uint32_t kv_batch_shape = bsz * max_tile_size_per_bs_kv;
    CUDA_CHECK(cudaMemsetAsync(
        kv_batch_ids.data<int>(), 0, kv_batch_shape * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(kv_tile_ids_per_batch.data<int>(),
                               0,
                               kv_batch_shape * sizeof(int32_t),
                               stream));
    auto kv_num_blocks_x =
        GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());

    split_kv_block<<<1, 32, 0, seq_lens_encoder.stream()>>>(
        seq_lens_decoder.data<int>(),
        seq_lens_encoder.data<int>(),
        kv_batch_ids.data<int>(),
        kv_tile_ids_per_batch.data<int>(),
        kv_num_blocks_x.data<int>(),
        bsz,
        block_size,
        block_size);

    kv_num_blocks_x_cpu.copy_(
        kv_num_blocks_x, kv_num_blocks_x_cpu.place(), false);
    // Clear buffer
    const uint32_t encoder_max_tile_size_per_bs_q =
        div_up((max_enc_dec_len_this_time * group_size), encoder_block_shape_q);
    const uint32_t encoder_batch_shape = bsz * encoder_max_tile_size_per_bs_q;
    CUDA_CHECK(cudaMemsetAsync(encoder_batch_ids.data<int>(),
                               0,
                               encoder_batch_shape * sizeof(int32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(encoder_tile_ids_per_batch.data<int>(),
                               0,
                               encoder_batch_shape * sizeof(int32_t),
                               stream));
    auto encoder_num_blocks_x =
        GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());
    split_q_block<<<1, 32, 0, stream>>>(seq_lens_encoder.data<int>(),
                                        nullptr,
                                        encoder_batch_ids.data<int>(),
                                        encoder_tile_ids_per_batch.data<int>(),
                                        encoder_num_blocks_x.data<int>(),
                                        bsz,
                                        encoder_block_shape_q,
                                        group_size);
    encoder_num_blocks_x_cpu.copy_(
        encoder_num_blocks_x, encoder_num_blocks_x_cpu.place(), false);
  }
}

std::vector<std::vector<int64_t>> GetBlockShapeAndSplitKVBlockInferShape(
    const std::vector<int64_t> &seq_lens_encoder,
    const std::vector<int64_t> &seq_lens_decoder,
    const std::vector<int64_t> &seq_lens_this_time,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int group_size,
    const int block_size) {
  return {};
}

std::vector<paddle::DataType> GetBlockShapeAndSplitKVBlockInferDtype(
    const paddle::DataType &seq_lens_encoder,
    const paddle::DataType &seq_lens_decoder,
    const paddle::DataType &seq_lens_this_time,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int group_size,
    const int block_size) {
  return {};
}

PD_BUILD_STATIC_OP(get_block_shape_and_split_kv_block)
    .Inputs({
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "decoder_batch_ids",
        "decoder_tile_ids_per_batch",
        "decoder_num_blocks_cpu",
        "decoder_num_blocks_device",
        "decoder_chunk_size_device",
        "max_len_tensor_cpu",
        "encoder_batch_ids",
        "encoder_tile_ids_per_batch",
        "encoder_num_blocks_x_cpu",
        "kv_batch_ids",
        "kv_tile_ids_per_batch",
        "kv_num_blocks_x_cpu",
    })
    .Outputs({

    })
    .Attrs({"encoder_block_shape_q: int",
            "decoder_block_shape_q: int",
            "group_size: int",
            "block_size: int"})
    .SetKernelFn(PD_KERNEL(GetBlockShapeAndSplitKVBlock))
    .SetInferShapeFn(PD_INFER_SHAPE(GetBlockShapeAndSplitKVBlockInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetBlockShapeAndSplitKVBlockInferDtype));
