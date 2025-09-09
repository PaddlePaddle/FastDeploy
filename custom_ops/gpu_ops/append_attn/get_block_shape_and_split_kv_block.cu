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

#include "helper.h"
#include "paddle/extension.h"
#include "paddle/phi/core/memory/memcpy.h"

template <int THREADBLOCK_SIZE>
__global__ void
GetMaxLenKernel(const int *seq_lens, const int *seq_lens_this_time,
                const int *seq_lens_encoder,
                const int *seq_lens_this_time_merged,
                const int *seq_lens_encoder_merged, const int *seq_mapping,
                const int *system_lens, int *max_lens, const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_time_this_thread = 0;
  int max_len_encoder_this_thread = 0;
  int max_len_decoder_this_thread = 0;
  int max_len_this_thread = 0;
  int max_just_dec_len_this_thread = 0;
  int max_just_dec_merged_len_this_time_this_thread = 0;
  int max_system_len_this_thread = 0;
  int max_dec_len_without_system_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    const int seq_len_this_time = seq_lens_this_time[i];
    max_len_this_time_this_thread =
        max(seq_len_this_time, max_len_this_time_this_thread);
    max_len_encoder_this_thread =
        max(seq_lens_encoder[i], max_len_encoder_this_thread);
    max_len_decoder_this_thread = max(seq_lens[i], max_len_decoder_this_thread);
    if (seq_len_this_time <= 0)
      continue;
    const int max_just_dec_len_now = seq_lens_encoder[i] > 0 ? 0 : seq_lens[i];
    max_len_this_thread =
        max(seq_lens[i] + seq_len_this_time, max_len_this_thread);
    max_just_dec_len_this_thread =
        max(max_just_dec_len_this_thread, max_just_dec_len_now);
    if (system_lens) {
      const int real_bid = seq_mapping[i];
      const int system_len_now = system_lens[real_bid];
      max_system_len_this_thread =
          max(max_system_len_this_thread, system_len_now);
      max_dec_len_without_system_this_thread =
          max(max_dec_len_without_system_this_thread,
              max_just_dec_len_now - system_len_now);
    }
  }
  if (system_lens) {
    for (int i = tid; i < batch_size; i += blockDim.x) {
      const int ori_seq_len_this_time = seq_lens_this_time_merged[i];
      if (ori_seq_len_this_time <= 0)
        continue;
      const int max_just_dec_merged_len_this_time_now =
          seq_lens_encoder_merged[i] > 0 ? 0 : ori_seq_len_this_time;
      max_just_dec_merged_len_this_time_this_thread =
          max(max_just_dec_merged_len_this_time_this_thread,
              max_just_dec_merged_len_this_time_now);
    }
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
  int total_just_dec_merged =
      BlockReduce(temp_storage)
          .Reduce(max_just_dec_merged_len_this_time_this_thread, MaxOp<int>());
  int total_system_len = BlockReduce(temp_storage)
                             .Reduce(max_system_len_this_thread, MaxOp<int>());
  int total_dec_len_without_system =
      BlockReduce(temp_storage)
          .Reduce(max_dec_len_without_system_this_thread, MaxOp<int>());
  if (tid == 0) {
    max_lens[0] = total_max_len_this_time;
    max_lens[1] = total_max_len_encoder;
    max_lens[2] = total_max_len_decoder;
    max_lens[3] = total;
    max_lens[4] = total_just_dec;
    max_lens[5] = total_just_dec_merged;
    max_lens[6] = total_system_len;
    max_lens[7] = total_dec_len_without_system;
  }
}

void GetMaxLen(const paddle::Tensor &seq_lens_tensor,
               const paddle::Tensor &seq_lens_this_time,
               const paddle::Tensor &seq_lens_encoder,
               paddle::Tensor &max_len_tensor, const int batch_size) {
  constexpr int blockSize = 1024;
  GetMaxLenKernel<blockSize><<<1, blockSize, 0, seq_lens_encoder.stream()>>>(
      seq_lens_tensor.data<int>(), seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(), nullptr, nullptr, nullptr, nullptr,
      max_len_tensor.data<int>(), batch_size);
}

__global__ void split_q_block(const int *__restrict__ seq_lens_q,
                              const int *__restrict__ seq_lens_encoder,
                              int *__restrict__ batch_ids,
                              int *__restrict__ tile_ids_per_batch,
                              int *__restrict__ num_blocks_x, const int bsz,
                              const int num_rows_per_block,
                              const int group_size) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
        seq_len = 0;
      }
      const int loop_times = div_up(seq_len * group_size, num_rows_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

__global__ void split_kv_block(const int *__restrict__ seq_lens_decoder,
                               const int *__restrict__ seq_lens_encoder,
                               int *__restrict__ batch_ids,
                               int *__restrict__ tile_ids_per_batch,
                               int *__restrict__ num_blocks_x, const int bsz,
                               const int pad_len, const int num_row_per_block) {
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

template <int THREADBLOCK_SIZE>
__global__ void
get_max_len_kv_ernel(int *max_seq_lens_out, const int *seq_lens_this_time,
                     const int *seq_lens_decoder, const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    if (seq_lens_decoder[i] == 0)
      continue;
    max_len_this_thread =
        max(seq_lens_this_time[i] + seq_lens_decoder[i], max_len_this_thread);
  }
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  if (tid == 0) {
    *max_seq_lens_out = total;
  }
}

void GetBlockShapeAndSplitKVBlock(
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_this_time,
    paddle::Tensor &decoder_batch_ids,          // Inplace
    paddle::Tensor &decoder_tile_ids_per_batch, // Inplace
    paddle::Tensor &decoder_num_blocks_x_cpu,   // Inplace, Pinned Memory
    paddle::Tensor &max_len_tensor_cpu,         // Inplace, CPU
    paddle::Tensor &encoder_batch_ids,          // Inplace
    paddle::Tensor &encoder_tile_ids_per_batch, // Inplace
    paddle::Tensor &encoder_num_blocks_x_cpu,   // Inplace, CPU
    paddle::Tensor &kv_batch_ids,               // Inplace
    paddle::Tensor &kv_tile_ids_per_batch,      // Inplace
    paddle::Tensor &kv_num_blocks_x_cpu,        // Inplace, CPU
    paddle::Tensor &max_len_kv_cpu,             // Inplace, CPU
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int group_size,
    const int block_size,
    const int decoder_step_token_num)
{
  auto stream = seq_lens_encoder.stream();
  int bsz = seq_lens_this_time.shape()[0];

  paddle::Tensor max_len_tensor_gpu = GetEmptyTensor({max_len_tensor_cpu.shape()[0]}, paddle::DataType::INT32, seq_lens_this_time.place());
  GetMaxLen(seq_lens_decoder, seq_lens_this_time, seq_lens_encoder,
            max_len_tensor_gpu, bsz);
  max_len_tensor_cpu.copy_(max_len_tensor_gpu, max_len_tensor_cpu.place(), false);

  auto max_len_cpu_ptr = max_len_tensor_cpu.data<int>();
  int max_len_this_time = max_len_cpu_ptr[0];
  int max_enc_len_this_time = max_len_cpu_ptr[1];
  int max_dec_len_this_time = max_len_cpu_ptr[2];
  int max_enc_dec_len_this_time = max_len_cpu_ptr[3];
  int max_just_dec_len_this_time = max_len_cpu_ptr[4];
  int max_just_dec_merged_len_this_time = max_len_cpu_ptr[5];
  int max_system_len = max_len_cpu_ptr[6];
  int max_just_dec_len_without_system = max_len_cpu_ptr[7];



  auto max_len_kv =
      GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_decoder.place());
  get_max_len_kv_ernel<128><<<1, 128, 0, stream>>>(
      max_len_kv.data<int>(), seq_lens_this_time.data<int>(),
      seq_lens_decoder.data<int>(), bsz);


  max_len_kv_cpu.copy_(max_len_kv, max_len_kv_cpu.place(), false);

  if (max_enc_len_this_time > 0) {
    const uint32_t max_tile_size_per_bs_kv = div_up(max_enc_dec_len_this_time, block_size);
    const uint32_t kv_batch_shape = bsz * max_tile_size_per_bs_kv;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(kv_batch_ids.data<int>(), 0, kv_batch_shape * sizeof(int32_t), stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(kv_tile_ids_per_batch.data<int>(), 0, kv_batch_shape * sizeof(int32_t), stream));
    auto kv_num_blocks_x =
        GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());

    split_kv_block<<<1, 32, 0, seq_lens_encoder.stream()>>>(
        seq_lens_decoder.data<int>(),
        // sequence_lengths->data<int>(),
        seq_lens_encoder.data<int>(), kv_batch_ids.data<int>(),
        kv_tile_ids_per_batch.data<int>(), kv_num_blocks_x.data<int>(), bsz,
        block_size, block_size);

    kv_num_blocks_x_cpu.copy_(kv_num_blocks_x, kv_num_blocks_x_cpu.place(), false);
    // Clear buffer
    const uint32_t encoder_max_tile_size_per_bs_q = div_up((max_enc_dec_len_this_time * group_size), encoder_block_shape_q);
    const uint32_t encoder_batch_shape = bsz * encoder_max_tile_size_per_bs_q;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(encoder_batch_ids.data<int>(), 0, encoder_batch_shape * sizeof(int32_t), stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(encoder_tile_ids_per_batch.data<int>(), 0, encoder_batch_shape * sizeof(int32_t), stream));
    auto encoder_num_blocks_x =
        GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());
    split_q_block<<<1, 32, 0, stream>>>(seq_lens_encoder.data<int>(), nullptr,
                                        encoder_batch_ids.data<int>(),
                                        encoder_tile_ids_per_batch.data<int>(),
                                        encoder_num_blocks_x.data<int>(), bsz,
                                        encoder_block_shape_q, group_size);
    encoder_num_blocks_x_cpu.copy_(encoder_num_blocks_x, encoder_num_blocks_x_cpu.place(), false);
  }

  if (max_just_dec_len_this_time > 0) {
    // Clear buffer
    const uint32_t decoder_max_tile_size_per_bs_q = div_up((decoder_step_token_num * group_size), decoder_block_shape_q);
    const uint32_t decoder_batch_shape = bsz * decoder_max_tile_size_per_bs_q;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(decoder_batch_ids.data<int>(), 0, decoder_batch_shape * sizeof(int32_t), stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(decoder_tile_ids_per_batch.data<int>(), 0, decoder_batch_shape * sizeof(int32_t), stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(decoder_num_blocks_x_cpu.data<int>(), 0, sizeof(int32_t), stream));

    auto decoder_num_blocks_x =
        GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());
    split_q_block<<<1, 32, 0, stream>>>(
        seq_lens_this_time.data<int>(),
        seq_lens_encoder.data<int>(),
        decoder_batch_ids.data<int>(),
        decoder_tile_ids_per_batch.data<int>(),
        decoder_num_blocks_x.data<int>(),
        bsz,
        decoder_block_shape_q,
        group_size);
    decoder_num_blocks_x_cpu.copy_(decoder_num_blocks_x, decoder_num_blocks_x_cpu.place(), false);
  }

}

PD_BUILD_STATIC_OP(get_block_shape_and_split_kv_block)
    .Inputs({
      "seq_lens_encoder",
      "seq_lens_decoder",
      "seq_lens_this_time",
      "decoder_batch_ids",
      "decoder_tile_ids_per_batch",
      "decoder_num_blocks_x_cpu",
      "max_len_tensor_cpu",
      "encoder_batch_ids",
      "encoder_tile_ids_per_batch",
      "encoder_num_blocks_x_cpu",
      "kv_batch_ids",
      "kv_tile_ids_per_batch",
      "kv_num_blocks_x_cpu",
      "max_len_kv_cpu"
    })
    .Outputs({

    })
    .Attrs({
      "encoder_block_shape_q: int",
      "decoder_block_shape_q: int",
      "group_size: int",
      "block_size: int",
      "decoder_step_token_num: int"
    })
    .SetKernelFn(PD_KERNEL(GetBlockShapeAndSplitKVBlock));
