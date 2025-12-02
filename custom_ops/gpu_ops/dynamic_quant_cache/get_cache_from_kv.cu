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
#include "paddle/extension.h"
#include "utils.hpp"

namespace dynamic_quant_cache {

template <typename T,
          typename ScaleType,
          int kBlockSize,
          int kHeadDim,
          int kThreads>
void __global__ get_kv_from_cache_kernel(T *k_input,
                                         T *v_input,
                                         uint8_t *cache_k_c2,
                                         uint8_t *cache_v_c2,
                                         T *cache_k_c16,
                                         T *cache_v_c16,
                                         const int *cu_seq_k,
                                         const int *encoder_seqs_len,
                                         const int *decoder_seqs_len,
                                         const int *block_tables,
                                         const int64_t *prompt_lens,
                                         const int max_num_blocks_per_seq,
                                         const int data_num_per_block,
                                         const int c16_remain_seq_len,
                                         const int head_num,
                                         const int kv_head_num) {
  const int bidb = blockIdx.x;
  const int bidh = blockIdx.y;
  const int block_idx = blockIdx.z;
  const int tidx = threadIdx.x;

  const int seq_len_encoder = encoder_seqs_len[bidb];
  const int seq_len_decoder = decoder_seqs_len[bidb];
  const int prompt_len = prompt_lens[bidb];

  const int token_idx = block_idx * kBlockSize;

  const int seq_len = seq_len_encoder + seq_len_decoder;

  if (seq_len_encoder == 0 || seq_len_decoder == 0 || token_idx >= seq_len) {
    return;
  }

  using pakc_half = typename PackedHalf<T>::Type;

  const int c16_cache_max_len = kBlockSize + c16_remain_seq_len;

  const int c16_cache_len = prompt_len < c16_cache_max_len
                                ? prompt_len
                                : c16_remain_seq_len + prompt_len % kBlockSize;

  const int c2_cache_len = prompt_len - c16_cache_len;

  constexpr int kPackSize = 16 / sizeof(T);
  constexpr int data_per_row = kHeadDim / kPackSize;
  const int copy_row_idx = tidx / data_per_row;
  const int copy_col = tidx % data_per_row;
  const int copy_col_idx = copy_col * kPackSize;
  const int all_rows = kThreads / data_per_row;

  if (token_idx >= c2_cache_len) {
    const int remain_tokens = seq_len - token_idx;
    if (remain_tokens <= 0) {
      return;
    }
    int store_idx = (cu_seq_k[bidb] + token_idx) * kv_head_num * kHeadDim +
                    bidh * kHeadDim + copy_col_idx;
    int load_idx = (bidb * c16_cache_max_len + token_idx - c2_cache_len) *
                       kv_head_num * kHeadDim +
                   bidh * kHeadDim + copy_col_idx;
    T *src = cache_k_c16;
    T *dst = k_input;
    if (bidh >= kv_head_num) {
      src = cache_v_c16;
      dst = v_input;
      store_idx -= kv_head_num * kHeadDim;
      load_idx -= kv_head_num * kHeadDim;
    }

    const int copy_tokens = min(remain_tokens, kBlockSize);
    for (int i = copy_row_idx; i < copy_tokens; i += all_rows) {
      *reinterpret_cast<int4 *>(dst + store_idx + i * kv_head_num * kHeadDim) =
          *reinterpret_cast<int4 *>(src + load_idx +
                                    i * kv_head_num * kHeadDim);
    }
    return;
  }

  extern __shared__ char smem_[];

  uint8_t *cache_smem = reinterpret_cast<uint8_t *>(smem_);
  uint8_t *cache_store_smem = cache_smem + data_num_per_block;

  const int *block_table = block_tables + bidb * max_num_blocks_per_seq;
  const int physical_block_number = block_table[block_idx];

  int cache_offset = physical_block_number * kv_head_num * data_num_per_block +
                     bidh * data_num_per_block;

  const int kPackSizeInt8 = 16 / sizeof(uint8_t);

  uint8_t *cache_global = cache_k_c2;

  if (bidh >= kv_head_num) {
    cache_global = cache_v_c2;
    cache_offset -= kv_head_num * data_num_per_block;
  }

  for (int i = tidx * kPackSizeInt8; i < data_num_per_block;
       i += kThreads * kPackSizeInt8) {
    *reinterpret_cast<int4 *>(cache_smem + i) =
        *reinterpret_cast<int4 *>(cache_global + cache_offset + i);
  }

  __syncthreads();

  pakc_half *scale_mem = reinterpret_cast<pakc_half *>(cache_smem) + 512;
  pakc_half *zp_mem = scale_mem + 128;

  const int warp_id = tidx / 32;
  const int lane_id = tidx % 32;
  const int row = lane_id / 4;
  const int col = lane_id % 4;
  const int row_offset = row * 4;
  constexpr int all_cols = (kHeadDim + (32 / 4) * 8) / 2;

  constexpr uint32_t mask = 0x03030303;

  if (bidh < kv_head_num) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      uint32_t c2_value =
          reinterpret_cast<uint32_t *>(cache_smem)[tidx + i * 64];

      for (int k = 1; k >= 0; --k) {
#pragma unroll
        for (int j = 1; j >= 0; --j) {
          uint32_t value = c2_value & mask;
          c2_value = c2_value >> 2;

          int2 half_data = Convert_from_fp8<ScaleType, T>()(value);

          pakc_half cur_value = reinterpret_cast<pakc_half *>(&half_data)[0];
          pakc_half next_value = reinterpret_cast<pakc_half *>(&half_data)[1];

          const int scale_idx = (i + k) * 8 + col + j * 4;
          const int idx =
              (row + warp_id * 8) * all_cols + row_offset + scale_idx;

          pakc_half cur_dequant_value = scale_mem[scale_idx];
          pakc_half cur_quant_zp = zp_mem[scale_idx];
          pakc_half next_dequant_value = scale_mem[scale_idx + 64];
          pakc_half next_quant_zp = zp_mem[scale_idx + 64];

          cur_value = cur_value * cur_dequant_value - cur_quant_zp;
          next_value = next_value * next_dequant_value - next_quant_zp;

          reinterpret_cast<pakc_half *>(cache_store_smem)[idx] = cur_value;
          reinterpret_cast<pakc_half *>(
              cache_store_smem)[idx + (kThreads / 32 * 8 * all_cols)] =
              next_value;
        }
      }
    }

    __syncthreads();

    const int store_idx =
        (cu_seq_k[bidb] + token_idx) * kv_head_num * kHeadDim +
        bidh * kHeadDim + copy_col_idx;
    for (int i = copy_row_idx; i < kBlockSize; i += all_rows) {
      const int offset = i % 8 * 8;
      *reinterpret_cast<int4 *>(k_input + store_idx +
                                i * kv_head_num * kHeadDim) =
          *reinterpret_cast<int4 *>(reinterpret_cast<T *>(cache_store_smem) +
                                    i * (all_cols * 2) + copy_col_idx + offset);
    }

  } else {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t c2_value =
          reinterpret_cast<uint32_t *>(cache_smem)[tidx + i * 128];
      pakc_half dequant_value[8];
      for (int j = 3; j >= 0; j--) {
        const int scale_idx = i * 8 + j * 32 + col;
        uint32_t value = c2_value & mask;
        c2_value = c2_value >> 2;

        int2 half_data = Convert_from_fp8<ScaleType, T>()(value);

        pakc_half value1 = *reinterpret_cast<pakc_half *>(&half_data.x);
        pakc_half value2 = *reinterpret_cast<pakc_half *>(&half_data.y);

        value1 = value1 * scale_mem[scale_idx] - zp_mem[scale_idx];
        value2 = value2 * scale_mem[scale_idx + 4] - zp_mem[scale_idx + 4];

        dequant_value[2 * j] = value1;
        dequant_value[2 * j + 1] = value2;
      }
      uint32_t smem_ptr = cast_smem_ptr_to_uint(
          reinterpret_cast<uint128_t *>(cache_store_smem + i * 16 * 128 * 2) +
          tidx);
      asm volatile(
          "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "
          "%4};\n" ::"r"(smem_ptr),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[0]),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[1]),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[2]),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[3]));

      smem_ptr = cast_smem_ptr_to_uint(
          reinterpret_cast<uint128_t *>(cache_store_smem + 8 * 128 * 2 +
                                        i * 16 * 128 * 2) +
          tidx);
      asm volatile(
          "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "
          "%4};\n" ::"r"(smem_ptr),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[4]),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[5]),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[6]),
          "r"(reinterpret_cast<uint32_t *>(dequant_value)[7]));
    }

    __syncthreads();

    const int store_idx =
        (cu_seq_k[bidb] + token_idx) * kv_head_num * kHeadDim +
        (bidh - kv_head_num) * kHeadDim + copy_col_idx;
    for (int i = copy_row_idx; i < kBlockSize; i += all_rows) {
      const int cur_idx = i * data_per_row + copy_col;
      int div_256 = cur_idx / 256 * 256;
      int mod_256 = cur_idx % 256;
      const int src_idx = mod_256 / 16 + div_256 + mod_256 % 8 / 4 * 16 +
                          mod_256 % 4 * 32 + copy_col / 8 * 128;
      *reinterpret_cast<int4 *>(v_input + store_idx +
                                i * kv_head_num * kHeadDim) =
          reinterpret_cast<int4 *>(cache_store_smem)[src_idx];
    }
  }
}

template <typename T, typename ScaleType>
void get_kv_from_cache(T *k_input,
                       T *v_input,
                       uint8_t *cache_k_c2,
                       uint8_t *cache_v_c2,
                       T *cache_k_c16,
                       T *cache_v_c16,
                       const int *cu_seq_k,
                       const int *encoder_seqs_len,
                       const int *decoder_seqs_len,
                       const int *block_tables,
                       const int64_t *prompt_lens,
                       const int max_num_blocks_per_seq,
                       const int data_num_per_block,
                       const int c16_remain_seq_len,
                       const int head_num,
                       const int kv_head_num,
                       const int max_seq_k,
                       const int bsz,
                       cudaStream_t stream) {
  constexpr int kBlockSize = 64;
  constexpr int kHeadDim = 128;
  constexpr int kThreads = 128;
  const int smem_size =
      kBlockSize * (kHeadDim + (32 / 4) * 8) * sizeof(T) + data_num_per_block;

  int block_num =
      (max_seq_k + c16_remain_seq_len + kBlockSize - 1) / kBlockSize;
  dim3 gird_dim;
  gird_dim.x = bsz;
  gird_dim.y = kv_head_num * 2;
  gird_dim.z = block_num;

  get_kv_from_cache_kernel<T, ScaleType, kBlockSize, kHeadDim, kThreads>
      <<<gird_dim, kThreads, smem_size, stream>>>(k_input,
                                                  v_input,
                                                  cache_k_c2,
                                                  cache_v_c2,
                                                  cache_k_c16,
                                                  cache_v_c16,
                                                  cu_seq_k,
                                                  encoder_seqs_len,
                                                  decoder_seqs_len,
                                                  block_tables,
                                                  prompt_lens,
                                                  max_num_blocks_per_seq,
                                                  data_num_per_block,
                                                  c16_remain_seq_len,
                                                  head_num,
                                                  kv_head_num);
}

void GetKVFromCache(const paddle::Tensor &k_input,
                    const paddle::Tensor &v_input,
                    const paddle::Tensor &cache_k_c2,
                    const paddle::Tensor &cache_v_c2,
                    const paddle::Tensor &cache_k_c16,
                    const paddle::Tensor &cache_v_c16,
                    const paddle::Tensor &cu_seq_k,
                    const paddle::Tensor &encoder_seqs_len,
                    const paddle::Tensor &decoder_seqs_len,
                    const paddle::Tensor &block_table,
                    const paddle::Tensor &prompt_lens,
                    const int c16_remain_seq_len,
                    const int head_num,
                    const int kv_head_num,
                    const int head_dim,
                    const int max_seq_k,
                    const std::string &cache_quant_type_str) {
  using scale_type = cutlass::float_e4m3_t;
  constexpr int kBlockSize = 64;
  const int max_num_blocks_per_seq = block_table.dims()[1];
  const int data_num_per_block =
      kBlockSize * head_dim / 4 + kBlockSize / 32 * head_dim * 4;
  const int bsz = encoder_seqs_len.dims()[0];

  if (k_input.dtype() == paddle::DataType::FLOAT16) {
    using input_type = phi::dtype::float16;
    get_kv_from_cache<input_type, scale_type>(
        const_cast<input_type *>(k_input.data<input_type>()),
        const_cast<input_type *>(v_input.data<input_type>()),
        const_cast<uint8_t *>(cache_k_c2.data<uint8_t>()),
        const_cast<uint8_t *>(cache_v_c2.data<uint8_t>()),
        const_cast<input_type *>(cache_k_c16.data<input_type>()),
        const_cast<input_type *>(cache_v_c16.data<input_type>()),
        cu_seq_k.data<int>(),
        encoder_seqs_len.data<int>(),
        decoder_seqs_len.data<int>(),
        block_table.data<int>(),
        prompt_lens.data<int64_t>(),
        max_num_blocks_per_seq,
        data_num_per_block,
        c16_remain_seq_len,
        head_num,
        kv_head_num,
        max_seq_k,
        bsz,
        k_input.stream());
  } else {
    PD_THROW("BF16 is not supported\n");
  }
}

}  // namespace dynamic_quant_cache

PD_BUILD_OP(dynamic_quant_get_kv_from_cache)
    .Inputs({"k_input",
             "v_input",
             "cache_k_c2",
             "cache_v_c2",
             "cache_k_c16",
             "cache_v_c16",
             "cu_seq_k",
             "encoder_seqs_len",
             "decoder_seqs_len",
             "block_table",
             "prompt_lens"})
    .Attrs({"c16_remain_seq_len: int",
            "head_num: int",
            "kv_head_num: int",
            "head_dim: int",
            "max_seq_k: int",
            "cache_quant_type_str: std::string"})
    .Outputs({"k_input_out", "v_input_out"})
    .SetInplaceMap({{"k_input", "k_input_out"}, {"v_input", "v_input_out"}})
    .SetKernelFn(PD_KERNEL(dynamic_quant_cache::GetKVFromCache));
