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

namespace dynamic_quant_cache_attn {

template <typename input_type, typename output_type, int tokens_per_block>
__global__ void split_qkv_and_rope_kernel(const input_type *qkv_input,
                                          const input_type *qkv_bias,
                                          output_type *q_input,
                                          output_type *k_input,
                                          output_type *v_input,
                                          const float *rope_sin_cos,
                                          const int *seq_len_encoder,
                                          const int *seq_len_decoder,
                                          const int *cu_seq_q,
                                          const int *cu_seq_k,
                                          const int max_seq_q,
                                          const int head_num,
                                          const int kv_head_num,
                                          const int max_input_length) {
  constexpr int kPackSize = 16 / sizeof(input_type);
  constexpr int kHeadDim = 128;

  using src_type = Vec<input_type, kPackSize>;
  using dst_type = Vec<output_type, kPackSize>;

  using rope_type = Vec<float, kPackSize / 2>;
  using pack_half =
      std::conditional_t<std::is_same<input_type, phi::dtype::float16>::value,
                         __half2,
                         nv_bfloat162>;

  const int bidb = blockIdx.x;
  const int bidh = blockIdx.y;
  const int bidt = blockIdx.z * tokens_per_block;
  const int tidx = threadIdx.x;
  const int lane_id = tidx % 32;
  const int warp_id = tidx / 32;
  const int seq_len = seq_len_encoder[bidb];
  const int seq_len_start = seq_len_decoder[bidb];

  if (seq_len == 0) {
    return;
  }

  const int all_head_num = head_num + 2 * kv_head_num;
  const int hidden = all_head_num * kHeadDim;

  const int row_idx = tidx / (kHeadDim / kPackSize);
  const int col_idx = tidx % (kHeadDim / kPackSize);

  const int bias_idx = bidh * kHeadDim + col_idx * kPackSize;

  src_type src, src_bias;
  dst_type dst;
  rope_type sin, cos;

  const bool need_add_bias = qkv_bias != nullptr;

  if (need_add_bias) {
    src_bias.load_from(qkv_bias + bias_idx);
  }

  const int cur_token = bidt + row_idx;

  if (cur_token < seq_len) {
    src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx +
                  cur_token * hidden);
    if (need_add_bias) {
      src.add(src_bias);
    }
  }

  if (bidh < head_num) {
    const float *cos_rope = rope_sin_cos +
                            (cur_token + seq_len_start) * (kHeadDim / 2) +
                            col_idx * (kPackSize / 2);
    const float *sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

    if (cur_token < seq_len) {
      sin.load_from(sin_rope);
      cos.load_from(cos_rope);
      apply_rotary_embedding<input_type, output_type, kPackSize>(
          src, dst, cos, sin);
      dst.store_to(q_input +
                   (cu_seq_q[bidb] + cur_token) * head_num * kHeadDim +
                   bias_idx);
    }
  } else if (bidh < head_num + kv_head_num) {
    const float *cos_rope = rope_sin_cos +
                            (cur_token + seq_len_start) * (kHeadDim / 2) +
                            col_idx * (kPackSize / 2);
    const float *sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

    if (cur_token < seq_len) {
      sin.load_from(sin_rope);
      cos.load_from(cos_rope);
      apply_rotary_embedding<input_type, output_type, kPackSize>(
          src, dst, cos, sin);
      dst.store_to(k_input +
                   (cu_seq_k[bidb] + cur_token) * kv_head_num * kHeadDim +
                   bias_idx - head_num * kHeadDim);
    }
  } else {
    if (cur_token < seq_len) {
      for (int i = 0; i < kPackSize; i++) {
        dst.data.elt[i] = static_cast<output_type>(src.data.elt[i]);
      }
      dst.store_to(v_input +
                   (cu_seq_k[bidb] + cur_token) * kv_head_num * kHeadDim +
                   bias_idx - (head_num + kv_head_num) * kHeadDim);
    }
  }
}

template <typename input_type, typename output_type>
void split_qkv_and_rope(const input_type *qkv_input,
                        const input_type *qkv_bias,
                        output_type *q_input,
                        output_type *k_input,
                        output_type *v_input,
                        const float *rope_sin_cos,
                        const int *seq_len_encoder,
                        const int *seq_len_decoder,
                        const int *cu_seq_q,
                        const int *cu_seq_k,
                        const int max_seq_q,
                        const int head_num,
                        const int kv_head_num,
                        const int max_input_length,
                        const int bsz,
                        cudaStream_t stream) {
  constexpr int kPackSize = 16 / sizeof(input_type);
  constexpr int kHeadDim = 128;
  constexpr int kThreads = 128;
  constexpr int tokens_per_block = kThreads / (kHeadDim / kPackSize);
  dim3 grid_dims;
  grid_dims.x = bsz;
  grid_dims.y = head_num + 2 * kv_head_num;
  grid_dims.z = (max_seq_q + tokens_per_block - 1) / tokens_per_block;

  split_qkv_and_rope_kernel<input_type, output_type, tokens_per_block>
      <<<grid_dims, kThreads, 0, stream>>>(qkv_input,
                                           qkv_bias,
                                           q_input,
                                           k_input,
                                           v_input,
                                           rope_sin_cos,
                                           seq_len_encoder,
                                           seq_len_decoder,
                                           cu_seq_q,
                                           cu_seq_k,
                                           max_seq_q,
                                           head_num,
                                           kv_head_num,
                                           max_input_length);
}

void SplitQKVAndRope(const paddle::Tensor &qkv_out,
                     const paddle::Tensor &q_input,
                     const paddle::Tensor &k_input,
                     const paddle::Tensor &v_input,
                     const paddle::Tensor &rotary_embs,
                     const paddle::Tensor &seq_len_encoder,
                     const paddle::Tensor &seq_len_decoder,
                     const paddle::Tensor &cu_seq_q,
                     const paddle::Tensor &cu_seq_k,
                     const paddle::optional<paddle::Tensor> &qkv_bias,
                     const int head_num,
                     const int kv_head_num,
                     const int head_dim,
                     const int max_seq_q,
                     const int max_input_length,
                     const std::string &cache_quant_type_str) {
  if (qkv_out.dtype() == paddle::DataType::FLOAT16) {
    PD_THROW("FLOAT16 is not supported\n");
  } else if (qkv_out.dtype() == paddle::DataType::BFLOAT16) {
    using T = phi::dtype::bfloat16;
    split_qkv_and_rope<T>(
        const_cast<T *>(qkv_out.data<T>()),
        qkv_bias ? qkv_bias.get().data<T>() : nullptr,
        const_cast<phi::dtype::float16 *>(q_input.data<phi::dtype::float16>()),
        const_cast<phi::dtype::float16 *>(k_input.data<phi::dtype::float16>()),
        const_cast<phi::dtype::float16 *>(v_input.data<phi::dtype::float16>()),
        rotary_embs.data<float>(),
        seq_len_encoder.data<int>(),
        seq_len_decoder.data<int>(),
        cu_seq_q.data<int>(),
        cu_seq_k.data<int>(),
        max_seq_q,
        head_num,
        kv_head_num,
        max_input_length,
        seq_len_encoder.dims()[0],
        qkv_out.stream());
  }
}

__global__ void get_cur_cu_seq_len_k_kernel(
    const int *__restrict__ seq_lens_encoder,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    int *__restrict__ cu_seqlens_k,
    int *__restrict__ qk_tokens,
    const int bsz) {
  cu_seqlens_k[0] = 0;
  qk_tokens[0] = 0;
  qk_tokens[1] = 0;
  qk_tokens[2] = 0;
  qk_tokens[3] = 0;
  int total_tokens = 0;

  for (uint32_t bid = 0; bid < bsz; bid++) {
    int cache_len = seq_lens_decoder[bid];
    const int q_len = seq_lens_encoder[bid];
    const int tokens = seq_lens_this_time[bid];

    qk_tokens[1] = max(qk_tokens[1], cache_len);
    if (q_len == 0) {
      cache_len = tokens;
    }
    total_tokens += (cache_len + q_len);
    cu_seqlens_k[bid + 1] = total_tokens;
    qk_tokens[0] = max(qk_tokens[0], q_len);

    qk_tokens[2] += tokens;
    qk_tokens[3] += (cache_len + q_len);
  }
}

std::vector<paddle::Tensor> GetQKTokenNum(
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_decoder) {
  auto stream = seq_lens_decoder.stream();
  auto place = seq_lens_decoder.place();
  int bsz = seq_lens_this_time.shape()[0];

  paddle::Tensor cu_seqlens_k =
      paddle::empty({bsz + 1}, paddle::DataType::INT32, place);
  paddle::Tensor qk_tokens = paddle::empty({4}, paddle::DataType::INT32, place);

  get_cur_cu_seq_len_k_kernel<<<1, 1, 0, stream>>>(
      seq_lens_encoder.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_decoder.data<int>(),
      cu_seqlens_k.data<int>(),
      qk_tokens.data<int>(),
      bsz);

  auto qk_tokens_cpu = qk_tokens.copy_to(paddle::CPUPlace(), true);
  return {cu_seqlens_k, qk_tokens_cpu};
}

}  // namespace dynamic_quant_cache_attn

PD_BUILD_OP(split_qkv_and_rope)
    .Inputs({"qkv_out",
             "q_input",
             "k_input",
             "v_input",
             "rotary_embs",
             "seq_len_encoder",
             "seq_len_decoder",
             "cu_seq_q",
             "cu_seq_k",
             paddle::Optional("qkv_bias")})
    .Attrs({"head_num: int",
            "kv_head_num: int",
            "head_dim: int",
            "max_seq_q: int",
            "max_input_length: int",
            "cache_quant_type_str: std::string"})
    .Outputs({"q_input_out", "k_input_out", "v_input_out"})
    .SetInplaceMap({{"q_input", "q_input_out"},
                    {"k_input", "k_input_out"},
                    {"v_input", "v_input_out"}})
    .SetKernelFn(PD_KERNEL(dynamic_quant_cache_attn::SplitQKVAndRope));

PD_BUILD_OP(get_qk_tokens_num)
    .Inputs({"seq_lens_encoder", "seq_lens_this_time", "seq_lens_decoder"})
    .Outputs({"cu_seqlens_k", "qk_tokens"})
    .SetKernelFn(PD_KERNEL(dynamic_quant_cache_attn::GetQKTokenNum));
