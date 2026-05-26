// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "flash_attn.h"
#include "helper.h"
#include "paddle/extension.h"

void flash_attn_varlen_forward(
    const paddle::Tensor& q,  // [num_tokens, num_heads, head_dim]
    const paddle::Tensor&
        k,  // [kv_num_tokens, kv_num_heads, head_dim] or [num_blocks,
            // block_size, kv_num_heads, head_dim]
    const paddle::Tensor&
        v,  // [kv_num_tokens, kv_num_heads, head_dim] or [num_blocks,
            // block_size, kv_num_heads, head_dim]
    const paddle::Tensor& cu_seqlens_q,  // [batch_size + 1,]
    const paddle::Tensor& cu_seqlens_k,  // [batch_size + 1,]
    const paddle::optional<paddle::Tensor>&
        block_tables,  // [batch_size, max_blocks_per_seq]
    const paddle::optional<paddle::Tensor>&
        alibi_slopes,     // [num_heads,] or [batch_size, num_heads]
    paddle::Tensor& out,  // [num_tokens, num_heads, head_dim]
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float dropout_p,
    const float softmax_scale,
    const bool causal,
    const int window_size_left,
    const int window_size_right,
    const float softcap,
    const cudaStream_t& stream) {
  if (q.dtype() != paddle::DataType::BFLOAT16 ||
      k.dtype() != paddle::DataType::BFLOAT16 ||
      v.dtype() != paddle::DataType::BFLOAT16) {
    PD_THROW("Only support q/k/v dtype of BF16");
  }
  using data_t = phi::dtype::bfloat16;

  const auto q_shape = q.shape();
  const auto k_shape = k.shape();
  const auto num_tokens = q_shape[0];
  const auto kv_num_tokens = k_shape[0];
  const auto num_heads = q_shape[1];
  const auto kv_num_heads = block_tables ? k_shape[2] : k_shape[1];
  const auto head_dim = q_shape[2];
  const auto num_blocks = k_shape[0];
  const auto block_size = k_shape[1];
  const auto batch_size = cu_seqlens_q.shape()[0] - 1;
  const auto max_blocks_per_seq = block_tables ? block_tables->shape()[1] : 0;

  auto softmax_lse = paddle::empty({batch_size, num_heads, max_seqlen_q},
                                   paddle::DataType::FLOAT32,
                                   q.place());

  auto tensor_q =
      make_contiguous_tensor3d(const_cast<data_t*>(q.data<data_t>()),
                               MCFLASHATTN_DATATYPE_BF16,
                               num_tokens,
                               num_heads,
                               head_dim);

  Tensor_t tensor_k = nullptr;
  if (k_shape.size() == 4) {
    tensor_k = make_contiguous_tensor4d(const_cast<data_t*>(k.data<data_t>()),
                                        MCFLASHATTN_DATATYPE_BF16,
                                        num_blocks,
                                        block_size,
                                        kv_num_heads,
                                        head_dim);
  } else {
    tensor_k = make_contiguous_tensor3d(const_cast<data_t*>(k.data<data_t>()),
                                        MCFLASHATTN_DATATYPE_BF16,
                                        kv_num_tokens,
                                        kv_num_heads,
                                        head_dim);
  }

  Tensor_t tensor_v = nullptr;
  if (k_shape.size() == 4) {
    tensor_v = make_contiguous_tensor4d(const_cast<data_t*>(v.data<data_t>()),
                                        MCFLASHATTN_DATATYPE_BF16,
                                        num_blocks,
                                        block_size,
                                        kv_num_heads,
                                        head_dim);
  } else {
    tensor_v = make_contiguous_tensor3d(const_cast<data_t*>(v.data<data_t>()),
                                        MCFLASHATTN_DATATYPE_BF16,
                                        kv_num_tokens,
                                        kv_num_heads,
                                        head_dim);
  }

  auto tensor_out = make_contiguous_tensor3d(out.data<data_t>(),
                                             MCFLASHATTN_DATATYPE_BF16,
                                             num_tokens,
                                             num_heads,
                                             head_dim);

  auto tensor_cu_seqlens_q = make_contiguous_tensor1d(
      const_cast<int32_t*>(cu_seqlens_q.data<int32_t>()),
      MCFLASHATTN_DATATYPE_INT32,
      batch_size + 1);

  auto tensor_cu_seqlens_k = make_contiguous_tensor1d(
      const_cast<int32_t*>(cu_seqlens_k.data<int32_t>()),
      MCFLASHATTN_DATATYPE_INT32,
      batch_size + 1);

  Tensor_t tensor_alibi_slopes = nullptr;
  if (alibi_slopes) {
    if (alibi_slopes->shape().size() == 1) {
      tensor_alibi_slopes = make_contiguous_tensor1d(
          const_cast<float*>(alibi_slopes->data<float>()),
          MCFLASHATTN_DATATYPE_FP32,
          num_heads);
    } else {
      tensor_alibi_slopes = make_contiguous_tensor2d(
          const_cast<float*>(alibi_slopes->data<float>()),
          MCFLASHATTN_DATATYPE_FP32,
          batch_size,
          num_heads);
    }
  }

  auto tensor_softmax_lse = make_contiguous_tensor3d(softmax_lse.data<float>(),
                                                     MCFLASHATTN_DATATYPE_FP32,
                                                     batch_size,
                                                     num_heads,
                                                     max_seqlen_q);

  auto extend_param = make_extend_param();
  set_extend_parameter_softcap(extend_param, softcap);
  if (block_tables) {
    set_extend_parameter_block_table(
        extend_param,
        const_cast<int*>(block_tables->data<int>()),
        batch_size,
        max_blocks_per_seq);
  }

  auto status = mha_varlen_fwd(batch_size,
                               num_tokens,
                               num_heads,
                               kv_num_tokens,
                               kv_num_heads,
                               head_dim,
                               tensor_q,
                               tensor_k,
                               tensor_v,
                               tensor_out,
                               tensor_cu_seqlens_q,
                               tensor_cu_seqlens_k,
                               nullptr,  // seqused_k
                               tensor_alibi_slopes,
                               tensor_softmax_lse,
                               nullptr,  // p
                               nullptr,  // rng_state
                               max_seqlen_q,
                               max_seqlen_k,
                               dropout_p,
                               softmax_scale,
                               causal,
                               window_size_left,
                               window_size_right,
                               stream,
                               extend_param);

  if (status != MCFLASHATTN_STATUS_SUCCESS) {
    phi::errors::External("Error in McFlashAttn, error code is %d", status);
  }

  release_tensor(tensor_q);
  release_tensor(tensor_k);
  release_tensor(tensor_v);
  release_tensor(tensor_out);
  release_tensor(tensor_cu_seqlens_q);
  release_tensor(tensor_cu_seqlens_k);
  if (tensor_alibi_slopes) release_tensor(tensor_alibi_slopes);
  release_tensor(tensor_softmax_lse);
  release_extend_param(extend_param);
}

void flash_attn_kvcache_forward(
    const paddle::Tensor& q,  // [batch_size, seqlen, num_heads, head_dim]
    const paddle::Tensor&
        k_cache,  // [batch_size_c, seqlen_c, kv_num_heads, head_dim]
                  // or [num_blocks, block_size, kv_num_heads, head_dim]
    const paddle::Tensor&
        v_cache,  // [batch_size_c, seqlen_c, kv_num_heads, head_dim]
                  // or [num_blocks, block_size, kv_num_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        k,  // [batch_size, seqlen_kv, kv_num_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        v,  // [batch_size, seqlen_kv, kv_num_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        rotary_cos,  // [seqlen_ro, rotary_dim / 2]
    const paddle::optional<paddle::Tensor>&
        rotary_sin,  // [seqlen_ro, rotary_dim / 2]
    const paddle::optional<paddle::Tensor>& cache_seqlens,    // [batch_size,]
    const paddle::optional<paddle::Tensor>& cache_batch_idx,  // [batch_size,]
    const paddle::optional<paddle::Tensor>& cache_leftpad,    // [batch_size,]
    const paddle::optional<paddle::Tensor>&
        block_tables,  // [batch_size, max_blocks_per_seq]
    const paddle::optional<paddle::Tensor>&
        alibi_slopes,     // [num_heads,] or [batch_size, num_heads]
    paddle::Tensor& out,  // [batch_size, seqlen, num_heads, head_dim]
    const float softmax_scale,
    const bool causal,
    const int window_size_left,
    const int window_size_right,
    const float softcap,
    const bool rotary_interleaved,
    const int num_splits,
    const cudaStream_t& stream) {
  if (q.dtype() != paddle::DataType::BFLOAT16 ||
      k_cache.dtype() != paddle::DataType::BFLOAT16 ||
      v_cache.dtype() != paddle::DataType::BFLOAT16) {
    PD_THROW("Only support q/k_cache/v_cache dtype of BF16");
  }
  using data_t = phi::dtype::bfloat16;

  const auto q_shape = q.shape();
  const auto k_cache_shape = k_cache.shape();
  const auto batch_size = q_shape[0];
  const auto batch_size_c = k_cache_shape[0];
  const auto seqlen = q_shape[1];
  const auto seqlen_c = k_cache_shape[1];
  const auto seqlen_kv = k ? k->shape()[1] : 0;
  const auto num_heads = q_shape[2];
  const auto kv_num_heads = k_cache_shape[2];
  const auto head_dim = q_shape[3];
  const auto seqlen_ro = rotary_cos ? rotary_cos->shape()[0] : 0;
  const auto half_rotary_dim = rotary_cos ? rotary_cos->shape()[1] : 0;
  const auto max_blocks_per_seq = block_tables ? block_tables->shape()[1] : 0;

  auto softmax_lse = paddle::empty(
      {batch_size, num_heads, seqlen}, paddle::DataType::FLOAT32, q.place());

  auto tensor_q =
      make_contiguous_tensor4d(const_cast<data_t*>(q.data<data_t>()),
                               MCFLASHATTN_DATATYPE_BF16,
                               batch_size,
                               seqlen,
                               num_heads,
                               head_dim);

  auto tensor_k_cache =
      make_contiguous_tensor4d(const_cast<data_t*>(k_cache.data<data_t>()),
                               MCFLASHATTN_DATATYPE_BF16,
                               batch_size_c,
                               seqlen_c,
                               kv_num_heads,
                               head_dim);

  auto tensor_v_cache =
      make_contiguous_tensor4d(const_cast<data_t*>(v_cache.data<data_t>()),
                               MCFLASHATTN_DATATYPE_BF16,
                               batch_size_c,
                               seqlen_c,
                               kv_num_heads,
                               head_dim);

  Tensor_t tensor_k = nullptr;
  if (k) {
    tensor_k = make_contiguous_tensor4d(const_cast<data_t*>(k->data<data_t>()),
                                        MCFLASHATTN_DATATYPE_BF16,
                                        batch_size,
                                        seqlen_kv,
                                        kv_num_heads,
                                        head_dim);
  }

  Tensor_t tensor_v = nullptr;
  if (v) {
    tensor_v = make_contiguous_tensor4d(const_cast<data_t*>(v->data<data_t>()),
                                        MCFLASHATTN_DATATYPE_BF16,
                                        batch_size,
                                        seqlen_kv,
                                        kv_num_heads,
                                        head_dim);
  }

  Tensor_t tensor_cache_seqlens = nullptr;
  if (cache_seqlens) {
    tensor_cache_seqlens =
        make_contiguous_tensor1d(const_cast<int*>(cache_seqlens->data<int>()),
                                 MCFLASHATTN_DATATYPE_INT32,
                                 batch_size);
  }

  Tensor_t tensor_rotary_cos = nullptr;
  if (rotary_cos) {
    tensor_rotary_cos = make_contiguous_tensor2d(
        const_cast<data_t*>(rotary_cos->data<data_t>()),
        MCFLASHATTN_DATATYPE_BF16,
        seqlen_ro,
        half_rotary_dim);
  }

  Tensor_t tensor_rotary_sin = nullptr;
  if (rotary_sin) {
    tensor_rotary_sin = make_contiguous_tensor2d(
        const_cast<data_t*>(rotary_sin->data<data_t>()),
        MCFLASHATTN_DATATYPE_BF16,
        seqlen_ro,
        half_rotary_dim);
  }

  Tensor_t tensor_cache_batch_idx = nullptr;
  if (cache_batch_idx) {
    tensor_cache_batch_idx =
        make_contiguous_tensor1d(const_cast<int*>(cache_batch_idx->data<int>()),
                                 MCFLASHATTN_DATATYPE_INT32,
                                 batch_size);
  }

  Tensor_t tensor_block_tables = nullptr;
  if (block_tables) {
    tensor_block_tables =
        make_contiguous_tensor2d(const_cast<int*>(block_tables->data<int>()),
                                 MCFLASHATTN_DATATYPE_INT32,
                                 batch_size,
                                 max_blocks_per_seq);
  }

  Tensor_t tensor_alibi_slopes = nullptr;
  if (alibi_slopes) {
    if (alibi_slopes->shape().size() == 1) {
      tensor_alibi_slopes = make_contiguous_tensor1d(
          const_cast<float*>(alibi_slopes->data<float>()),
          MCFLASHATTN_DATATYPE_FP32,
          num_heads);
    } else {
      tensor_alibi_slopes = make_contiguous_tensor2d(
          const_cast<float*>(alibi_slopes->data<float>()),
          MCFLASHATTN_DATATYPE_FP32,
          batch_size,
          num_heads);
    }
  }

  auto tensor_out = make_contiguous_tensor4d(out.data<data_t>(),
                                             MCFLASHATTN_DATATYPE_BF16,
                                             batch_size,
                                             seqlen,
                                             num_heads,
                                             head_dim);

  auto tensor_softmax_lse = make_contiguous_tensor3d(softmax_lse.data<float>(),
                                                     MCFLASHATTN_DATATYPE_FP32,
                                                     batch_size,
                                                     num_heads,
                                                     seqlen);

  auto extend_param = make_extend_param();
  set_extend_parameter_softcap(extend_param, softcap);
  if (cache_leftpad) {
    set_extend_parameter_leftpad(
        extend_param, const_cast<int*>(cache_leftpad->data<int>()), batch_size);
  }

  auto status = mha_fwd_kvcache(tensor_q,
                                tensor_k_cache,
                                tensor_v_cache,
                                tensor_k,
                                tensor_v,
                                tensor_cache_seqlens,
                                tensor_rotary_cos,
                                tensor_rotary_sin,
                                tensor_cache_batch_idx,
                                tensor_block_tables,
                                tensor_alibi_slopes,
                                tensor_softmax_lse,
                                tensor_out,
                                softmax_scale,
                                causal,
                                window_size_left,
                                window_size_right,
                                rotary_interleaved,
                                stream,
                                num_splits,
                                nullptr,  // softmax_lse_accum
                                nullptr,  // out_accum
                                extend_param);

  if (status != MCFLASHATTN_STATUS_SUCCESS) {
    phi::errors::External("Error in McFlashAttn, error code is %d", status);
  }

  release_tensor(tensor_q);
  release_tensor(tensor_k_cache);
  release_tensor(tensor_v_cache);
  if (tensor_k) release_tensor(tensor_k);
  if (tensor_v) release_tensor(tensor_v);
  if (tensor_cache_seqlens) release_tensor(tensor_cache_seqlens);
  if (tensor_rotary_cos) release_tensor(tensor_rotary_cos);
  if (tensor_rotary_sin) release_tensor(tensor_rotary_sin);
  if (tensor_cache_batch_idx) release_tensor(tensor_cache_batch_idx);
  if (tensor_block_tables) release_tensor(tensor_block_tables);
  if (tensor_alibi_slopes) release_tensor(tensor_alibi_slopes);
  release_tensor(tensor_softmax_lse);
  release_tensor(tensor_out);
  release_extend_param(extend_param);
}

void FlashAttnVarlenForward(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    const paddle::optional<paddle::Tensor>& block_tables,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    paddle::Tensor& out,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float dropout_p,
    const float softmax_scale,
    const bool causal,
    const int window_size_left,
    const int window_size_right,
    const float softcap) {
  flash_attn_varlen_forward(q,
                            k,
                            v,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            block_tables,
                            alibi_slopes,
                            out,
                            max_seqlen_q,
                            max_seqlen_k,
                            dropout_p,
                            softmax_scale,
                            causal,
                            window_size_left,
                            window_size_right,
                            softcap,
                            q.stream());
}

void FlashAttnKVCacheForward(
    const paddle::Tensor& q,
    const paddle::Tensor& k_cache,
    const paddle::Tensor& v_cache,
    const paddle::optional<paddle::Tensor>& k,
    const paddle::optional<paddle::Tensor>& v,
    const paddle::optional<paddle::Tensor>& rotary_cos,
    const paddle::optional<paddle::Tensor>& rotary_sin,
    const paddle::optional<paddle::Tensor>& cache_seqlens,
    const paddle::optional<paddle::Tensor>& cache_batch_idx,
    const paddle::optional<paddle::Tensor>& cache_leftpad,
    const paddle::optional<paddle::Tensor>& block_tables,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    paddle::Tensor& out,
    const float softmax_scale,
    const bool causal,
    const int window_size_left,
    const int window_size_right,
    const float softcap,
    const bool rotary_interleaved,
    const int num_splits) {
  flash_attn_kvcache_forward(q,
                             k_cache,
                             v_cache,
                             k,
                             v,
                             rotary_cos,
                             rotary_sin,
                             cache_seqlens,
                             cache_batch_idx,
                             cache_leftpad,
                             block_tables,
                             alibi_slopes,
                             out,
                             softmax_scale,
                             causal,
                             window_size_left,
                             window_size_right,
                             softcap,
                             rotary_interleaved,
                             num_splits,
                             q.stream());
}

PD_BUILD_STATIC_OP(flash_attn_varlen_forward)
    .Inputs({"q",
             "k",
             "v",
             "cu_seqlens_q",
             "cu_seqlens_k",
             paddle::Optional("block_tables"),
             paddle::Optional("alibi_slopes"),
             "out"})
    .Outputs({"mha_out"})
    .Attrs({"max_seqlen_q:int",
            "max_seqlen_k:int",
            "dropout_p:float",
            "softmax_scale:float",
            "causal:bool",
            "window_size_left:int",
            "window_size_right:int",
            "softcap:float"})
    .SetInplaceMap({{"out", "mha_out"}})
    .SetKernelFn(PD_KERNEL(FlashAttnVarlenForward));

PD_BUILD_STATIC_OP(flash_attn_kvcache_forward)
    .Inputs({"q",
             "k_cache",
             "v_cache",
             paddle::Optional("k"),
             paddle::Optional("v"),
             paddle::Optional("rotary_cos"),
             paddle::Optional("rotary_sin"),
             paddle::Optional("cache_seqlens"),
             paddle::Optional("cache_batch_idx"),
             paddle::Optional("cache_leftpad"),
             paddle::Optional("block_tables"),
             paddle::Optional("alibi_slopes"),
             "out"})
    .Outputs({"mha_out"})
    .Attrs({"softmax_scale:float",
            "causal:bool",
            "window_size_left:int",
            "window_size_right:int",
            "softcap:float",
            "rotary_interleaved:bool",
            "num_splits:int"})
    .SetInplaceMap({{"out", "mha_out"}})
    .SetKernelFn(PD_KERNEL(FlashAttnKVCacheForward));
