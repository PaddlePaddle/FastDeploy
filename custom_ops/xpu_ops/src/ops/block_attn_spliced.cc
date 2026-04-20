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

#include <blocks/core_attention_block.h>
#include <core/check.h>
#include <core/context.h>
#include <core/param.h>
#include <core/types.h>
#include <flash_api.h>
#include <infer_ops.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xft_api.h>

#include "ops/pybind/cachekv_signal_thread_worker.h"
#include "ops/remote_cache_kv_ipc.h"
#include "ops/utility/env.h"
#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

XPU_DECLARE_BOOL(fmt_write_cache_completed_signal, false);
XPU_DECLARE_BOOL(use_pd_disaggregation_per_chunk, false);
XPU_DECLARE_BOOL(encoder_splice, false);
XPU_DECLARE_BOOL(decoder_splice, false);
XPU_DECLARE_BOOL(use_sdnn_rmsnorm, false);

namespace xftblock = baidu::xpu::xftblock;
namespace api = baidu::xpu::api;

template <typename TC, typename TS>
struct SplitRopeTypeTrait {
  using E_Scale = TS;
  using D_Scale = TS;
};
template <>
struct SplitRopeTypeTrait<bfloat16, bfloat16> {
  using E_Scale = bfloat16;
  using D_Scale = float;
};
template <>
struct SplitRopeTypeTrait<int8_t, bfloat16> {
  using E_Scale = bfloat16;
  using D_Scale = bfloat16;
};

void do_add_zero(api::Context* xpu_ctx,
                 paddle::Place place,
                 bfloat16* x,
                 int64_t token_num,
                 int64_t kv_head_num,
                 int64_t head_dim,
                 const float* cache_zero) {
  if (cache_zero == nullptr) {
    return;
  }
  std::vector<int64_t> x_shape = {token_num, kv_head_num * head_dim};
  std::vector<int64_t> cache_zero_shape = {1, kv_head_num * head_dim};
  int64_t x_numel = token_num * kv_head_num * head_dim;

  int ret;
  auto x_fp32 = paddle::empty(x_shape, paddle::DataType::FLOAT32, place);
  auto x_fp32_ptr = const_cast<float*>(x_fp32.data<float>());

  ret = api::cast<bfloat16, float>(xpu_ctx, x, x_fp32_ptr, x_numel);
  PD_CHECK(ret == api::SUCCESS, "api::cast failed.");
  ret = api::broadcast_add<float>(
      xpu_ctx, x_fp32_ptr, cache_zero, x_fp32_ptr, x_shape, cache_zero_shape);
  PD_CHECK(ret == api::SUCCESS, "api::broadcast_add failed.");
  ret = api::cast<float, bfloat16>(xpu_ctx, x_fp32_ptr, x, x_numel);
  PD_CHECK(ret == api::SUCCESS, "api::cast failed.");
}

template <typename TKV_CACHE, typename TID, typename TSCALE, typename TZERO>
void store_paged_kv_cache_wrapper(api::Context* xpu_ctx,
                                  paddle::Place place,
                                  bfloat16* k,
                                  bfloat16* v,
                                  TKV_CACHE* key_cache,
                                  TKV_CACHE* value_cache,
                                  TID* slot_mapping,
                                  int64_t num_blocks,
                                  int64_t token_num,
                                  int64_t kv_head_num,
                                  int64_t head_dim,
                                  int64_t block_size,
                                  const TSCALE* k_cache_scale,
                                  const TSCALE* v_cache_scale,
                                  const TZERO* k_cache_zero,
                                  const TZERO* v_cache_zero) {
  std::vector<int64_t> cache_zero_shape = {1, kv_head_num * head_dim};
  int64_t cache_zero_numel = kv_head_num * head_dim;

  int ret;
  paddle::Tensor k_cache_scale_fp32, v_cache_scale_fp32, k_cache_zero_fp32,
      v_cache_zero_fp32;
  float* k_cache_scale_fp32_ptr = nullptr;
  float* v_cache_scale_fp32_ptr = nullptr;
  float* k_cache_zero_fp32_ptr = nullptr;
  float* v_cache_zero_fp32_ptr = nullptr;

  if (k_cache_scale != nullptr) {
    if (!std::is_same<TSCALE, float>::value) {
      k_cache_scale_fp32 =
          paddle::empty(cache_zero_shape, paddle::DataType::FLOAT32, place);
      v_cache_scale_fp32 =
          paddle::empty(cache_zero_shape, paddle::DataType::FLOAT32, place);
      k_cache_scale_fp32_ptr =
          const_cast<float*>(k_cache_scale_fp32.data<float>());
      v_cache_scale_fp32_ptr =
          const_cast<float*>(v_cache_scale_fp32.data<float>());
      ret = api::cast<TSCALE, float>(
          xpu_ctx, k_cache_scale, k_cache_scale_fp32_ptr, cache_zero_numel);
      PD_CHECK(ret == api::SUCCESS, "api::cast failed.");
      ret = api::cast<TSCALE, float>(
          xpu_ctx, v_cache_scale, v_cache_scale_fp32_ptr, cache_zero_numel);
      PD_CHECK(ret == api::SUCCESS, "api::cast failed.");
    } else {
      k_cache_scale_fp32_ptr =
          const_cast<float*>(reinterpret_cast<const float*>(k_cache_scale));
      v_cache_scale_fp32_ptr =
          const_cast<float*>(reinterpret_cast<const float*>(v_cache_scale));
    }
  }
  if (k_cache_zero != nullptr) {
    if (!std::is_same<TZERO, float>::value) {
      k_cache_zero_fp32 =
          paddle::empty(cache_zero_shape, paddle::DataType::FLOAT32, place);
      v_cache_zero_fp32 =
          paddle::empty(cache_zero_shape, paddle::DataType::FLOAT32, place);
      k_cache_zero_fp32_ptr =
          const_cast<float*>(k_cache_zero_fp32.data<float>());
      v_cache_zero_fp32_ptr =
          const_cast<float*>(v_cache_zero_fp32.data<float>());
      ret = api::cast<TZERO, float>(
          xpu_ctx, k_cache_zero, k_cache_zero_fp32_ptr, cache_zero_numel);
      PD_CHECK(ret == api::SUCCESS, "api::cast failed.");
      ret = api::cast<TZERO, float>(
          xpu_ctx, v_cache_zero, v_cache_zero_fp32_ptr, cache_zero_numel);
      PD_CHECK(ret == api::SUCCESS, "api::cast failed.");
    } else {
      k_cache_zero_fp32_ptr =
          const_cast<float*>(reinterpret_cast<const float*>(k_cache_zero));
      v_cache_zero_fp32_ptr =
          const_cast<float*>(reinterpret_cast<const float*>(v_cache_zero));
    }
  }

  if (k_cache_zero != nullptr) {
    do_add_zero(xpu_ctx,
                place,
                k,
                token_num,
                kv_head_num,
                head_dim,
                k_cache_zero_fp32_ptr);
    do_add_zero(xpu_ctx,
                place,
                v,
                token_num,
                kv_head_num,
                head_dim,
                v_cache_zero_fp32_ptr);
  }

  ret = infer_ops::store_paged_kv_cache<bfloat16, TKV_CACHE, TID>(
      xpu_ctx,
      k,
      v,
      key_cache,
      value_cache,
      slot_mapping,
      k_cache_scale_fp32_ptr,
      v_cache_scale_fp32_ptr,
      token_num,
      kv_head_num,
      head_dim,
      num_blocks,
      block_size);
  PD_CHECK(ret == api::SUCCESS, "store_paged_kv_cache failed.");
}

template <typename TQKV,
          typename TR,
          typename TKV_CACHE,
          typename TID,
          typename TSCALE>
void split_kvcache_encoder(api::Context* xpu_ctx,
                           xftblock::XFTContext& xctx,
                           const paddle::Tensor& qkv,
                           const paddle::Tensor& rotary_embs,
                           const paddle::Tensor& q,
                           const paddle::Tensor& k,
                           const paddle::Tensor& v,
                           const paddle::Tensor& key_cache,
                           const paddle::Tensor& value_cache,
                           const paddle::Tensor& block_tables,
                           const paddle::Tensor& slot_mapping,
                           int64_t batch_size,
                           int64_t token_num,
                           int64_t q_num_heads,
                           int64_t kv_num_heads,
                           int64_t head_dim,
                           int64_t rope_head_dim,
                           int64_t hidden_dim,
                           int64_t rope_max_seqlen,
                           int64_t block_size,
                           int64_t num_blocks,
                           int64_t block_batch,
                           int64_t max_block_per_seq,
                           const api::VectorParam<int32_t>& seq_lod,
                           const api::VectorParam<int32_t>& start_tokens,
                           const api::VectorParam<int32_t>& real_batch,
                           int64_t qkv_offset,
                           const float* k_cache_scale_inv,
                           const float* v_cache_scale_inv,
                           const TSCALE* intx_k_pc_scale,
                           const TSCALE* intx_v_pc_scale,
                           const TSCALE* intx_k_pc_zero,
                           const TSCALE* intx_v_pc_zero,
                           const float* q_norm_weight,
                           const float* k_norm_weight,
                           std::string pos_emb_type,
                           bool rope_3d,
                           bool use_neox_rotary_style) {
  int ret;
  int64_t real_kv_num_heads = (kv_num_heads == -1) ? q_num_heads : kv_num_heads;
  // TODO: spliced split kvcache should support rope3d
  if (FLAGS_encoder_splice && !rope_3d) {
    if (rope_3d) {
      PD_THROW("split_kvcache_encoder does not support rope_3d == true!");
    }
    paddle::Place place = qkv.place();
    xftblock::DataType KV_BUF_TYPE = std::is_same<bfloat16, TQKV>::value
                                         ? xftblock::DataType::DT_BFLOAT16
                                         : xftblock::DataType::DT_FLOAT16;
    auto q_split = paddle::empty({token_num, hidden_dim}, qkv.type(), place);
    auto k_split = paddle::empty(
        {token_num, real_kv_num_heads * head_dim}, qkv.type(), place);
    xftblock::Tensor qkv_xft_tensor(
        const_cast<void*>(qkv.data() + qkv_offset * sizeof(TQKV)),
        KV_BUF_TYPE,
        {token_num, (q_num_heads + 2 * real_kv_num_heads) * head_dim});
    xftblock::Tensor q_xft_tensor(
        q_split.data(), KV_BUF_TYPE, {token_num, hidden_dim});
    xftblock::Tensor k_xft_tensor(
        k_split.data(), KV_BUF_TYPE, {token_num, real_kv_num_heads * head_dim});
    xftblock::Tensor v_xft_tensor(const_cast<void*>(v.data()),
                                  KV_BUF_TYPE,
                                  {token_num, real_kv_num_heads * head_dim});

    ret = xftblock::split_qkv_block<TQKV>(&xctx,
                                          &qkv_xft_tensor,
                                          &q_xft_tensor,
                                          &k_xft_tensor,
                                          &v_xft_tensor,
                                          token_num,
                                          q_num_heads,
                                          real_kv_num_heads,
                                          head_dim);
    PD_CHECK(ret == api::SUCCESS, "split_qkv_block failed.");

    if (!use_neox_rotary_style) {
      ret = infer_ops::vsl_rotary_embedding_gptj<TQKV, TR, TID>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(q_split.data()),
          reinterpret_cast<const TQKV*>(k_split.data()),
          reinterpret_cast<const float*>(rotary_embs.data<float>()),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          seq_lod,
          1,
          rope_max_seqlen,
          q_num_heads,
          head_dim,
          "BLHD",
          start_tokens,
          "NORMAL",
          real_kv_num_heads,
          false);
      PD_CHECK(ret == api::SUCCESS, "vsl_rotary_embedding_gptj failed.");
    } else {
      ret = infer_ops::vsl_rotary_embedding_neox<TQKV, TR, TID>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(q_split.data()),
          reinterpret_cast<const TQKV*>(k_split.data()),
          reinterpret_cast<const float*>(rotary_embs.data<float>()),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          seq_lod,
          1,
          rope_max_seqlen,
          q_num_heads,
          head_dim,
          rope_head_dim,
          "BLHD",
          start_tokens,
          "NORMAL",
          real_kv_num_heads,
          false);
      PD_CHECK(ret == api::SUCCESS, "vsl_rotary_embedding_neox failed.");
    }

    if (q_norm_weight) {
      ret = infer_ops::qkrmsnorm<TQKV, float>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(q.data()),
          q_num_heads * head_dim,
          head_dim,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          q_num_heads * head_dim,
          head_dim,
          head_dim,
          token_num,
          q_num_heads,
          1e-5,
          q_norm_weight,
          nullptr,  // not supported yet
          false,    // not supported yet
          FLAGS_use_sdnn_rmsnorm,
          false);
    }
    if (k_norm_weight) {
      ret = infer_ops::qkrmsnorm<TQKV, float>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(k.data()),
          real_kv_num_heads * head_dim,
          head_dim,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          real_kv_num_heads * head_dim,
          head_dim,
          head_dim,
          token_num,
          real_kv_num_heads,
          1e-5,
          k_norm_weight,
          nullptr,  // not supported yet
          false,    // not supported yet
          FLAGS_use_sdnn_rmsnorm,
          false);
    }

    // write to cache
    if (std::is_same<TKV_CACHE, int8_t>::value && intx_k_pc_scale &&
        intx_v_pc_scale) {
      store_paged_kv_cache_wrapper<TKV_CACHE, TID, TSCALE, TSCALE>(
          xpu_ctx,
          place,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(v.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
          const_cast<TID*>(slot_mapping.data<TID>()),
          num_blocks,
          token_num,
          real_kv_num_heads,
          head_dim,
          block_size,
          intx_k_pc_scale,
          intx_v_pc_scale,
          intx_k_pc_zero,
          intx_v_pc_zero);
    } else {
      float* k_scale_cache_ptr = nullptr;
      float* v_scale_cache_ptr = nullptr;
      paddle::Tensor k_scale_cache, v_scale_cache;
      if (k_cache_scale_inv) {
        k_scale_cache = paddle::empty(
            {real_kv_num_heads}, paddle::DataType::FLOAT32, place);
        k_scale_cache_ptr = const_cast<float*>(k_scale_cache.data<float>());
        ret = api::reciprocal<float>(
            xpu_ctx, k_cache_scale_inv, k_scale_cache_ptr, real_kv_num_heads);
        PD_CHECK(ret == api::SUCCESS, "api::reciprocal failed.");
      }
      if (v_cache_scale_inv) {
        v_scale_cache = paddle::empty(
            {real_kv_num_heads}, paddle::DataType::FLOAT32, place);
        v_scale_cache_ptr = const_cast<float*>(v_scale_cache.data<float>());
        ret = api::reciprocal<float>(
            xpu_ctx, v_cache_scale_inv, v_scale_cache_ptr, real_kv_num_heads);
        PD_CHECK(ret == api::SUCCESS, "api::reciprocal failed.");
      }
      store_paged_kv_cache_wrapper<TKV_CACHE, TID, float, float>(
          xpu_ctx,
          place,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(v.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
          const_cast<TID*>(slot_mapping.data<TID>()),
          num_blocks,
          token_num,
          real_kv_num_heads,
          head_dim,
          block_size,
          k_scale_cache_ptr,
          v_scale_cache_ptr,
          nullptr,
          nullptr);
    }
  } else {
    if (use_neox_rotary_style) {
      ret = infer_ops::split_neox_cache_kv_encoder<TQKV, float, TKV_CACHE, int>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(qkv.data()) + qkv_offset,  // qkv
          reinterpret_cast<const float*>(
              rotary_embs.data<float>()),  // rotary_pos_emb
          reinterpret_cast<const int*>(
              block_tables.data<int>()),  // block_table
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(v.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
          seq_lod,          // seq_lod
          real_batch,       // real_batch
          start_tokens,     // start_tokens
          batch_size,       // batch_size
          1,                // emb_batch_size
          rope_max_seqlen,  // max_seqlen
          q_num_heads,
          real_kv_num_heads,
          head_dim,
          rope_head_dim,
          block_batch,
          block_size,
          max_block_per_seq,
          "BLHD",
          "HLD",
          pos_emb_type,
          nullptr,  // k_cache_scale_inv - use for per head
          nullptr,  // v_cache_scale_inv - use for per head
          nullptr,  // intx_k_pc_scale
          nullptr,  // intx_v_pc_scale
          nullptr,  // intx_k_pc_zero
          nullptr,  // intx_v_pc_zero
          rope_3d);
      PD_CHECK(ret == api::SUCCESS, "split_neox_cache_kv_encoder failed.");
    } else {
      ret = infer_ops::
          split_rope_cache_kv_encoder<TQKV, float, TKV_CACHE, int, TSCALE>(
              xpu_ctx,
              reinterpret_cast<const TQKV*>(qkv.data()) + qkv_offset,  // qkv
              reinterpret_cast<const float*>(
                  rotary_embs.data<float>()),  // rotary_pos_emb
              reinterpret_cast<const int*>(
                  block_tables.data<int>()),  // block_table
              const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
              const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
              const_cast<TQKV*>(reinterpret_cast<const TQKV*>(v.data())),
              const_cast<TKV_CACHE*>(
                  reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
              const_cast<TKV_CACHE*>(
                  reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
              seq_lod,          // seq_lod
              real_batch,       // real_batch
              start_tokens,     // start_tokens
              batch_size,       // batch_size
              1,                // emb_batch_size
              rope_max_seqlen,  // max_seqlen
              q_num_heads,
              real_kv_num_heads,
              head_dim,
              block_batch,
              block_size,
              max_block_per_seq,
              "BLHD",
              "HLD",
              pos_emb_type,
              k_cache_scale_inv,  // k_cache_scale_inv - use for per head
              v_cache_scale_inv,  // v_cache_scale_inv - use for per head
              intx_k_pc_scale,    // intx_k_pc_scale
              intx_v_pc_scale,    // intx_v_pc_scale
              intx_k_pc_zero,     // intx_k_pc_zero
              intx_v_pc_zero,     // intx_v_pc_zero
              q_norm_weight,
              k_norm_weight,
              rope_3d);
      PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_encoder failed.");
    }
  }
}

template <typename TQKV,
          typename TR,
          typename TKV_CACHE,
          typename TID,
          typename TSCALE>
void split_kvcache_decoder(api::Context* xpu_ctx,
                           xftblock::XFTContext& xctx,
                           const paddle::Tensor& qkv,
                           const paddle::Tensor& rotary_embs,
                           const paddle::Tensor& q,
                           const paddle::Tensor& k,
                           const paddle::Tensor& v,
                           const paddle::Tensor& key_cache,
                           const paddle::Tensor& value_cache,
                           const paddle::Tensor& block_tables,
                           const paddle::Tensor& slot_mapping,
                           int64_t batch_size,
                           int64_t token_num,
                           int64_t q_num_heads,
                           int64_t kv_num_heads,
                           int64_t head_dim,
                           int64_t rope_head_dim,
                           int64_t hidden_dim,
                           int64_t rope_max_seqlen,
                           int64_t block_size,
                           int64_t num_blocks,
                           int64_t block_batch,
                           int64_t max_block_per_seq,
                           const api::VectorParam<int32_t>& seq_lod,
                           const api::VectorParam<int32_t>& seq_lod_for_fused,
                           const api::VectorParam<int32_t>& start_tokens,
                           const api::VectorParam<int32_t>& real_batch,
                           int64_t qkv_offset,
                           const TSCALE* k_cache_scale_inv,
                           const TSCALE* v_cache_scale_inv,
                           const TSCALE* k_pc_zero,
                           const TSCALE* v_pc_zero,
                           const float* q_norm_weight,
                           const float* k_norm_weight,
                           std::string pos_emb_type,
                           bool rope_3d,
                           bool b_c8_pc,
                           bool use_neox_rotary_style) {
  int64_t real_kv_num_heads = (kv_num_heads == -1) ? q_num_heads : kv_num_heads;
  int ret;
  // TODO: spliced split kvcache should support rope3d
  if (FLAGS_decoder_splice && !rope_3d) {
    // not yet supported
    if (rope_3d) {
      PD_THROW("split_kvcache_decoder does not support rope_3d == true!");
    }
    if (std::is_same<TKV_CACHE, int8_t>::value &&
        (k_cache_scale_inv == nullptr || v_cache_scale_inv == nullptr)) {
      PD_THROW(
          "split_kvcache_decoder of kv_cache type int8_t does not "
          "support nullptr for k_cache_scale_inv or v_cache_scale_inv!");
    }

    xftblock::DataType KV_BUF_TYPE = std::is_same<bfloat16, TQKV>::value
                                         ? xftblock::DataType::DT_BFLOAT16
                                         : xftblock::DataType::DT_FLOAT16;

    paddle::Place place = qkv.place();

    auto q_split = paddle::empty({token_num, hidden_dim}, qkv.type(), place);
    auto k_split = paddle::empty(
        {token_num, real_kv_num_heads * head_dim}, qkv.type(), place);
    xftblock::Tensor qkv_xft_tensor(
        const_cast<void*>(qkv.data() + qkv_offset * sizeof(TQKV)),
        KV_BUF_TYPE,
        {token_num, (q_num_heads + 2 * real_kv_num_heads) * head_dim});
    xftblock::Tensor q_xft_tensor(
        q_split.data(), KV_BUF_TYPE, {token_num, hidden_dim});
    xftblock::Tensor k_xft_tensor(
        k_split.data(), KV_BUF_TYPE, {token_num, real_kv_num_heads * head_dim});
    xftblock::Tensor v_xft_tensor(const_cast<void*>(v.data()),
                                  KV_BUF_TYPE,
                                  {token_num, real_kv_num_heads * head_dim});

    ret = xftblock::split_qkv_block<TQKV>(&xctx,
                                          &qkv_xft_tensor,
                                          &q_xft_tensor,
                                          &k_xft_tensor,
                                          &v_xft_tensor,
                                          token_num,
                                          q_num_heads,
                                          real_kv_num_heads,
                                          head_dim);
    PD_CHECK(ret == api::SUCCESS, "split_qkv_block failed.");

    if (!use_neox_rotary_style) {
      ret = infer_ops::vsl_rotary_embedding_gptj<TQKV, TR, TID>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(q_split.data()),
          reinterpret_cast<const TQKV*>(k_split.data()),
          reinterpret_cast<const float*>(rotary_embs.data<float>()),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          seq_lod,
          1,
          rope_max_seqlen,
          q_num_heads,
          head_dim,
          "BLHD",
          start_tokens,
          "NORMAL",
          real_kv_num_heads,
          false);
      PD_CHECK(ret == api::SUCCESS, "vsl_rotary_embedding_gptj failed.");
    } else {
      ret = infer_ops::vsl_rotary_embedding_neox<TQKV, TR, TID>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(q_split.data()),
          reinterpret_cast<const TQKV*>(k_split.data()),
          reinterpret_cast<const float*>(rotary_embs.data<float>()),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          seq_lod,
          1,
          rope_max_seqlen,
          q_num_heads,
          head_dim,
          rope_head_dim,
          "BLHD",
          start_tokens,
          "NORMAL",
          real_kv_num_heads,
          false);
      PD_CHECK(ret == api::SUCCESS, "vsl_rotary_embedding_neox failed.");
    }

    if (q_norm_weight) {
      ret = infer_ops::qkrmsnorm<TQKV, float>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(q.data()),
          q_num_heads * head_dim,
          head_dim,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
          q_num_heads * head_dim,
          head_dim,
          head_dim,
          token_num,
          q_num_heads,
          1e-5,
          q_norm_weight,
          nullptr,  // not supported yet
          false,    // not supported yet
          FLAGS_use_sdnn_rmsnorm,
          false);
    }
    if (k_norm_weight) {
      ret = infer_ops::qkrmsnorm<TQKV, float>(
          xpu_ctx,
          reinterpret_cast<const TQKV*>(k.data()),
          real_kv_num_heads * head_dim,
          head_dim,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          real_kv_num_heads * head_dim,
          head_dim,
          head_dim,
          token_num,
          real_kv_num_heads,
          1e-5,
          k_norm_weight,
          nullptr,  // not supported yet
          false,    // not supported yet
          FLAGS_use_sdnn_rmsnorm,
          false);
    }

    // write to cache
    float* k_cache_scale_fp32_ptr = nullptr;
    float* v_cache_scale_fp32_ptr = nullptr;
    paddle::Tensor k_scale_cache, v_scale_cache;
    int64_t cache_scale_zero_len =
        b_c8_pc ? real_kv_num_heads * head_dim : real_kv_num_heads;
    if (k_cache_scale_inv) {
      k_scale_cache = paddle::empty(
          {cache_scale_zero_len}, paddle::DataType::FLOAT32, place);
      k_cache_scale_fp32_ptr = const_cast<float*>(k_scale_cache.data<float>());
      ret = api::cast<TSCALE, float>(xpu_ctx,
                                     k_cache_scale_inv,
                                     k_cache_scale_fp32_ptr,
                                     cache_scale_zero_len);
      if (!b_c8_pc) {
        ret = api::reciprocal<float>(xpu_ctx,
                                     k_cache_scale_fp32_ptr,
                                     k_cache_scale_fp32_ptr,
                                     cache_scale_zero_len);
        PD_CHECK(ret == api::SUCCESS, "api::reciprocal failed.");
      }
    }
    if (v_cache_scale_inv) {
      v_scale_cache = paddle::empty(
          {cache_scale_zero_len}, paddle::DataType::FLOAT32, place);
      v_cache_scale_fp32_ptr = const_cast<float*>(v_scale_cache.data<float>());
      ret = api::cast<TSCALE, float>(xpu_ctx,
                                     v_cache_scale_inv,
                                     v_cache_scale_fp32_ptr,
                                     cache_scale_zero_len);
      if (!b_c8_pc) {
        ret = api::reciprocal<float>(xpu_ctx,
                                     v_cache_scale_fp32_ptr,
                                     v_cache_scale_fp32_ptr,
                                     cache_scale_zero_len);
        PD_CHECK(ret == api::SUCCESS, "api::reciprocal failed.");
      }
    }
    if (std::is_same<TKV_CACHE, int8_t>::value && b_c8_pc) {
      store_paged_kv_cache_wrapper<TKV_CACHE, TID, float, TSCALE>(
          xpu_ctx,
          place,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(v.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
          const_cast<TID*>(slot_mapping.data<TID>()),
          num_blocks,
          token_num,
          real_kv_num_heads,
          head_dim,
          block_size,
          k_cache_scale_fp32_ptr,
          v_cache_scale_fp32_ptr,
          k_pc_zero,
          v_pc_zero);
    } else {
      store_paged_kv_cache_wrapper<TKV_CACHE, TID, float, float>(
          xpu_ctx,
          place,
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(k.data())),
          const_cast<TQKV*>(reinterpret_cast<const TQKV*>(v.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
          const_cast<TKV_CACHE*>(
              reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
          const_cast<TID*>(slot_mapping.data<TID>()),
          num_blocks,
          token_num,
          real_kv_num_heads,
          head_dim,
          block_size,
          k_cache_scale_fp32_ptr,
          v_cache_scale_fp32_ptr,
          nullptr,
          nullptr);
    }
  } else {
    if (use_neox_rotary_style) {
      ret = infer_ops::
          split_neox_cache_kv_decoder<TQKV, float, TKV_CACHE, TSCALE, int>(
              xpu_ctx,
              reinterpret_cast<const TQKV*>(qkv.data()) + qkv_offset,  // qkv
              reinterpret_cast<const float*>(
                  rotary_embs.data<float>()),  // rotary_pos_emb
              reinterpret_cast<const int*>(
                  block_tables.data<int>()),  // block_table
              const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
              nullptr,
              nullptr,
              const_cast<TKV_CACHE*>(
                  reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
              const_cast<TKV_CACHE*>(
                  reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
              seq_lod_for_fused,  // seq_lod
              real_batch,         // real_batch
              batch_size,         // batch_size
              1,                  // emb_batch_size = rotary_embs.dims()[1] = 1
              rope_max_seqlen,    // max_seqlen
              q_num_heads,
              real_kv_num_heads,
              head_dim,
              rope_head_dim,
              block_batch,
              block_size,
              max_block_per_seq,
              "BLHD",
              "HLD",
              pos_emb_type,
              k_cache_scale_inv,  // k_cache_scale_inv
              v_cache_scale_inv,  // v_cache_scale_inv
              k_pc_zero,          // k_cache_zp
              v_pc_zero,          // v_cache_zp
              rope_3d);
    } else {
      ret = infer_ops::
          split_rope_cache_kv_decoder<TQKV, float, TKV_CACHE, TSCALE, int>(
              xpu_ctx,
              reinterpret_cast<const TQKV*>(qkv.data()) + qkv_offset,  // qkv
              reinterpret_cast<const float*>(
                  rotary_embs.data<float>()),  // rotary_pos_emb
              reinterpret_cast<const int*>(
                  block_tables.data<int>()),  // block_table
              const_cast<TQKV*>(reinterpret_cast<const TQKV*>(q.data())),
              nullptr,
              nullptr,
              const_cast<TKV_CACHE*>(
                  reinterpret_cast<const TKV_CACHE*>(key_cache.data())),
              const_cast<TKV_CACHE*>(
                  reinterpret_cast<const TKV_CACHE*>(value_cache.data())),
              seq_lod_for_fused,  // seq_lod
              real_batch,         // real_batch
              batch_size,         // batch_size
              1,                  // emb_batch_size = rotary_embs.dims()[1] = 1
              rope_max_seqlen,    // max_seqlen
              q_num_heads,
              real_kv_num_heads,
              head_dim,
              block_batch,
              block_size,
              max_block_per_seq,
              "BLHD",
              "HLD",
              pos_emb_type,
              k_cache_scale_inv,  // k_cache_scale_inv
              v_cache_scale_inv,  // v_cache_scale_inv
              k_pc_zero,          // k_cache_zp
              v_pc_zero,          // v_cache_zp
              q_norm_weight,
              k_norm_weight,
              b_c8_pc,  // bool b_c8_pc
              rope_3d);
      PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_decoder failed.");
    }
  }
}

/**
 * qkv shape: [token_num, (num_heads + 2 * kv_num_heads) * head_dim]
 * k_scales/v_scales value: 127 / max (type = TS)
 * k_scales_inv/v_scales_inv value:
 *   1. perchannel with zp: max / 127 (type = TS)
 *   2. perchannel without zp: max (type = float)
 **/
template <typename TX, typename TC, typename TS>
std::vector<paddle::Tensor> SplitEmbeddingKVCache(
    api::Context* xpu_ctx,
    xftblock::XFTContext& xctx,
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& len_info_cpu,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_kv_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& prefix_len_cpu,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_kv_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& prefix_len,
    const paddle::Tensor& slot_mapping_enc,
    const paddle::Tensor& slot_mapping_dec,
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_scales_inv,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const bool use_neox_rotary_style,
    const bool rope_3d) {
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  using XPU_XType = typename XPUTypeTrait<TX>::Type;
  using XPU_CType = typename XPUTypeTrait<TC>::Type;
  using XPU_SType = typename XPUTypeTrait<TS>::Type;
  using E_Scale = typename SplitRopeTypeTrait<XPU_CType, XPU_SType>::E_Scale;
  using D_Scale = typename SplitRopeTypeTrait<XPU_CType, XPU_SType>::D_Scale;
  typedef TX data_t;
  typedef TC cdata_t;
  typedef TS sdata_t;
  xftblock::DataType KV_BUF_TYPE = std::is_same<bfloat16, XPU_XType>::value
                                       ? xftblock::DataType::DT_BFLOAT16
                                       : xftblock::DataType::DT_FLOAT16;
  auto qkv_shape = qkv.dims();
  auto cache_shape = key_cache.dims();
  auto block_table_shape = block_tables.dims();
  const int block_batch = block_table_shape[0];
  const int max_block_per_seq = block_table_shape[1];
  const int num_blocks = cache_shape[0];
  const int kv_num_heads = cache_shape[1];
  const int block_size = cache_shape[2];
  const int head_dim = cache_shape[3];
  const int max_seq_len = block_size * max_block_per_seq;

  const int token_num = qkv_shape[0];
  const int total_num_head = qkv_shape[qkv_shape.size() - 1] / head_dim;
  const int num_heads = total_num_head - 2 * kv_num_heads;
  const int hidden_dim = num_heads * head_dim;

  int enc_batch = len_info_cpu.data<int32_t>()[0];
  int dec_batch = len_info_cpu.data<int32_t>()[1];
  int total_enc_len = len_info_cpu.data<int32_t>()[2];
  int total_dec_len = token_num - total_enc_len;
  int max_enc_len = len_info_cpu.data<int32_t>()[3];
  int max_kv_len = len_info_cpu.data<int32_t>()[4];
  int prefix_block_num_per_seq = len_info_cpu.data<int32_t>()[5];

  int rope_max_seqlen = 0;
  int rope_head_dim = 0;
  if (rope_3d) {
    PD_CHECK(rotary_embs.dims().size() == 6,
             "rotary_embs dim size should be 6 in multi-modal model");
    rope_max_seqlen = rotary_embs.dims()[3];
    rope_head_dim = rotary_embs.dims()[5];
  } else {
    PD_CHECK(rotary_embs.dims().size() == 5,
             "rotary_embs dim size should be 5 in language model");
    rope_max_seqlen = rotary_embs.dims()[2];
    rope_head_dim = rotary_embs.dims()[4];
  }
  std::string pos_emb_type;
  if (use_neox_rotary_style) {
    pos_emb_type = "NEOX";
  } else if (rope_head_dim == head_dim / 2) {
    // vl model use this
    pos_emb_type = "HALF_HEAD_DIM";
  } else {
    pos_emb_type = "NORMAL";
  }

  // TODO(lizanz03): only support c8 zp per channel
  bool is_cache_int8 = std::is_same<int8_t, XPU_CType>::value;
  bool has_zp = k_zeros && v_zeros;
  XPU_SType *quant_k_scale{nullptr}, *quant_v_scale{nullptr},
      *quant_k_scale_inv_zp{nullptr}, *quant_k_zp{nullptr},
      *quant_v_zp{nullptr};
  // maxptr for xfa
  float* quant_v_scale_inv{nullptr};
  if (is_cache_int8) {
    // only support c8 per channel
    quant_k_scale = reinterpret_cast<XPU_SType*>(
        const_cast<sdata_t*>(k_scales.get().data<sdata_t>()));
    quant_v_scale = reinterpret_cast<XPU_SType*>(
        const_cast<sdata_t*>(v_scales.get().data<sdata_t>()));
    if (has_zp) {
      quant_k_scale_inv_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(k_scales_inv.get().data<sdata_t>()));
      quant_k_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(k_zeros.get().data<sdata_t>()));
      quant_v_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(v_zeros.get().data<sdata_t>()));
    }
  }
  const float *q_norm_weight_data{nullptr}, *k_norm_weight_data{nullptr};
  if (q_norm_weight) {
    q_norm_weight_data = q_norm_weight.get().data<float>();
  }
  if (k_norm_weight) {
    k_norm_weight_data = k_norm_weight.get().data<float>();
  }
  PD_CHECK(!(pos_emb_type == "NEOX" && q_norm_weight_data != nullptr),
           "split_neox_cache_kv_encoder not support q/k norm weight");

  int ret;
  auto q_enc_tensor =
      paddle::empty({total_enc_len, hidden_dim}, qkv.type(), qkv.place());
  auto k_enc_tensor = paddle::empty(
      {total_enc_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
  auto v_enc_tensor = paddle::empty(
      {total_enc_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
  auto q_dec_tensor =
      paddle::empty({total_dec_len, hidden_dim}, qkv.type(), qkv.place());
  auto k_dec_tensor = paddle::empty(
      {total_dec_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
  auto v_dec_tensor = paddle::empty(
      {total_dec_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());

  if (enc_batch > 0) {
    xftblock::Tensor q_enc_xft_tensor(
        q_enc_tensor.data(), KV_BUF_TYPE, {total_enc_len, hidden_dim});
    xftblock::Tensor k_enc_xft_tensor(k_enc_tensor.data(),
                                      KV_BUF_TYPE,
                                      {total_enc_len, kv_num_heads * head_dim});
    xftblock::Tensor v_enc_xft_tensor(v_enc_tensor.data(),
                                      KV_BUF_TYPE,
                                      {total_enc_len, kv_num_heads * head_dim});

    api::VectorParam<int32_t> seqlod_vp = {
        const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()),
        enc_batch + 1,
        const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
    api::VectorParam<int32_t> real_batch_vp = {
        const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()),
        enc_batch,
        const_cast<int32_t*>(encoder_batch_map.data<int32_t>())};  // real batch
    baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp{
        const_cast<int32_t*>(prefix_len_cpu.data<int32_t>()),
        enc_batch,
        const_cast<int32_t*>(prefix_len.data<int32_t>())};

    // split, rotary embedding and write to kv cache
    split_kvcache_encoder<XPU_XType, float, XPU_CType, int, E_Scale>(
        xpu_ctx,
        xctx,
        qkv,
        rotary_embs,
        q_enc_tensor,
        k_enc_tensor,
        v_enc_tensor,
        key_cache,
        value_cache,
        block_tables,
        slot_mapping_enc,
        enc_batch,
        total_enc_len,
        num_heads,
        kv_num_heads,
        head_dim,
        rope_head_dim,
        hidden_dim,
        rope_max_seqlen,
        block_size,
        num_blocks,
        block_batch,
        max_block_per_seq,
        seqlod_vp,
        prefix_lens_vp,
        real_batch_vp,
        0,
        nullptr,        // k_cache_scale_inv - use for per head
        nullptr,        // v_cache_scale_inv - use for per head
        quant_k_scale,  // intx_k_pc_scale
        quant_v_scale,  // intx_v_pc_scale
        quant_k_zp,     // intx_k_pc_zero
        quant_v_zp,     // intx_v_pc_zero
        q_norm_weight_data,
        k_norm_weight_data,
        pos_emb_type,
        rope_3d,
        use_neox_rotary_style);

    // pd split
    if (FLAGS_fmt_write_cache_completed_signal) {
      XPUEvent write_event = nullptr;
      ret = xpu_event_create(&write_event);
      PD_CHECK(ret == 0, "xpu_event_create write_event failed.");

      ret = xpu_event_record(write_event, xctx.get_main_stream());
      PD_CHECK(ret == 0, "xpu_event_record failed.");

      PD_CHECK(cachekv_signal_thread_cpu,
               "cachekv_signal_thread should not be nullptr");
      auto worker = reinterpret_cast<CacheKvSignalThreadWorker*>(
          cachekv_signal_thread_cpu.get().data<int64_t>()[0]);
      PD_CHECK(worker != nullptr,
               "cachekv_signal_thread should not be nullptr");

      if (FLAGS_use_pd_disaggregation_per_chunk) {
        worker->push_signal_task_per_query(write_event, nullptr);
      } else {
        // If use micro batch:
        //     micro_batch_0 do nothing.
        //     micro_batch_1 write kv signal.
        if (kv_signal_data_cpu) {
          worker->push_signal_task(
              write_event,
              reinterpret_cast<void*>((const_cast<int64_t*>(
                  kv_signal_data_cpu.get().data<int64_t>()))));
        }
      }
    }

    bool is_prefix_cache = prefix_block_num_per_seq > 0;
    if (is_cache_int8 && has_zp && is_prefix_cache) {
      // assume q_layout is BLHD, q = q * k_scales_inv
      ret = api::broadcast_mul<XPU_XType>(
          xpu_ctx,
          q_enc_xft_tensor.data<XPU_XType>(),
          quant_k_scale_inv_zp,
          q_enc_xft_tensor.data<XPU_XType>(),
          {total_enc_len, kv_num_heads, num_heads / kv_num_heads, head_dim},
          {1, kv_num_heads, 1, head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
    }
  }

  if (dec_batch > 0) {
    xftblock::Tensor q_dec_xft_tensor(
        q_dec_tensor.data(), KV_BUF_TYPE, {total_dec_len, hidden_dim});
    xftblock::Tensor k_dec_xft_tensor(k_dec_tensor.data(),
                                      KV_BUF_TYPE,
                                      {total_dec_len, kv_num_heads * head_dim});
    xftblock::Tensor v_dec_xft_tensor(v_dec_tensor.data(),
                                      KV_BUF_TYPE,
                                      {total_dec_len, kv_num_heads * head_dim});

    api::VectorParam<int32_t> decoder_context_len_cache_vp = {
        const_cast<int32_t*>(decoder_context_len_cache_cpu.data<int32_t>()),
        dec_batch,
        const_cast<int32_t*>(decoder_context_len_cache.data<int32_t>())};
    api::VectorParam<int32_t> real_batch_vp = {
        const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()),
        dec_batch,
        const_cast<int32_t*>(decoder_batch_map.data<int32_t>())};
    api::VectorParam<int32_t> seqlod_vp = {
        const_cast<int32_t*>(decoder_seq_lod_cpu.data<int32_t>()),
        dec_batch + 1,
        const_cast<int32_t*>(decoder_seq_lod.data<int32_t>())};
    api::VectorParam<int32_t> seqlod_for_fused_vp = {
        const_cast<int32_t*>(decoder_context_len_cpu.data<int32_t>()),
        dec_batch,
        const_cast<int32_t*>(decoder_context_len.data<int32_t>())};

    // split, rotary embedding and write to kv cache
    if (total_dec_len != dec_batch) {
      // mtp branch
      split_kvcache_encoder<XPU_XType, float, XPU_CType, int, E_Scale>(
          xpu_ctx,
          xctx,
          qkv,
          rotary_embs,
          q_dec_tensor,
          k_dec_tensor,
          v_dec_tensor,
          key_cache,
          value_cache,
          block_tables,
          slot_mapping_dec,
          dec_batch,
          total_dec_len,
          num_heads,
          kv_num_heads,
          head_dim,
          rope_head_dim,
          hidden_dim,
          rope_max_seqlen,
          block_size,
          num_blocks,
          block_batch,
          max_block_per_seq,
          seqlod_vp,                     // seq_lod
          decoder_context_len_cache_vp,  // start_tokens (prefix len)
          real_batch_vp,                 // real_batch
          total_enc_len * qkv_shape[qkv_shape.size() - 1],
          nullptr,        // k_cache_scale_inv - use for per head
          nullptr,        // v_cache_scale_inv - use for per head
          quant_k_scale,  // intx_k_pc_scale
          quant_v_scale,  // intx_v_pc_scale
          quant_k_zp,     // intx_k_pc_zero
          quant_v_zp,     // intx_v_pc_zero
          q_norm_weight_data,
          k_norm_weight_data,
          pos_emb_type,
          rope_3d,
          use_neox_rotary_style);
    } else {
      // non mtp branch
      split_kvcache_decoder<XPU_XType, float, XPU_CType, int, D_Scale>(
          xpu_ctx,
          xctx,
          qkv,
          rotary_embs,
          q_dec_tensor,
          k_dec_tensor,
          v_dec_tensor,
          key_cache,
          value_cache,
          block_tables,
          slot_mapping_dec,
          dec_batch,
          total_dec_len,
          num_heads,
          kv_num_heads,
          head_dim,
          rope_head_dim,
          hidden_dim,
          rope_max_seqlen,
          block_size,
          num_blocks,
          block_batch,
          max_block_per_seq,
          seqlod_vp,
          seqlod_for_fused_vp,
          decoder_context_len_cache_vp,
          real_batch_vp,
          total_enc_len * qkv_shape[qkv_shape.size() - 1],
          reinterpret_cast<D_Scale*>(quant_k_scale),  // k_cache_scale_inv
          reinterpret_cast<D_Scale*>(quant_v_scale),  // v_cache_scale_inv
          reinterpret_cast<D_Scale*>(quant_k_zp),     // k_cache_zp
          reinterpret_cast<D_Scale*>(quant_v_zp),     // v_cache_zp
          q_norm_weight_data,
          k_norm_weight_data,
          pos_emb_type,
          rope_3d,
          is_cache_int8,  // bool b_c8_pc
          use_neox_rotary_style);
    }

    if (is_cache_int8 && has_zp) {
      // q = q * k_scales_inv
      ret = api::broadcast_mul<XPU_XType>(
          xpu_ctx,
          q_dec_xft_tensor.data<XPU_XType>(),
          quant_k_scale_inv_zp,
          q_dec_xft_tensor.data<XPU_XType>(),
          {total_dec_len, kv_num_heads, num_heads / kv_num_heads, head_dim},
          {1, kv_num_heads, 1, head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
    }
  }
  return {q_enc_tensor, k_enc_tensor, v_enc_tensor, q_dec_tensor};
}

template <typename TX, typename TC, typename TS>
std::vector<paddle::Tensor> BlockAttn(
    api::Context* xpu_ctx,
    xftblock::XFTContext& xctx,
    const paddle::Tensor& q_enc_tensor,
    const paddle::Tensor& k_enc_tensor,
    const paddle::Tensor& v_enc_tensor,
    const paddle::Tensor& q_dec_tensor,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& prefix_block_tables,
    const paddle::Tensor& len_info_cpu,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_kv_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& prefix_len_cpu,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_kv_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& prefix_len,
    const paddle::optional<paddle::Tensor>& k_scales_inv,
    const paddle::optional<paddle::Tensor>& v_scales_inv,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& shift,
    const paddle::optional<paddle::Tensor>& smooth) {
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  using XPU_XType = typename XPUTypeTrait<TX>::Type;
  using XPU_CType = typename XPUTypeTrait<TC>::Type;
  using XPU_SType = typename XPUTypeTrait<TS>::Type;
  using E_Scale = typename SplitRopeTypeTrait<XPU_CType, XPU_SType>::E_Scale;
  using D_Scale = typename SplitRopeTypeTrait<XPU_CType, XPU_SType>::D_Scale;
  typedef TX data_t;
  typedef TC cdata_t;
  typedef TS sdata_t;
  xftblock::DataType KV_BUF_TYPE = std::is_same<bfloat16, XPU_XType>::value
                                       ? xftblock::DataType::DT_BFLOAT16
                                       : xftblock::DataType::DT_FLOAT16;

  auto cache_shape = key_cache.dims();
  auto block_table_shape = block_tables.dims();
  const int block_batch = block_table_shape[0];
  const int max_block_per_seq = block_table_shape[1];
  const int kv_num_heads = cache_shape[1];
  const int block_size = cache_shape[2];
  const int head_dim = cache_shape[3];
  const int max_seq_len = block_size * max_block_per_seq;

  const int token_num = q_enc_tensor.dims()[0] + q_dec_tensor.dims()[0];
  const int hidden_dim = q_enc_tensor.dims()[q_enc_tensor.dims().size() - 1];
  const int num_heads = hidden_dim / head_dim;
  const int total_num_head = num_heads + 2 * kv_num_heads;

  int enc_batch = len_info_cpu.data<int32_t>()[0];
  int dec_batch = len_info_cpu.data<int32_t>()[1];
  int total_enc_len = len_info_cpu.data<int32_t>()[2];
  int total_dec_len = token_num - total_enc_len;
  int max_enc_len = len_info_cpu.data<int32_t>()[3];
  int max_kv_len = len_info_cpu.data<int32_t>()[4];
  int prefix_block_num_per_seq = len_info_cpu.data<int32_t>()[5];
  int max_dec_len = len_info_cpu.data<int32_t>()[6];

  auto block_attn_out = paddle::empty(
      {token_num, hidden_dim}, q_enc_tensor.type(), q_enc_tensor.place());

  // TODO(lizanz03): only support c8 zp per channel
  bool is_cache_int8 = std::is_same<int8_t, XPU_CType>::value;
  bool has_zp = k_zeros && v_zeros;
  XPU_SType *quant_v_scale_inv_zp{nullptr}, *quant_v_zp{nullptr};
  // maxptr for xfa
  float *quant_k_scale_inv{nullptr}, *quant_v_scale_inv{nullptr};
  XPU_XType *p_shift{nullptr}, *p_smooth{nullptr};
  if (is_cache_int8) {
    if (shift) {
      p_shift = reinterpret_cast<XPU_XType*>(
          const_cast<data_t*>(shift.get().data<data_t>()));
    }
    if (smooth) {
      p_smooth = reinterpret_cast<XPU_XType*>(
          const_cast<data_t*>(smooth.get().data<data_t>()));
    }
    if (has_zp) {
      quant_v_scale_inv_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(v_scales_inv.get().data<sdata_t>()));
      quant_v_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(v_zeros.get().data<sdata_t>()));
    } else {
      quant_k_scale_inv = reinterpret_cast<float*>(
          const_cast<float*>(k_scales_inv.get().data<float>()));
      quant_v_scale_inv = reinterpret_cast<float*>(
          const_cast<float*>(v_scales_inv.get().data<float>()));
    }
  }

  int ret;

  if (enc_batch > 0) {
    xftblock::TransformerParam param;
    xftblock::TransformerVsl vsl;
    param.batch_size = enc_batch;
    param.head_num = num_heads;
    param.kv_head_num = kv_num_heads;
    param.head_dim = head_dim;
    param.max_batch_size = block_batch;
    param.max_seq_len = max_seq_len;
    param.use_cache_per_channel =
        is_cache_int8 && !has_zp;  // only support c8 per channel

    vsl.usual_lod_vp = {
        const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()),
        enc_batch + 1,
        const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
    vsl.kv_lod_vp = {const_cast<int32_t*>(encoder_kv_lod_cpu.data<int32_t>()),
                     enc_batch + 1,
                     const_cast<int32_t*>(encoder_kv_lod.data<int32_t>())};
    vsl.slot_mapping_vp = {
        const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()),
        enc_batch,
        const_cast<int32_t*>(encoder_batch_map.data<int32_t>())};  // real batch
    param.max_valid_seqlen = max_enc_len;
    param.max_kv_valid_seqlen = max_kv_len;
    // setting for prefix cache
    bool is_prefix_cache = prefix_block_num_per_seq > 0;
    param.prefill_len = is_prefix_cache ? param.max_valid_seqlen : -1;
    param.page_attn.block_size = block_size;
    param.page_attn.max_num_blocks_per_seq = prefix_block_num_per_seq;
    // prefix_block_tables is a subset of block_tables, which is used for
    // prefix cache
    xftblock::Tensor prefix_block_tables_tensor(
        is_prefix_cache ? reinterpret_cast<void*>(const_cast<int32_t*>(
                              prefix_block_tables.data<int32_t>()))
                        : nullptr,
        xftblock::DataType::DT_INT32,
        {prefix_block_tables.dims()[0], prefix_block_num_per_seq});
    param.page_attn.block_table = &prefix_block_tables_tensor;
    baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp{
        const_cast<int32_t*>(prefix_len_cpu.data<int32_t>()),
        enc_batch,
        const_cast<int32_t*>(prefix_len.data<int32_t>())};

    float* fake_perhead_scale = nullptr;
    if (is_cache_int8 && has_zp && is_prefix_cache) {
      fake_perhead_scale = RAII_GUARD.alloc<float>(param.kv_head_num);
      // set fake_perhead_scale to ones
      ret = api::constant<float>(
          xpu_ctx, fake_perhead_scale, param.kv_head_num, 127.f);
      PD_CHECK(ret == api::SUCCESS, "api::constant failed.");
    }
    // buf tensor
    xftblock::Tensor q_enc_xft_tensor(const_cast<void*>(q_enc_tensor.data()),
                                      KV_BUF_TYPE,
                                      {total_enc_len, hidden_dim});
    xftblock::Tensor k_enc_xft_tensor(const_cast<void*>(k_enc_tensor.data()),
                                      KV_BUF_TYPE,
                                      {total_enc_len, kv_num_heads * head_dim});
    xftblock::Tensor v_enc_xft_tensor(const_cast<void*>(v_enc_tensor.data()),
                                      KV_BUF_TYPE,
                                      {total_enc_len, kv_num_heads * head_dim});

    // kv cache tensor
    xftblock::Tensor key_cache_tensor(
        reinterpret_cast<void*>(
            const_cast<cdata_t*>(key_cache.data<cdata_t>())),  // src_data
        nullptr,                                               // max_data
        has_zp                                                 // pc_scale
            ? fake_perhead_scale
            : quant_k_scale_inv,
        is_cache_int8  // cache type
            ? xftblock::DataType::DT_INT8
            : KV_BUF_TYPE,
        {cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]});
    xftblock::Tensor value_cache_tensor(
        reinterpret_cast<void*>(
            const_cast<cdata_t*>(value_cache.data<cdata_t>())),  // src_data
        nullptr,                                                 // max_data
        has_zp                                                   // pc_scale
            ? fake_perhead_scale
            : quant_v_scale_inv,
        is_cache_int8  // cache type
            ? xftblock::DataType::DT_INT8
            : KV_BUF_TYPE,
        {cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]});

    xftblock::Tensor encode_output(reinterpret_cast<void*>(const_cast<data_t*>(
                                       block_attn_out.data<data_t>())),
                                   KV_BUF_TYPE,
                                   {total_enc_len, hidden_dim});

    // attn encode
    if (is_prefix_cache) {
      ret =
          xftblock::xft_context_core_attenion_block<XPU_XType,
                                                    XPU_CType,
                                                    float>(&xctx,
                                                           &q_enc_xft_tensor,
                                                           &key_cache_tensor,
                                                           &value_cache_tensor,
                                                           &encode_output,
                                                           param,
                                                           vsl);
    } else {
      ret = xftblock::xft_context_core_attenion_block<XPU_XType,
                                                      XPU_XType,
                                                      float>(&xctx,
                                                             &q_enc_xft_tensor,
                                                             &k_enc_xft_tensor,
                                                             &v_enc_xft_tensor,
                                                             &encode_output,
                                                             param,
                                                             vsl);
    }
    PD_CHECK(ret == api::SUCCESS,
             "xftblock::xft_context_core_attenion_block failed.");

    if (is_cache_int8 && has_zp && is_prefix_cache) {
      int64_t q_head_num = param.head_num;
      int64_t kv_head_num = param.kv_head_num;
      // out = (out - v_zeros) * v_scales_inv
      ret = api::broadcast_sub<XPU_XType>(xpu_ctx,
                                          encode_output.data<XPU_XType>(),
                                          quant_v_zp,
                                          encode_output.data<XPU_XType>(),
                                          {total_enc_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_sub failed.");
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx,
                                          encode_output.data<XPU_XType>(),
                                          quant_v_scale_inv_zp,
                                          encode_output.data<XPU_XType>(),
                                          {total_enc_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
    }
    if (p_shift != nullptr) {
      ret = api::broadcast_add<XPU_XType>(xpu_ctx,
                                          p_shift,
                                          encode_output.data<XPU_XType>(),
                                          encode_output.data<XPU_XType>(),
                                          {1, hidden_dim},
                                          {total_enc_len, hidden_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_add for shift failed.");
    }
    if (p_smooth != nullptr) {
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx,
                                          p_smooth,
                                          encode_output.data<XPU_XType>(),
                                          encode_output.data<XPU_XType>(),
                                          {1, hidden_dim},
                                          {total_enc_len, hidden_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul for smooth failed.");
    }
  }

  if (dec_batch > 0) {
    xftblock::TransformerParam param;
    xftblock::TransformerVsl vsl;
    param.batch_size = dec_batch;
    param.head_num = num_heads;
    param.kv_head_num = kv_num_heads;
    param.head_dim = head_dim;
    param.max_batch_size = block_batch;
    param.max_seq_len = max_seq_len;
    param.use_page_attn = true;
    xftblock::Tensor decode_output(
        reinterpret_cast<void*>(
            const_cast<data_t*>(block_attn_out.data<data_t>()) +
            total_enc_len * hidden_dim),
        KV_BUF_TYPE,
        {total_dec_len, hidden_dim});
    // buf tensor
    xftblock::Tensor q_dec_xft_tensor(const_cast<void*>(q_dec_tensor.data()),
                                      KV_BUF_TYPE,
                                      {total_dec_len, hidden_dim});

    float* fake_perhead_scale = nullptr;
    if (is_cache_int8 && has_zp) {
      int64_t kv_head_num = param.kv_head_num;
      fake_perhead_scale = RAII_GUARD.alloc<float>(kv_head_num);
      // set fake_perhead_scale to ones
      ret =
          api::constant<float>(xpu_ctx, fake_perhead_scale, kv_head_num, 127.f);
      PD_CHECK(ret == api::SUCCESS, "api::constant failed.");
    }

    if (total_dec_len != dec_batch) {
      api::VectorParam<int32_t> decoder_context_len_vp = {
          const_cast<int32_t*>(decoder_context_len_cpu.data<int32_t>()),
          dec_batch,
          const_cast<int32_t*>(
              decoder_context_len
                  .data<int32_t>())};  // use for speculative_attention_decoder
                                       // seq_len in MTP
      api::VectorParam<int32_t> decoder_batch_map_vp = {
          const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()),
          dec_batch,
          const_cast<int32_t*>(
              decoder_batch_map.data<int32_t>())};  // real batch
      api::VectorParam<int32_t> decoder_seq_lod_vp = {
          const_cast<int32_t*>(decoder_seq_lod_cpu.data<int32_t>()),
          dec_batch + 1,
          const_cast<int32_t*>(
              decoder_seq_lod
                  .data<int32_t>())};  // use for split rope enc as lod in MTP

      XPU_XType* q_dec_xft_tensor_ptr = q_dec_xft_tensor.data<XPU_XType>();
      XPU_XType* decode_output_ptr = decode_output.data<XPU_XType>();
      using TGEMM = std::conditional_t<std::is_same_v<XPU_XType, XPU_CType>,
                                       tfloat32,
                                       int8_wo_t>;
      constexpr int quant_mode = std::is_same_v<XPU_CType, int8_t> ? 3 : 0;
      ret = baidu::xpu::xfa::speculative_attention_decoder<XPU_XType,
                                                           XPU_CType,
                                                           XPU_XType,
                                                           TGEMM,
                                                           TGEMM,
                                                           float,
                                                           int32_t,
                                                           quant_mode>(
          xpu_ctx,
          decode_output_ptr,     // out
          q_dec_xft_tensor_ptr,  // q
          nullptr,               // k
          nullptr,               // v
          reinterpret_cast<const XPU_CType*>(
              key_cache.data<cdata_t>()),  // k_cache
          reinterpret_cast<const XPU_CType*>(
              value_cache.data<cdata_t>()),  // v_cache
          reinterpret_cast<const int32_t*>(
              block_tables.data<int32_t>()),  // block_tables
          decoder_context_len_vp,             // seq_lengths
          decoder_batch_map_vp,               // valid_batch
          param.max_batch_size,               // batch_num
          max_dec_len,                        // qlen
          max_seq_len,                        // max_seq_len
          param.head_num,                     // head_num
          param.head_dim,                     // head_dim
          param.kv_head_num,                  // kv_head_num
          nullptr,                            // attn_mask
          1.0f /
              std::sqrt(static_cast<float>(param.head_dim)),  // scale 【check】
          block_size,                                         // block_size
          max_block_per_seq,  // max_blocks_per_seq
          -1,                 // max_window_size
          nullptr,            // q_maxptr
          has_zp              // k_cache_maxptr
              ? fake_perhead_scale
              : quant_k_scale_inv,
          has_zp  // v_cache_maxptr
              ? fake_perhead_scale
              : quant_v_scale_inv,
          nullptr,              // o_maxptr
          param.head_dim,       // vo_head_dim
          decoder_seq_lod_vp);  // qlod
      PD_CHECK(ret == api::SUCCESS,
               "xfa::speculative_attention_decoder failed.");
    } else {
      vsl.usual_lod_vp = {
          const_cast<int32_t*>(decoder_context_len_cpu.data<int32_t>()),
          dec_batch,
          const_cast<int32_t*>(decoder_context_len.data<int32_t>())};
      vsl.slot_mapping_vp = {
          const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()),
          dec_batch,
          const_cast<int32_t*>(
              decoder_batch_map.data<int32_t>())};  // real batch
      // can not set to nullptr and 0, which will cause inference interrupt
      //   vsl.slot_mapping_vp = {nullptr, 0, nullptr};  // real batch

      xftblock::Tensor block_table_tensor(
          reinterpret_cast<void*>(
              const_cast<int32_t*>(block_tables.data<int32_t>())),
          xftblock::DataType::DT_INT32,
          {block_table_shape[0], block_table_shape[1]});

      // normal setting
      param.use_cache_per_channel =
          is_cache_int8 && !has_zp;  // only support c8 per channel
      param.prefill_len = -1;
      param.page_attn.block_size = block_size;
      param.page_attn.max_context_len = max_seq_len;
      param.page_attn.max_num_blocks_per_seq = max_block_per_seq;
      param.page_attn.block_table = &block_table_tensor;

      // kv cache tensor
      xftblock::Tensor key_cache_tensor(
          reinterpret_cast<void*>(
              const_cast<cdata_t*>(key_cache.data<cdata_t>())),  // src_data
          nullptr,                                               // max_data
          has_zp                                                 // pc_scale
              ? fake_perhead_scale
              : quant_k_scale_inv,
          is_cache_int8  // cache type
              ? xftblock::DataType::DT_INT8
              : KV_BUF_TYPE,
          {cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]});
      xftblock::Tensor value_cache_tensor(
          reinterpret_cast<void*>(
              const_cast<cdata_t*>(value_cache.data<cdata_t>())),  // src_data
          nullptr,                                                 // max_data
          has_zp                                                   // pc_scale
              ? fake_perhead_scale
              : quant_v_scale_inv,
          is_cache_int8  // cache type
              ? xftblock::DataType::DT_INT8
              : KV_BUF_TYPE,
          {cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]});

      // attn decode
      ret = xftblock::xft_decoder_core_attenion_block<
          XPU_XType,
          XPU_CType,
          XPU_XType>(  // TGEMM = XPU_XType TODOlizan03: used high
                       // precision
          &xctx,
          &q_dec_xft_tensor,
          &key_cache_tensor,
          &value_cache_tensor,
          &decode_output,
          param,
          vsl);
      PD_CHECK(ret == api::SUCCESS,
               "xftblock::xft_decoder_core_attenion_block failed.");
    }
    if (is_cache_int8 && has_zp) {
      int64_t q_head_num = param.head_num;
      int64_t kv_head_num = param.kv_head_num;
      // out = (out - v_zeros) * v_scales_inv
      if (quant_v_zp) {
        ret =
            api::broadcast_sub<XPU_XType>(xpu_ctx,
                                          decode_output.data<XPU_XType>(),
                                          quant_v_zp,
                                          decode_output.data<XPU_XType>(),
                                          {total_dec_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
        PD_CHECK(ret == api::SUCCESS, "api::broadcast_sub failed.");
      }
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx,
                                          decode_output.data<XPU_XType>(),
                                          quant_v_scale_inv_zp,
                                          decode_output.data<XPU_XType>(),
                                          {total_dec_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
    }
    if (p_shift != nullptr) {
      ret = api::broadcast_add<XPU_XType>(xpu_ctx,
                                          p_shift,
                                          decode_output.data<XPU_XType>(),
                                          decode_output.data<XPU_XType>(),
                                          {1, hidden_dim},
                                          {total_dec_len, hidden_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_add for shift failed.");
    }
    if (p_smooth != nullptr) {
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx,
                                          p_smooth,
                                          decode_output.data<XPU_XType>(),
                                          decode_output.data<XPU_XType>(),
                                          {1, hidden_dim},
                                          {total_dec_len, hidden_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul for smooth failed.");
    }
  }
  return {block_attn_out};
}

std::vector<paddle::Tensor> SplitEmbeddingKVCacheBlockAttn(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& prefix_block_tables,
    const paddle::Tensor& len_info_cpu,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_kv_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& prefix_len_cpu,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_kv_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& prefix_len,
    const paddle::Tensor& slot_mapping_enc,
    const paddle::Tensor& slot_mapping_dec,
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_scales_inv,
    const paddle::optional<paddle::Tensor>& v_scales_inv,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& shift,
    const paddle::optional<paddle::Tensor>& smooth,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const bool use_neox_rotary_style,
    const bool rope_3d = false) {
#define APPLY_SPLITKVCACHE(TX, TC, TS)                                 \
  std::vector<paddle::Tensor> split_qkv =                              \
      SplitEmbeddingKVCache<TX, TC, TS>(xpu_ctx,                       \
                                        xctx,                          \
                                        qkv,                           \
                                        key_cache,                     \
                                        value_cache,                   \
                                        rotary_embs,                   \
                                        block_tables,                  \
                                        len_info_cpu,                  \
                                        encoder_seq_lod_cpu,           \
                                        decoder_seq_lod_cpu,           \
                                        encoder_kv_lod_cpu,            \
                                        encoder_batch_map_cpu,         \
                                        decoder_context_len_cpu,       \
                                        decoder_context_len_cache_cpu, \
                                        decoder_batch_map_cpu,         \
                                        prefix_len_cpu,                \
                                        encoder_seq_lod,               \
                                        decoder_seq_lod,               \
                                        encoder_kv_lod,                \
                                        encoder_batch_map,             \
                                        decoder_context_len,           \
                                        decoder_context_len_cache,     \
                                        decoder_batch_map,             \
                                        prefix_len,                    \
                                        slot_mapping_enc,              \
                                        slot_mapping_dec,              \
                                        k_scales,                      \
                                        v_scales,                      \
                                        k_scales_inv,                  \
                                        k_zeros,                       \
                                        v_zeros,                       \
                                        q_norm_weight,                 \
                                        k_norm_weight,                 \
                                        kv_signal_data_cpu,            \
                                        cachekv_signal_thread_cpu,     \
                                        use_neox_rotary_style,         \
                                        rope_3d);
#define APPLY_BLOCKATTN(TX, TC, TS)                           \
  return BlockAttn<TX, TC, TS>(xpu_ctx,                       \
                               xctx,                          \
                               split_qkv[0],                  \
                               split_qkv[1],                  \
                               split_qkv[2],                  \
                               split_qkv[3],                  \
                               key_cache,                     \
                               value_cache,                   \
                               rotary_embs,                   \
                               block_tables,                  \
                               prefix_block_tables,           \
                               len_info_cpu,                  \
                               encoder_seq_lod_cpu,           \
                               decoder_seq_lod_cpu,           \
                               encoder_kv_lod_cpu,            \
                               encoder_batch_map_cpu,         \
                               decoder_context_len_cpu,       \
                               decoder_context_len_cache_cpu, \
                               decoder_batch_map_cpu,         \
                               prefix_len_cpu,                \
                               encoder_seq_lod,               \
                               decoder_seq_lod,               \
                               encoder_kv_lod,                \
                               encoder_batch_map,             \
                               decoder_context_len,           \
                               decoder_context_len_cache,     \
                               decoder_batch_map,             \
                               prefix_len,                    \
                               k_scales_inv,                  \
                               v_scales_inv,                  \
                               k_zeros,                       \
                               v_zeros,                       \
                               shift,                         \
                               smooth);

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx)->x_context();
  xftblock::XFTContext xctx(xpu_ctx, nullptr);
  const auto cache_dtype = key_cache.dtype();

  if (cache_dtype == paddle::DataType::BFLOAT16) {
    APPLY_SPLITKVCACHE(paddle::bfloat16, paddle::bfloat16, paddle::bfloat16);
    APPLY_BLOCKATTN(paddle::bfloat16, paddle::bfloat16, paddle::bfloat16);
  } else if (cache_dtype == paddle::DataType::INT8) {
    APPLY_SPLITKVCACHE(paddle::bfloat16, int8_t, paddle::bfloat16);
    APPLY_BLOCKATTN(paddle::bfloat16, int8_t, paddle::bfloat16);
  } else {
    PD_THROW("block_attn not support cache_dtype==%d",
             static_cast<int>(cache_dtype));
    return {};
  }

#undef APPLY_SPLITKVCACHE
#undef APPLY_BLOCKATTN
}

std::vector<std::vector<int64_t>> SplitEmbeddingKVCacheBlockAttnInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape) {
  const int token_num = qkv_shape[0];
  const int kv_num_heads = key_cache_shape[1];
  int head_dim = key_cache_shape[3];
  //   if (cache_quant_type_str == "cache_int4_zp") {
  //     head_dim *= 2;
  //   }
  const int total_num_head = qkv_shape[qkv_shape.size() - 1] / head_dim;
  const int num_heads = total_num_head - 2 * kv_num_heads;
  return {{token_num, num_heads * head_dim}};
}

std::vector<paddle::DataType> SplitEmbeddingKVCacheBlockAttnInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(block_attn)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "rotary_embs",
             "block_tables",
             "prefix_block_tables",
             "len_info_cpu",
             "encoder_seq_lod_cpu",
             "decoder_seq_lod_cpu",
             "encoder_kv_lod_cpu",
             "encoder_batch_map_cpu",
             "decoder_context_len_cpu",
             "decoder_context_len_cache_cpu",
             "decoder_batch_map_cpu",
             "prefix_len_cpu",
             "encoder_seq_lod",
             "decoder_seq_lod",
             "encoder_kv_lod",
             "encoder_batch_map",
             "decoder_context_len",
             "decoder_context_len_cache",
             "decoder_batch_map",
             "prefix_len",
             "slot_mapping_enc",
             "slot_mapping_dec",
             paddle::Optional("k_scales"),
             paddle::Optional("v_scales"),
             paddle::Optional("k_scales_inv"),
             paddle::Optional("v_scales_inv"),
             paddle::Optional("k_zeros"),
             paddle::Optional("v_zeros"),
             paddle::Optional("shift"),
             paddle::Optional("smooth"),
             paddle::Optional("q_norm_weight"),
             paddle::Optional("k_norm_weight"),
             paddle::Optional("kv_signal_data_cpu"),
             paddle::Optional("cachekv_signal_thread_cpu")})
    .Attrs({"use_neox_rotary_style:bool", "rope_3d:bool"})
    .Outputs({"block_attn_out"})
    .SetKernelFn(PD_KERNEL(SplitEmbeddingKVCacheBlockAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(SplitEmbeddingKVCacheBlockAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SplitEmbeddingKVCacheBlockAttnInferDtype));
