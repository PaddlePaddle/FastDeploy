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

namespace xftblock = baidu::xpu::xftblock;

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

/**
 * qkv shape: [token_num, (num_heads + 2 * kv_num_heads) * head_dim]
 * k_scales/v_scales value: 127 / max (type = TS)
 * k_scales_inv/v_scales_inv value:
 *   1. perchannel with zp: max / 127 (type = TS)
 *   2. perchannel without zp: max (type = float)
 **/
/**
 * @brief Split QKV, apply RoPE and write to KV cache
 *
 * This kernel performs the following operations on XPU:
 * 1. Split the input QKV tensor into Q, K, V tensors
 * 2. Apply rotary position embedding (RoPE) to Q and K
 * 3. Write K and V to the KV cache for efficient inference
 *
 * @tparam TX Input QKV tensor data type (float16/bfloat16/int8_t)
 * @tparam TC Cache data type (float16/bfloat16)
 * @tparam TS Scale data type (float/bfloat16)
 *
 * @param qkv Input QKV tensor, shape: [token_num, (num_heads + 2 * kv_num_heads) * head_dim]
 * @param key_cache Output key cache tensor
 * @param value_cache Output value cache tensor
 * @param cum_offsets Cumulative offsets for each sequence
 * @param rotary_embs Rotary position embeddings
 * @param block_tables Block table for paged KV cache
 * @param len_info_cpu Length information on CPU: [enc_batch, dec_batch, total_enc_len, max_enc_len, max_kv_len, prefix_block_num_per_seq]
 * @param encoder_seq_lod_cpu Encoder sequence lod on CPU
 * @param decoder_seq_lod_cpu Decoder sequence lod on CPU
 * @param encoder_batch_map_cpu Encoder batch map on CPU
 * @param decoder_context_len_cpu Decoder context length on CPU
 * @param decoder_context_len_cache_cpu Decoder context length cache on CPU
 * @param decoder_batch_map_cpu Decoder batch map on CPU
 * @param prefix_len_cpu Prefix length on CPU
 * @param encoder_seq_lod Encoder sequence lod
 * @param decoder_seq_lod Decoder sequence lod
 * @param encoder_batch_map Encoder batch map
 * @param decoder_context_len Decoder context length
 * @param decoder_context_len_cache Decoder context length cache
 * @param decoder_batch_map Decoder batch map
 * @param prefix_len Prefix length
 * @param k_scales Optional K channel scales for quantization
 * @param v_scales Optional V channel scales for quantization
 * @param k_zeros Optional K zeros for quantization
 * @param v_zeros Optional V zeros for quantization
 * @param q_norm_weight Optional Q normalization weight
 * @param k_norm_weight Optional K normalization weight
 * @param kv_signal_data_cpu Optional KV signal data on CPU
 * @param cachekv_signal_thread_cpu Optional cache KV signal thread on CPU
 * @param use_neox_rotary_style Whether to use NEOX style rotary embedding
 * @param rope_3d Whether to use 3D rotary embedding
 *
 * @return std::vector<paddle::Tensor> Vector of output tensors
 */
template <typename TX, typename TC, typename TS>
std::vector<paddle::Tensor> SplitRopeKVCacheKernel(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& len_info_cpu,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& prefix_len_cpu,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& prefix_len,
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const bool use_neox_rotary_style,
    const bool rope_3d) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());
  xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
  auto rt_guard = xctx.get_rt_guard();

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
  const int bsz = cum_offsets.dims()[0];
  const int block_batch = block_table_shape[0];
  const int max_block_per_seq = block_table_shape[1];
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
  if (use_neox_rotary_style == true) {
    pos_emb_type = "NEOX";
  } else if (rope_head_dim == head_dim / 2) {
    pos_emb_type = "HALF_HEAD_DIM";
  } else {
    pos_emb_type = "NORMAL";
  }

  // TODO(lizanz03): only support c8 zp per channel
  bool is_cache_int8 = std::is_same<int8_t, XPU_CType>::value;
  bool has_zp = k_zeros && v_zeros;
  XPU_SType *quant_k_scale{nullptr}, *quant_v_scale{nullptr},
      *quant_v_scale_inv_zp{nullptr},
      *quant_k_zp{nullptr}, *quant_v_zp{nullptr};
  if (is_cache_int8) {
    // only support c8 per channel
    quant_k_scale = reinterpret_cast<XPU_SType*>(
        const_cast<sdata_t*>(k_scales.get().data<sdata_t>()));
    quant_v_scale = reinterpret_cast<XPU_SType*>(
        const_cast<sdata_t*>(v_scales.get().data<sdata_t>()));
    if (has_zp) {
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
  auto k_enc_tensor =
      paddle::empty({total_enc_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
  auto v_enc_tensor =
      paddle::empty({total_enc_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
  auto q_dec_tensor =
      paddle::empty({total_dec_len, hidden_dim}, qkv.type(), qkv.place());
  auto k_dec_tensor =
      paddle::empty({total_dec_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
  auto v_dec_tensor =
      paddle::empty({total_dec_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());

  if (enc_batch > 0) {
    xftblock::TransformerParam param;
    xftblock::TransformerVsl vsl;
    param.batch_size = enc_batch;
    param.head_num = num_heads;
    param.kv_head_num = kv_num_heads;
    param.head_dim = head_dim;
    param.max_batch_size = block_batch;

    vsl.usual_lod_vp = {
        const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()),
        enc_batch + 1,
        const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
    vsl.slot_mapping_vp = {
        const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()),
        enc_batch,
        const_cast<int32_t*>(encoder_batch_map.data<int32_t>())};  // real batch
        
    baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp{
        const_cast<int32_t*>(prefix_len_cpu.data<int32_t>()),
        enc_batch,
        const_cast<int32_t*>(prefix_len.data<int32_t>())};

    // buf tensor
    xftblock::Tensor q_enc_xft_tensor(q_enc_tensor.data(), KV_BUF_TYPE, {total_enc_len, hidden_dim});
    xftblock::Tensor k_enc_xft_tensor(k_enc_tensor.data(), KV_BUF_TYPE, {total_enc_len, kv_num_heads * head_dim});
    xftblock::Tensor v_enc_xft_tensor(v_enc_tensor.data(), KV_BUF_TYPE, {total_enc_len, kv_num_heads * head_dim});

    // rope + cache
    if (pos_emb_type == "NEOX") {
      ret = infer_ops::
          split_neox_cache_kv_encoder<XPU_XType, float, XPU_CType, int>(
              xpu_ctx->x_context(),
              reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()),  // qkv
              reinterpret_cast<const float*>(
                  rotary_embs.data<float>()),  // rotary_pos_emb
              reinterpret_cast<const int*>(
                  block_tables.data<int>()),  // block_table
              q_enc_xft_tensor.data<XPU_XType>(),
              k_enc_xft_tensor.data<XPU_XType>(),
              v_enc_xft_tensor.data<XPU_XType>(),
              const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                  key_cache.data<cdata_t>())),
              const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                  value_cache.data<cdata_t>())),
              vsl.usual_lod_vp,     // seq_lod
              vsl.slot_mapping_vp,  // real_batch
              param.batch_size,     // batch_size
              1,                    // emb_batch_size
              rope_max_seqlen,      // max_seqlen
              param.head_num,
              param.kv_head_num,
              param.head_dim,
              param.max_batch_size,
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
      auto q_enc_split_tensor =
          paddle::empty({total_enc_len, hidden_dim}, qkv.type(), qkv.place());
      auto k_enc_split_tensor =
          paddle::empty({total_enc_len, kv_num_heads * head_dim}, qkv.type(), qkv.place());
      // buf tensor
      xftblock::Tensor q_enc_split_xft_tensor(q_enc_split_tensor.data(), KV_BUF_TYPE, {total_enc_len, hidden_dim});
      xftblock::Tensor k_enc_split_xft_tensor(k_enc_split_tensor.data(), KV_BUF_TYPE, {total_enc_len, kv_num_heads * head_dim});
      xftblock::Tensor qkv_enc_xft_tensor(const_cast<void*>(qkv.data()), KV_BUF_TYPE, qkv.shape());
      ret = xftblock::split_qkv_block<XPU_XType>(
        &xctx,
        &qkv_enc_xft_tensor,
        &q_enc_split_xft_tensor,
        &k_enc_split_xft_tensor,
        &v_enc_xft_tensor,
        total_enc_len,
        hidden_dim / head_dim,
        kv_num_heads,
        head_dim);
      PD_CHECK(ret == api::SUCCESS, "split_qkv_block failed.");
      
      ret = infer_ops::vsl_rotary_embedding_gptj<XPU_XType, float, int32_t>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPU_XType*>(q_enc_split_tensor.data()),
        reinterpret_cast<const XPU_XType*>(k_enc_split_tensor.data()),
        reinterpret_cast<const float*>(rotary_embs.data<float>()),
        const_cast<XPU_XType*>(reinterpret_cast<const XPU_XType*>(q_enc_tensor.data())),
        const_cast<XPU_XType*>(reinterpret_cast<const XPU_XType*>(k_enc_tensor.data())),
        vsl.usual_lod_vp,
        param.batch_size,
        rope_max_seqlen,
        param.head_num,
        param.head_dim,
        "BLHD",
        prefix_lens_vp,
        "NORMAL",
        param.kv_head_num,
        false);
      PD_CHECK(ret == api::SUCCESS, "vsl_rotary_embedding_gptj failed.");

      // write to cache
      ret = infer_ops::reshape_and_cached_lod<float16, float16, int32_t>(
        xpu_ctx->x_context(),
        reinterpret_cast<const float16*>(k_enc_tensor.data()),
        reinterpret_cast<const float16*>(v_enc_tensor.data()),
        const_cast<float16*>(reinterpret_cast<const float16*>(key_cache.data())),
        const_cast<float16*>(reinterpret_cast<const float16*>(value_cache.data())),
        block_tables.data<int>(),
        vsl.usual_lod_vp,
        prefix_lens_vp,
        vsl.slot_mapping_vp,
        param.batch_size,
        param.kv_head_num,
        param.head_dim,
        rope_max_seqlen,
        block_size,
        max_block_per_seq,
        "BLHD",
        "HLD",
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
      );
      PD_CHECK(ret == api::SUCCESS, "reshape_and_cached_lod failed.");
    }

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
  }

  if (dec_batch > 0) {
    xftblock::TransformerParam param;
    xftblock::TransformerVsl vsl;
    param.batch_size = dec_batch;
    param.head_num = num_heads;
    param.kv_head_num = kv_num_heads;
    param.head_dim = head_dim;
    param.max_batch_size = block_batch;
    
    if (total_dec_len != dec_batch) {
      // buf tensor
      xftblock::Tensor q_dec_xft_tensor(q_dec_tensor.data(), KV_BUF_TYPE, {total_dec_len, hidden_dim});
      xftblock::Tensor k_dec_xft_tensor(k_dec_tensor.data(), KV_BUF_TYPE, {total_dec_len, kv_num_heads * head_dim});
      xftblock::Tensor v_dec_xft_tensor(v_dec_tensor.data(), KV_BUF_TYPE, {total_dec_len, kv_num_heads * head_dim});

      api::VectorParam<int32_t> decoder_context_len_cache_vp = {
          const_cast<int32_t*>(decoder_context_len_cache_cpu.data<int32_t>()),
          dec_batch,
          const_cast<int32_t*>(
              decoder_context_len_cache
                  .data<int32_t>())};  // use for split rope enc as prefix cache
                                       // len in MTP
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

      // rope + cache
      if (pos_emb_type == "NEOX") {
        ret = infer_ops::
            split_neox_cache_kv_encoder<XPU_XType, float, XPU_CType, int>(
                xpu_ctx->x_context(),
                reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()),  // qkv
                reinterpret_cast<const float*>(
                    rotary_embs.data<float>()),  // rotary_pos_emb
                reinterpret_cast<const int*>(
                    block_tables.data<int>()),  // block_table
                q_dec_xft_tensor.data<XPU_XType>(),
                k_dec_xft_tensor.data<XPU_XType>(),
                v_dec_xft_tensor.data<XPU_XType>(),
                const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                    key_cache.data<cdata_t>())),
                const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                    value_cache.data<cdata_t>())),
                decoder_seq_lod_vp,    // seq_lod
                decoder_batch_map_vp,  // real_batch
                param.batch_size,      // batch_size
                1,                     // emb_batch_size
                rope_max_seqlen,       // max_seqlen
                param.head_num,
                param.kv_head_num,
                param.head_dim,
                param.max_batch_size,
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
        ret = infer_ops::split_rope_cache_kv_encoder<XPU_XType,
                                                     float,
                                                     XPU_CType,
                                                     int,
                                                     E_Scale>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()) +
                total_enc_len * qkv_shape[qkv_shape.size() - 1],  // qkv
            reinterpret_cast<const float*>(
                rotary_embs.data<float>()),  // rotary_pos_emb
            reinterpret_cast<const int*>(
                block_tables.data<int>()),  // block_table
            q_dec_xft_tensor.data<XPU_XType>(),
            k_dec_xft_tensor.data<XPU_XType>(),
            v_dec_xft_tensor.data<XPU_XType>(),
            const_cast<XPU_CType*>(
                reinterpret_cast<const XPU_CType*>(key_cache.data<cdata_t>())),
            const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                value_cache.data<cdata_t>())),
            decoder_seq_lod_vp,            // seq_lod
            decoder_batch_map_vp,          // real_batch
            decoder_context_len_cache_vp,  // start_tokens (prefix len)
            param.batch_size,              // batch_size
            1,                             // emb_batch_size
            rope_max_seqlen,               // max_seqlen
            param.head_num,
            param.kv_head_num,
            param.head_dim,
            param.max_batch_size,
            block_size,
            max_block_per_seq,
            "BLHD",
            "HLD",
            pos_emb_type,
            nullptr,        // k_cache_scale_inv - use for per head
            nullptr,        // v_cache_scale_inv - use for per head
            quant_k_scale,  // intx_k_pc_scale
            quant_v_scale,  // intx_v_pc_scale
            quant_k_zp,     // intx_k_pc_zero
            quant_v_zp,     // intx_v_pc_zero
            q_norm_weight_data,
            k_norm_weight_data,
            rope_3d);
        PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_encoder failed.");
      }
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

      xftblock::Tensor q_dec_xft_tensor(q_dec_tensor.data(), KV_BUF_TYPE, {total_dec_len, hidden_dim});

      // rope + cache
      if (pos_emb_type == "NEOX") {
        ret = infer_ops::split_neox_cache_kv_decoder<XPU_XType,
                                                     float,
                                                     XPU_CType,
                                                     D_Scale,
                                                     int>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()) +
                total_enc_len * qkv_shape[qkv_shape.size() - 1],  // qkv
            reinterpret_cast<const float*>(
                rotary_embs.data<float>()),  // rotary_pos_emb
            reinterpret_cast<const int*>(
                block_tables.data<int>()),  // block_table
            q_dec_xft_tensor.data<XPU_XType>(),
            nullptr,
            nullptr,
            const_cast<XPU_CType*>(
                reinterpret_cast<const XPU_CType*>(key_cache.data<cdata_t>())),
            const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                value_cache.data<cdata_t>())),
            vsl.usual_lod_vp,     // seq_lod
            vsl.slot_mapping_vp,  // real_batch
            param.batch_size,     // batch_size
            1,                    // emb_batch_size = rotary_embs.dims()[1] = 1
            rope_max_seqlen,      // max_seqlen
            param.head_num,
            param.kv_head_num,
            param.head_dim,
            param.max_batch_size,
            block_size,
            max_block_per_seq,
            "BLHD",
            "HLD",
            pos_emb_type,
            reinterpret_cast<D_Scale*>(quant_k_scale),  // k_cache_scale_inv
            reinterpret_cast<D_Scale*>(quant_v_scale),  // v_cache_scale_inv
            reinterpret_cast<D_Scale*>(quant_k_zp),     // k_cache_zp
            reinterpret_cast<D_Scale*>(quant_v_zp),     // v_cache_zp
            rope_3d);
        PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_decoder failed.");
      } else {
        ret = infer_ops::split_rope_cache_kv_decoder<XPU_XType,
                                                     float,
                                                     XPU_CType,
                                                     D_Scale,
                                                     int>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()) +
                total_enc_len * qkv_shape[qkv_shape.size() - 1],  // qkv
            reinterpret_cast<const float*>(
                rotary_embs.data<float>()),  // rotary_pos_emb
            reinterpret_cast<const int*>(
                block_tables.data<int>()),  // block_table
            q_dec_xft_tensor.data<XPU_XType>(),
            nullptr,
            nullptr,
            const_cast<XPU_CType*>(
                reinterpret_cast<const XPU_CType*>(key_cache.data<cdata_t>())),
            const_cast<XPU_CType*>(reinterpret_cast<const XPU_CType*>(
                value_cache.data<cdata_t>())),
            vsl.usual_lod_vp,     // seq_lod
            vsl.slot_mapping_vp,  // real_batch
            param.batch_size,     // batch_size
            1,                    // emb_batch_size = rotary_embs.dims()[1] = 1
            rope_max_seqlen,      // max_seqlen
            param.head_num,
            param.kv_head_num,
            param.head_dim,
            param.max_batch_size,
            block_size,
            max_block_per_seq,
            "BLHD",
            "HLD",
            pos_emb_type,
            reinterpret_cast<D_Scale*>(quant_k_scale),  // k_cache_scale_inv
            reinterpret_cast<D_Scale*>(quant_v_scale),  // v_cache_scale_inv
            reinterpret_cast<D_Scale*>(quant_k_zp),     // k_cache_zp
            reinterpret_cast<D_Scale*>(quant_v_zp),     // v_cache_zp
            q_norm_weight_data,
            k_norm_weight_data,
            is_cache_int8,  // bool b_c8_pc
            rope_3d);
        PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_decoder failed.");
      }
    }
  }

  return {q_enc_tensor, k_enc_tensor, v_enc_tensor, q_dec_tensor, k_dec_tensor, v_dec_tensor};
}

std::vector<paddle::Tensor> SplitRopeKVCache(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& len_info_cpu,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& prefix_len_cpu,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& prefix_len,
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const bool use_neox_rotary_style,
    const bool rope_3d = false) {
#define APPLY_KERNEL(TX, TC, TS)                                    \
  return SplitRopeKVCacheKernel<TX, TC, TS>(qkv,                           \
                                     key_cache,                     \
                                     value_cache,                   \
                                     cum_offsets,                   \
                                     rotary_embs,                   \
                                     block_tables,                  \
                                     len_info_cpu,                  \
                                     encoder_seq_lod_cpu,           \
                                     decoder_seq_lod_cpu,           \
                                     encoder_batch_map_cpu,         \
                                     decoder_context_len_cpu,       \
                                     decoder_context_len_cache_cpu, \
                                     decoder_batch_map_cpu,         \
                                     prefix_len_cpu,                \
                                     encoder_seq_lod,               \
                                     decoder_seq_lod,               \
                                     encoder_batch_map,             \
                                     decoder_context_len,           \
                                     decoder_context_len_cache,     \
                                     decoder_batch_map,             \
                                     prefix_len,                    \
                                     k_scales,                      \
                                     v_scales,                      \
                                     k_zeros,                       \
                                     v_zeros,                       \
                                     q_norm_weight,                 \
                                     k_norm_weight,                 \
                                     kv_signal_data_cpu,            \
                                     cachekv_signal_thread_cpu,     \
                                     use_neox_rotary_style,         \
                                     rope_3d);

  const auto cache_dtype = key_cache.dtype();
  if (cache_dtype == paddle::DataType::BFLOAT16) {
    APPLY_KERNEL(paddle::bfloat16, paddle::bfloat16, paddle::bfloat16);
  } else if (cache_dtype == paddle::DataType::INT8) {
    APPLY_KERNEL(paddle::bfloat16, int8_t, paddle::bfloat16);
  } else {
    PD_THROW("split_rope_kvcache not support cache_dtype==%d",
             static_cast<int>(cache_dtype));
    return {};
  }

#undef APPLY_KERNEL
}

PD_BUILD_STATIC_OP(split_rope_kvcache)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "cum_offsets",
             "rotary_embs",
             "block_tables",
             "len_info_cpu",
             "encoder_seq_lod_cpu",
             "decoder_seq_lod_cpu",
             "encoder_batch_map_cpu",
             "decoder_context_len_cpu",
             "decoder_context_len_cache_cpu",
             "decoder_batch_map_cpu",
             "prefix_len_cpu",
             "encoder_seq_lod",
             "decoder_seq_lod",
             "encoder_batch_map",
             "decoder_context_len",
             "decoder_context_len_cache",
             "decoder_batch_map",
             "prefix_len",
             paddle::Optional("k_scales"),
             paddle::Optional("v_scales"),
             paddle::Optional("k_zeros"),
             paddle::Optional("v_zeros"),
             paddle::Optional("q_norm_weight"),
             paddle::Optional("k_norm_weight"),
             paddle::Optional("kv_signal_data_cpu"),
             paddle::Optional("cachekv_signal_thread_cpu")})
    .Attrs({"use_neox_rotary_style:bool", "rope_3d:bool"})
    .Outputs({"q_enc_tensor", "k_enc_tensor", "v_enc_tensor", "q_dec_tensor", "k_dec_tensor", "v_dec_tensor"})
    .SetKernelFn(PD_KERNEL(SplitRopeKVCache));
