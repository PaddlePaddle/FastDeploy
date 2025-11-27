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
template <typename TX, typename TC, typename TS>
std::vector<paddle::Tensor> BlockAttnKernel(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cum_offsets,
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
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_scales_inv,
    const paddle::optional<paddle::Tensor>& v_scales_inv,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& shift,
    const paddle::optional<paddle::Tensor>& smooth,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const std::string& pos_emb_type,
    bool rope_3d) {
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
  int rope_3d_num_seqs = 1;
  if (rope_3d) {
    rope_max_seqlen = rotary_embs.dims()[3];
    rope_3d_num_seqs = rotary_embs.dims()[0];
  } else {
    rope_max_seqlen = rotary_embs.dims()[2];
  }

  auto block_attn_out =
      paddle::empty({token_num, hidden_dim}, qkv.type(), qkv.place());

  // TODO(lizanz03): only support c8 zp per channel
  bool is_cache_int8 = std::is_same<int8_t, XPU_CType>::value;
  bool has_zp = k_zeros && v_zeros;
  XPU_SType *quant_k_scale{nullptr}, *quant_v_scale{nullptr},
      *quant_k_scale_inv_zp{nullptr}, *quant_v_scale_inv_zp{nullptr},
      *quant_k_zp{nullptr}, *quant_v_zp{nullptr};
  // maxptr for xfa
  float *quant_k_scale_inv{nullptr}, *quant_v_scale_inv{nullptr};
  XPU_XType *p_shift{nullptr}, *p_smooth{nullptr};
  if (is_cache_int8) {
    // only support c8 per channel
    quant_k_scale = reinterpret_cast<XPU_SType*>(
        const_cast<sdata_t*>(k_scales.get().data<sdata_t>()));
    quant_v_scale = reinterpret_cast<XPU_SType*>(
        const_cast<sdata_t*>(v_scales.get().data<sdata_t>()));
    if (shift) {
      p_shift = reinterpret_cast<XPU_XType*>(
          const_cast<data_t*>(shift.get().data<data_t>()));
    }
    if (smooth) {
      p_smooth = reinterpret_cast<XPU_XType*>(
          const_cast<data_t*>(smooth.get().data<data_t>()));
    }
    if (has_zp) {
      quant_k_scale_inv_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(k_scales_inv.get().data<sdata_t>()));
      quant_v_scale_inv_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(v_scales_inv.get().data<sdata_t>()));
      quant_k_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(k_zeros.get().data<sdata_t>()));
      quant_v_zp = reinterpret_cast<XPU_SType*>(
          const_cast<sdata_t*>(v_zeros.get().data<sdata_t>()));
    } else {
      quant_k_scale_inv = reinterpret_cast<float*>(
          const_cast<float*>(k_scales_inv.get().data<float>()));
      quant_v_scale_inv = reinterpret_cast<float*>(
          const_cast<float*>(v_scales_inv.get().data<float>()));
    }
  }
  int ret = 0;
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
        nullptr};
    vsl.kv_lod_vp = {const_cast<int32_t*>(encoder_kv_lod_cpu.data<int32_t>()),
                     enc_batch + 1,
                     nullptr};
    vsl.slot_mapping_vp = {
        const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()),
        enc_batch,
        nullptr};  // real batch
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
        nullptr};

    float* fake_perhead_scale = nullptr;
    if (is_cache_int8 && has_zp && is_prefix_cache) {
      fake_perhead_scale = RAII_GUARD.alloc<float>(param.kv_head_num);
      // set fake_perhead_scale to ones
      int ret = api::constant<float>(
          xpu_ctx->x_context(), fake_perhead_scale, param.kv_head_num, 127.f);
      PD_CHECK(ret == api::SUCCESS, "api::constant failed.");
    }
    // buf tensor
    bool use_l3 = false;
    xftblock::Tensor q_buf(
        rt_guard, KV_BUF_TYPE, {total_enc_len, hidden_dim}, false, false);
    xftblock::Tensor k_buf(
        rt_guard->alloc(                              // src_data
            KV_BUF_TYPE,                              // data_type
            total_enc_len * kv_num_heads * head_dim,  // numel
            xftblock::AllocType::GM,                  // alloc_type
            &use_l3),                                 // use_l3
        nullptr,                                      // max_data
        nullptr,                                      // pc_scale
        KV_BUF_TYPE,                                  // type
        {total_enc_len, kv_num_heads * head_dim});
    xftblock::Tensor v_buf(
        rt_guard->alloc(                              // src_data
            KV_BUF_TYPE,                              // data_type
            total_enc_len * kv_num_heads * head_dim,  // numel
            xftblock::AllocType::GM,                  // alloc_type
            &use_l3),                                 // use_l3
        nullptr,                                      // max_data
        nullptr,                                      // pc_scale
        KV_BUF_TYPE,                                  // type
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
    // rope + cache
    int ret = 0;
    if (pos_emb_type == "NEOX") {
      ret = infer_ops::
          split_neox_cache_kv_encoder<XPU_XType, float, XPU_CType, int>(
              xpu_ctx->x_context(),
              reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()),  // qkv
              reinterpret_cast<const float*>(
                  rotary_embs.data<float>()),  // rotary_pos_emb
              reinterpret_cast<const int*>(
                  block_tables.data<int>()),  // block_table
              q_buf.data<XPU_XType>(),
              k_buf.data<XPU_XType>(),
              v_buf.data<XPU_XType>(),
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
      ret = infer_ops::split_rope_cache_kv_encoder<XPU_XType,
                                                   float,
                                                   XPU_CType,
                                                   int,
                                                   E_Scale>(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()),  // qkv
          reinterpret_cast<const float*>(
              rotary_embs.data<float>()),  // rotary_pos_emb
          reinterpret_cast<const int*>(
              block_tables.data<int>()),  // block_table
          q_buf.data<XPU_XType>(),
          k_buf.data<XPU_XType>(),
          v_buf.data<XPU_XType>(),
          const_cast<XPU_CType*>(
              reinterpret_cast<const XPU_CType*>(key_cache.data<cdata_t>())),
          const_cast<XPU_CType*>(
              reinterpret_cast<const XPU_CType*>(value_cache.data<cdata_t>())),
          vsl.usual_lod_vp,     // seq_lod
          vsl.slot_mapping_vp,  // real_batch
          prefix_lens_vp,       // start_tokens
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
          nullptr,        // k_cache_scale_inv - use for per head
          nullptr,        // v_cache_scale_inv - use for per head
          quant_k_scale,  // intx_k_pc_scale
          quant_v_scale,  // intx_v_pc_scale
          quant_k_zp,     // intx_k_pc_zero
          quant_v_zp,     // intx_v_pc_zero
          rope_3d);
      PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_encoder failed.");
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

    if (is_cache_int8 && has_zp && is_prefix_cache) {
      int64_t q_head_num = param.head_num;
      int64_t kv_head_num = param.kv_head_num;
      // assume q_layout is BLHD, q = q * k_scales_inv
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
                                          q_buf.data<XPU_XType>(),
                                          quant_k_scale_inv_zp,
                                          q_buf.data<XPU_XType>(),
                                          {total_enc_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
    }

    // attn encode
    if (is_prefix_cache) {
      ret =
          xftblock::xft_context_core_attenion_block<XPU_XType,
                                                    XPU_CType,
                                                    float>(&xctx,
                                                           &q_buf,
                                                           &key_cache_tensor,
                                                           &value_cache_tensor,
                                                           &encode_output,
                                                           param,
                                                           vsl);
    } else {
      ret = xftblock::
          xft_context_core_attenion_block<XPU_XType, XPU_XType, float>(
              &xctx, &q_buf, &k_buf, &v_buf, &encode_output, param, vsl);
    }
    PD_CHECK(ret == api::SUCCESS,
             "xftblock::xft_context_core_attenion_block failed.");

    if (is_cache_int8 && has_zp && is_prefix_cache) {
      int64_t q_head_num = param.head_num;
      int64_t kv_head_num = param.kv_head_num;
      // out = (out - v_zeros) * v_scales_inv
      ret = api::broadcast_sub<XPU_XType>(xpu_ctx->x_context(),
                                          encode_output.data<XPU_XType>(),
                                          quant_v_zp,
                                          encode_output.data<XPU_XType>(),
                                          {total_enc_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_sub failed.");
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
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
      ret = api::broadcast_add<XPU_XType>(xpu_ctx->x_context(),
                                          p_shift,
                                          encode_output.data<XPU_XType>(),
                                          encode_output.data<XPU_XType>(),
                                          {1, hidden_dim},
                                          {total_enc_len, hidden_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_add for shift failed.");
    }
    if (p_smooth != nullptr) {
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
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

    if (total_dec_len != dec_batch) {
      bool Eq_len = (total_dec_len % dec_batch == 0);
      // only support draft token num == 1, used in draft model
      int q_len = Eq_len ? total_dec_len / dec_batch : 1;
      xftblock::Tensor q_buf(
          rt_guard, KV_BUF_TYPE, {total_dec_len, hidden_dim}, false, false);
      xftblock::Tensor k_buf(rt_guard,
                             KV_BUF_TYPE,
                             {total_dec_len, kv_num_heads * head_dim},
                             false,
                             false);
      xftblock::Tensor v_buf(rt_guard,
                             KV_BUF_TYPE,
                             {total_dec_len, kv_num_heads * head_dim},
                             false,
                             false);

      api::VectorParam<int32_t> decoder_context_len_vp = {
          const_cast<int32_t*>(decoder_context_len_cpu.data<int32_t>()),
          dec_batch,
          nullptr};  // use for speculative_attention_decoder seq_len in
                     // MTP
      api::VectorParam<int32_t> decoder_context_len_cache_vp = {
          const_cast<int32_t*>(decoder_context_len_cache_cpu.data<int32_t>()),
          dec_batch,
          nullptr};  // use for split rope enc as prefix cache len in MTP
      api::VectorParam<int32_t> decoder_batch_map_vp = {
          const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()),
          dec_batch,
          nullptr};  // real batch
      api::VectorParam<int32_t> decoder_seq_lod_vp = {
          const_cast<int32_t*>(decoder_seq_lod_cpu.data<int32_t>()),
          dec_batch + 1,
          nullptr};  // use for split rope enc as lod in MTP

      // rope + cache
      int ret = 0;
      if (pos_emb_type == "NEOX") {
        ret = infer_ops::
            split_neox_cache_kv_encoder<XPU_XType, float, XPU_CType, int>(
                xpu_ctx->x_context(),
                reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()),  // qkv
                reinterpret_cast<const float*>(
                    rotary_embs.data<float>()),  // rotary_pos_emb
                reinterpret_cast<const int*>(
                    block_tables.data<int>()),  // block_table
                q_buf.data<XPU_XType>(),
                k_buf.data<XPU_XType>(),
                v_buf.data<XPU_XType>(),
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
            reinterpret_cast<const XPU_XType*>(qkv.data<data_t>()),  // qkv
            reinterpret_cast<const float*>(
                rotary_embs.data<float>()),  // rotary_pos_emb
            reinterpret_cast<const int*>(
                block_tables.data<int>()),  // block_table
            q_buf.data<XPU_XType>(),
            k_buf.data<XPU_XType>(),
            v_buf.data<XPU_XType>(),
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
            rope_3d);
        PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_encoder failed.");
      }

      float* fake_perhead_scale = nullptr;
      if (is_cache_int8 && has_zp) {
        int64_t q_head_num = param.head_num;
        int64_t kv_head_num = param.kv_head_num;
        fake_perhead_scale = RAII_GUARD.alloc<float>(kv_head_num);
        // q = q * k_scales_inv
        ret =
            api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
                                          q_buf.data<XPU_XType>(),
                                          quant_k_scale_inv_zp,
                                          q_buf.data<XPU_XType>(),
                                          {total_dec_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
        PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
        // set fake_perhead_scale to ones
        ret = api::constant<float>(
            xpu_ctx->x_context(), fake_perhead_scale, kv_head_num, 127.f);
        PD_CHECK(ret == api::SUCCESS, "api::constant failed.");
      }

      XPU_XType* q_buf_ptr = q_buf.data<XPU_XType>();
      XPU_XType* decode_output_ptr = decode_output.data<XPU_XType>();
      const int* decoder_context_len_ptr =
          decoder_context_len_cpu.data<int32_t>();
      const int* decoder_context_len_cache_ptr =
          decoder_context_len_cache_cpu.data<int32_t>();
      std::vector<int> lody_vec(dec_batch + 1);
      std::vector<int> offset_vec(dec_batch, 0);
      std::vector<int> lod_ref_vec(dec_batch + 1, 0);
      if (!Eq_len) {
        q_buf_ptr = RAII_GUARD.alloc<XPU_XType>(dec_batch * hidden_dim);
        decode_output_ptr = RAII_GUARD.alloc<XPU_XType>(dec_batch * hidden_dim);
        std::iota(lody_vec.begin(), lody_vec.end(), 0);  // 从0开始填充
        for (int i = 0; i < dec_batch; ++i) {
          int seq_len_this_time =
              decoder_context_len_ptr[i] - decoder_context_len_cache_ptr[i];
          offset_vec[i] = seq_len_this_time - 1;
          lod_ref_vec[i + 1] = lod_ref_vec[i] + seq_len_this_time;
        }
        ret = api::sequence_slice<float16, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<float16*>(q_buf.data<XPU_XType>()),
            // {total_dec_len, hidden_dim}
            reinterpret_cast<float16*>(q_buf_ptr),
            // {dec_batch, hidden_dim}
            decoder_seq_lod_vp,
            {offset_vec.data(), dec_batch, nullptr},
            {lody_vec.data(), dec_batch + 1, nullptr},
            hidden_dim);
        PD_CHECK(ret == api::SUCCESS, "api::sequence_slice failed.");
      }
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
          xpu_ctx->x_context(),
          decode_output_ptr,  // out
          q_buf_ptr,          // q
          nullptr,            // k
          nullptr,            // v
          reinterpret_cast<const XPU_CType*>(
              key_cache.data<cdata_t>()),  // k_cache
          reinterpret_cast<const XPU_CType*>(
              value_cache.data<cdata_t>()),  // v_cache
          reinterpret_cast<const int32_t*>(
              block_tables.data<int32_t>()),  // block_tables
          decoder_context_len_vp,             // seq_lengths
          decoder_batch_map_vp,               // valid_batch
          param.max_batch_size,               // batch_num
          q_len,                              // qlen
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
          nullptr,          // o_maxptr
          param.head_dim);  // vo_head_dim
      PD_CHECK(ret == api::SUCCESS,
               "xfa::speculative_attention_decoder failed.");
      if (!Eq_len) {
        ret = api::sequence_expand<float16, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<float16*>(decode_output_ptr),
            // {dec_batch, hidden_dim}
            reinterpret_cast<float16*>(decode_output.data<XPU_XType>()),
            // {total_dec_len, hidden_dim}
            {lody_vec.data(), dec_batch + 1, nullptr},
            decoder_seq_lod_vp,
            {lod_ref_vec.data(), dec_batch + 1, nullptr},
            hidden_dim);
        PD_CHECK(ret == api::SUCCESS, "api::sequence_expand failed.");
      }
    } else {
      vsl.usual_lod_vp = {
          const_cast<int32_t*>(decoder_context_len_cpu.data<int32_t>()),
          dec_batch,
          nullptr};
      vsl.slot_mapping_vp = {
          const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()),
          dec_batch,
          nullptr};  // real batch

      xftblock::Tensor q_buf(
          rt_guard, KV_BUF_TYPE, {total_dec_len, hidden_dim}, false, false);
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

      // rope + cache
      int ret = 0;
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
            q_buf.data<XPU_XType>(),
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
            q_buf.data<XPU_XType>(),
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
            is_cache_int8,                              // bool b_c8_pc
            rope_3d);
        PD_CHECK(ret == api::SUCCESS, "split_rope_cache_kv_decoder failed.");
      }

      float* fake_perhead_scale = nullptr;
      if (is_cache_int8 && has_zp) {
        int64_t q_head_num = param.head_num;
        int64_t kv_head_num = param.kv_head_num;
        fake_perhead_scale = RAII_GUARD.alloc<float>(kv_head_num);
        // q = q * k_scales_inv
        ret =
            api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
                                          q_buf.data<XPU_XType>(),
                                          quant_k_scale_inv_zp,
                                          q_buf.data<XPU_XType>(),
                                          {total_dec_len,
                                           kv_head_num,
                                           q_head_num / kv_head_num,
                                           param.head_dim},
                                          {1, kv_head_num, 1, param.head_dim});
        PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
        // set fake_perhead_scale to ones
        ret = api::constant<float>(
            xpu_ctx->x_context(), fake_perhead_scale, kv_head_num, 127.f);
        PD_CHECK(ret == api::SUCCESS, "api::constant failed.");
      }

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
          &q_buf,
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
            api::broadcast_sub<XPU_XType>(xpu_ctx->x_context(),
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
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
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
      ret = api::broadcast_add<XPU_XType>(xpu_ctx->x_context(),
                                          p_shift,
                                          decode_output.data<XPU_XType>(),
                                          decode_output.data<XPU_XType>(),
                                          {1, hidden_dim},
                                          {total_dec_len, hidden_dim});
      PD_CHECK(ret == api::SUCCESS, "api::broadcast_add for shift failed.");
    }
    if (p_smooth != nullptr) {
      ret = api::broadcast_mul<XPU_XType>(xpu_ctx->x_context(),
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

std::vector<paddle::Tensor> BlockAttn(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cum_offsets,
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
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_scales_inv,
    const paddle::optional<paddle::Tensor>& v_scales_inv,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& shift,
    const paddle::optional<paddle::Tensor>& smooth,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const std::string& pos_emb_type = "NORMAL",
    bool rope_3d = false) {
#define APPLY_KERNEL(TX, TC, TS)                                    \
  return BlockAttnKernel<TX, TC, TS>(qkv,                           \
                                     key_cache,                     \
                                     value_cache,                   \
                                     cum_offsets,                   \
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
                                     k_scales,                      \
                                     v_scales,                      \
                                     k_scales_inv,                  \
                                     v_scales_inv,                  \
                                     k_zeros,                       \
                                     v_zeros,                       \
                                     shift,                         \
                                     smooth,                        \
                                     kv_signal_data_cpu,            \
                                     cachekv_signal_thread_cpu,     \
                                     pos_emb_type,                  \
                                     rope_3d);

  const auto cache_dtype = key_cache.dtype();
  if (cache_dtype == paddle::DataType::BFLOAT16) {
    APPLY_KERNEL(paddle::bfloat16, paddle::bfloat16, paddle::bfloat16);
  } else if (cache_dtype == paddle::DataType::INT8) {
    APPLY_KERNEL(paddle::bfloat16, int8_t, paddle::bfloat16);
  } else {
    PD_THROW("block_attn not support cache_dtype==%d",
             static_cast<int>(cache_dtype));
    return {};
  }

#undef APPLY_KERNEL
}

std::vector<std::vector<int64_t>> BlockAttnInferShape(
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

std::vector<paddle::DataType> BlockAttnInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(block_attn)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "cum_offsets",
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
             paddle::Optional("k_scales"),
             paddle::Optional("v_scales"),
             paddle::Optional("k_scales_inv"),
             paddle::Optional("v_scales_inv"),
             paddle::Optional("k_zeros"),
             paddle::Optional("v_zeros"),
             paddle::Optional("shift"),
             paddle::Optional("smooth"),
             paddle::Optional("kv_signal_data_cpu"),
             paddle::Optional("cachekv_signal_thread_cpu")})
    .Attrs({"pos_emb_type:std::string", "rope_3d:bool"})
    .Outputs({"block_attn_out"})
    .SetKernelFn(PD_KERNEL(BlockAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(BlockAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BlockAttnInferDtype));
