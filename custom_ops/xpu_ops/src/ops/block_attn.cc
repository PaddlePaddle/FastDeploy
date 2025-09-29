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
#include <infer_ops.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xft_api.h>

#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

namespace xftblock = baidu::xpu::xftblock;

std::vector<paddle::Tensor> BlockAttnKernel(
    const paddle::Tensor
        &qkv, // [token_num, (num_heads + 2 * kv_num_heads) * head_dim]
    const paddle::Tensor &key_cache, const paddle::Tensor &value_cache,
    const paddle::Tensor &cum_offsets, const paddle::Tensor &rotary_embs,
    const paddle::Tensor &block_tables,
    const paddle::Tensor &prefix_block_tables,
    const paddle::Tensor &p_kcache_perhead_scale,
    const paddle::Tensor &p_vcache_perhead_scale,
    const paddle::Tensor &enc_batch_tensor,
    const paddle::Tensor &dec_batch_tensor,
    const paddle::Tensor &total_enc_len_tensor,
    const paddle::Tensor &encoder_seq_lod_cpu,
    const paddle::Tensor &encoder_batch_map_cpu,
    const paddle::Tensor &decoder_context_len_cpu,
    const paddle::Tensor &decoder_batch_map_cpu,
    const std::string &pos_emb_type="NORMAL",
    bool rope_3d=false) {
    phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
    auto dev_ctx =
        paddle::experimental::DeviceContextPool::Instance().Get(place);
    auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
    xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
    auto rt_guard = xctx.get_rt_guard();

    using XPUType = typename XPUTypeTrait<bfloat16>::Type;
    using CType = typename XPUTypeTrait<bfloat16>::Type;
    typedef paddle::bfloat16 data_t;
    typedef paddle::bfloat16 cdata_t;
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

    int enc_batch = enc_batch_tensor.data<int32_t>()[0];
    int dec_batch = dec_batch_tensor.data<int32_t>()[0];
    int total_enc_len = total_enc_len_tensor.data<int32_t>()[0];
    int rope_max_seqlen = 0;
    int rope_3d_num_seqs = 1;
    if (rope_3d) {
        rope_max_seqlen = rotary_embs.dims()[3];
        rope_3d_num_seqs = rotary_embs.dims()[0];
    } else {
        rope_max_seqlen = rotary_embs.dims()[2];
    }

    auto block_attn_out =
        paddle::full({token_num, hidden_dim}, -1, qkv.type(), qkv.place());

    if (enc_batch > 0) {
        xftblock::TransformerParam param;
        xftblock::TransformerVsl vsl;
        param.batch_size = enc_batch;
        param.head_num = num_heads;
        param.kv_head_num = kv_num_heads;
        param.head_dim = head_dim;
        param.max_batch_size = block_batch;
        param.max_seq_len = max_seq_len;

        vsl.usual_lod_vp = {
            const_cast<int32_t *>(encoder_seq_lod_cpu.data<int32_t>()),
            enc_batch + 1, nullptr};
        vsl.slot_mapping_vp = {
            const_cast<int32_t *>(encoder_batch_map_cpu.data<int32_t>()),
            enc_batch, nullptr}; // real batch

        param.max_valid_seqlen = vsl.usual_lod_vp.cpu[enc_batch];
        // normal setting
        param.use_cache_per_channel = false;

        // setting for prefix cache
        param.prefill_len = -1;
        param.page_attn.block_size = block_size;
        param.page_attn.max_num_blocks_per_seq = max_block_per_seq;
        xftblock::Tensor *prefix_block_tables_ptr = nullptr;
        if (prefix_block_tables.defined()) {
            prefix_block_tables_ptr = new xftblock::Tensor(
                reinterpret_cast<void *>(
                    const_cast<int32_t *>(prefix_block_tables.data<int32_t>())),
                xftblock::DataType::DT_INT32,
                {prefix_block_tables.dims()[0], prefix_block_tables.dims()[1]});
        }
        param.page_attn.block_table = prefix_block_tables_ptr;
        vsl.kv_lod_vp = {
            const_cast<int32_t *>(encoder_seq_lod_cpu.data<int32_t>()),
            enc_batch + 1, nullptr};

        baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp{
        nullptr,
        0,
        nullptr};

        xftblock::Tensor q_buf(rt_guard, xftblock::DataType::DT_BFLOAT16,
                               {total_enc_len, hidden_dim}, false, false);
        xftblock::Tensor k_buf(rt_guard, xftblock::DataType::DT_BFLOAT16,
                               {total_enc_len, kv_num_heads * head_dim}, false,
                               false);
        xftblock::Tensor v_buf(rt_guard, xftblock::DataType::DT_BFLOAT16,
                               {total_enc_len, kv_num_heads * head_dim}, false,
                               false);
        xftblock::Tensor encode_output(
            reinterpret_cast<void *>(
                const_cast<data_t *>(block_attn_out.data<data_t>())),
            xftblock::DataType::DT_BFLOAT16, {total_enc_len, hidden_dim});
        // rope + cache
        int ret =
            infer_ops::split_rope_cache_kv_encoder<XPUType, float, CType, int, bfloat16>( // TSCALE bfloat16 means no-use
                xpu_ctx->x_context(),
                reinterpret_cast<const XPUType *>(qkv.data<data_t>()), // qkv
                reinterpret_cast<const float *>(
                    rotary_embs.data<float>()), // rotary_pos_emb
                reinterpret_cast<const int *>(
                    block_tables.data<int>()), // block_table
                q_buf.data<XPUType>(), k_buf.data<XPUType>(),
                v_buf.data<XPUType>(),
                const_cast<CType *>(
                    reinterpret_cast<const CType *>(key_cache.data<cdata_t>())),
                const_cast<CType *>(reinterpret_cast<const CType *>(
                    value_cache.data<cdata_t>())),
                vsl.usual_lod_vp,      // seq_lod
                vsl.slot_mapping_vp,   // real_batch
                prefix_lens_vp,        // start_tokens
                param.batch_size,      // batch_size
                1,                     // emb_batch_size
                rope_max_seqlen,       // max_seqlen
                param.head_num, param.kv_head_num, param.head_dim,
                param.max_batch_size, block_size, max_block_per_seq, "BLHD",
                "HLD", pos_emb_type,
                !p_kcache_perhead_scale.defined()
                    ? nullptr
                    : p_kcache_perhead_scale.data<float>() +
                          param.kv_head_num, // k_cache_scale_inv
                !p_vcache_perhead_scale.defined()
                    ? nullptr
                    : p_vcache_perhead_scale.data<float>() +
                          param.kv_head_num, // v_cache_scale_inv
                nullptr,                     // int4_k_cache_scale
                nullptr,                     // int4_v_cache_scale
                nullptr,                     // int4_k_cache_zero
                nullptr);                    // int4_v_cache_zero
        XFTBLOCK_CHECK_EQ(ret, api::SUCCESS);

        // attn encode
        ret = xftblock::xft_context_core_attenion_block<XPUType, CType, float>(
            &xctx, &q_buf, &k_buf, &v_buf, &encode_output, param, vsl);
        XFTBLOCK_CHECK_EQ(ret, api::SUCCESS);
        if (prefix_block_tables_ptr)
            delete prefix_block_tables_ptr;
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

        vsl.usual_lod_vp = {
            const_cast<int32_t *>(decoder_context_len_cpu.data<int32_t>()),
            dec_batch, nullptr};
        vsl.slot_mapping_vp = {
            const_cast<int32_t *>(decoder_batch_map_cpu.data<int32_t>()),
            dec_batch, nullptr}; // real batch

        xftblock::Tensor q_buf(rt_guard, xftblock::DataType::DT_BFLOAT16,
                               {dec_batch, hidden_dim}, false, false);
        xftblock::Tensor key_cache_tensor(
            reinterpret_cast<void *>(
                const_cast<cdata_t *>(key_cache.data<cdata_t>())),
            xftblock::DataType::DT_BFLOAT16, // cache type
            {cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]});
        xftblock::Tensor value_cache_tensor(
            reinterpret_cast<void *>(
                const_cast<cdata_t *>(value_cache.data<cdata_t>())),
            xftblock::DataType::DT_BFLOAT16, // cache type
            {cache_shape[0], cache_shape[1], cache_shape[2], cache_shape[3]});
        xftblock::Tensor block_table_tensor(
            reinterpret_cast<void *>(
                const_cast<int32_t *>(block_tables.data<int32_t>())),
            xftblock::DataType::DT_INT32,
            {block_table_shape[0], block_table_shape[1]});
        xftblock::Tensor decode_output(
            reinterpret_cast<void *>(
                const_cast<data_t *>(block_attn_out.data<data_t>()) +
                total_enc_len * hidden_dim),
            xftblock::DataType::DT_BFLOAT16, {dec_batch, hidden_dim});

        // normal setting
        param.use_cache_per_channel = false;
        param.prefill_len = -1;
        param.page_attn.block_size = block_size;
        param.page_attn.max_context_len = max_seq_len;
        param.page_attn.max_num_blocks_per_seq = max_block_per_seq;
        param.page_attn.block_table = &block_table_tensor;

        int ret = 0;
        // rope + cache
        ret = infer_ops::split_rope_cache_kv_decoder<XPUType, float, CType,
                                                         float, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType *>(qkv.data<data_t>()) +
                total_enc_len * qkv_shape[qkv_shape.size() - 1], // qkv
            reinterpret_cast<const float *>(
                rotary_embs.data<float>()), // rotary_pos_emb
            reinterpret_cast<const int *>(
                block_tables.data<int>()), // block_table
            q_buf.data<XPUType>(), nullptr, nullptr,
            const_cast<CType *>(
                reinterpret_cast<const CType *>(key_cache.data<cdata_t>())),
            const_cast<CType *>(
                reinterpret_cast<const CType *>(value_cache.data<cdata_t>())),
            vsl.usual_lod_vp,      // seq_lod
            vsl.slot_mapping_vp,   // real_batch
            param.batch_size,      // batch_size
            1,                     // emb_batch_size
            rope_max_seqlen,       // max_seqlen
            param.head_num, param.kv_head_num, param.head_dim,
            param.max_batch_size, block_size, max_block_per_seq, "BLHD", "HLD",
            pos_emb_type,
            !p_kcache_perhead_scale.defined()
                ? nullptr
                : p_kcache_perhead_scale.data<float>() +
                      param.kv_head_num, // k_cache_scale_inv
            !p_vcache_perhead_scale.defined()
                ? nullptr
                : p_vcache_perhead_scale.data<float>() +
                      param.kv_head_num, // v_cache_scale_inv
            nullptr,                     // k_cache_zp
            nullptr,                     // v_cache_zp
            false,                       // b_c8_pc
            rope_3d,                     // rope_3d
            rope_3d_num_seqs);
        XFTBLOCK_CHECK_EQ(ret, api::SUCCESS);

        // attn decode
        ret = xftblock::xft_decoder_core_attenion_block<XPUType, CType, float>(
            &xctx, &q_buf, &key_cache_tensor, &value_cache_tensor,
            &decode_output, param, vsl);
        XFTBLOCK_CHECK_EQ(ret, api::SUCCESS);
    }

    return {block_attn_out};
}

std::vector<std::vector<int64_t>>
BlockAttnInferShape(const std::vector<int64_t> &qkv_shape,
                    const std::vector<int64_t> &key_cache_shape,
                    const std::vector<int64_t> &value_cache_shape) {
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

std::vector<paddle::DataType>
BlockAttnInferDtype(const paddle::DataType &qkv_dtype,
                    const paddle::DataType &key_cache_dtype,
                    const paddle::DataType &value_cache_dtype) {
    return {qkv_dtype};
}

PD_BUILD_OP(block_attn)
    .Inputs({
        "qkv",
        "key_cache",
        "value_cache",
        "cum_offsets",
        "rotary_embs",
        "block_tables",
        "prefix_block_tables",
        "p_kcache_perhead_scale",
        "p_vcache_perhead_scale",
        "enc_batch_tensor",
        "dec_batch_tensor",
        "total_enc_len_tensor",
        "encoder_seq_lod_cpu",
        "encoder_batch_map_cpu",
        "decoder_context_len_cpu",
        "decoder_batch_map_cpu",
    })
    .Attrs({"pos_emb_type:std::string", "rope_3d:bool"})
    .Outputs({"block_attn_out"})
    .SetKernelFn(PD_KERNEL(BlockAttnKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(BlockAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BlockAttnInferDtype));
