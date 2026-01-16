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

#include "indexer_decode_rope_norm_cache_kernel.h"
#include "append_attn/utils.cuh"


#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T>
class type2value;

template <>
class type2value<phi::dtype::bfloat16> {
    public:
    static constexpr paddle::DataType value = paddle::DataType::BFLOAT16;
};

template <>
class type2value<phi::dtype::float16> {
    public:
    static constexpr paddle::DataType value = paddle::DataType::FLOAT16;
};

void IndexerDecoderRopeNormWriteCache(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const bool speculate_decoder) {

    AppendAttnMetaData meta_data;
    const auto& qkv_dims = qkv.dims();
    const auto& key_cache_dims = key_cache.dims();
    meta_data.token_nums = qkv_dims[0];
    meta_data.kv_num_heads = key_cache_dims[1];
    meta_data.head_dims = key_cache_dims[3];
    // TODO: trick method support c4, add attr head_dims in the future
    if (cache_quant_type_str == "cache_int4_zp") {
        meta_data.head_dims *= 2;
    }
    const int total_num_head =
        qkv_dims[qkv_dims.size() - 1] / meta_data.head_dims;
    meta_data.q_num_heads = total_num_head -  meta_data.kv_num_heads;

    meta_data.max_blocks_per_seq = block_tables.dims()[1];
    meta_data.block_size = key_cache.dims()[2];
    meta_data.batch_size = seq_lens_this_time.dims()[0];


    // template dtype generation
    phi::DataType dtype_id;
    switch (qkv.dtype()) {
        case paddle::DataType::FLOAT16:  {dtype_id = phi::DataType::FLOAT16;  break;}
        case paddle::DataType::BFLOAT16: {dtype_id = phi::DataType::BFLOAT16; break;}
        case paddle::DataType::INT32: {
        if (compute_dtype == "bf16") {
            dtype_id = phi::DataType::BFLOAT16;
            break;
        } else if (compute_dtype == "fp16") {
            dtype_id = phi::DataType::FLOAT16;
            break;
        } else {
            PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
            break;
        }
        }
        default: {
        PD_THROW(
            "NOT supported data type. "
            "Only float16 and bfloat16 are supported. ");
        break;
        }
    }

    typedef PDTraits<phi::DataType::BFLOAT16> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;


    auto main_stream = qkv.stream();
    cudaStream_t exec_stream = main_stream;

    paddle::Tensor qkv_out;
    qkv_out = qkv;

    IndexerDecoderRoPENormWriteCacheKernel<data_t, data_t>(
        meta_data,
        qkv_out,  // [token_num, num_heads, head_dim]
        seq_lens_decoder,
        seq_lens_encoder,
        cu_seqlens_q,
        block_tables,
        rotary_embs,
        qkv_out_scales,
        qkv_bias,
        cache_k_quant_scales,
        cache_k_zp,
        cache_quant_type_str,
        use_neox_rotary_style,
        rope_3d,
        max_input_length,
        exec_stream,
        &qkv_out,
        const_cast<paddle::Tensor*>(&key_cache),
        q_norm_weight,
        k_norm_weight,
        rms_norm_eps);
    return;
}

// PD_BUILD_STATIC_OP(indexer_decoder_rope_norm_write_cache)
//     .Inputs({"qkv",
//              "key_cache",
//              "seq_lens_encoder",
//              "seq_lens_decoder",
//              "seq_lens_this_time",
//              "batch_id_per_token",
//              "cu_seqlens_q",
//              "block_tables",
//              paddle::Optional("rotary_embs"),
//              paddle::Optional("qkv_bias"),
//              paddle::Optional("qkv_out_scales"),
//              paddle::Optional("cache_k_quant_scales"),
//              paddle::Optional("cache_k_dequant_scales"),
//              paddle::Optional("cache_k_zp"),
//              paddle::Optional("kv_signal_data"),
//              paddle::Optional("q_norm_weight"),
//              paddle::Optional("k_norm_weight")})
//     .Attrs({"rms_norm_eps: float",
//             "compute_type: std::string",
//             "cache_quant_type: std::string",
//             "use_neox_rotary_style: bool",
//             "rope_3d: bool",
//             "max_input_length: int",
//             "speculate_decoder: bool",})
//     .SetKernelFn(PD_KERNEL(IndexerDecoderRopeNormWriteCache));

