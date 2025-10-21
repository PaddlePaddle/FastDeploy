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

#include "paddle/extension.h"
#include "moba_attn/moba_attn_utils.hpp"
#include "moba_attn/moba_attn.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename input_type, int moba_block_size, int kBlockM, int kMaxN, int tokens_per_block, bool need_k_mean>
__global__ void fused_block_mean_and_rope_kernel(
        const input_type *qkv_input,
        const input_type *qkv_bias,
        input_type *k_gate_mean,
        input_type *q_input,
        input_type *k_input,
        input_type *v_input,
        const float *rope_sin_cos,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int *cu_seq_q,
        const int *cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int max_input_length) {

    constexpr int kPackSize = 16 / sizeof(input_type);
    constexpr int kHeadDim = 128;

    using src_type = Vec<input_type, kPackSize>;

    using rope_type = Vec<float, kPackSize / 2>;
    using pack_half = std::conditional_t<std::is_same<input_type, cutlass::half_t>::value, __half2, nv_bfloat162>;

    __align__(16) __shared__ input_type local_sum_mem[128 / 32 * kHeadDim];

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidt_q = blockIdx.z * tokens_per_block;
    const int bidt_v = blockIdx.z * tokens_per_block;
    const int bidt_k = need_k_mean ? blockIdx.z * moba_block_size : blockIdx.z * tokens_per_block;
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
    rope_type sin, cos;

    const bool need_add_bias = qkv_bias != nullptr;

    if (need_add_bias) {
        src_bias.load_from(qkv_bias + bias_idx);
    }

    if (bidh < head_num) {
        const int cur_token = bidt_q + row_idx;
        const float * cos_rope = rope_sin_cos + (cur_token + seq_len_start) * (kHeadDim / 2) + col_idx * (kPackSize / 2);
        const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

        if (cur_token < seq_len) {
            src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx + cur_token * hidden);

            if (need_add_bias) {
                src.add(src_bias);
            }

            sin.load_from(sin_rope);
            cos.load_from(cos_rope);
            apply_rotary_embedding<input_type, kPackSize>(src, cos, sin);

            src.store_to(q_input + (cu_seq_q[bidb] + cur_token) * head_num * kHeadDim + bias_idx);
        }
    } else if (bidh < head_num + kv_head_num) {
        if constexpr (!need_k_mean) {
            const int cur_token = bidt_k + row_idx;
            const float * cos_rope = rope_sin_cos + (cur_token + seq_len_start) * (kHeadDim / 2) + col_idx * (kPackSize / 2);
            const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

            if (cur_token < seq_len) {
                src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx + cur_token * hidden);

                if (need_add_bias) {
                    src.add(src_bias);
                }

                sin.load_from(sin_rope);
                cos.load_from(cos_rope);
                apply_rotary_embedding<input_type, kPackSize>(src, cos, sin);

                src.store_to(k_input + (cu_seq_k[bidb] + cur_token) * head_num * kHeadDim + bias_idx- head_num * kHeadDim);
            }
        } else {
            if (bidt_k >= seq_len) {
                return;
            }

            src_type local_sum;
            local_sum.set_zero();

            const input_type* qkv = qkv_input + cu_seq_q[bidb] * hidden + bias_idx;

            for (int i = 0; i < moba_block_size; i += tokens_per_block) {
                const int cur_token = bidt_k + i + row_idx;
                if (cur_token < seq_len) {
                    src.load_from(qkv + cur_token * hidden);

                    if (need_add_bias) {
                        src.add(src_bias);
                    }
                    const float * cos_rope = rope_sin_cos + (cur_token + seq_len_start) * (kHeadDim / 2) + col_idx * (kPackSize / 2);
                    const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);
                    sin.load_from(sin_rope);
                    cos.load_from(cos_rope);

                    apply_rotary_embedding<input_type, kPackSize>(src, cos, sin);

                    src.store_to(k_input + (cu_seq_k[bidb] + cur_token) * kv_head_num * kHeadDim + bias_idx - head_num * kHeadDim);

                    local_sum.add(src);
                }
            }

            src_type neighbor;

            #pragma unroll
            for (int i = 0; i < kPackSize; i+=2) {
                *reinterpret_cast<int32_t*>(neighbor.data.elt + i) = __shfl_down_sync(0xffffffff, *reinterpret_cast<int32_t*>(local_sum.data.elt + i), 16);
            }

            local_sum.add(neighbor);

            if (lane_id < 16) {
                local_sum.store_to(local_sum_mem + warp_id * kHeadDim + lane_id * kPackSize);
            }

            __syncthreads();

            pack_half * local_sum_mem_half = reinterpret_cast<pack_half*>(local_sum_mem);

            pack_half local_sum_half = local_sum_mem_half[tidx];


            if (tidx < kHeadDim / 2) {

                #pragma unroll
                for (int i = 1; i < 4; i++) {
                    local_sum_half += local_sum_mem_half[tidx + i * (kHeadDim / 2)];
                }

                float inv_tokens_sum = fdividef(1.0f, min(seq_len - bidt_k, moba_block_size));

                local_sum_half *= float_2_half2<input_type>(inv_tokens_sum);

                const int store_mean_idx = ((bidb * kMaxN + blockIdx.z + seq_len_start / moba_block_size) * kv_head_num * kHeadDim + (bidh - head_num) * kHeadDim) / 2 + tidx;

                reinterpret_cast<pack_half*>(k_gate_mean)[store_mean_idx] = local_sum_half;
            }
        }
    } else {
        const int cur_token = bidt_v + row_idx;

        if (cur_token < seq_len) {
            src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx + cur_token * hidden);
            if (need_add_bias) {
                src.add(src_bias);
            }

            src.store_to(v_input + (cu_seq_k[bidb] + cur_token) * kv_head_num * kHeadDim + bias_idx - (head_num + kv_head_num) * kHeadDim);
        }
    }
}

template <typename input_type, int moba_block_size, int kBlockM, int kMaxN>
void fused_block_mean_and_rope(
        const input_type *qkv_input,
        const input_type *qkv_bias,
        input_type *k_gate_mean,
        input_type *q_input,
        input_type *k_input,
        input_type *v_input,
        const float *rope_sin_cos,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int *cu_seq_q,
        const int *cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int bsz,
        const int max_input_length,
        cudaStream_t stream) {

    static_assert(moba_block_size >= 64, "moba_block_size must be at least 64");
    constexpr int kPackSize = 16 / sizeof(input_type);
    constexpr int kHeadDim = 128;
    constexpr int kThreads = 128;
    constexpr int tokens_per_block = kThreads / (kHeadDim / kPackSize);
    dim3 grid_dims;
    grid_dims.x = bsz;
    grid_dims.y = head_num + 2 * kv_head_num;
    grid_dims.z = (max_seq_q + tokens_per_block - 1) / tokens_per_block;

    if (k_gate_mean != nullptr) {
        fused_block_mean_and_rope_kernel<input_type, moba_block_size, kBlockM, kMaxN, tokens_per_block, true>
        <<<grid_dims, kThreads, 0, stream>>>(
            qkv_input,
            qkv_bias,
            k_gate_mean,
            q_input,
            k_input,
            v_input,
            rope_sin_cos,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            max_input_length);
    } else {
        fused_block_mean_and_rope_kernel<input_type, moba_block_size, kBlockM, kMaxN, tokens_per_block, false>
        <<<grid_dims, kThreads, 0, stream>>>(
            qkv_input,
            qkv_bias,
            k_gate_mean,
            q_input,
            k_input,
            v_input,
            rope_sin_cos,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            max_input_length);
    }
}

void FusedBlockMeanAndRope(
        const paddle::Tensor& qkv_out,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& rotary_embs,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::optional<paddle::Tensor>& qkv_bias,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length,
        const int max_seq_q,
        const int max_seq_k,
        const std::string &cache_quant_type_str) {

    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;

    if (k_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        using cute_type = typename cuteType<T>::type;
        fused_block_mean_and_rope<cute_type, kMobaBlockSize, kBlockM, kMaxN>(
            reinterpret_cast<cute_type *>(const_cast<T*>(qkv_out.data<T>())),
            qkv_bias ? reinterpret_cast<cute_type *>(const_cast<T*>(qkv_bias.get().data<T>())) : nullptr,
            reinterpret_cast<cute_type *>(const_cast<T*>(k_block_means.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(q_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            rotary_embs.data<float>(),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            seq_len_encoder.dims()[0],
            max_input_length,
            qkv_out.stream());
    } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        using cute_type = typename cuteType<T>::type;
        fused_block_mean_and_rope<cute_type, kMobaBlockSize, kBlockM, kMaxN>(
            reinterpret_cast<cute_type *>(const_cast<T*>(qkv_out.data<T>())),
            qkv_bias ? reinterpret_cast<cute_type *>(const_cast<T*>(qkv_bias.get().data<T>())) : nullptr,
            reinterpret_cast<cute_type *>(const_cast<T*>(k_block_means.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(q_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            rotary_embs.data<float>(),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            seq_len_encoder.dims()[0],
            max_input_length,
            qkv_out.stream());
    }
}



PD_BUILD_STATIC_OP(fused_block_mean_and_rope)
    .Inputs({
        "qkv_out",
        "k_block_means",
        "q_input",
        "k_input",
        "v_input",
        "rotary_embs",
        "seq_len_encoder",
        "seq_len_decoder",
        "cu_seq_q",
        "cu_seq_k",
        paddle::Optional("qkv_bias")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_input_length: int",
        "max_seq_q: int",
        "max_seq_k: int",
        "cache_quant_type_str: std::string"})
    .Outputs({"q_input_out", "k_input_out", "v_input_out", "k_block_means_out"})
    .SetInplaceMap({{"q_input", "q_input_out"},
                    {"k_input", "k_input_out"},
                    {"v_input", "v_input_out"},
                    {"k_block_means", "k_block_means_out"}})
    .SetKernelFn(PD_KERNEL(FusedBlockMeanAndRope));
