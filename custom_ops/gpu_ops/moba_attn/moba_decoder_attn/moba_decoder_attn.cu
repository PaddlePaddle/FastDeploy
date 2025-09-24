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

#pragma once
#include "paddle/extension.h"
#include "moba_decoder_attn_kernel.h"
#include "moba_attn/moba_attn.h"


template<bool Is_first, int kMiLen, typename Tensor0, typename Tensor1, typename T>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &acc_o, const T *scores_max, const T *scores_max_prev, T * scores_sum, const float softmax_scale) {
    if (Is_first) {
        scale_apply_exp2<kMiLen>(scores, scores_max, scores_sum, softmax_scale);
    } else {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < kMiLen; ++mi) {
            const float scores_scale = expf((scores_max_prev[mi] - scores_max[mi]) * softmax_scale);
            scores_sum[mi] *= scores_scale;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale;
            }
        }
        scale_apply_exp2<kMiLen>(scores, scores_max, scores_sum, softmax_scale);
    }
};

template<typename Kernel_traits, typename ParamType>
__global__ __launch_bounds__(Kernel_traits::kNThreads) void moba_decoder_attention_kernel(ParamType params) {
    using cuteType = typename Kernel_traits::cuteType;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using CacheKV_traits = typename Kernel_traits::CacheKV_traits;
    constexpr int32_t kHeadDim = Kernel_traits::kHeadDim;
    constexpr int32_t kHeadDimKV = Kernel_traits::kHeadDimKV;
    constexpr int32_t kBlockM = Kernel_traits::kBlockM;
    constexpr int32_t kBlockSize = Kernel_traits::kBlockSize;
    constexpr int32_t kGqaGroupSize = Kernel_traits::kGqaGroupSize;
    constexpr int32_t kNWarps = Kernel_traits::kNWarps;
    constexpr int32_t kTileN = Kernel_traits::kTileN;
    constexpr int32_t kBlockN = kTileN * kBlockSize;
    constexpr int32_t kDataBits = Kernel_traits::kDataBits;
    constexpr int32_t kMiLen = (kGqaGroupSize + 7) / 8;

    const int32_t bi = blockIdx.y;
    const int32_t tidx = threadIdx.x;
    const int32_t partition_idx = blockIdx.x;
    const int32_t kv_head_idx = blockIdx.z;
    const int32_t q_head_idx = kv_head_idx * kGqaGroupSize;

    const int32_t seq_len = params.seq_lens_decoder[bi] == 0 ? 0 : params.seq_lens_decoder[bi] + 1;

    const int32_t head_num = params.head_num;
    const int32_t kv_head_num = params.kv_head_num;

    const int32_t partition_num = (seq_len + kBlockN - 1) / kBlockN;

    if (seq_len == 0 || partition_idx >= partition_num) {
        return;
    }

    if (seq_len >= params.use_moba_seq_limit && params.qk_gate_topk_idx_ptr[(bi * kv_head_num + kv_head_idx) * Kernel_traits::kMaxN + partition_idx] == 0) {
        return;
    }


    const int q_bias_offset = q_head_idx * kHeadDim;

    cuteType * q_input = reinterpret_cast<cuteType *>(params.q_input) + params.cu_seq_q[bi] * head_num * kHeadDim;

    Tensor gQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<const cuteType *>(q_input) + q_bias_offset),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        Stride<Int<kHeadDim>, _1>{});

    const int32_t block_idx = partition_idx * kTileN;
    const int* block_table = params.block_table + bi * params.max_num_blocks_per_seq + block_idx;
    const int32_t physical_block_number = block_table[0];

    const int32_t cache_offset = (physical_block_number * kv_head_num + kv_head_idx) * kBlockSize * kHeadDimKV;

    Tensor gK = make_tensor(
        make_gmem_ptr(reinterpret_cast<const cuteType *>(params.cache_k) + cache_offset),
        Shape<Int<kBlockSize>, Int<kHeadDimKV>>{},
        Stride<Int<kHeadDimKV>, _1>{});

    Tensor gV = make_tensor(
        make_gmem_ptr(reinterpret_cast<const cuteType *>(params.cache_v) + cache_offset),
        Shape<Int<kBlockSize>, Int<kHeadDimKV>>{},
        Stride<Int<kHeadDimKV>, _1>{});

    extern __shared__ char smem_[];
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<cuteType *>(smem_)),
        typename Kernel_traits::SmemLayoutQ{});
    Tensor sQK = make_tensor(
        sQ.data() + size(sQ),
        typename Kernel_traits::SmemLayoutQK{});

    Tensor sK = make_tensor(sQK.data() + size(sQK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    __shared__ ElementAccum scores_warp[kNWarps][kMiLen * kBlockM];

    auto gmem_tiled_copy_Q = typename Kernel_traits::GmemTiledCopyQ{};
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    auto gmem_tiled_copy_KV = typename Kernel_traits::GmemTiledCopyKV{};
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);

    Tensor cQ = make_identity_tensor(make_shape(kBlockM, kHeadDim));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);

    Tensor cKV = make_identity_tensor(make_shape(kBlockSize, kHeadDim));
    Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);

    typename Kernel_traits::TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using SmemCopyAtom = typename Kernel_traits::SmemCopyAtom;
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);

    Tensor tSsQK = smem_thr_copy_Q.partition_S(sQK);
    Tensor tSrQK = thr_mma.partition_fragment_A(sQK);

    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);

    copy<false>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, kGqaGroupSize);


    cute::cp_async_fence();
    cp_async_wait<0>();

    const int32_t remain_seq_len = seq_len - partition_idx * kTileN * kBlockSize;

    copy(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV);

    cute::cp_async_fence();

    const int32_t warp_id = tidx / 32;
    const int32_t lane_id = tidx % 32;
    const int32_t row = lane_id / 4;
    const int32_t col = lane_id % 4;
    const int row_idx = tidx / 4;

    using scale_k_vec = Vec<cuteType, 32>;
    using scale_v_vec = Vec<cuteType, 4>;

    scale_k_vec scale_k;
    scale_k_vec zp_k;
    scale_v_vec scale_v;
    scale_v_vec zp_v;
    if constexpr (kDataBits == 4) {
        scale_k = *reinterpret_cast<const scale_k_vec*>(params.cache_k_dequant_scale + kv_head_idx * kHeadDim + col * 32);
        zp_k = *reinterpret_cast<const scale_k_vec*>(params.cache_k_zp + kv_head_idx * kHeadDim + col * 32);
        scale_v = *reinterpret_cast<const scale_v_vec*>(params.cache_v_dequant_scale + kv_head_idx * kHeadDim + row_idx * 4);
        zp_v = *reinterpret_cast<const scale_v_vec*>(params.cache_v_zp + kv_head_idx * kHeadDim + row_idx * 4);
    }

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockSize>>{});

    ElementAccum scores_max[kMiLen];
    ElementAccum scores_max_prev[kMiLen];
    ElementAccum scores_sum[kMiLen];

    #pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
        scores_max[mi] = -INFINITY;
        scores_sum[mi] = 0;
    }

    const int cache_offset_step = kv_head_num * kBlockSize * kHeadDimKV;

    #pragma unroll
    for (int n = 0; n < kTileN; ++n) {
        const int cur_remain_seq_len = remain_seq_len - n * kBlockSize;

        if (cur_remain_seq_len <= 0) {
            break;
        }

        clear(acc_s);
        cp_async_wait<0>();
        __syncthreads();

        if (n > 0) {
            tVgV.data() = tVgV.data() + (block_table[n] - block_table[n - 1]) * cache_offset_step;
        }

        copy(gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV);

        cute::cp_async_fence();

        if constexpr (kDataBits == 16) {
            if (n == 0) {
                gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_thr_copy_Q, smem_thr_copy_K, smem_tiled_copy_Q, smem_tiled_copy_K);
            } else {
                gemm<true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_thr_copy_Q, smem_thr_copy_K, smem_tiled_copy_Q, smem_tiled_copy_K);
            }
        } else {
            Tensor tSrKQuant = make_tensor<cuteType>(
                Layout<
                    Shape<Shape<_2, _2>, Int<kBlockSize / 32>>,
                    Stride<Shape<_1, _2>, _4>>{});
            if (n == 0) {
                gemm_qk_quant<CacheKV_traits, cuteType, kHeadDim, kDataBits>(acc_s, tSrQ, tSsQ, tSrKQuant, sK, tiled_mma, smem_thr_copy_Q, smem_tiled_copy_Q, tidx, scale_k.data.elt, zp_k.data.elt);
            } else {
                gemm_qk_quant<CacheKV_traits, cuteType, kHeadDim, kDataBits, true>(acc_s, tSrQ, tSsQ, tSrKQuant, sK, tiled_mma, smem_thr_copy_Q, smem_tiled_copy_Q, tidx, scale_k.data.elt, zp_k.data.elt);
            }
        }
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

        if (partition_idx == partition_num - 1 && cur_remain_seq_len < kBlockSize) {
            apply_mask<kMiLen>(scores, warp_id, col, cur_remain_seq_len);
        }

        #pragma unroll
        for (int mi = 0; mi < kMiLen; ++mi) {
            scores_max_prev[mi] = scores_max[mi];
        }

        reduce_max<kMiLen>(scores, scores_max);

        if (col == 0) {
            scores_warp[warp_id][row] = scores_max[0];
            if constexpr (kMiLen > 1) {
                scores_warp[warp_id][row + 8] = scores_max[1];
            }
        }

        __syncthreads();

        MaxOp<ElementAccum> max_op;

        if (tidx < kGqaGroupSize) {
            float cur_max = scores_warp[0][tidx];
            #pragma unroll
            for (uint32_t i = 1; i < kNWarps; ++i) {
                cur_max = max_op(scores_warp[i][tidx], cur_max);
            }
            scores_warp[0][tidx] = cur_max;
        }

        cp_async_wait<0>();
        __syncthreads();

        if (cur_remain_seq_len > kBlockSize && n < kTileN - 1) {
            tKgK.data() = tKgK.data() + (block_table[n + 1] - block_table[n]) * cache_offset_step;
            copy(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV);
            cute::cp_async_fence();
        }

        #pragma unroll
        for (int mi = 0; mi < kMiLen; ++mi) {
            scores_max[mi] = scores_warp[0][row + mi * 8];
        }

        if (n == 0) {
            softmax_rescale_o<true, kMiLen>(scores, acc_o, scores_max, scores_max_prev, scores_sum, params.inv_sqrt_dh);
        } else {
            softmax_rescale_o<false, kMiLen>(scores, acc_o, scores_max, scores_max_prev, scores_sum, params.inv_sqrt_dh);
        }

        Tensor rS = convert_type<cuteType>(acc_s);

        Tensor trQK = smem_thr_copy_O.retile_S(rS);
        Tensor tsQK = smem_thr_copy_O.partition_D(sQK);
        cute::copy(smem_tiled_copy_O, trQK, tsQK);

        __syncthreads();

        if constexpr (kDataBits == 16) {
            gemm(acc_o, tSrQK, tOrVt, tSsQK, tOsVt, tiled_mma, smem_thr_copy_Q, smem_thr_copy_V, smem_tiled_copy_Q, smem_tiled_copy_V);
        } else {
            Tensor tSrVQuant = make_tensor<cuteType>(
                Layout<
                    Shape<_4, Shape<_2, _2>>,
                    Stride<_1, Shape<_4, _8>>>{});
            gemm_value_quant<CacheKV_traits, cuteType, kHeadDim, kDataBits>(acc_o, tSrQK, tSsQK, tSrVQuant, sV, tiled_mma, smem_thr_copy_Q, smem_tiled_copy_Q, tidx, scale_v.data.elt, zp_v.data.elt);
        }
    }

    const uint32_t pack_max_partition_num = (params.max_num_partitions + 3) / 4 * 4;
    uint32_t max_sum_offset = bi * pack_max_partition_num * head_num + (tidx + q_head_idx) * pack_max_partition_num + partition_idx;

    if (tidx < kGqaGroupSize) {
        params.maxs[max_sum_offset] = scores_warp[0][tidx] * params.inv_sqrt_dh;
    }

    SumOp<ElementAccum> sum_op;
    #pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
        scores_sum[mi] = Allreduce<4>::run(scores_sum[mi], sum_op);
    }
    __syncthreads();

    if (col == 0) {
        scores_warp[warp_id][row] = scores_sum[0];
        if constexpr (kMiLen > 1) {
            scores_warp[warp_id][row + 8] = scores_sum[1];
        }
    }


    Tensor rO = convert_type<cuteType>(acc_o);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sQ);

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    __syncthreads();

    if (tidx < kGqaGroupSize) {
        float cur_sum = scores_warp[0][tidx];
        #pragma unroll
        for (uint32_t i = 1; i < kNWarps; ++i) {
            cur_sum = sum_op(scores_warp[i][tidx], cur_sum);
        }
        scores_warp[0][tidx] = cur_sum;
    }

    Tensor gO = make_tensor(
        make_gmem_ptr(reinterpret_cast<cuteType *>(params.partition_attn_out) + ((bi * params.max_num_partitions + partition_idx) * head_num  + q_head_idx)* kHeadDim),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            Stride<Int<kHeadDim>, _1>{});

    auto gmem_tiled_copy_O = typename Kernel_traits::GmemTiledCopyO{};
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sQ);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    constexpr int32_t copy_size = kGqaGroupSize * 16;
    __syncthreads();

    if (tidx < copy_size) {
        cute::copy(gmem_tiled_copy_O, tOsO(_, 0, _), tOgO(_, 0, _));
    }

    if constexpr (kMiLen > 1) {
        if (tidx < copy_size - 128) {
            cute::copy(gmem_tiled_copy_O, tOsO(_, 1, _), tOgO(_, 1, _));
        }
    }

    if (tidx < kGqaGroupSize) {
        params.sums[max_sum_offset] = scores_warp[0][tidx];
    }
}


template<typename Kernel_traits, typename ParamType>
inline __device__ float calculate_logit_scale(const int partition_num, const int pack_max_partition_num, ParamType &params, char * shared_mem, const int seq_len, const int *qk_gate_topk_idx_ptr) {
    constexpr int32_t kNFloatPacksize = 16 / sizeof(float);
    constexpr int32_t kNReduceThreads = Kernel_traits::kNReduceThreads;
    const int32_t bi = blockIdx.z;
    const int32_t tidx = threadIdx.x;
    const int32_t head_idx = blockIdx.y;
    const int32_t head_num = params.head_num;

    using float_vec = Vec<float, kNFloatPacksize>;
    const int32_t offset = bi * head_num * pack_max_partition_num + head_idx * pack_max_partition_num;

    float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
    const float* max_logits_ptr = params.maxs + offset;
    float global_max_logit = -FLT_MAX;

    int32_t idx = tidx * kNFloatPacksize;
    #pragma unroll
    for (; idx <= partition_num - kNFloatPacksize; idx += kNReduceThreads * kNFloatPacksize) {
        float_vec cur_max = *reinterpret_cast<const float_vec*>(max_logits_ptr + idx);
        #pragma unroll
        for (int32_t j = 0; j < kNFloatPacksize; ++j) {
            if (seq_len >= params.use_moba_seq_limit) {
                if (qk_gate_topk_idx_ptr[idx + j] != 0) {
                    global_max_logit = fmaxf(global_max_logit, cur_max.data.elt[j]);
                }
            } else {
                global_max_logit = fmaxf(global_max_logit, cur_max.data.elt[j]);
            }
        }
        cur_max.store_to(shared_max_logits + idx);
    }

    const int32_t packed_data_num = partition_num / kNFloatPacksize * kNFloatPacksize;

    idx = packed_data_num + tidx;
    #pragma unroll
    for (; idx < partition_num; idx += kNReduceThreads) {
        if (seq_len >= params.use_moba_seq_limit) {
            if (qk_gate_topk_idx_ptr[idx] != 0) {
                float cur_max = max_logits_ptr[idx];
                global_max_logit = fmaxf(global_max_logit, cur_max);
                shared_max_logits[idx] = cur_max;
            }
        } else {
            float cur_max = max_logits_ptr[idx];
            global_max_logit = fmaxf(global_max_logit, cur_max);
            shared_max_logits[idx] = cur_max;
        }
    }
    __syncthreads();

    global_max_logit = BlockAllReduce<float, MaxOp<float>, kNReduceThreads>(global_max_logit);

    float* share_sum_scale = reinterpret_cast<float*>(shared_mem + sizeof(float) * pack_max_partition_num);
    const float* exp_sums_ptr = params.sums + offset;
    float global_exp_sum = 0.0f;

    idx = tidx * kNFloatPacksize;
    #pragma unroll
    for (; idx <= partition_num - kNFloatPacksize; idx += kNReduceThreads * kNFloatPacksize) {
        float_vec share_max = *reinterpret_cast<const float_vec*>(shared_max_logits + idx);
        #pragma unroll
        for (int32_t j = 0; j < kNFloatPacksize; ++j) {
            if (seq_len >= params.use_moba_seq_limit) {
                if (qk_gate_topk_idx_ptr[idx + j] != 0) {
                    float exp_sub_max = expf(share_max.data.elt[j] - global_max_logit);
                    float rescaled_exp_sum = exp_sums_ptr[idx + j] * exp_sub_max;
                    global_exp_sum += rescaled_exp_sum;
                    share_max.data.elt[j] = exp_sub_max;
                }
            } else {
                float exp_sub_max = expf(share_max.data.elt[j] - global_max_logit);
                float rescaled_exp_sum = exp_sums_ptr[idx + j] * exp_sub_max;
                global_exp_sum += rescaled_exp_sum;
                share_max.data.elt[j] = exp_sub_max;
            }
        }
        share_max.store_to(share_sum_scale + idx);
    }

    idx = packed_data_num + tidx;
    #pragma unroll
    for (; idx < partition_num; idx += kNReduceThreads) {
        if (seq_len >= params.use_moba_seq_limit) {
            if (qk_gate_topk_idx_ptr[idx] != 0) {
                float share_max = shared_max_logits[idx];
                float exp_sub_max = expf(share_max - global_max_logit);
                float rescaled_exp_sum = exp_sums_ptr[idx] * exp_sub_max;
                global_exp_sum += rescaled_exp_sum;
                share_sum_scale[idx] = exp_sub_max;
            }
        } else {
            float share_max = shared_max_logits[idx];
            float exp_sub_max = expf(share_max - global_max_logit);
            float rescaled_exp_sum = exp_sums_ptr[idx] * exp_sub_max;
            global_exp_sum += rescaled_exp_sum;
            share_sum_scale[idx] = exp_sub_max;
        }
    }
    __syncthreads();

    global_exp_sum = BlockAllReduce<float, SumOp<float>, kNReduceThreads>(global_exp_sum);

    const float inv_global_exp_sum = fdividef(1.0f, global_exp_sum + 1e-6f);
    return inv_global_exp_sum;
}

template<typename Kernel_traits, typename ParamType>
__global__ void __launch_bounds__(Kernel_traits::kNReduceThreads) moba_decoder_attention_merge_kernel(ParamType params) {
    using cuteType = typename Kernel_traits::cuteType;
    constexpr int32_t kBlockN = Kernel_traits::kTileN * Kernel_traits::kBlockSize;
    constexpr int32_t kNReducePacksize = 16 / sizeof(cuteType);
    constexpr int32_t kNFloatPacksize = 16 / sizeof(float);
    constexpr int32_t kNReduceWarps = Kernel_traits::kNReduceWarps;
    constexpr int32_t kHeadDim = Kernel_traits::kHeadDim;
    const int32_t bi = blockIdx.z;
    const int32_t headdim_idx = kNReducePacksize * kNReduceWarps * blockIdx.x;
    const int32_t tidx = threadIdx.x;
    const int32_t head_idx = blockIdx.y;
    const int32_t warp_id = tidx / 32;
    const int32_t lane_id = tidx % 32;
    const int32_t seq_len = params.seq_lens_decoder[bi] + 1;
    const int32_t head_num = params.head_num;
    using pack_half = typename PackedHalf<cuteType>::Type;


    if (params.seq_lens_decoder[bi] == 0) {
        return;
    }

    extern __shared__ char shared_mem[];

    const int32_t partition_num = (seq_len + kBlockN - 1) / kBlockN;
    const int32_t pack_max_partition_num = (params.max_num_partitions + kNFloatPacksize - 1) / kNFloatPacksize * kNFloatPacksize;

    float* share_sum_scale = reinterpret_cast<float*>(shared_mem + sizeof(float) * pack_max_partition_num);

    constexpr int32_t kGqaGroupSize = Kernel_traits::kGqaGroupSize;
    const int kv_head_idx = head_idx / Kernel_traits::kGqaGroupSize;
    const int * qk_gate_topk_idx_ptr = params.qk_gate_topk_idx_ptr + (bi * params.kv_head_num + kv_head_idx) * Kernel_traits::kMaxN;

    float inv_global_exp_sum = calculate_logit_scale<Kernel_traits>(partition_num, pack_max_partition_num, params, shared_mem, seq_len, qk_gate_topk_idx_ptr);


    using T_vec = Vec<cuteType, kNReducePacksize>;

    cuteType* partition_attn_out = reinterpret_cast<cuteType*>(params.partition_attn_out) + bi * head_num * params.max_num_partitions * kHeadDim + head_idx * kHeadDim + headdim_idx;

    Vec<float, kNReducePacksize> acc;
    acc.set_zero();
    #pragma unroll
    for (int idx = lane_id; idx < partition_num; idx += 32) {
        if (seq_len >= params.use_moba_seq_limit && qk_gate_topk_idx_ptr[idx] == 0) {
            continue;
        }
        T_vec sub_logits = *reinterpret_cast<T_vec*>(&partition_attn_out[idx * head_num * kHeadDim + warp_id * kNReducePacksize]);
        float scale = share_sum_scale[idx];
        #pragma unroll
        for (int k = 0; k < kNReducePacksize; ++k) {
            acc.data.elt[k] += static_cast<float>(sub_logits.data.elt[k]) * scale;
        }
    }

    __syncthreads();

    T_vec out;
    #pragma unroll
    for (int k = 0; k < kNReducePacksize; ++k) {
        out.data.elt[k] = static_cast<cuteType>(WarpAllReduce<float, SumOp<float>>(acc.data.elt[k]) * inv_global_exp_sum);
    }

    const int ori_token_idx = params.cu_seq_q[bi];
    cuteType * attn_out = reinterpret_cast<cuteType *>(params.attn_out) + ori_token_idx * head_num * kHeadDim + head_idx * kHeadDim + headdim_idx + warp_id * kNReducePacksize;

    if (lane_id == 0) {
        out.store_to(attn_out);
    }
}


template<typename Kernel_traits, typename ParamType>
void run_moba_decoder_attn(ParamType &params, cudaStream_t stream) {
    dim3 grid;
    grid.x = params.max_num_partitions;
    grid.y = params.batch_size;
    grid.z = params.kv_head_num;
    constexpr int smem_size = Kernel_traits::kShareMemSize;
    constexpr auto kernel = &moba_decoder_attention_kernel<Kernel_traits, ParamType>;
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);

    int32_t reduce_shared_mem_size = 2 * (params.max_num_partitions + 4) * sizeof(float);
    constexpr int32_t pack_size = 16 / sizeof(typename Kernel_traits::cuteType);
    static_assert(Kernel_traits::kHeadDim % pack_size == 0);
    static_assert((Kernel_traits::kHeadDim / Kernel_traits::kNReduceWarps) % pack_size == 0);
    grid.x = Kernel_traits::kHeadDim / Kernel_traits::kNReduceWarps / pack_size;
    grid.y = params.head_num;
    grid.z = params.batch_size;
    auto reduce_kernel = &moba_decoder_attention_merge_kernel<Kernel_traits, ParamType>;

    if (reduce_shared_mem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            reduce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, reduce_shared_mem_size);
    }
    reduce_kernel<<<grid, Kernel_traits::kNReduceThreads, reduce_shared_mem_size, stream>>>(params);
}


template<typename cute_type, int kCacheBits, int kBlockN, int kMaxN, typename ParamType>
void run_moba_decoder_attn_hdim128(ParamType &params, cudaStream_t stream) {
    const int gqaGroupSize = params.head_num / params.kv_head_num;
    using CacheKVTraits = CacheKV_quant_traits<cute_type, kCacheBits>;
    constexpr int kTileN = kBlockN / CacheKVTraits::kBlockSize;
    switch (gqaGroupSize) {
        case 12: {
            run_moba_decoder_attn<moba_decoder_attn_kernel_traits<12, kTileN, kMaxN,CacheKVTraits>>(params, stream);
            break;
        }
        case 8: {
            run_moba_decoder_attn<moba_decoder_attn_kernel_traits<8, kTileN, kMaxN,CacheKVTraits>>(params, stream);
            break;
        }
        case 7: {
            run_moba_decoder_attn<moba_decoder_attn_kernel_traits<7, kTileN, kMaxN,CacheKVTraits>>(params, stream);
            break;
        }
        case 6: {
            run_moba_decoder_attn<moba_decoder_attn_kernel_traits<6, kTileN, kMaxN,CacheKVTraits>>(params, stream);
            break;
        }
        case 5: {
            run_moba_decoder_attn<moba_decoder_attn_kernel_traits<5, kTileN, kMaxN,CacheKVTraits>>(params, stream);
            break;
        }
        case 4: {
            run_moba_decoder_attn<moba_decoder_attn_kernel_traits<4, kTileN, kMaxN,CacheKVTraits>>(params, stream);
            break;
        }
        default: {
            PADDLE_THROW(phi::errors::Unimplemented(
            "DecoderBlockAttention not implemented for gqaGroupSize = %d", gqaGroupSize));
        }
    }
}


template <typename T>
void DispatchMobaDecoderAttn(
        const paddle::Tensor& q_input,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cache_k,
        const paddle::Tensor& cache_v,
        const paddle::Tensor& block_tables,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& out,
        const paddle::Tensor& qk_gate_topk_idx,
        const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_q,
        const int max_seq_k,
        const int batch_size,
        const int max_input_length,
        const int use_moba_seq_limit,
        const std::string &cache_quant_type_str) {

    using cute_type = typename cuteType<T>::type;
    const int kMobaBlockSize = 128;
    const int kMaxN = 1024;

    constexpr int max_seq_per_block = kMobaBlockSize;
    moba_decoder_attn_params<cute_type> params;
    memset(&params, 0, sizeof(params));
    const uint32_t max_num_partitions = (max_seq_k + max_seq_per_block) / max_seq_per_block;
    assert(head_dim == 128);

    paddle::Tensor maxs = paddle::empty({batch_size, head_num, (max_num_partitions + 3) / 4 * 4}, paddle::DataType::FLOAT32, q_input.place());
    paddle::Tensor sums = paddle::empty({batch_size, head_num, (max_num_partitions + 3) / 4 * 4}, paddle::DataType::FLOAT32, q_input.place());
    paddle::Tensor partition_attn_out = paddle::empty({batch_size, max_num_partitions, head_num, head_dim}, q_input.dtype(), q_input.place());

    params.q_input = reinterpret_cast<cute_type *>(const_cast<T*>(q_input.data<T>()));
    params.attn_out = reinterpret_cast<cute_type *>(const_cast<T*>(out.data<T>()));
    params.seq_lens_encoder = const_cast<int*>(seq_len_encoder.data<int>());
    params.seq_lens_decoder = const_cast<int*>(seq_len_decoder.data<int>());
    params.block_table = const_cast<int*>(block_tables.data<int>());
    params.max_input_length = max_input_length;
    params.head_num = head_num;
    params.kv_head_num = kv_head_num;
    params.max_num_blocks_per_seq = block_tables.dims()[1];
    params.batch_size = batch_size;
    params.inv_sqrt_dh = 1.0f / std::sqrt(head_dim);
    params.max_num_partitions = max_num_partitions;
    params.maxs = reinterpret_cast<float*>(maxs.data<float>());
    params.sums = reinterpret_cast<float*>(sums.data<float>());
    params.partition_attn_out = reinterpret_cast<cute_type *>(partition_attn_out.data<T>());
    params.qk_gate_topk_idx_ptr = const_cast<int*>(qk_gate_topk_idx.data<int>());
    params.use_moba_seq_limit = use_moba_seq_limit;
    params.cu_seq_q = const_cast<int*>(cu_seq_q.data<int>());


    if (cache_quant_type_str == "none") {
        params.cache_k = reinterpret_cast<cute_type *>(const_cast<T*>(cache_k.data<T>()));
        params.cache_v = reinterpret_cast<cute_type *>(const_cast<T*>(cache_v.data<T>()));
        run_moba_decoder_attn_hdim128<cute_type, 16, max_seq_per_block, kMaxN>(params, q_input.stream());
    } else {
        params.cache_k = const_cast<uint8_t*>(cache_k.data<uint8_t>());
        params.cache_v = const_cast<uint8_t*>(cache_v.data<uint8_t>());
        params.cache_k_quant_scale = reinterpret_cast<cute_type *>(const_cast<T*>(cache_k_quant_scale.get().data<T>()));
        params.cache_v_quant_scale = reinterpret_cast<cute_type *>(const_cast<T*>(cache_v_quant_scale.get().data<T>()));
        params.cache_k_dequant_scale = reinterpret_cast<cute_type *>(const_cast<T*>(cache_k_dequant_scale.get().data<T>()));
        params.cache_v_dequant_scale = reinterpret_cast<cute_type *>(const_cast<T*>(cache_v_dequant_scale.get().data<T>()));
        params.cache_k_zp = reinterpret_cast<cute_type *>(const_cast<T*>(cache_k_zero_points.get().data<T>()));
        params.cache_v_zp = reinterpret_cast<cute_type *>(const_cast<T*>(cache_v_zero_points.get().data<T>()));
        if (cache_quant_type_str == "cache_int8_zp") {
            run_moba_decoder_attn_hdim128<cute_type, 8, max_seq_per_block, kMaxN>(params, q_input.stream());
        } else if (cache_quant_type_str == "cache_int4_zp") {
            run_moba_decoder_attn_hdim128<cute_type, 4, max_seq_per_block, kMaxN>(params, q_input.stream());
        } else {
            PADDLE_THROW(phi::errors::Unimplemented(
            "GQA Attention not implemented for cache_quant_type_str = %s", cache_quant_type_str.c_str()));
        }
    }
}

void MobaDecoderAttn(
        const paddle::Tensor& q_input,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cache_k,
        const paddle::Tensor& cache_v,
        const paddle::Tensor& block_tables,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& out,
        const paddle::Tensor& qk_gate_topk_idx,
        const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length,
        const int use_moba_seq_limit,
        const int max_seq_q,
        const int max_seq_k,
        const std::string &cache_quant_type_str) {

    const int batch_size = block_tables.dims()[0];
    if (q_input.dtype() == paddle::DataType::FLOAT16) {
        return DispatchMobaDecoderAttn<phi::dtype::float16>(
            q_input,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cache_k,
            cache_v,
            block_tables,
            k_block_means,
            out,
            qk_gate_topk_idx,
            cache_k_quant_scale,
            cache_v_quant_scale,
            cache_k_dequant_scale,
            cache_v_dequant_scale,
            cache_k_zero_points,
            cache_v_zero_points,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_q,
            max_seq_k,
            batch_size,
            max_input_length,
            use_moba_seq_limit,
            cache_quant_type_str);
    } else if (q_input.dtype() == paddle::DataType::BFLOAT16) {
        return DispatchMobaDecoderAttn<phi::dtype::bfloat16>(
            q_input,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cache_k,
            cache_v,
            block_tables,
            k_block_means,
            out,
            qk_gate_topk_idx,
            cache_k_quant_scale,
            cache_v_quant_scale,
            cache_k_dequant_scale,
            cache_v_dequant_scale,
            cache_k_zero_points,
            cache_v_zero_points,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_q,
            max_seq_k,
            batch_size,
            max_input_length,
            use_moba_seq_limit,
            cache_quant_type_str);
    }
}
