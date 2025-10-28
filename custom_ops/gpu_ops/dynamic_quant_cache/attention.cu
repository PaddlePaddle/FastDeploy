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

#include "attention.h"

namespace dynamic_quant_cache {
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

template <typename Kernel_traits, int cache_bits>
void __global__ multi_block_gqa_attention_kernel(Block_attn_params params) {
    using input_type = typename Kernel_traits::input_type;
    using output_type = typename Kernel_traits::output_type;
    using scale_type = typename Kernel_traits::scale_type;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using GmemTiledCopy = typename Kernel_traits::GmemTiledCopy;
    using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyO;
    using Gmem_copy_struct = typename Kernel_traits::Gmem_copy_struct;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVtransposed = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtransposedNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;
    using TiledMma = typename Kernel_traits::TiledMma;
    using SmemCopyAtom = typename Kernel_traits::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Kernel_traits::SmemCopyAtomTransposed;
    using pakc_half = typename PackedHalf<input_type>::Type;
    using SmemLayoutQK = typename Kernel_traits::SmemLayoutQK;
    using SmemCopyAtomO = typename Kernel_traits::SmemCopyAtomO;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    constexpr int32_t kGqaGroupSize = Kernel_traits::kGqaGroupSize;
    constexpr int32_t kBlockSize = Kernel_traits::kBlockSize;
    constexpr int32_t kBlockM = Kernel_traits::kBlockM;
    constexpr int32_t kHeadDim = Kernel_traits::kHeadDim;
    constexpr int32_t kTileN = Kernel_traits::kTileN;
    constexpr int32_t kNThreads = Kernel_traits::kNThreads;
    constexpr int32_t kShareMemSize = Kernel_traits::kShareMemSizeC2;
    constexpr int32_t kMiLen = (kGqaGroupSize + 7) / 8;
    constexpr int32_t kNWarps = Kernel_traits::kNWarps;
    constexpr int32_t kBlockN = kTileN * kBlockSize;

    const int32_t partition_idx = blockIdx.x;
    const int32_t bidb = blockIdx.y;
    const int32_t kv_head_idx = blockIdx.z;
    const int32_t tidx = threadIdx.x;
    const int32_t q_head_idx = kv_head_idx * kGqaGroupSize;


    const int32_t warp_id = tidx / 32;
    const int32_t lane_id = tidx % 32;
    const int32_t row = lane_id / 4;
    const int32_t col = lane_id % 4;

    const int32_t seq_len_decoder = params.seq_lens_decoder[bidb] + 1;
    const int c16_remain_seq_len = params.c16_remain_seq_len;

    const int c16_cache_max_len = c16_remain_seq_len + kBlockSize;
    const int c16_cache_len = seq_len_decoder < c16_cache_max_len ? seq_len_decoder : c16_remain_seq_len + seq_len_decoder % kBlockSize;

    const int c2_cache_len = seq_len_decoder - c16_cache_len;

    int partition_num;

    if constexpr (cache_bits == 2) {
        partition_num = (c2_cache_len + kBlockN - 1) / kBlockN;
        if (seq_len_decoder == 1 || partition_idx >= partition_num) {
            return;
        }
    } else {
        partition_num = (c16_cache_len + kBlockN - 1) / kBlockN;
        if (seq_len_decoder == 1 || partition_idx >= partition_num) {
            return;
        }
    }

    const int32_t head_num = params.head_num;
    const int32_t kv_head_num = params.kv_head_num;
    const int data_num_per_block = cache_bits == 2 ? params.data_num_per_block : kBlockSize * kHeadDim;

    const int32_t q_offset = bidb * head_num * kHeadDim + q_head_idx * kHeadDim;

    Tensor gQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<const input_type *>(params.q_input) + q_offset),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        Stride<Int<kHeadDim>, _1>{});

    const int32_t block_idx = partition_idx * kTileN;
    const int* block_table = params.block_table + bidb * params.max_num_blocks_per_seq + block_idx;
    const int32_t physical_block_number = block_table[0];
    int cache_offset;
    if constexpr (cache_bits == 2) {
        cache_offset = (physical_block_number * kv_head_num + kv_head_idx) * data_num_per_block;
    } else {
        cache_offset = (bidb * c16_cache_max_len + block_idx * kBlockSize) * kv_head_num * kHeadDim + kv_head_idx * kHeadDim;
    }

    uint8_t *gK = params.cache_k_c2 + cache_offset;
    uint8_t *gV = params.cache_v_c2 + cache_offset;

    Tensor gK_tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<const input_type *>(params.cache_k_c16) + cache_offset),
        Shape<Int<kBlockSize>, Int<kHeadDim>>{},
        make_stride(kHeadDim * kv_head_num, _1{}));

    Tensor gV_tensor = make_tensor(
        make_gmem_ptr(reinterpret_cast<const input_type *>(params.cache_v_c16) + cache_offset),
        Shape<Int<kBlockSize>, Int<kHeadDim>>{},
        make_stride(kHeadDim * kv_head_num, _1{}));

    extern __shared__ char smem_[];
    __shared__ ElementAccum scores_warp[kNWarps][kMiLen * kBlockM];

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<input_type *>(smem_)), SmemLayoutQ{});
    Tensor sQK = make_tensor(make_smem_ptr(reinterpret_cast<input_type *>(smem_)), SmemLayoutQK{});
    Tensor sK_tensor = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});
    Tensor sV_tensor = make_tensor(sK_tensor.data() + size(sK_tensor), SmemLayoutKV{});
    Tensor sVt = make_tensor(sV_tensor.data(), SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV_tensor.data(), SmemLayoutVtransposedNoSwizzle{});


    uint8_t *sK = reinterpret_cast<uint8_t *>(smem_) + kShareMemSize;
    uint8_t *sV = sK + data_num_per_block;

    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy.partition_S(gK_tensor);
    Tensor tKsK = gmem_thr_copy.partition_D(sK_tensor);
    Tensor tVgV = gmem_thr_copy.partition_S(gV_tensor);
    Tensor tVsV = gmem_thr_copy.partition_D(sV_tensor);

    Tensor cK = make_identity_tensor(make_shape(size<0>(sK_tensor), size<1>(sK_tensor)));
    Tensor tKcK = gmem_thr_copy.partition_S(cK);

    constexpr int32_t copy_size = kGqaGroupSize * 8;

    // copy q to smem
    if (tidx < copy_size) {
        cute::copy(gmem_tiled_copy, tQgQ, tQsQ);
    }

    // copy k to smem
    if constexpr (cache_bits == 2) {
        copy_kv<Gmem_copy_struct, kNThreads>(tidx, data_num_per_block, gK, sK);
    } else {
        copy(gmem_tiled_copy, tKgK, tKsK, tKcK);
    }

    cute::cp_async_fence();

    const int cache_offset_step = kv_head_num * data_num_per_block;

    const int remain_seq_len = cache_bits == 2 ? c2_cache_len - partition_idx * kBlockN : c16_cache_len - partition_idx * kBlockN;

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockSize>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSsQK = smem_thr_copy_Q.partition_S(sQK);
    Tensor tSrQK = thr_mma.partition_fragment_A(sQK);
    Tensor tSrK = thr_mma.partition_fragment_B(sK_tensor);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK_tensor);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor tCrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    cp_async_wait<0>();
    __syncthreads();
    copy(smem_tiled_copy_Q, tSsQ, tCrQ_copy_view);

    ElementAccum scores_max[kMiLen];
    ElementAccum scores_max_prev[kMiLen];
    ElementAccum scores_sum[kMiLen];


    #pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
        scores_max[mi] = -INFINITY;
        scores_sum[mi] = 0;
    }


    #pragma unroll
    for (int n = 0; n < kTileN; ++n) {
        const int cur_remain_seq_len = remain_seq_len - n * kBlockSize;

        if (cur_remain_seq_len <= 0) {
            break;
        }

        clear(acc_s);
        cp_async_wait<0>();
        __syncthreads();

        if constexpr (cache_bits == 2) {
            if (n > 0) {
                gV = gV + (block_table[n] - block_table[n - 1]) * cache_offset_step;
            }
            copy_kv<Gmem_copy_struct, kNThreads>(tidx, data_num_per_block, gV, sV);
        } else {
            if (n > 0) {
                tVgV.data() = tVgV.data() + cache_offset_step;
            }
            if (cur_remain_seq_len < kBlockSize) {
                copy<false, true>(gmem_tiled_copy, tVgV, tVsV, tKcK, cur_remain_seq_len);
            } else {
                copy(gmem_tiled_copy, tVgV, tVsV, tKcK);
            }
        }

        cute::cp_async_fence();

        if constexpr (cache_bits == 2) {
            gemm_qk<input_type, scale_type>(acc_s, tSrQ, tSrK, sK, tiled_mma, tidx);
        } else {
            gemm<true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_thr_copy_Q, smem_thr_copy_K, smem_tiled_copy_Q, smem_tiled_copy_K);
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

        cute::cp_async_wait<0>();
        __syncthreads();

        if (cur_remain_seq_len > kBlockSize && n < kTileN - 1) {
            if constexpr (cache_bits == 2) {
                gK = gK + (block_table[n + 1] - block_table[n]) * cache_offset_step;
                copy_kv<Gmem_copy_struct, kNThreads>(tidx, data_num_per_block, gK, sK);
            } else {
                tKgK.data() = tKgK.data() + cache_offset_step;
                copy(gmem_tiled_copy, tKgK, tKsK, tKcK);
            }

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

        Tensor rS = convert_type<input_type>(acc_s);

        Tensor trQK = smem_thr_copy_O.retile_S(rS);
        Tensor tsQK = smem_thr_copy_O.partition_D(sQK);
        cute::copy(smem_tiled_copy_O, trQK, tsQK);

        __syncthreads();

        if constexpr (cache_bits == 2) {
            gemm_v<input_type, scale_type>(acc_o, tSrQK, tSsQK, tOrVt, sV, tiled_mma, smem_thr_copy_Q, smem_tiled_copy_Q, tidx);
        } else {
            gemm(acc_o, tSrQK, tOrVt, tSsQK, tOsVt, tiled_mma, smem_thr_copy_Q, smem_thr_copy_V, smem_tiled_copy_Q, smem_tiled_copy_V);
        }
    }

    int store_partition_idx;

    if constexpr (cache_bits == 2) {
        store_partition_idx = partition_idx;
    } else {
        const int c2_partition_num = (c2_cache_len + kBlockN - 1) / kBlockN;
        store_partition_idx = partition_idx + c2_partition_num;
    }

    const uint32_t max_partition_num = params.max_num_partitions;
    uint32_t max_sum_offset = bidb * max_partition_num * head_num + (tidx + q_head_idx) * max_partition_num + store_partition_idx;

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


    Tensor rO = convert_type<output_type>(acc_o);
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
        make_gmem_ptr(reinterpret_cast<output_type *>(params.partition_attn_out) +
        ((bidb * max_partition_num + store_partition_idx) * head_num  + q_head_idx) * kHeadDim),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        Stride<Int<kHeadDim>, _1>{});

    GmemTiledCopyO gmem_tiled_copy_o;
    auto gmem_thr_copy_o = gmem_tiled_copy_o.get_thread_slice(tidx);

    Tensor tOgO = gmem_thr_copy_o.partition_S(gO);
    Tensor tOsO = gmem_thr_copy_o.partition_D(sQ);
    __syncthreads();

    if (tidx < copy_size) {
        cute::copy(gmem_tiled_copy_o, tOsO, tOgO);
    }

    if (tidx < kGqaGroupSize) {
        params.sums[max_sum_offset] = scores_warp[0][tidx];
    }
}

template<typename Kernel_traits>
__global__ void __launch_bounds__(Kernel_traits::kNReduceThreads) multi_block_attention_reduce_kernel(Block_attn_params params) {
    using output_type = typename Kernel_traits::output_type;
    constexpr int kBlockSize = Kernel_traits::kBlockSize;
    constexpr int32_t kBlockN = Kernel_traits::kTileN * kBlockSize;
    constexpr int32_t kHeadDim = Kernel_traits::kHeadDim;
    constexpr int32_t kNReducePacksize = kHeadDim / 32;
    constexpr int32_t kNReduceWarps = Kernel_traits::kNReduceWarps;
    constexpr int32_t kNReduceThreads = Kernel_traits::kNReduceThreads;
    static_assert(kHeadDim == kNReduceThreads);

    const int32_t head_idx = blockIdx.x;
    const int32_t bidb = blockIdx.y;
    const int32_t tidx = threadIdx.x;

    const int32_t seq_len_decoder = params.seq_lens_decoder[bidb] + 1;
    const int32_t head_num = params.head_num;
    const int32_t warp_id = tidx / 32;
    const int32_t lane_id = tidx % 32;

    if (seq_len_decoder == 1) {
        return;
    }

    extern __shared__ char shared_mem[];

    const int c16_remain_seq_len = params.c16_remain_seq_len;

    const int c16_cache_max_len = c16_remain_seq_len + kBlockSize;
    const int c16_cache_len = seq_len_decoder < c16_cache_max_len ? seq_len_decoder : c16_remain_seq_len + seq_len_decoder % kBlockSize;

    const int c2_cache_len = seq_len_decoder - c16_cache_len;

    const int32_t partition_num = (c2_cache_len + kBlockN - 1) / kBlockN + (c16_cache_len + kBlockN - 1) / kBlockN;
    const int32_t max_partition_num = params.max_num_partitions;

    const int32_t offset = bidb * head_num * max_partition_num + head_idx * max_partition_num;

    float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
    const float* max_logits_ptr = params.maxs + offset;
    float global_max_logit = -100000000000.0f;

    #pragma unroll
    for (int32_t idx = tidx; idx < partition_num; idx += kNReduceThreads) {
        float cur_max = max_logits_ptr[idx];
        global_max_logit = fmaxf(global_max_logit, cur_max);
        shared_max_logits[idx] = cur_max;
    }

    __syncthreads();

    global_max_logit = BlockAllReduce<float, MaxOp<float>, kNReduceThreads>(global_max_logit);

    float* share_sum_scale = reinterpret_cast<float*>(shared_mem + sizeof(float) * max_partition_num);
    const float* exp_sums_ptr = params.sums + offset;
    float global_exp_sum = 0.0f;

    for (int idx = tidx; idx < partition_num; idx += kNReduceThreads) {
        float share_max = shared_max_logits[idx];
        float exp_sub_max = expf(share_max - global_max_logit);
        float rescaled_exp_sum = exp_sums_ptr[idx] * exp_sub_max;
        global_exp_sum += rescaled_exp_sum;
        share_sum_scale[idx] = exp_sub_max;
    }
    __syncthreads();


    global_exp_sum = BlockAllReduce<float, SumOp<float>, kNReduceThreads>(global_exp_sum);
    const float inv_global_exp_sum = fdividef(1.0f, global_exp_sum);

    output_type* partition_attn_out = reinterpret_cast<output_type*>(params.partition_attn_out) + bidb * head_num * max_partition_num * kHeadDim + head_idx * kHeadDim;

    output_type * attn_out = reinterpret_cast<output_type *>(params.attn_out) + params.cu_seq_q[bidb] * head_num * kHeadDim + head_idx * kHeadDim;

    static_assert(kNReduceThreads == kHeadDim);
    float acc[kNReducePacksize] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i = warp_id; i < partition_num; i += kNReduceWarps) {
        int2 sub_logits = *reinterpret_cast<int2*>(&partition_attn_out[i * head_num * kHeadDim + lane_id * kNReducePacksize]);
        float scale = share_sum_scale[i];
        #pragma unroll
        for (int k = 0; k < kNReducePacksize; ++k) {
            acc[k] += static_cast<float>(reinterpret_cast<output_type*>(&sub_logits)[k]) * scale;
        }
    }

    float* acc_warp = reinterpret_cast<float*>(shared_mem + 2 * sizeof(float) * ((max_partition_num + 3) / 4 * 4));

    *reinterpret_cast<int4*>(acc_warp + warp_id * kHeadDim + lane_id * kNReducePacksize) = *reinterpret_cast<int4*>(&acc);

    __syncthreads();

    float warp_sum = acc_warp[tidx];

    #pragma unroll
    for (int i = 1; i < kNReduceWarps; ++i) {
        warp_sum += acc_warp[i * kHeadDim + tidx];
    }

    warp_sum *= inv_global_exp_sum;

    reinterpret_cast<output_type*>(acc_warp)[tidx] = static_cast<output_type>(warp_sum);

    __syncthreads();

    if (warp_id == 0) {
        reinterpret_cast<int2*>(attn_out)[lane_id] = reinterpret_cast<int2*>(acc_warp)[lane_id];
    }
}

template<typename input_type, typename output_type, typename scale_type, int stage, int kGqaGroupSize>
void run_block_attn(Block_attn_params &params, cudaStream_t stream) {
    dim3 grid;
    grid.x = params.max_num_partitions;
    grid.y = params.batch_size;
    grid.z = params.kv_head_num;

    constexpr int kNWarps = 4;
    constexpr int kTileN = stage;
    constexpr int kHeadDim = 128;

    using Kernel_traits = Block_attn_kernel_traits<
        kGqaGroupSize,
        kNWarps,
        kTileN,
        kHeadDim,
        input_type,
        output_type,
        scale_type>;
    const int smem_size_c2 = Kernel_traits::kShareMemSizeC2 + params.data_num_per_block * 2;

    constexpr auto kernel_c2 = &multi_block_gqa_attention_kernel<Kernel_traits, 2>;
    if (smem_size_c2 >= 48 * 1024) {
        cudaFuncSetAttribute(kernel_c2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_c2);
    }
    kernel_c2<<<grid, Kernel_traits::kNThreads, smem_size_c2, stream>>>(params);


    const int smem_size_c16 = Kernel_traits::kShareMemSizeC16;

    grid.x = (params.c16_remain_seq_len + Kernel_traits::kBlockSize + Kernel_traits::kTileN * Kernel_traits::kBlockSize - 1) / (Kernel_traits::kTileN * Kernel_traits::kBlockSize);

    constexpr auto kernel_c16 = &multi_block_gqa_attention_kernel<Kernel_traits, 16>;
    if (smem_size_c16 >= 48 * 1024) {
        cudaFuncSetAttribute(kernel_c16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_c16);
    }
    kernel_c16<<<grid, Kernel_traits::kNThreads, smem_size_c16, stream>>>(params);

    int32_t reduce_shared_mem_size = 2 * (params.max_num_partitions + 4) * sizeof(float) + (Kernel_traits::kNReduceWarps * Kernel_traits::kNReduceThreads) * sizeof(float);
    constexpr int32_t pack_size = 16 / sizeof(typename Kernel_traits::output_type);

    static_assert(kHeadDim % pack_size == 0);
    static_assert((kHeadDim / Kernel_traits::kNReduceWarps) % pack_size == 0);
    grid.x = params.head_num;
    grid.y = params.batch_size;
    grid.z = 1;

    auto reduce_kernel = &multi_block_attention_reduce_kernel<Kernel_traits>;

    reduce_kernel<<<grid, Kernel_traits::kNReduceThreads, reduce_shared_mem_size, stream>>>(params);
}

void DecoderAttention(
        const paddle::Tensor& q_input,
        const paddle::Tensor& cache_k_c2,
        const paddle::Tensor& cache_v_c2,
        const paddle::Tensor& cache_k_c16,
        const paddle::Tensor& cache_v_c16,
        const paddle::Tensor& attn_out,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& encoder_seqs_len,
        const paddle::Tensor& decoder_seqs_len,
        const paddle::Tensor& block_table,
        const int c16_remain_seq_len,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_k,
        const int max_input_length,
        const std::string& cache_quant_type_str) {
    using scale_type = cutlass::float_e4m3_t;
    using input_type = cutlass::half_t;
    using output_type = cutlass::bfloat16_t;

    constexpr int kBlockSize = 64;
    constexpr int stage = 8;
    constexpr int max_seq_per_block = stage * kBlockSize;

    const int data_num_per_block = kBlockSize * head_dim / 4 + kBlockSize / 32 * head_dim * 4;
    const int c16_cache_max_len = c16_remain_seq_len + kBlockSize;
    const int cache_seq_len = max_seq_k + 1;
    const int c16_cache_len = cache_seq_len < c16_cache_max_len ? cache_seq_len : c16_remain_seq_len + cache_seq_len % kBlockSize;
    const int c2_cache_len = cache_seq_len - c16_cache_len;
    const uint32_t max_num_partitions = (c16_cache_len + max_seq_per_block - 1) / max_seq_per_block + (c2_cache_len + max_seq_per_block - 1) / max_seq_per_block;
    const int bsz = decoder_seqs_len.dims()[0];

    paddle::Tensor partition_attn_out = paddle::empty({bsz, head_num, max_num_partitions, head_dim}, paddle::DataType::FLOAT16, q_input.place());
    paddle::Tensor maxs = paddle::empty({bsz, head_num, max_num_partitions}, paddle::DataType::FLOAT32, q_input.place());
    paddle::Tensor sums = paddle::empty({bsz, head_num, max_num_partitions}, paddle::DataType::FLOAT32, q_input.place());

    Block_attn_params params;
    memset(&params, 0, sizeof(params));
    params.q_input = const_cast<phi::dtype::float16*>(q_input.data<phi::dtype::float16>());
    params.cache_k_c2 = const_cast<uint8_t*>(cache_k_c2.data<uint8_t>());
    params.cache_v_c2 = const_cast<uint8_t*>(cache_v_c2.data<uint8_t>());
    params.cache_k_c16 = const_cast<phi::dtype::float16*>(cache_k_c16.data<phi::dtype::float16>());
    params.cache_v_c16 = const_cast<phi::dtype::float16*>(cache_v_c16.data<phi::dtype::float16>());
    params.attn_out = const_cast<phi::dtype::bfloat16*>(attn_out.data<phi::dtype::bfloat16>());
    params.partition_attn_out = partition_attn_out.data<phi::dtype::float16>();
    params.cu_seq_q = const_cast<int*>(cu_seq_q.data<int>());
    params.sums = sums.data<float>();
    params.maxs = maxs.data<float>();
    params.seq_lens_encoder = const_cast<int*>(encoder_seqs_len.data<int>());
    params.seq_lens_decoder = const_cast<int*>(decoder_seqs_len.data<int>());
    params.block_table = const_cast<int*>(block_table.data<int>());
    params.max_input_length = max_input_length;
    params.head_num = head_num;
    params.kv_head_num = kv_head_num;
    params.max_num_blocks_per_seq = block_table.dims()[1];
    params.batch_size = decoder_seqs_len.dims()[0];
    params.max_num_partitions = max_num_partitions;
    params.inv_sqrt_dh = 1.0f / std::sqrt(head_dim);
    params.data_num_per_block = data_num_per_block;
    params.c16_remain_seq_len = c16_remain_seq_len;

    const int gqa_group_size = head_num / kv_head_num;
    if (gqa_group_size == 5) {
        run_block_attn<input_type, output_type, scale_type, stage, 5>(params, q_input.stream());
    } else if (gqa_group_size == 8) {
        run_block_attn<input_type, output_type, scale_type, stage, 8>(params, q_input.stream());
    } else if (gqa_group_size == 14) {
        run_block_attn<input_type, output_type, scale_type, stage, 14>(params, q_input.stream());
    } else {
        PD_THROW("gqa_group_size is not supported :%d\n", gqa_group_size);
    }
}
}

PD_BUILD_OP(dynamic_quant_cache_decoder_attention)
    .Inputs({
        "q_input",
        "cache_k_c2",
        "cache_v_c2",
        "cache_k_c16",
        "cache_v_c16",
        "attn_out",
        "cu_seq_q",
        "encoder_seqs_len",
        "decoder_seqs_len",
        "block_table"})
    .Attrs({
        "c16_remain_seq_len: int",
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_seq_k: int",
        "max_input_length: int",
        "cache_quant_type_str: std::string"})
    .Outputs({"out"})
    .SetInplaceMap({
        {"attn_out", "out"}})
    .SetKernelFn(PD_KERNEL(dynamic_quant_cache::DecoderAttention));
