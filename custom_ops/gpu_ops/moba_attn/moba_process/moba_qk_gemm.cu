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
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/reg_reconfig.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename input_type, int kBlockM, int kBlockN, int kMobaBlockSize, int kMaxN, int kHeadDim, bool is_split_kv>
__global__ void qk_gemm_kernel(
        const input_type *q_input,
        const input_type *k_gate_mean,
        input_type *qk_gate_weight,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int *cu_seq_q,
        const int *cu_seq_k,
        const int use_moba_seq_limit,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int kGQA_groupsize) {

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    using SmemLayoutAtomQ = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K, input_type,
            decltype(cute::get<0>(TileShape_MNK{})),
            decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K, input_type, decltype(cute::get<1>(TileShape_MNK{})),
            decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomQK = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, input_type,
        decltype(cute::get<0>(TileShape_MNK{})),
        decltype(cute::get<1>(TileShape_MNK{}))>());

    using SmemLayoutQK = decltype(tile_to_shape(SmemLayoutAtomQK{}, select<0, 1>(TileShape_MNK{})));


    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<input_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;

    using ValLayoutMNK = std::conditional_t<
        is_split_kv,
        Layout<Shape<_1,_4,_1>>,
        Layout<Shape<_4,_1,_1>>
    >;

    using PermutationMNK = std::conditional_t<
        is_split_kv,
        Tile<_16,_64,_16>,
        Tile<_64,_16,_16>
    >;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        ValLayoutMNK,
        PermutationMNK>;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, input_type>;
    using SmemCopyAtomQK = Copy_Atom<cute::SM90_U32x4_STSM_N, input_type>;

    constexpr int kNThreads = 128;
    constexpr int kThreadPerValue = 16 / sizeof(input_type);
    constexpr int kThreadsPerRow = kHeadDim / kThreadPerValue;
    constexpr int kThreadsPerRowQK = kBlockN / kThreadPerValue;

    using GmemLayoutAtom = Layout<
        Shape <Int<kNThreads / kThreadsPerRow>, Int<kThreadsPerRow>>,
        Stride<Int<kThreadsPerRow>, _1>>;

    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<
            SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, input_type>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kThreadPerValue>>>{}));

    using GmemLayoutAtomQK = Layout<
        Shape <Int<kNThreads / kThreadsPerRowQK>, Int<kThreadsPerRowQK>>,
        Stride<Int<kThreadsPerRowQK>, _1>>;

    using GmemTiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<
            UniversalCopy<cutlass::uint128_t>, input_type>{},
            GmemLayoutAtomQK{},
            Layout<Shape<_1, Int<kThreadPerValue>>>{}));

    int mn_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const int bidh_k = bidh / kGQA_groupsize;
    const int tidx = threadIdx.x;

    extern __shared__ char smem_[];

    const int seq_len_q = seq_len_encoder[bidb];
    const int seq_len_k = seq_len_decoder[bidb];
    const int seq_len_qk = seq_len_q + seq_len_k;

    int q_head_stride;
    const int k_head_stride = kv_head_num * kHeadDim;
    int qk_head_stride;
    int offset_q;
    int offset_k;
    int offset_qk;
    int remain_q_seq;

    if constexpr (is_split_kv) {
        if (seq_len_k < use_moba_seq_limit || seq_len_k == 0) {
            return;
        }
        mn_block *= kBlockN;
        q_head_stride = kHeadDim;
        qk_head_stride = kMaxN;
        if (mn_block >= (seq_len_k + kMobaBlockSize - 1) / kMobaBlockSize) {
            return;
        }
        offset_q = cu_seq_q[bidb] * head_num * kHeadDim + bidh * kGQA_groupsize * kHeadDim;
        offset_k = (bidb * kMaxN + mn_block) * k_head_stride + bidh * kHeadDim;
        offset_qk = bidb * head_num * kMaxN + bidh * kGQA_groupsize * kMaxN + mn_block;
        remain_q_seq = kGQA_groupsize;
    } else {
        if (seq_len_q == 0 || seq_len_qk < use_moba_seq_limit) {
            return;
        }
        q_head_stride = head_num * kHeadDim;
        qk_head_stride = head_num * kMaxN;
        mn_block *= kBlockM;
        if (mn_block >= seq_len_q) {
            return;
        }
        offset_q = (cu_seq_q[bidb] + mn_block) * q_head_stride + bidh * kHeadDim;
        offset_k = bidb * kMaxN * k_head_stride + bidh_k * kHeadDim;
        offset_qk = (cu_seq_q[bidb] + mn_block) * qk_head_stride + bidh * kMaxN;
        remain_q_seq = seq_len_q - mn_block;
    }

    Tensor gQ = make_tensor(make_gmem_ptr(q_input + offset_q),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(q_head_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(k_gate_mean + offset_k),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(k_head_stride, _1{}));
    Tensor gQK = make_tensor(make_gmem_ptr(qk_gate_weight + offset_qk),
        Shape<Int<kBlockM>, Int<kBlockN>>{},
        make_stride(qk_head_stride, _1{}));

    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<input_type *>(smem_)), SmemLayoutK{});
    Tensor sQ = make_tensor(sK.data() + size(sK), SmemLayoutQ{});
    Tensor sQK = make_tensor(sK.data() + size(sK), SmemLayoutQK{});

    auto gmem_tiled_copy = GmemTiledCopy{};
    auto gmem_tiled_copy_qk = GmemTiledCopyQK{};
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);
    auto gmem_thr_copy_qk = gmem_tiled_copy_qk.get_thread_slice(tidx);


    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);

    Tensor tKgK = gmem_thr_copy.partition_S(gK);
    Tensor tKsK = gmem_thr_copy.partition_D(sK);

    Tensor tQKgQK = gmem_thr_copy_qk.partition_S(gQK);
    Tensor tQKsQK = gmem_thr_copy_qk.partition_D(sQK);


    Tensor cQ = make_identity_tensor(make_shape(kBlockM, kHeadDim));
    Tensor tQcQ = gmem_thr_copy.partition_S(cQ);

    Tensor cK = make_identity_tensor(make_shape(kBlockN, kHeadDim));
    Tensor tKcK = gmem_thr_copy.partition_S(cK);

    Tensor cQK = make_identity_tensor(make_shape(kBlockM, kBlockN));
    Tensor tQKcQK = gmem_thr_copy.partition_S(cQK);

    if (remain_q_seq >= kBlockM) {
        copy(gmem_tiled_copy, tQgQ, tQsQ, tQcQ);
    } else {
        copy<false>(gmem_tiled_copy, tQgQ, tQsQ, tQcQ, remain_q_seq);
    }
    copy(gmem_tiled_copy, tKgK, tKsK, tKcK);

    cute::cp_async_fence();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma).get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_QK = make_tiled_copy_C(SmemCopyAtomQK{}, tiled_mma);
    auto smem_thr_copy_QK = smem_tiled_copy_QK.get_thread_slice(tidx);
    Tensor tsQK = smem_thr_copy_QK.partition_D(sQK);

    const int n_blocks = is_split_kv ? 1 : cute::ceil_div(cute::ceil_div(seq_len_qk, kMobaBlockSize), kBlockN);

    #pragma unroll
    for (int n_block = 0; n_block < n_blocks; ++n_block) {
        clear(acc_s);
        cp_async_wait<0>();
        __syncthreads();
        if (n_block == 0) {
            gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_thr_copy_Q, smem_thr_copy_K, smem_tiled_copy_Q, smem_tiled_copy_K);
        } else {
            gemm<true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_thr_copy_Q, smem_thr_copy_K, smem_tiled_copy_Q, smem_tiled_copy_K);
        }
        if constexpr (!is_split_kv) {
            if (n_block < n_blocks - 1) {
                __syncthreads();
                tKgK.data() = tKgK.data() + kBlockN * k_head_stride;
                copy(gmem_tiled_copy, tKgK, tKsK, tKcK);
                cute::cp_async_fence();
            }
        }

        Tensor rS = convert_type<input_type>(acc_s);
        Tensor trQK = smem_thr_copy_QK.retile_S(rS);
        cute::copy(smem_tiled_copy_QK, trQK, tsQK);

        __syncthreads();
        if (remain_q_seq >= kBlockM) {
            copy(gmem_tiled_copy_qk, tQKsQK, tQKgQK, tQKcQK);
        } else {
            copy<false>(gmem_tiled_copy_qk, tQKsQK, tQKgQK, tQKcQK, remain_q_seq);
        }
        if constexpr (!is_split_kv) {
            __syncthreads();
            tQKgQK.data() = tQKgQK.data() + kBlockN;
        }
    }
}

template <typename input_type, int kBlockM, int kBlockN, int kMobaBlockSize, int kMaxN, bool is_split_kv>
void qk_gemm(
        const input_type *q_input,
        const input_type *k_gate_mean,
        input_type *qk_gate_weight,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int *cu_seq_q,
        const int *cu_seq_k,
        const int use_moba_seq_limit,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int bsz,
        cudaStream_t stream) {

    const int gqa_group_size = head_num / kv_head_num;

    dim3 grid_dims;
    const int num_m_block = (max_seq_q + kBlockM - 1) / kBlockM;
    const int num_n_block = ((max_seq_k + kMobaBlockSize - 1) / kMobaBlockSize + kBlockN - 1) / kBlockN;

    if (is_split_kv) {
        grid_dims.x = num_n_block;
        grid_dims.z = kv_head_num;
    } else {
        grid_dims.x = num_m_block;
        grid_dims.z = head_num;
    }
    grid_dims.y = bsz;

    constexpr int kHeadDim = 128;
    constexpr int smemq = kBlockM * kHeadDim * sizeof(input_type);
    constexpr int smemk = kBlockN * kHeadDim * sizeof(input_type);
    constexpr int smemqk = kBlockM * kBlockN * sizeof(input_type);
    const int smem_size = smemk + max(smemq, smemqk);

    auto kernel = &qk_gemm_kernel<input_type, kBlockM, kBlockN, kMobaBlockSize, kMaxN, kHeadDim, is_split_kv>;

    if (smem_size >= 48 * 1024) {
       cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid_dims, 128, smem_size, stream>>>(
        q_input,
        k_gate_mean,
        qk_gate_weight,
        seq_len_encoder,
        seq_len_decoder,
        cu_seq_q,
        cu_seq_k,
        use_moba_seq_limit,
        max_seq_q,
        max_seq_k,
        head_num,
        kv_head_num,
        gqa_group_size);
}


template <typename T>
std::vector<paddle::Tensor> DispatchMobaQKGemm(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const bool is_split_kv,
        const int use_moba_seq_limit) {

    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;
    const int batch_size = seq_len_encoder.dims()[0];
    using cute_type = typename cuteType<T>::type;
    if (is_split_kv) {
        paddle::Tensor qk_gate_weight = paddle::empty({batch_size, head_num, kMaxN}, q_input.dtype(), q_input.place());
        qk_gemm<cute_type, 16, kMobaBlockSize, kMobaBlockSize, kMaxN, true>(
            reinterpret_cast<const cute_type*>(q_input.data<T>()),
            reinterpret_cast<const cute_type*>(k_block_means.data<T>()),
            reinterpret_cast<cute_type*>(qk_gate_weight.data<T>()),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            use_moba_seq_limit,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            batch_size,
            q_input.stream()
        );
        return {qk_gate_weight};
    } else {
        constexpr int kBlockM = 128;
        constexpr int kBlockN = 128;
        const int token_num = q_input.dims()[0];
        paddle::Tensor qk_gate_weight = paddle::empty({token_num, head_num, kMaxN}, q_input.dtype(), q_input.place());
        qk_gemm<cute_type, kBlockM, kBlockN, kMobaBlockSize, kMaxN, false>(
            reinterpret_cast<cute_type *>(const_cast<T*>(q_input.data<T>())),
            reinterpret_cast<cute_type *>(const_cast<T*>(k_block_means.data<T>())),
            reinterpret_cast<cute_type *>(qk_gate_weight.data<T>()),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            use_moba_seq_limit,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            batch_size,
            q_input.stream());
        return {qk_gate_weight};
    }
}

std::vector<paddle::Tensor> MobaQKGemm(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const bool is_split_kv,
        const int use_moba_seq_limit) {

    if (q_input.dtype() == paddle::DataType::FLOAT16) {
        return std::move(
            DispatchMobaQKGemm<phi::dtype::float16>(
                q_input,
                k_block_means,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                is_split_kv,
                use_moba_seq_limit
            )
        );
    } else if (q_input.dtype() == paddle::DataType::BFLOAT16) {
        return std::move(
            DispatchMobaQKGemm<phi::dtype::bfloat16>(
                q_input,
                k_block_means,
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_q,
                cu_seq_k,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                is_split_kv,
                use_moba_seq_limit
            )
        );
    }
}

PD_BUILD_STATIC_OP(moba_qk_gemm)
    .Inputs({
        "q_input",
        "k_block_means",
        "seq_len_encoder",
        "seq_len_decoder",
        "cu_seq_q",
        "cu_seq_k"})
    .Attrs({
        "max_seq_q: int",
        "max_seq_k: int",
        "head_num: int",
        "kv_head_num: int",
        "is_split_kv: bool",
        "use_moba_seq_limit: int"})
    .Outputs({"qk_gate_weight"})
    .SetKernelFn(PD_KERNEL(MobaQKGemm));
