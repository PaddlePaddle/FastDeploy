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

#include "paddle/extension.h"
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#  include "cutlass/util/cublas_wrappers.hpp"
#endif
#include "moba_attn/moba_attn_utils.hpp"
#include "moba_attn/moba_attn.h"
#include "kernel_traits.h"
#include "mainloop_attn.hpp"
#include "softmax.hpp"
#include "cutlass/arch/reg_reconfig.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <int kHeadDim>
auto get_gmem_layout(int token_num, int head_num) {
    return make_layout(
        make_shape(token_num, kHeadDim, head_num),
        make_stride(head_num * kHeadDim, _1{}, kHeadDim));
}

template <typename Ktraits>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    moba_encoder_attention_kernel(
        CUTE_GRID_CONSTANT typename CollectiveMainloopAttn<Ktraits>::Params const mainloop_params,
        CUTE_GRID_CONSTANT moba_encoder_attn_params const data_params) {

    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    constexpr int kHeadDim = Ktraits::kHeadDim;
    constexpr int kMaxN = Ktraits::kMaxN;

    using CollectiveMainloop = CollectiveMainloopAttn<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;

    const int seq_len_q = data_params.seq_len_encoder[bidb];
    const int seq_len_k = data_params.cu_seq_k[bidb + 1] - data_params.cu_seq_k[bidb];


    if (seq_len_q == 0) {
        return;
    }

    __align__(16) __shared__ int qk_gate_topk_idx[kMaxN];
    const int *qk_gate_idx_cur_offset = data_params.qk_gate_topk_idx + data_params.cu_seq_q_pack[bidb] / kBlockM * data_params.head_num * kMaxN + (m_block * data_params.head_num + bidh) * kMaxN;

    #pragma unroll
    for (int i = threadIdx.x; i < kMaxN / 4; i += Ktraits::kNWarps * cutlass::NumThreadsPerWarp) {
        reinterpret_cast<int4*>(qk_gate_topk_idx)[i] = reinterpret_cast<const int4*>(qk_gate_idx_cur_offset)[i];
    }


    const int n_block_max = min(cute::ceil_div((m_block + 1) * kBlockM + seq_len_k - seq_len_q, kBlockN), cute::ceil_div(seq_len_k, kBlockN));

    if (m_block * kBlockM >= seq_len_q) {
        return;
    }

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1);
    }

    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    __syncthreads();

    CollectiveMainloop collective_mainloop;

    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 8 ? 56 : 24>();

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0) {
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            collective_mainloop.load<Ktraits::UseMoba>(
                mainloop_params,
                pipeline_k,
                pipeline_v,
                smem_pipe_write_k,
                smem_pipe_write_v,
                shared_storage,
                qk_gate_topk_idx,
                n_block_max,
                m_block,
                bidh,
                bidb,
                data_params.cu_seq_q,
                data_params.cu_seq_k,
                seq_len_q,
                seq_len_k);
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 8 ? 256 : 240>();
        typename Ktraits::TiledMma1 tiled_mma1;

        collective_mainloop.mma_init();

        PipelineState smem_pipe_read_k, smem_pipe_read_v;

        Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
        Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax;

        collective_mainloop.mma<Ktraits::UseMoba>(
            mainloop_params,
            pipeline_k,
            pipeline_v,
            smem_pipe_read_k,
            smem_pipe_read_v,
            tOrO,
            softmax,
            qk_gate_topk_idx,
            n_block_max,
            threadIdx.x - NumCopyThreads,
            m_block,
            seq_len_q,
            seq_len_k,
            shared_storage);

        const int o_head_stride = data_params.head_num * kHeadDim;
        const int store_offset = (data_params.cu_seq_q[bidb] + m_block * kBlockM) * o_head_stride + bidh * kHeadDim;

        const int real_seq = seq_len_q - m_block * kBlockM;

        collective_mainloop.store<NumMmaThreads>(
            mainloop_params,
            tOrO,
            shared_storage,
            tiled_mma1,
            threadIdx.x - NumCopyThreads,
            o_head_stride,
            real_seq,
            reinterpret_cast<Element*>(data_params.o_ptr) + store_offset);
    }

}


template<typename Kernel_traits>
void run_moba_decoder_attn(moba_encoder_attn_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    using CollectiveMainloop = CollectiveMainloopAttn<Kernel_traits>;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),
            get_gmem_layout<kHeadDim>(params.max_seq_q * params.batch_size, params.head_num),
            static_cast<Element const*>(params.k_ptr),
            get_gmem_layout<kHeadDim>(params.max_seq_k * params.batch_size, params.kv_head_num),
            static_cast<Element const*>(params.v_ptr),
            get_gmem_layout<kHeadDim>(params.max_seq_k * params.batch_size, params.kv_head_num),
            params.scale_softmax_log2
        });

    int num_blocks_m = cutlass::ceil_div(params.max_seq_q, Kernel_traits::kBlockM);

    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});

    void *kernel;
    kernel = (void *)moba_encoder_attention_kernel<Kernel_traits>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);

    if (smem_size >= 48 * 1024) {
       cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 grid_dims;
    grid_dims.x = num_blocks_m;
    grid_dims.y = params.head_num;
    grid_dims.z = params.batch_size;

    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, params);
}


template <int kBlockM, int kBlockN, int kMaxN, typename InputType>
void run_moba_encoder_attn_hdim128(moba_encoder_attn_params &params, cudaStream_t stream) {

    constexpr static int Headdim = 128;
    constexpr static int kNWarps = kBlockM / 16 + 4;
    constexpr static int kStages = 2;

    using Ktraits = moba_encoder_attn_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, kStages, kMaxN, true, InputType>;
    run_moba_decoder_attn<Ktraits>(params, stream);
}

template <typename T>
void DispatchMobaEncoderAttn(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& qk_gate_topk_idx,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& out,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int batch_size,
        const int max_input_length) {

    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;

    using cute_type = typename cuteType<T>::type;

    moba_encoder_attn_params params;
    memset(&params, 0, sizeof(moba_encoder_attn_params));

    params.q_ptr = reinterpret_cast<cute_type*>(const_cast<T*>(q_input.data<T>()));
    params.k_ptr = reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>()));
    params.v_ptr = reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>()));
    params.o_ptr = reinterpret_cast<cute_type*>(const_cast<T*>(out.data<T>()));
    params.cu_seq_q = const_cast<int*>(cu_seq_q.data<int>());
    params.cu_seq_k = const_cast<int*>(cu_seq_k.data<int>());
    params.head_num = head_num;
    params.kv_head_num = kv_head_num;
    params.max_seq_q = max_seq_q;
    params.max_seq_k = max_seq_k;
    params.batch_size = batch_size;
    params.gqa_group_size = head_num / kv_head_num;
    constexpr float kLog2e = 1.4426950408889634074;
    params.scale_softmax_log2 = 1.0f / std::sqrt(head_dim) * kLog2e;
    params.qk_gate_topk_idx = const_cast<int*>(qk_gate_topk_idx.data<int>());
    params.seq_len_encoder = const_cast<int*>(seq_len_encoder.data<int>());
    params.cu_seq_q_pack = const_cast<int*>(cu_seq_q_pack.data<int>());

    run_moba_encoder_attn_hdim128<kBlockM, kBlockN, kMaxN, cute_type>(params, out.stream());
}

void MobaEncoderAttn(
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& qk_gate_topk_idx,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& out,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length) {

    const int batch_size = seq_len_encoder.dims()[0];
    if (q_input.dtype() == paddle::DataType::FLOAT16) {
        return
            DispatchMobaEncoderAttn<phi::dtype::float16>(
                q_input,
                k_input,
                v_input,
                qk_gate_topk_idx,
                cu_seq_q,
                cu_seq_k,
                cu_seq_q_pack,
                seq_len_encoder,
                seq_len_decoder,
                out,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                head_dim,
                batch_size,
                max_input_length);
    } else if (q_input.dtype() == paddle::DataType::BFLOAT16) {
        return
            DispatchMobaEncoderAttn<phi::dtype::bfloat16>(
                q_input,
                k_input,
                v_input,
                qk_gate_topk_idx,
                cu_seq_q,
                cu_seq_k,
                cu_seq_q_pack,
                seq_len_encoder,
                seq_len_decoder,
                out,
                max_seq_q,
                max_seq_k,
                head_num,
                kv_head_num,
                head_dim,
                batch_size,
                max_input_length);
    }
}


PD_BUILD_STATIC_OP(moba_encoder_attn)
    .Inputs({
        "q_input",
        "k_input",
        "v_input",
        "qk_gate_topk_idx",
        "cu_seq_q",
        "cu_seq_k",
        "cu_seq_q_pack",
        "seq_len_encoder",
        "seq_len_decoder",
        "out"})
    .Attrs({
        "max_seq_q: int",
        "max_seq_k: int",
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_input_length: int"})
    .Outputs({"attn_out"})
    .SetInplaceMap({{"out", "attn_out"}})
    .SetKernelFn(PD_KERNEL(MobaEncoderAttn));
