#pragma once
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/reg_reconfig.h"

#include "kernel_traits.h"
#include "mainloop_fwd.h"

#define N_SWITCH(_N, ...)                                                    \
    [&] {                                                                    \
        if (_N == 16) {                                                      \
            constexpr static int GemmN = 16;                                 \
            return __VA_ARGS__();                                            \
        } else if (_N == 32) {                                               \
            constexpr static int GemmN = 32;                                 \
            return __VA_ARGS__();                                            \
        } else if (_N == 48) {                                               \
            constexpr static int GemmN = 48;                                 \
            return __VA_ARGS__();                                            \
        } else if (_N == 64) {                                               \
            constexpr static int GemmN = 64;                                 \
            return __VA_ARGS__();                                            \
        } else if (_N == 80) {                                               \
            constexpr static int GemmN = 80;                                 \
            return __VA_ARGS__();                                            \
        } else if (_N == 96) {                                               \
            constexpr static int GemmN = 96;                                 \
            return __VA_ARGS__();                                            \
        } else if (_N == 112) {                                              \
            constexpr static int GemmN = 112;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 128) {                                              \
            constexpr static int GemmN = 128;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 144) {                                              \
            constexpr static int GemmN = 144;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 160) {                                              \
            constexpr static int GemmN = 160;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 176) {                                              \
            constexpr static int GemmN = 176;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 192) {                                              \
            constexpr static int GemmN = 192;                                \
            return __VA_ARGS__();                                            \
        }  else if (_N == 208) {                                             \
            constexpr static int GemmN = 208;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 224) {                                              \
            constexpr static int GemmN = 224;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 240) {                                              \
            constexpr static int GemmN = 240;                                \
            return __VA_ARGS__();                                            \
        } else if (_N == 256) {                                              \
            constexpr static int GemmN = 256;                                \
            return __VA_ARGS__();                                            \
        } else {                                                             \
            constexpr static int GemmN = 256;                               \
            return __VA_ARGS__();                                            \
        }                                                                    \
    }()



template<int splitK, typename InputType>
void __global__ element_add_kernel(const InputType *src, InputType * dst, const uint32_t step) {
    constexpr uint32_t kPacketSize = 16 / sizeof(InputType);
    const uint32_t idx = kPacketSize * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= step) return;
    float4 dst_vec;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        reinterpret_cast<uint32_t*>(&dst_vec)[i] = 0;
    }

    #pragma unroll
    for (int i = 0; i < 1; ++i) {
        const float4 src_vec = *reinterpret_cast<const float4*>(src + idx);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            reinterpret_cast<nv_bfloat162*>(&dst_vec)[j] = reinterpret_cast<const nv_bfloat162*>(&src_vec)[j] + reinterpret_cast<nv_bfloat162*>(&dst_vec)[j];
        }
    }
    *reinterpret_cast<float4*>(dst + idx) = dst_vec;
    
}

template <typename Ktraits>
void  __global__ __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1) w4afp8_geem_kernel(
        CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits>::Params const mainloop_params) {
    
    using Element = typename Ktraits::Element;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma{});
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int M = Ktraits::M;
    static constexpr int TokenPaddingSize = Ktraits::TokenPaddingSize;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using ElementOutput = typename Ktraits::ElementOutput;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    const int bidm = blockIdx.x;
    const int bidn = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    if (tidx == 0) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    
    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesA + CollectiveMainloop::TmaTransactionBytesB;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;

    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    const int pre_fix_tokens = TokenPaddingSize == 0 ? mainloop_params.tokens[bidb] : 0;

    const int tokens = TokenPaddingSize == 0 ? mainloop_params.tokens[bidb + 1] - pre_fix_tokens : mainloop_params.tokens[bidb];
    

    if (bidn * kBlockN >= tokens) {
        return;
    }

    __align__(16) __shared__ float input_row_sum[kBlockN];

    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 40 : 32>();
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>(); 
        collective_mainloop.load(
                mainloop_params, 
                pipeline, 
                smem_pipe_write,
                shared_storage,
                tokens,
                pre_fix_tokens,
                bidm,
                bidn,
                bidb,
                tidx);
    } else {
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 232 : 160>(); 
        PipelineState smem_pipe_read;

        typename Ktraits::TiledMma tiled_mma;

        Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{})); 

        const int mma_tidx = tidx - NumCopyThreads;
        const int lane_id = mma_tidx % 4 * 2;

        const float2 weight_scale = reinterpret_cast<const float2*>(mainloop_params.weight_scale + bidb * M + bidm * kBlockM)[mma_tidx / 4]; 

        if constexpr (TokenPaddingSize == 0) {
            const int input_sum_idx = pre_fix_tokens + bidn * kBlockN;
            if (mma_tidx < kBlockN) {
                reinterpret_cast<float*>(input_row_sum)[mma_tidx] = reinterpret_cast<const float*>(mainloop_params.input_row_sum + input_sum_idx)[mma_tidx];
            }
        } else {
            const int input_sum_idx = bidb * TokenPaddingSize + bidn * kBlockN;
            if (mma_tidx < kBlockN / 4) {
                reinterpret_cast<float4*>(input_row_sum)[mma_tidx] = reinterpret_cast<const float4*>(mainloop_params.input_row_sum + input_sum_idx)[mma_tidx];
            }
        }
        

        collective_mainloop.mma(
            mainloop_params,
            pipeline,  
            smem_pipe_read,
            shared_storage,
            tSrS,
            mma_tidx); 
        
        collective_mainloop.store(
            mainloop_params, 
            tSrS, 
            shared_storage, 
            tiled_mma,
            input_row_sum + lane_id,
            reinterpret_cast<const float*>(&weight_scale),
            tokens,
            pre_fix_tokens,         
            bidm,
            bidn,
            bidb,
            mma_tidx);
    }

}

template <int Batch>
auto get_gmem_layout(const int Rows, const int Cols) {
    return  make_layout(
                make_shape(
                    static_cast<int64_t>(Rows),
                    static_cast<int64_t>(Cols),
                    static_cast<int64_t>(Batch)),
                make_stride(
                    static_cast<int64_t>(Cols),
                    cute::_1{},
                    static_cast<int64_t>(Rows * Cols)));
}


template <typename InputType, typename OutputType, typename Kernel_traits, int M, int K, int Batch, int TokenPaddingSize>
void run_gemm(const InputType * A, const InputType * B, OutputType * C, const float *weight_scale,
        const float *input_row_sum, const int * tokens, const int max_tokens, cudaStream_t stream) {

    using ElementOutput = typename Kernel_traits::ElementOutput;
    using Element = typename Kernel_traits::Element;
    using CollectiveMainloop = CollectiveMainloopFwd<Kernel_traits>;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    constexpr int M_nums = (M + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int N_nums = (max_tokens + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(A),
            get_gmem_layout<Batch>(M, K / 2),
            static_cast<Element const*>(B),
            get_gmem_layout<Batch>(TokenPaddingSize == 0 ? max_tokens * Batch : TokenPaddingSize, K),
            static_cast<ElementOutput*>(C),
            get_gmem_layout<Batch>(M, TokenPaddingSize == 0 ? max_tokens : TokenPaddingSize),
            weight_scale,
            input_row_sum,
            tokens
        });

    void *kernel;
    kernel = (void *)w4afp8_geem_kernel<Kernel_traits>;
    
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);

    if (smem_size >= 48 * 1024) {
       cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 grid_dims;
    grid_dims.x = M_nums;
    grid_dims.y = N_nums;
    grid_dims.z = Batch;
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(
        launch_params, kernel, mainloop_params);
}

template <typename InputType, typename OutputType, int M, int K, int Batch, int TokenPaddingSize>
void w4afp8_gemm(
        const InputType * weight, 
        const InputType * input, 
        OutputType * out, 
        const float *weight_scale,
        const float *input_row_sum, 
        const int *tokens, 
        const int max_tokens, 
        cudaStream_t stream) {
    constexpr static int kBlockM = 128;
    constexpr static int kBlockK = 128;
    constexpr static int kNWarps = 4 + kBlockM / 16;
    constexpr static int kStages = 5;
    constexpr int kCluster = 1;
    static_assert(K % kBlockK == 0);
    constexpr int kTiles = K / kBlockK;
    const int N = (max_tokens + 15) / 16 * 16;

    N_SWITCH(N, [&] {
        using Kernel_traits = Kernel_traits<kBlockM, GemmN, kBlockK, kNWarps, kStages, kTiles, M, TokenPaddingSize, kCluster, InputType, OutputType>;
        run_gemm<InputType, OutputType, Kernel_traits, M, K, Batch, TokenPackSize>(weight, input, out, weight_scale, input_row_sum, tokens, max_tokens, stream);
    });
    
}

