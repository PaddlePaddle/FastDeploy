/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Is_exact_streaming>
__global__ void flash_fwd_block_kernel(Flash_fwd_params params) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    FLASH_NAMESPACE::compute_block_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax, Is_exact_streaming>(params);
}

// blocksparse
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_block(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {

            BOOL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {

                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    BOOL_SWITCH(params.is_exact_streaming, Is_exact_streaming, [&] {
                    // Will only return softmax if dropout, to reduce compilation time.
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    // If Is_local, set Is_causal to false

                    auto kernel = &flash_fwd_block_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local && !Is_causal, false, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, ReturnSoftmaxConst && Is_dropout, Is_exact_streaming && Is_causal>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}


template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}


template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
}  // namespace FLASH_NAMESPACE
