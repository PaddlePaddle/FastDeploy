/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_bwd_launch_template.h
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_bwd_kernel.h"

namespace FLASH_NAMESPACE {

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(Flash_bwd_params params) {
    FLASH_NAMESPACE::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

//add by JXGuo: not used
template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(Flash_bwd_params params) {
    FLASH_NAMESPACE::clear_dKVaccum<Kernel_traits>(params);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(Flash_bwd_params params) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    FLASH_NAMESPACE::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K>(params);
}


// for blocksparse-flash-attention2
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K>
__global__ void flash_bwd_block_dq_dk_dv_loop_seqk_parallel_kernel(Flash_bwd_params params) {
    static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
    FLASH_NAMESPACE::compute_block_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K>(params);
}


template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(Flash_bwd_params params, const int nsplits) {
    FLASH_NAMESPACE::convert_dQ<Kernel_traits>(params, nsplits);
}

// add by JXGuo: not used
template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(Flash_bwd_params params) {
    FLASH_NAMESPACE::convert_dKV<Kernel_traits>(params);
}


// for blocksparse-flash-attention2
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_block_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        int num_sm = get_num_sm(get_current_device());
        gridDimx = (num_sm + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                // If Is_local, set Is_causal to false
                auto kernel = &flash_bwd_block_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local && !Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst>;
                if (smem_size_dq_dk_dv >= 48 * 1024)  {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// for blocksparse-flash-attention2
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_block(Flash_bwd_params &params, cudaStream_t stream) {
    run_flash_bwd_block_seqk_parallel<Kernel_traits, Is_dropout, Is_causal>(params, stream);
}

// for blocksparse-flash-attention2
template<typename T, bool Is_causal>
void run_mha_bwd_block_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
            if constexpr(!Is_dropout) {  // We can afford more registers to keep V in registers
                run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {  // 96 KB
            run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_block_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // Changing AtomLayoutMdQ from 2 to 4 takes the same time
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream, configure);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, true, false, T>>(params, stream, configure);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 2, 4, 4, false, false, T>>(params, stream, configure);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream, configure);
        // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // This has a lot of register spilling
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
        } else {
            // if (params.h == params.h_k) {
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout>(params, stream, configure);
            // } else {
            //     run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream, configure);
            // }
        }
    });
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, 2, 2, 2, true, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 128, 4, 1, 4, 1, false, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 16, 128, 4, 1, 4, 1, false, false, T>>(params, stream, configure);
    // M=128, N=64 is quite slow, I think because we need to read/write dQaccum twice as many times
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 2, 2, 2, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream, configure);

    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 4, 4, 2, 4, false, false, T>>(params, stream, configure);
}

template<typename T, bool Is_causal>
void run_mha_bwd_block_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (params.h == params.h_k) {
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 128, 8, 2, 2, 2, false, false, T>>(params, stream, configure);
            // This is faster, in the case of sequence-parallel bwd (where we need fewer registers).
            // Out of these three, the 2nd one is slightly faster (2% faster than the first). Idk why.
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 2, 2, false, false, T>>(params, stream, configure);
            if (max_smem_per_block >= 144 * 1024) {
                run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_bwd_seqk_parallel<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd_seqk_parallel<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, true, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, true, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 2, true, false, T>, Is_dropout>(params, stream, configure);
            } else {
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream, configure);
                run_flash_bwd_block<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>>(params, stream, configure);

            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
        // } else {
            // run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
        // }
    });
}

}  // namespace FLASH_NAMESPACE