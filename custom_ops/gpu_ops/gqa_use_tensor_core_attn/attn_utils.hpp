/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif



#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "../moba_attention/moba_encoder_utils.hpp"
namespace gqa_attention {

template<typename T>
struct HalfSub;

template<>
struct HalfSub<cutlass::half_t> {
    inline __device__ void operator()(uint32_t* result_ptr, const uint32_t magic_num) {
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(*result_ptr) : "r"(*result_ptr), "r"(magic_num));
    }
};

template<>
struct HalfSub<cutlass::bfloat16_t> {
    inline __device__ void operator()(uint32_t* result_ptr, const uint32_t magic_num) {
        *reinterpret_cast<nv_bfloat162*>(result_ptr) -= *reinterpret_cast<const nv_bfloat162*>(&magic_num);
    }
};

template<typename T>
struct HalfMul;

template<>
struct HalfMul<cutlass::half_t> {
    inline __device__ void operator()(uint32_t* result_ptr, const uint32_t magic_num) {
        asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(*result_ptr) : "r"(*result_ptr), "r"(magic_num));
    }
};

template<>
struct HalfMul<cutlass::bfloat16_t> {
    inline __device__ void operator()(uint32_t* result_ptr, const uint32_t magic_num) {
        *reinterpret_cast<nv_bfloat162*>(result_ptr) *= *reinterpret_cast<const nv_bfloat162*>(&magic_num);
    }
};


template<typename T>
struct HalfMax;
template<>
struct HalfMax<cutlass::half_t> {
    inline __device__ __half2 operator()(const __half2 x, const __half2 y) {
        __half2 res;
        asm volatile("max.f16x2 %0, %1, %2;\n" :
            "=r"(*reinterpret_cast<uint32_t*>(&res)) :
            "r"(*reinterpret_cast<const uint32_t*>(&x)),
            "r"(*reinterpret_cast<const uint32_t*>(&y)));
        return res;
    }
};

template<>
struct HalfMax<cutlass::bfloat16_t> {
    inline __device__ nv_bfloat162 operator()(const nv_bfloat162 x, const nv_bfloat162 y) {
        nv_bfloat162 res;
        asm volatile("max.bf16x2 %0, %1, %2;\n" :
            "=r"(*reinterpret_cast<uint32_t*>(&res)) :
            "r"(*reinterpret_cast<const uint32_t*>(&x)),
            "r"(*reinterpret_cast<const uint32_t*>(&y)));
        return res;
    }
};


template<typename T>
struct HalfMin;
template<>
struct HalfMin<cutlass::half_t> {
    inline __device__ __half2 operator()(const __half2 x, const __half2 y) {
        __half2 res;
        asm volatile("min.f16x2 %0, %1, %2;\n" :
            "=r"(*reinterpret_cast<uint32_t*>(&res)) :
            "r"(*reinterpret_cast<const uint32_t*>(&x)),
            "r"(*reinterpret_cast<const uint32_t*>(&y)));
        return res;
    }
};

template<>
struct HalfMin<cutlass::bfloat16_t> {
    inline __device__ nv_bfloat162 operator()(const nv_bfloat162 x, const nv_bfloat162 y) {
        nv_bfloat162 res;
        asm volatile("min.bf16x2 %0, %1, %2;\n" :
            "=r"(*reinterpret_cast<uint32_t*>(&res)) :
            "r"(*reinterpret_cast<const uint32_t*>(&x)),
            "r"(*reinterpret_cast<const uint32_t*>(&y)));
        return res;
    }
};


template<typename T, bool Is_K>
inline __device__ static void convert_c8_2_half(uint32_t *src, T *dst, const T *cache_scale, const T* cache_zp) {
    uint32_t* half_result_ptr = reinterpret_cast<uint32_t*>(dst);
    if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
        static constexpr uint32_t fp32_base = 0x4B000000;
        float fp32_intermediates[4];

        uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
        fp32_intermediates_casted[0] = __byte_perm(*src, fp32_base, 0x7650);
        fp32_intermediates_casted[1] = __byte_perm(*src, fp32_base, 0x7651);
        fp32_intermediates_casted[2] = __byte_perm(*src, fp32_base, 0x7652);
        fp32_intermediates_casted[3] = __byte_perm(*src, fp32_base, 0x7653);

        #pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            fp32_intermediates[ii] -= 8388608.f;
        }

        #pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            half_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
        }
    } else {
        static constexpr uint32_t head_for_fp16 = 0x64006400;
        half_result_ptr[0] = __byte_perm(*src, head_for_fp16, 0x7150);
        half_result_ptr[1] = __byte_perm(*src, head_for_fp16, 0x7352);
    }

    using pack_half = typename moba::PackedHalf<T>::Type;
    #pragma unroll
    for (int i = 0; i < 2; i++){
        if constexpr (Is_K) {
            HalfSub<T>()(half_result_ptr + i, *reinterpret_cast<const uint32_t*>(cache_zp + i * 2));
            HalfMul<T>()(half_result_ptr + i, *reinterpret_cast<const uint32_t*>(cache_scale + i * 2));
        } else {
            pack_half bias;
            pack_half scale;
            bias.x = cache_zp[0];
            bias.y = cache_zp[0];
            scale.x = cache_scale[0];
            scale.y = cache_scale[0];
            HalfSub<T>()(half_result_ptr + i, *reinterpret_cast<const uint32_t*>(&bias));
            HalfMul<T>()(half_result_ptr + i, *reinterpret_cast<const uint32_t*>(&scale));
        }
    }
}

template<typename T, bool Is_K>
inline __device__ static void convert_c4_2_half(uint32_t *src, T *dst, const T *cache_scale, const T* cache_zp) {
    using pack_half = typename moba::PackedHalf<T>::Type;
    static constexpr uint32_t MASK = 0x0f0f0f0f;
    static constexpr uint32_t head_for_fp16 = std::is_same_v<T, cutlass::bfloat16_t> ? 0x43004300 : 0x64006400;
    static constexpr uint32_t mask_for_c42fp16_one = 0x7253;
    static constexpr uint32_t mask_for_c42fp16_two = 0x7051;
    uint32_t* result_ptr = reinterpret_cast<uint32_t*>(dst);
    uint32_t source = *reinterpret_cast<uint32_t const*>(src);
    // source = {e0 e4 e1 e5 e2 e6 e3 e7}
    uint32_t bottom_i4s = source & MASK;
    // bottom_i4s = {0 e4 0 e5 0 e6 0 e7}
    uint32_t top_i4s = (source >> 4) & MASK;
    // top_i4s = {0 e0 0 e1 0 e2 0 e3}
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(result_ptr[0]) : "r"(top_i4s), "n"(head_for_fp16), "n"(mask_for_c42fp16_one));
    // result_ptr[0] = {e0 e1}
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(result_ptr[1]) : "r"(top_i4s), "n"(head_for_fp16), "n"(mask_for_c42fp16_two));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(result_ptr[2]) : "r"(bottom_i4s), "n"(head_for_fp16), "n"(mask_for_c42fp16_one));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(result_ptr[3]) : "r"(bottom_i4s), "n"(head_for_fp16), "n"(mask_for_c42fp16_two));

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (Is_K) {
            const int ith_col = i % 2 * 2;
            HalfSub<T>()(result_ptr + i, *reinterpret_cast<const uint32_t*>(cache_zp + ith_col));
            HalfMul<T>()(result_ptr + i, *reinterpret_cast<const uint32_t*>(cache_scale + ith_col));
        } else {
            const int ith_col = i / 2;
            pack_half bias;
            pack_half scale;
            bias.x = cache_zp[ith_col];
            bias.y = cache_zp[ith_col];
            scale.x = cache_scale[ith_col];
            scale.y = cache_scale[ith_col];
            HalfSub<T>()(result_ptr + i, *reinterpret_cast<const uint32_t*>(&bias));
            HalfMul<T>()(result_ptr + i, *reinterpret_cast<const uint32_t*>(&scale));
        }
    }
}

template<typename CacheKV_traits, typename T, int kHeadDim, int kDataNumPer2Byte, bool A_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename ThrCopy0, typename TiledCopy0>
inline __device__ void gemm_qk_quant(
        Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCsA, Tensor3 &tCrB,
        Tensor4 const& sB, TiledMma tiled_mma,
        ThrCopy0 smem_thr_copy_A,
        TiledCopy0 smem_tiled_copy_A,
        const int32_t tidx,
        const T * cache_scale, const T * cache_zp) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    if (!A_in_regs) {
        copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    uint32_t *sBdata = reinterpret_cast<uint32_t *>(sB.data().get()) + tidx * (kDataNumPer2Byte / 4);

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) {
                copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
        }
        if constexpr (kDataNumPer2Byte == 4) {
            convert_c4_2_half<T, true>(sBdata + i * kHeadDim, tCrB.data(), cache_scale + i * 4, cache_zp + i * 4);
        } else {
            convert_c8_2_half<T, true>(sBdata + i * (kHeadDim * 2), tCrB.data(), cache_scale + i * 4, cache_zp + i * 4);
            convert_c8_2_half<T, true>(sBdata + i * (kHeadDim * 2) + 1, tCrB.data() + 4, cache_scale + i * 4, cache_zp + i * 4);
        }

        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB, acc);
    }
}

template<typename CacheKV_traits, typename T, int kHeadDim, int kDataNumPer2Byte, bool A_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename ThrCopy0, typename TiledCopy0>
inline __device__ void gemm_value_quant(
        Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCsA, Tensor3 &tCrB,
        Tensor4 const& sB, TiledMma tiled_mma,
        ThrCopy0 smem_thr_copy_A,
        TiledCopy0 smem_tiled_copy_A,
        int32_t tidx,
        const T * cache_scale, const T * cache_zp) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    if (!A_in_regs) {
        copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    uint32_t *sBdata = reinterpret_cast<uint32_t *>(sB.data().get()) + tidx * (2 * kDataNumPer2Byte / 4);

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        const int cur_idx = i * kHeadDim * (2 * kDataNumPer2Byte / 4);

        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) {
                copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
        }
        if constexpr (kDataNumPer2Byte == 4) {
            convert_c4_2_half<T, false>(sBdata + cur_idx, tCrB.data(), cache_scale, cache_zp);
            convert_c4_2_half<T, false>(sBdata + cur_idx + 1, tCrB.data() + 8, cache_scale + 2, cache_zp + 2);
        } else {
            convert_c8_2_half<T, false>(sBdata + cur_idx, tCrB.data(), cache_scale, cache_zp);
            convert_c8_2_half<T, false>(sBdata + cur_idx + 1, tCrB.data() + 4, cache_scale + 1, cache_zp + 1);
            convert_c8_2_half<T, false>(sBdata + cur_idx + 2, tCrB.data() + 8, cache_scale + 2, cache_zp + 2);
            convert_c8_2_half<T, false>(sBdata + cur_idx + 3, tCrB.data() + 12, cache_scale + 3, cache_zp + 3);
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB, acc);
    }
}

template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};


template<int kMiLen, typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout> &scores, const uint32_t warp_id, const uint32_t col, const uint32_t reamin_seq_len) {
    const int cols = size<1>(scores) / 2;
    #pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < cols; ++ni) {
            const int col_index = warp_id * 8 + ni * 32 + col * 2;
            if (col_index >= reamin_seq_len) {
                scores(mi, ni * 2) = -INFINITY;
            }
            if (col_index + 1 >= reamin_seq_len) {
                scores(mi, ni * 2 + 1) = -INFINITY;
            }
        }
    }
}


template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

template<int kMiLen, typename Engine0, typename Layout0, typename T>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, T *scores_max){
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    moba::MaxOp<T> max_op;
    #pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni++) {
            scores_max[mi] = max_op(scores_max[mi], tensor(mi, ni));
        }
        scores_max[mi] = Allreduce<4>::run(scores_max[mi], max_op);
    }
}

template <int kMiLen, typename Engine0, typename Layout0, typename T>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, T const *max, T *sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    #pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
        const float max_scaled = max[mi] * scale;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
            sum[mi] += tensor(mi, ni);
        }
    }
}


}
