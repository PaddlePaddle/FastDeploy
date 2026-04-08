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
#include "cute/int_tuple.hpp"
#include <cute/arch/cluster_sm90.hpp>
#include <cub/cub.cuh>
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

using namespace cute;

template<typename T>
struct PackedHalf;

template<>
struct PackedHalf<cutlass::half_t> {
    using Type = __half2;
};

template<>
struct PackedHalf<cutlass::bfloat16_t> {
    using Type = nv_bfloat162;
};

template<>
struct PackedHalf<phi::dtype::float16> {
    using Type = __half2;
};

template<>
struct PackedHalf<phi::dtype::bfloat16> {
    using Type = nv_bfloat162;
};


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


template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template<typename T>
struct MinOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x < y ? x : y; }
};

template <>
struct MinOp<float> {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return min(x, y); }
};


template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
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

    using pack_half = typename PackedHalf<T>::Type;
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
    using pack_half = typename PackedHalf<T>::Type;
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
    MaxOp<T> max_op;
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


template <typename paddle_type>
struct cuteType;

template <>
struct cuteType<phi::dtype::float16> {
    using type = cutlass::half_t;
};

template <>
struct cuteType<phi::dtype::bfloat16> {
    using type = cutlass::bfloat16_t;
};

template<typename T>
__forceinline__ __device__ auto float_2_half2(const float x) {
    if constexpr (std::is_same<T, cutlass::half_t>::value) {
        return __float2half2_rn(x);
    } else {
        return __float2bfloat162_rn(x);
    }
}


struct uint16 {
    uint4 u;
    uint4 v;
    uint4 s;
    uint4 t;
};


struct uint8 {
    uint4 u;
    uint4 v;
};

template<int BYTES>
struct BytesToType {};

template<>
struct BytesToType<64> {
    using Type = uint16;
    static_assert(sizeof(Type) == 64);
};

template<>
struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
};

template<>
struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<>
struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<>
struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<>
struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<>
struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

template<typename Elt_type, uint32_t NUM_ELT>
struct Vec {

    enum { BYTES = NUM_ELT * sizeof(Elt_type) };

    using Vec_type = typename BytesToType<BYTES>::Type;

    using Alias_type = union {
        Vec_type vec;
        Elt_type elt[NUM_ELT];
    };

    Alias_type data;

    inline __device__ Vec() {}

    template<typename S>
    inline __device__ void to(Vec<S, NUM_ELT> &other) {
        #pragma unroll
        for( int it = 0; it < NUM_ELT; it++ ) {
            other.data.elt[it] = S(this->data.elt[it]);
        }
    }

    template<typename Op>
    inline __device__ void assign(const Op &op) {
        #pragma unroll
        for( int it = 0; it < NUM_ELT; it++ ) {
            this->data.elt[it] = op(it);
        }
    }

    inline __device__ void load_from(const void *base_ptr) {
        this->data.vec = *reinterpret_cast<const Vec_type *>(base_ptr);
    }


    inline __device__ void store_to(void *base_ptr) {
        *reinterpret_cast<Vec_type *>(base_ptr) = this->data.vec;
    }

    inline __device__ void add(const Vec<Elt_type, NUM_ELT> &other) {
        static_assert(NUM_ELT % 2 == 0);
        using type = typename PackedHalf<Elt_type>::Type;
        #pragma unroll
        for (int it = 0; it < NUM_ELT / 2; it++) {
            type b = *reinterpret_cast<const type *>(other.data.elt + it * 2);
            *reinterpret_cast<type *>(this->data.elt + it * 2) += b;
        }
    }

    inline __device__ void set_zero() {
        constexpr int size = sizeof(Vec_type) / sizeof(int);
        #pragma unroll
        for (int i = 0; i < size; ++i) {
            (reinterpret_cast<int *>(this->data.elt))[i] = 0;
        }
    }

    inline __device__ void fma(const Vec<Elt_type, NUM_ELT> &scale, const Vec<Elt_type, NUM_ELT> &bias) {
        static_assert(NUM_ELT % 2 == 0);
        using type = typename PackedHalf<Elt_type>::Type;
        #pragma unroll
        for (int it = 0; it < NUM_ELT / 2; it++) {
            type a = *reinterpret_cast<const type *>(scale.data.elt + it * 2);
            type b = *reinterpret_cast<const type *>(bias.data.elt + it * 2);
            *reinterpret_cast<type *>(this->data.elt + it * 2) += a * b;
        }
    }
};

template<typename T, int PackSize>
inline __device__ void apply_rotary_embedding(Vec<T, PackSize>& vec, Vec<float, PackSize / 2>& cos, Vec<float, PackSize / 2>& sin) {
    static_assert(PackSize % 2 == 0);
    #pragma unroll
    for (int i = 0; i < PackSize / 2; i++) {
        const float cos_inv_freq = cos.data.elt[i];
        const float sin_inv_freq = sin.data.elt[i];
        const float v1 = static_cast<float>(vec.data.elt[2 * i]);
        const float v2 = static_cast<float>(vec.data.elt[2 * i + 1]);
        vec.data.elt[2 * i] = static_cast<T>(cos_inv_freq * v1 - sin_inv_freq * v2);
        vec.data.elt[2 * i + 1] = static_cast<T>(sin_inv_freq * v1 + cos_inv_freq * v2);
    }
}

template <bool Is_even_MN=true, typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy(
        TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
        Tensor<Engine1, Layout1> &D,
        Tensor<Engine2, Layout2> const &identity_MN,
        const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        }
    }
}

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename ThrCopy0, typename ThrCopy1,
         typename TiledCopy0, typename TiledCopy1>
inline __device__ void gemm(
        Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
        Tensor4 const& tCsB, TiledMma tiled_mma,
        ThrCopy0 &smem_thr_copy_A, ThrCopy1 &smem_thr_copy_B,
        TiledCopy0 &smem_tiled_copy_A, TiledCopy1 &smem_tiled_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));

    if (!A_in_regs) { copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template<typename T, typename ReductionOp, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp());
    if (threadIdx.x == 0) { result_broadcast = result; }
    __syncthreads();
    return result_broadcast;
}

template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
        auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
        return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout), make_layout(get<2, 1>(l), get<2>(acc_layout)));
    } else {  // SM80
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    }
};

template <bool zero_init=false, int wg_wait=0, bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC) {
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}


template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = acc_layout;
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
    } else {  // SM80
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    }
};

template<typename T, typename ReductionOp, int thread_group_width = 32>
__inline__ __device__ T WarpAllReduce(T val) {
    ReductionOp op;
    #pragma unroll
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = op(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


template <int kPackSize, int knthreads>
__device__ inline int get_data_count(const float * src, const float limit_value) {
    int count = 0;
    #pragma unroll
    for (int i = 0; i < kPackSize; i++) {
        if (src[i] >= limit_value) {
            count++;
        }
    }
    count = BlockAllReduce<int, SumOp<int>, knthreads>(count);
    return count;
}
