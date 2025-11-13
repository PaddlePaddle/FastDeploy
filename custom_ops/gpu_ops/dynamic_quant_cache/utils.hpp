
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
#include "../moba_attn/moba_attn_utils.hpp"

namespace dynamic_quant_cache {
using namespace cute;

template <typename SrcType, typename DstType>
struct Convert_to_fp8;
template <>
struct Convert_to_fp8<cutlass::half_t, cutlass::float_e4m3_t> {
  __forceinline__ __device__ uint32_t operator()(const uint32_t src1,
                                                 const uint32_t src2) {
    uint32_t dst;
    asm volatile(
        "{\n"
        ".reg .b16 lo;\n"
        ".reg .b16 hi;\n"
        "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n"
        "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}"
        : "=r"(dst)
        : "r"(src1), "r"(src2));
    return dst;
  }
};

template <>
struct Convert_to_fp8<phi::dtype::float16, cutlass::float_e4m3_t> {
  __forceinline__ __device__ uint32_t operator()(const uint32_t src1,
                                                 const uint32_t src2) {
    Convert_to_fp8<cutlass::half_t, cutlass::float_e4m3_t> convert;
    return convert(src1, src2);
  }
};

template <>
struct Convert_to_fp8<cutlass::bfloat16_t, cutlass::float_e4m3_t> {
  __forceinline__ __device__ uint32_t operator()(const uint32_t src1,
                                                 const uint32_t src2) {
    float2 res1 =
        __bfloat1622float2(reinterpret_cast<const nv_bfloat162 &>(src1));
    float2 res2 =
        __bfloat1622float2(reinterpret_cast<const nv_bfloat162 &>(src2));
    uint32_t dst;
    asm volatile(
        "{\n"
        ".reg .b16 lo;\n"
        ".reg .b16 hi;\n"
        "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
        "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}"
        : "=r"(dst)
        : "f"(res1.x), "f"(res1.y), "f"(res2.x), "f"(res2.y));
    return dst;
  }
};

template <>
struct Convert_to_fp8<phi::dtype::bfloat16, cutlass::float_e4m3_t> {
  __forceinline__ __device__ uint32_t operator()(const uint32_t src1,
                                                 const uint32_t src2) {
    Convert_to_fp8<cutlass::bfloat16_t, cutlass::float_e4m3_t> convert;
    return convert(src1, src2);
  }
};

template <typename SrcType, typename DstType>
struct Convert_from_fp8;
template <>
struct Convert_from_fp8<cutlass::float_e4m3_t, cutlass::half_t> {
  __forceinline__ __device__ int2 operator()(const uint32_t src) {
    int2 dst;
    asm volatile(
        "{\n"
        ".reg .b16 lo, hi;\n"
        "mov.b32 {lo, hi}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
        "}\n"
        : "=r"(dst.x), "=r"(dst.y)
        : "r"(src));
    return dst;
  }
};

template <>
struct Convert_from_fp8<cutlass::float_e4m3_t, phi::dtype::float16> {
  __forceinline__ __device__ int2 operator()(const uint32_t src) {
    Convert_from_fp8<cutlass::float_e4m3_t, cutlass::half_t> convert;
    return convert(src);
  }
};

template <>
struct Convert_from_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t> {
  __forceinline__ __device__ int2 operator()(const uint32_t src) {
    int2 dst;
    asm volatile(
        "{\n"
        ".reg .b16 lo, hi;\n"
        "mov.b32 {lo, hi}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
        "}\n"
        : "=r"(dst.x), "=r"(dst.y)
        : "r"(src));

    float2 res_float1 = __half22float2(*reinterpret_cast<__half2 *>(&dst.x));
    float2 res_float2 = __half22float2(*reinterpret_cast<__half2 *>(&dst.y));
    *reinterpret_cast<nv_bfloat162 *>(&dst.x) =
        __float22bfloat162_rn(res_float1);
    *reinterpret_cast<nv_bfloat162 *>(&dst.y) =
        __float22bfloat162_rn(res_float2);
    return dst;
  }
};

template <>
struct Convert_from_fp8<cutlass::float_e4m3_t, phi::dtype::bfloat16> {
  __forceinline__ __device__ int2 operator()(const uint32_t src) {
    Convert_from_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t> convert;
    return convert(src);
  }
};

template <typename T,
          typename scale_type,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename TiledMma>
__forceinline__ __device__ void gemm_qk(Tensor0 &acc,
                                        Tensor1 &tCrA,
                                        Tensor2 &tCrB,
                                        uint8_t *smem_b,
                                        TiledMma tiled_mma,
                                        const int tidx) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

  using pakc_half = __half2;
  uint32_t *scale_mem = reinterpret_cast<uint32_t *>(smem_b) + 512;
  const int col = tidx % 32 / 8 * 16;

  constexpr uint32_t mask = 0x03030303;

#pragma unroll
  for (int i = 0; i < size<2>(tCrA); i += 2) {
    uint32_t c2_value = reinterpret_cast<uint32_t *>(smem_b)[tidx + i * 64];

    for (int k = 1; k >= 0; --k) {
#pragma unroll
      for (int j = 1; j >= 0; --j) {
        uint32_t value = c2_value & mask;
        c2_value = c2_value >> 2;

        int2 half_data = Convert_from_fp8<scale_type, T>()(value);

        pakc_half cur_value = reinterpret_cast<pakc_half *>(&half_data)[0];
        pakc_half next_value = reinterpret_cast<pakc_half *>(&half_data)[1];

        uint32_t dst0, dst1, dst2, dst3;
        const int scale_idx = (i + k) * 8 + j * 4;

        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(
            reinterpret_cast<uint128_t *>(scale_mem + scale_idx) + col);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(smem_int_ptr));

        cur_value = cur_value * reinterpret_cast<pakc_half *>(&dst0)[0] -
                    reinterpret_cast<pakc_half *>(&dst2)[0];
        next_value = next_value * reinterpret_cast<pakc_half *>(&dst1)[0] -
                     reinterpret_cast<pakc_half *>(&dst3)[0];

        reinterpret_cast<pakc_half *>(tCrB(_, _, i + k).data())[j] = cur_value;
        reinterpret_cast<pakc_half *>(tCrB(_, _, i + k).data())[j + 2] =
            next_value;
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    cute::gemm(tiled_mma, tCrA(_, _, i + 1), tCrB(_, _, i + 1), acc);
  }
}

template <typename T,
          typename scale_type,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename Tensor3,
          typename TiledMma,
          typename ThrCopy,
          typename TiledCopy>
__forceinline__ __device__ void gemm_v(Tensor0 &acc,
                                       Tensor1 &tCrA,
                                       Tensor2 &tCsA,
                                       Tensor3 &tCrB,
                                       uint8_t *smem_b,
                                       TiledMma tiled_mma,
                                       ThrCopy thr_copy_A,
                                       TiledCopy tiled_copy_A,
                                       const int tidx) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  using pakc_half = __half2;
  Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
  copy(tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));

  pakc_half *scale_mem = reinterpret_cast<pakc_half *>(smem_b) + 512;
  pakc_half *zp_mem = scale_mem + 128;

  const int col = tidx % 4;

  constexpr uint32_t mask = 0x03030303;

#pragma unroll
  for (int i = 0; i < size<2>(tCrA); i++) {
    if (i < size<2>(tCrA) - 1) {
      copy(tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
    }
    uint32_t c2_value = reinterpret_cast<uint32_t *>(smem_b)[tidx + i * 128];
    for (int j = 3; j >= 0; j--) {
      const int scale_idx = i * 8 + j * 32 + col;
      uint32_t value = c2_value & mask;
      c2_value = c2_value >> 2;

      int2 half_data = Convert_from_fp8<scale_type, T>()(value);

      pakc_half value1 = *reinterpret_cast<pakc_half *>(&half_data.x);
      pakc_half value2 = *reinterpret_cast<pakc_half *>(&half_data.y);

      value1 = value1 * scale_mem[scale_idx] - zp_mem[scale_idx];
      value2 = value2 * scale_mem[scale_idx + 4] - zp_mem[scale_idx + 4];

      reinterpret_cast<pakc_half *>(raw_pointer_cast(tCrB(_, j, i).data()))[0] =
          value1;
      reinterpret_cast<pakc_half *>(raw_pointer_cast(tCrB(_, j, i).data()))[1] =
          value2;
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <typename Gmem_copy_struct, int kNThreads>
__forceinline__ __device__ void copy_kv(const int tidx,
                                        const int data_num_per_block,
                                        const uint8_t *g_mem,
                                        uint8_t *s_mem) {
  constexpr int32_t kPackSize = 16 / sizeof(uint8_t);

  for (int i = tidx * kPackSize; i < data_num_per_block;
       i += kNThreads * kPackSize) {
    Gmem_copy_struct::copy(
        *reinterpret_cast<const cute::uint128_t *>(g_mem + i),
        *reinterpret_cast<cute::uint128_t *>(s_mem + i));
  }
}

template <int kMiLen, typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout> &scores,
                                  const uint32_t warp_id,
                                  const uint32_t col,
                                  const uint32_t reamin_seq_len) {
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

template <typename T, typename out_T, int PackSize>
inline __device__ void apply_rotary_embedding(Vec<T, PackSize> &vec,
                                              Vec<out_T, PackSize> &out,
                                              Vec<float, PackSize / 2> &cos,
                                              Vec<float, PackSize / 2> &sin) {
  static_assert(PackSize % 2 == 0);
#pragma unroll
  for (int i = 0; i < PackSize / 2; i++) {
    const float cos_inv_freq = cos.data.elt[i];
    const float sin_inv_freq = sin.data.elt[i];
    const float v1 = static_cast<float>(vec.data.elt[2 * i]);
    const float v2 = static_cast<float>(vec.data.elt[2 * i + 1]);
    out.data.elt[2 * i] =
        static_cast<out_T>(cos_inv_freq * v1 - sin_inv_freq * v2);
    out.data.elt[2 * i + 1] =
        static_cast<out_T>(sin_inv_freq * v1 + cos_inv_freq * v2);
  }
}

}  // namespace dynamic_quant_cache
