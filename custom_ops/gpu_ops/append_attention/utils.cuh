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
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "helper.h"
#include "mem_util.cuh"

#define NUM_WARPS_PER_BLOCK 4
#define NUM_THREADS_PER_BLOCK 128
#define kWarpSize 32

#define HOSTDEVICE __host__ __device__

/*-------------------------------------traits-----------------------------------------*/
template <typename T>
struct type_traits {
  using paddle_type = T;
  using phi_type = T;
  using nv_type = T;
  using nv2_type = T;
};

// template <>
// struct type_traits<paddle::DataType::FLOAT16> {
//   using paddle_type = paddle::DataType::FLOAT16;
//   using phi_type = phi::dtype::float16;
//   using nv_type = half;
//   using nv2_type = half2;
// };

template <>
struct type_traits<phi::dtype::float16> {
  // using paddle_type = paddle::DataType::FLOAT16;
  using phi_type = phi::dtype::float16;
  using nv_type = half;
  using nv2_type = half2;
};

template <>
struct type_traits<half> {
  // using paddle_type = paddle::DataType::FLOAT16;
  using phi_type = phi::dtype::float16;
  using nv_type = half;
  using nv2_type = half2;
};

template <>
struct type_traits<half2> {
  // using paddle_type = paddle::DataType::FLOAT16;
  using phi_type = phi::dtype::float16;
  using nv_type = half;
  using nv2_type = half2;
};

// template <>
// struct type_traits<paddle::DataType::BFLOAT16> {
//   using paddle_type = paddle::DataType::FLOAT16;
//   using phi_type = phi::dtype::bfloat16;
//   using nv_type = __nv_bfloat16;
//   using nv2_type = __nv_bfloat162;
// };

template <>
struct type_traits<phi::dtype::bfloat16> {
  // using paddle_type = paddle::DataType::FLOAT16;
  using phi_type = phi::dtype::bfloat16;
  using nv_type = __nv_bfloat16;
  using nv2_type = __nv_bfloat162;
};

template <>
struct type_traits<__nv_bfloat16> {
  // using paddle_type = paddle::DataType::FLOAT16;
  using phi_type = phi::dtype::bfloat16;
  using nv_type = __nv_bfloat16;
  using nv2_type = __nv_bfloat162;
};

template <>
struct type_traits<__nv_bfloat162> {
  // using paddle_type = paddle::DataType::FLOAT16;
  using phi_type = phi::dtype::bfloat16;
  using nv_type = __nv_bfloat16;
  using nv2_type = __nv_bfloat162;
};

// template <>
// struct type_traits<paddle::DataType::FLOAT8_E4M3FN> {
//   using paddle_type = paddle::DataType::FLOAT8_E4M3FN;
//   using phi_type = phi::dtype::float8_e4m3fn;
//   using nv_type = __nv_fp8_e4m3;
//   using nv2_type = __nv_fp8x2_e4m3;
// };

template <>
struct type_traits<phi::dtype::float8_e4m3fn> {
  // using paddle_type = paddle::DataType::FLOAT8_E4M3FN;
  using phi_type = phi::dtype::float8_e4m3fn;
  using nv_type = __nv_fp8_e4m3;
  using nv2_type = __nv_fp8x2_e4m3;
};

template <>
struct type_traits<__nv_fp8_e4m3> {
  // using paddle_type = paddle::DataType::FLOAT8_E4M3FN;
  using phi_type = phi::dtype::float8_e4m3fn;
  using nv_type = __nv_fp8_e4m3;
  using nv2_type = __nv_fp8x2_e4m3;
};

template <>
struct type_traits<__nv_fp8x2_e4m3> {
  // using paddle_type = paddle::DataType::FLOAT8_E4M3FN;
  using phi_type = phi::dtype::float8_e4m3fn;
  using nv_type = __nv_fp8_e4m3;
  using nv2_type = __nv_fp8x2_e4m3;
};
/*---------------------------------1. type
 * traits--------------------------------------*/

/*---------------------------------2. fast
 * convert--------------------------------------*/
inline __device__ static void convert_fp8(half* result,
                                          const uint32_t& source) {
  printf("Do not support fp8 to half although it's very easy.\n");
}

inline __device__ static void convert_fp8(__nv_bfloat16* result,
                                          const uint32_t& source) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  uint32_t dest0;
  uint32_t dest1;
  asm volatile(
      "{\n"
      ".reg .b16 lo, hi;\n"
      "mov.b32 {lo, hi}, %2;\n"
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
      "}\n"
      : "=r"(dest0), "=r"(dest1)
      : "r"(source));

  ((nv_bfloat162*)(result))[0] =
      __float22bfloat162_rn(__half22float2(((half2*)(&dest0))[0]));
  ((nv_bfloat162*)(result))[1] =
      __float22bfloat162_rn(__half22float2(((half2*)(&dest1))[0]));
#else
  printf("Do not support fp8 in arch < 890\n");
  asm("trap;");
#endif
}

inline __device__ static void convert_int8(
    half* result, const uint32_t& source) {  // 4 int8 each time
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(fp16_result_ptr[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(fp16_result_ptr[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(fp16_result_ptr[0])
               : "r"(fp16_result_ptr[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(fp16_result_ptr[1])
               : "r"(fp16_result_ptr[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

inline __device__ static void convert_int8(
    __nv_bfloat16* result, const uint32_t& source) {  // 4 int8 each time
  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[4];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
  fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
  fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

#pragma unroll
  for (int ii = 0; ii < 4; ++ii) {
    fp32_intermediates[ii] -= 8388736.f;  // (8388608.f + 128.f);
  }

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                      fp32_intermediates_casted[2 * ii + 1],
                                      0x7632);
  }
}
/*---------------------------------2. fast
 * convert--------------------------------------*/

/*---------------------------------3. vector
 * cast--------------------------------------*/
template <typename dst_t, typename src_t, size_t vec_size>
__forceinline__ HOSTDEVICE void vec_cast(dst_t* dst, const src_t* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    dst[i] = src[i];
  }
}

template <size_t vec_size>
__forceinline__ HOSTDEVICE void vec_cast<float, half>(float* dst,
                                                      const half* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ HOSTDEVICE void vec_cast<half, float>(half* dst,
                                                      const float* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((half2*)dst)[i] = __float22half2_rn(((float2*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ HOSTDEVICE void vec_cast<float, nv_bfloat16>(
    float* dst, const nv_bfloat16* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((float2*)dst)[i] = __bfloat1622float2(((nv_bfloat162*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ HOSTDEVICE void vec_cast<nv_bfloat16, float>(nv_bfloat16* dst,
                                                             const float* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((nv_bfloat162*)dst)[i] = __float22bfloat162_rn(((float2*)src)[i]);
  }
}
/*---------------------------------3. vector
 * cast--------------------------------------*/

/*-------------------------------------4.
 * func-----------------------------------------*/
__forceinline__ HOSTDEVICE int div_up(int a, int b) { return (a + b - 1) / b; }

template <typename T>
__inline__ __device__ T Rsqrt(T x);

template <>
__inline__ __device__ float Rsqrt<float>(float x) {
  return rsqrt(x);
}

template <>
__inline__ __device__ double Rsqrt<double>(double x) {
  return rsqrt(x);
}

__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x,
                                                           uint32_t y) {
  return (x > y) ? x - y : 0U;
}

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T, bool is_need_kv_quant, bool IsFP8, int RoundType = 0>
HOSTDEVICE __forceinline__ uint8_t QuantToC8(const T scale,
                                             const T value,
                                             const float max_bound,
                                             const float min_bound) {
  uint8_t eight_bits;
  float quant_value;
  if constexpr (is_need_kv_quant) {
    quant_value = static_cast<float>(scale * value);
  } else {
    quant_value = static_cast<float>(value);
  }
  if constexpr (RoundType == 0) {
    quant_value = roundWithTiesToEven(quant_value);
  } else {
    quant_value = round(quant_value);
  }

  if constexpr (IsFP8) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    quant_value = quant_value > 448.0f ? 448.0f : quant_value;
    quant_value = quant_value < -448.0f ? -448.0f : quant_value;
    auto tmp = static_cast<__nv_fp8_e4m3>(quant_value);
    eight_bits = *(reinterpret_cast<uint8_t*>(&tmp));
#else
    printf("Do not support fp8 in arch < 890\n");
    asm("trap;");
#endif
  } else {
    quant_value = quant_value > 127.0f ? 127.0f : quant_value;
    quant_value = quant_value < -127.0f ? -127.0f : quant_value;
    eight_bits = static_cast<uint8_t>(quant_value + 128.0f);
  }
  return eight_bits;
}

template <typename T, bool IsFP8>
inline __device__ static void convert_c8(T* result, const uint32_t& source) {
  if constexpr (IsFP8) {
    convert_fp8(result, source);
  } else {
    convert_int8(result, source);
  }
}

template <typename T>
inline __device__ void WelfordCombine1(T b_m2, T* m2) {
  *m2 += b_m2;
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_m2, T* m2) {
  *m2 = thread_m2;
  for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
    T b_m2 = __shfl_xor_sync(0xffffffff, *m2, mask);
    WelfordCombine1(b_m2, m2);
  }
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(T thread_m2, T* m2) {
  WelfordWarpReduce<T, thread_group_width>(thread_m2, m2);
}

#define CHECK_CUDA_CALL(func, ...)                                      \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (e != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e \
                << ") " << __FILE__ << ": line " << __LINE__            \
                << " at function " << STR(func) << std::endl;           \
      return e;                                                         \
    }                                                                   \
  }

__device__ __forceinline__ float2 fast_float2_mul(const float2& a,
                                                  const float2& b) {
  float2 res;
  // 使用向量化PTX指令同时处理x/y分量
  asm volatile(
      "{\n"
      "  fma.rn.f32 %0, %2, %4, 0.0;\n"  // res.x = a.x * b.x
      "  fma.rn.f32 %1, %3, %5, 0.0;\n"  // res.y = a.y * b.y
      "}"
      : "=f"(res.x), "=f"(res.y)                // 输出操作数
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y)  // 输入操作数
  );
  return res;
}

__device__ __forceinline__ float2 fast_float2_fma(float2& a,
                                                  const float2& b,
                                                  const float2& c) {
  float2 res;
  // 使用向量化PTX指令同时处理x/y分量
  asm volatile(
      "{\n"
      "  fma.rn.f32 %0, %2, %4, %6;\n"  // res.x = a.x * b.x
      "  fma.rn.f32 %1, %3, %5, %7;\n"  // res.y = a.y * b.y
      "}"
      : "=f"(res.x), "=f"(res.y)  // 输出操作数
      : "f"(a.x),
        "f"(a.y),
        "f"(b.x),
        "f"(b.y),
        "f"(c.x),
        "f"(c.y)  // 输入操作数
  );
  return res;
}

// __device__ __forceinline__ float2 fast_bfloat162_fma(__nv_bfloat162& a_bf162,
// const __nv_bfloat162& b_bf162, const __nv_bfloat162& c_bf162) {
//     // 使用向量化PTX指令同时处理x/y分量
//     asm volatile (
//         "{\n"
//         "  fma.rn.b16 %0, %2, %4, %0;\n"   // res.x = a.x * b.x
//         "  fma.rn.b16 %1, %3, %5, %1;\n"   // res.y = a.y * b.y
//         "}"
//         : "=r"(a_bf162.x), "=r"(a_bf162.y)          // 输出操作数
//         : "r"(b_bf162.x), "r"(b_bf162.y),
//           "r"(c_bf162.x), "r"(c_bf162.y) // 输入操作数
//     );
//     float2 res = __bfloat1622float2_rn(a_bf162);
//     return res;
// }

__device__ __forceinline__ float2 fast_float2_sub_expf(const float2& a,
                                                       const float2& b) {
  float2 res;
  // 使用向量化减法指令（PTX sub.rn.f32）
  asm volatile(
      "{\n"
      "  sub.f32 %0, %2, %4;\n"  // res.x = a.x - b.x
      "  sub.f32 %1, %3, %5;\n"  // res.y = a.y - b.y
      "}"
      : "=f"(res.x), "=f"(res.y)                // 输出操作数
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y)  // 输入操作数
  );
  res.x = expf(res.x);
  res.y = expf(res.y);
  return res;
}

template <typename T, int VEC_SIZE, typename OutT>
struct StoreFunc {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<OutT, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    out_vec[i] = static_cast<OutT>(ori_out_vec[i]);
    printf("Fatal! Unimplemented StoreFunc for cascade append attention\n");
  }
};

template <typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, int8_t> {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<int8_t, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    float quant_value =
        127.0f *
        static_cast<float>((ori_out_vec[i] + shift_bias_vec[i]) *
                           smooth_weight_vec[i]) *
        in_scale;
    quant_value = rintf(quant_value);
    quant_value = quant_value > 127.0f ? 127.0f : quant_value;
    quant_value = quant_value < -127.0f ? -127.0f : quant_value;
    out_vec[i] = static_cast<int8_t>(quant_value);
  }
};

template <typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, __nv_fp8_e4m3> {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<__nv_fp8_e4m3, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    float quant_value =
        quant_max_bound * static_cast<float>(ori_out_vec[i]) * in_scale;
    quant_value = quant_value > quant_max_bound ? quant_max_bound : quant_value;
    quant_value = quant_value < quant_min_bound ? quant_min_bound : quant_value;
    out_vec[i] = static_cast<__nv_fp8_e4m3>(quant_value);
  }
};

template <typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, T> {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<T, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    out_vec[i] = ori_out_vec[i];
  }
};
/*-------------------------------------4.
 * func-----------------------------------------*/

/*-----------------------------------5.
 * dispatch---------------------------------------*/
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...) \
  switch (head_dim) {                              \
    case 128: {                                    \
      constexpr size_t HEAD_DIM = 128;             \
      __VA_ARGS__                                  \
      break;                                       \
    }                                              \
    default: {                                     \
      PD_THROW("not support the head_dim");        \
    }                                              \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 3) {                              \
    constexpr size_t GROUP_SIZE = 3;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 5) {                              \
    constexpr size_t GROUP_SIZE = 5;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 6) {                              \
    constexpr size_t GROUP_SIZE = 6;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 7) {                              \
    constexpr size_t GROUP_SIZE = 7;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 12) {                             \
    constexpr size_t GROUP_SIZE = 12;                        \
    __VA_ARGS__                                              \
  } else if (group_size == 14) {                             \
    constexpr size_t GROUP_SIZE = 14;                        \
    __VA_ARGS__                                              \
  } else if (group_size == 16) {                             \
    constexpr size_t GROUP_SIZE = 16;                        \
    __VA_ARGS__                                              \
  } else {                                                   \
    PD_THROW("not support the group_size", group_size);      \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 12) {                             \
    constexpr size_t GROUP_SIZE = 12;                        \
    __VA_ARGS__                                              \
  } else if (group_size == 14) {                             \
    constexpr size_t GROUP_SIZE = 14;                        \
    __VA_ARGS__                                              \
  } else if (group_size == 16) {                             \
    constexpr size_t GROUP_SIZE = 16;                        \
    __VA_ARGS__                                              \
  } else {                                                   \
    PD_THROW("not support the group_size", group_size);      \
  }

#define DISPATCH_BLOCKSHAPE_Q(block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, ...) \
  if (block_shape_q <= 16) {                                                 \
    constexpr size_t BLOCK_SHAPE_Q = 16;                                     \
    constexpr size_t NUM_WARP_Q = 1;                                         \
    __VA_ARGS__                                                              \
  } else if (block_shape_q <= 32) {                                          \
    constexpr size_t BLOCK_SHAPE_Q = 32;                                     \
    constexpr size_t NUM_WARP_Q = 1;                                         \
    __VA_ARGS__                                                              \
  }

#define DISPATCH_Q_TILE_SIZE(                           \
    group_size, max_tokens_per_batch, Q_TILE_SIZE, ...) \
  if (group_size * max_tokens_per_batch <= 16) {        \
    constexpr size_t Q_TILE_SIZE = 16;                  \
    __VA_ARGS__                                         \
  } else {                                              \
    constexpr size_t Q_TILE_SIZE = 32;                  \
    __VA_ARGS__                                         \
  }

#define DISPATCH_CAUSAL(causal, CAUSAL, ...) \
  if (causal) {                              \
    constexpr bool CAUSAL = true;            \
    __VA_ARGS__                              \
  } else {                                   \
    constexpr bool CAUSAL = false;           \
    __VA_ARGS__                              \
  }

#define DISPATCH_BLOCKSHAPE_Q_SYSTEM(              \
    block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, ...) \
  if (block_shape_q <= 16) {                       \
    constexpr size_t BLOCK_SHAPE_Q = 16;           \
    constexpr size_t NUM_WARP_Q = 1;               \
    __VA_ARGS__                                    \
  } else if (block_shape_q <= 32) {                \
    constexpr size_t BLOCK_SHAPE_Q = 32;           \
    constexpr size_t NUM_WARP_Q = 1;               \
    __VA_ARGS__                                    \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...) \
  if (block_size == 64) {                                \
    constexpr size_t BLOCK_SIZE = 64;                    \
    __VA_ARGS__                                          \
  }

#define DISPATCH_DyCfp8(is_dynamic_cfp8, IsDynamicC8, ...) \
  if (is_dynamic_cfp8) {                                   \
    constexpr bool IsDynamicC8 = true;                     \
    __VA_ARGS__                                            \
  } else {                                                 \
    constexpr bool IsDynamicC8 = false;                    \
    __VA_ARGS__                                            \
  }

#define DISPATCH_IS_FP8(is_fp8, IS_FP8, ...) \
  if (causal) {                              \
    constexpr bool IS_FP8 = true;            \
    __VA_ARGS__                              \
  } else {                                   \
    constexpr bool IS_FP8 = false;           \
    __VA_ARGS__                              \
  }

struct AppendAttnMetaData {
  int batch_size;
  int block_size;
  int q_num_heads;
  int kv_num_heads;
  int token_num;
  int head_dims;
  int head_dims_v;
  int max_blocks_per_seq;
  const int* mask_offset = nullptr;
};

template <typename T, typename CacheT>
struct AttentionParams {
  T* __restrict__ qkv;
  CacheT* __restrict__ cache_k;
  CacheT* __restrict__ cache_v;
  T* __restrict__ cache_k_scale;
  T* __restrict__ cache_v_scale;
  int* __restrict__ seq_lens_q;
  int* __restrict__ seq_lens_kv;
  int* __restrict__ block_indices;
  int* __restrict__ num_blocks_ptr;
  int* __restrict__ chunk_size_ptr;
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ block_table;
  int* __restrict__ mask_offset;
  bool* __restrict__ attn_mask;
  T* __restrict__ tmp_o;
  float* __restrict__ tmp_m;
  float* __restrict__ tmp_d;
  int max_model_len;
  int max_kv_len;
  int max_blocks_per_seq;
  float softmax_scale;
  float quant_max_bound;
  float quant_min_bound;
  int num_blocks_x;
  int attn_mask_len;
  bool sliding_window;
  int q_num_heads;
  int kv_num_heads;
  int max_num_chunks;
  int max_tile_q;
  int batch_size;
  int token_num;
  int head_dims;
  int max_tokens_per_batch;
};
