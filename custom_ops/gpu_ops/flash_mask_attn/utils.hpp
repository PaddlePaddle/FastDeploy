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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <iostream>

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename T>
struct PackedHalf;

template <>
struct PackedHalf<cutlass::half_t> {
  using Type = __half2;
};

template <>
struct PackedHalf<cutlass::bfloat16_t> {
  using Type = nv_bfloat162;
};

template <typename T>
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

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<64> {
  using Type = uint16;
  static_assert(sizeof(Type) == 64);
};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <typename Elt_type, uint32_t NUM_ELT>
struct Vec {
  enum { BYTES = NUM_ELT * sizeof(Elt_type) };

  using Vec_type = typename BytesToType<BYTES>::Type;

  using Alias_type = union {
    Vec_type vec;
    Elt_type elt[NUM_ELT];
  };

  Alias_type data;

  inline __device__ Vec() {}

  template <typename S>
  inline __device__ void to(Vec<S, NUM_ELT> &other) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      other.data.elt[it] = S(this->data.elt[it]);
    }
  }

  template <typename Op>
  inline __device__ void assign(const Op &op) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
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

  inline __device__ void fma(const Vec<Elt_type, NUM_ELT> &scale,
                             const Vec<Elt_type, NUM_ELT> &bias) {
    static_assert(NUM_ELT % 2 == 0);
    using type = typename PackedHalf<Elt_type>::Type;
#pragma unroll
    for (int it = 0; it < NUM_ELT / 2; it++) {
      type a = *reinterpret_cast<const type *>(scale.data.elt + it * 2);
      type b = *reinterpret_cast<const type *>(bias.data.elt + it * 2);
      *reinterpret_cast<type *>(this->data.elt + it * 2) += a * b;
    }
  }

  inline __device__ void set_zero() {
    constexpr int size = sizeof(Vec_type) / sizeof(int);
#pragma unroll
    for (int i = 0; i < size; ++i) {
      (reinterpret_cast<int *>(this->data.elt))[i] = 0;
    }
  }
};

template <typename T, int PackSize>
inline __device__ void apply_rotary_embedding(Vec<T, PackSize> &vec,
                                              Vec<float, PackSize / 2> &cos,
                                              Vec<float, PackSize / 2> &sin) {
  static_assert(PackSize % 2 == 0);
#pragma unroll
  for (int i = 0; i < PackSize / 2; i++) {
    const float cos_inv_freq = cos.data.elt[i];
    const float sin_inv_freq = sin.data.elt[i];
    const float v1 = static_cast<float>(vec.data.elt[2 * i]);
    const float v2 = static_cast<float>(vec.data.elt[2 * i + 1]);
    vec.data.elt[2 * i] = static_cast<T>(cos_inv_freq * v1 - sin_inv_freq * v2);
    vec.data.elt[2 * i + 1] =
        static_cast<T>(sin_inv_freq * v1 + cos_inv_freq * v2);
  }
}

template <typename Tensor>
__forceinline__ __device__ void app_mask(Tensor &tSrS,
                                         const int *mask,
                                         const int &mask_row_id,
                                         const int &col_base) {
  const float mask_value = -1000000.0f;
  for (int i = 0; i < size(tSrS); i += 8) {
    const int col = i * 2 + col_base;
    if (col >= mask[mask_row_id]) {
      tSrS(i) = mask_value;
    }
    if (col + 1 >= mask[mask_row_id]) {
      tSrS(i + 1) = mask_value;
    }
    if (col >= mask[mask_row_id + 8]) {
      tSrS(i + 2) = mask_value;
    }
    if (col + 1 >= mask[mask_row_id + 8]) {
      tSrS(i + 3) = mask_value;
    }
    if (col + 8 >= mask[mask_row_id]) {
      tSrS(i + 4) = mask_value;
    }
    if (col + 9 >= mask[mask_row_id]) {
      tSrS(i + 5) = mask_value;
    }
    if (col + 8 >= mask[mask_row_id + 8]) {
      tSrS(i + 6) = mask_value;
    }
    if (col + 9 >= mask[mask_row_id + 8]) {
      tSrS(i + 7) = mask_value;
    }
  }
}

template <typename T>
struct HalfMax;
template <>
struct HalfMax<cutlass::half_t> {
  inline __device__ __half2 operator()(const __half2 x, const __half2 y) {
    __half2 res;
    asm volatile("max.f16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <>
struct HalfMax<cutlass::bfloat16_t> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162 x,
                                            const nv_bfloat162 y) {
    nv_bfloat162 res;
    asm volatile("max.bf16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <typename T>
struct HalfMin;
template <>
struct HalfMin<cutlass::half_t> {
  inline __device__ __half2 operator()(const __half2 x, const __half2 y) {
    __half2 res;
    asm volatile("min.f16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <>
struct HalfMin<cutlass::bfloat16_t> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162 x,
                                            const nv_bfloat162 y) {
    nv_bfloat162 res;
    asm volatile("min.bf16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <bool Is_even_MN = true,
          typename TiledCopy,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1,
          typename Engine2,
          typename Layout2>
__forceinline__ __device__ void copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &S,
    Tensor<Engine1, Layout1> &D,
    Tensor<Engine2, Layout2> const &identity_MN,
    const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
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

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename T, typename ReductionOp, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp());
  if (threadIdx.x == 0) {
    result_broadcast = result;
  }
  __syncthreads();
  return result_broadcast;
}

template <typename T, int block_size>
__inline__ __device__ T BlockScanSum(T val) {
  typedef cub::BlockScan<int, block_size> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  int aggregate;
  BlockScanT(temp_storage).ExclusiveSum(val, val, aggregate);
  __syncthreads();
  return val;
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const &x, float const &y) {
    return max(x, y);
  }
};

template <typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x < y ? x : y;
  }
};

template <>
struct MinOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const &x, float const &y) {
    return min(x, y);
  }
};

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x + y;
  }
};

template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    auto l = logical_divide(get<0>(acc_layout),
                            Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
    return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)),
                       get<1>(acc_layout),
                       make_layout(get<2, 1>(l), get<2>(acc_layout)));
  } else {  // SM80
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
      return acc_layout;
    } else {
      auto l = logical_divide(
          acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
      return make_layout(
          make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
  }
};

template <bool zero_init = false,
          int wg_wait = 0,
          bool arrive = true,
          bool commit = true,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma,
                                     Tensor0 const &tCrA,
                                     Tensor1 const &tCrB,
                                     Tensor2 &tCrC) {
  constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator,
                                           typename TiledMma::FrgTypeA>::value;
  // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take
  // const
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (arrive) {
    warpgroup_arrive();
  }
  if constexpr (zero_init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  }
  if constexpr (commit) {
    warpgroup_commit_batch();
  }
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
  }
}

template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
  } else {  // SM80
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
  }
};

template <typename T, typename ReductionOp, int thread_group_width = 32>
__inline__ __device__ T WarpAllReduce(T val) {
  ReductionOp op;
#pragma unroll
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = op(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}
