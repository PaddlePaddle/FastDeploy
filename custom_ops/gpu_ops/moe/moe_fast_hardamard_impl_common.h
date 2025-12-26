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

#include <string>
#include <vector>
#include "helper.h"

#define FULL_MASK 0xffffffff

struct uint8 {
  uint4 u;
  uint4 v;
};

template <int BYTES>
struct BytesToType {};

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

template <typename T>
struct nv_type_traits {
  using type = T;
};

template <>
struct nv_type_traits<phi::dtype::float16> {
  using type = half;
};

template <>
struct nv_type_traits<phi::dtype::bfloat16> {
  using type = __nv_bfloat16;
};

template <>
struct nv_type_traits<int8_t> {
  using type = int8_t;
};

#define DISPATCH_SP_logN(logN, kLogN, ...)                              \
  if (logN == 10) {                                                     \
    constexpr int kLogN = 10;                                           \
    __VA_ARGS__                                                         \
  } else if (logN == 9) {                                               \
    constexpr int kLogN = 9;                                            \
    __VA_ARGS__                                                         \
  } else if (logN == 8) {                                               \
    constexpr int kLogN = 8;                                            \
    __VA_ARGS__                                                         \
  } else if (logN == 7) {                                               \
    constexpr int kLogN = 7;                                            \
    __VA_ARGS__                                                         \
  } else {                                                              \
    PADDLE_THROW(                                                       \
        phi::errors::Unimplemented("logN = %d is unsupported!", logN)); \
  }

#define DISPATCH_SP_VS(vec_size, VEC_SIZE, ...)                              \
  if (vec_size == 16) {                                                      \
    constexpr int VEC_SIZE = 16;                                             \
    __VA_ARGS__                                                              \
  } else if (vec_size == 8) {                                                \
    constexpr int VEC_SIZE = 8;                                              \
    __VA_ARGS__                                                              \
  } else if (vec_size == 4) {                                                \
    constexpr int VEC_SIZE = 4;                                              \
    __VA_ARGS__                                                              \
  } else if (vec_size == 2) {                                                \
    constexpr int VEC_SIZE = 2;                                              \
    __VA_ARGS__                                                              \
  } else if (vec_size == 1) {                                                \
    constexpr int VEC_SIZE = 1;                                              \
    __VA_ARGS__                                                              \
  } else {                                                                   \
    PADDLE_THROW(phi::errors::Unimplemented("vec_size = %d is unsupported!", \
                                            vec_size));                      \
  }

#define DISPATCH_logN(logN, kLogN, ...)                           \
  if (logN == 11) {                                               \
    constexpr int kLogN = 11;                                     \
    __VA_ARGS__                                                   \
  } else if (logN == 12) {                                        \
    constexpr int kLogN = 12;                                     \
    __VA_ARGS__                                                   \
  } else if (logN == 13) {                                        \
    constexpr int kLogN = 13;                                     \
    __VA_ARGS__                                                   \
  } else if (logN == 14) {                                        \
    constexpr int kLogN = 14;                                     \
    __VA_ARGS__                                                   \
  } else {                                                        \
    PADDLE_THROW(phi::errors::Unimplemented("unsupported logN")); \
  }

template <typename T,
          typename OutT,
          int kLogN,
          int VecSize,
          int kNChunks,
          int kThreads,
          bool UseDiagonalBlockMatrix>
void MoeFastHardamardImplWrapper(const T *x,
                                 const int64_t *expert_idx_per_token,
                                 const int64_t *recv_expert_count,
                                 const T *shift,
                                 const T *smooth,
                                 const float *quant_scales,
                                 const int quant_round_type,
                                 const float quant_max_bound,
                                 const float quant_min_bound,
                                 const int64_t token_num,
                                 const int64_t dim,
                                 const int num_max_tokens_per_expert,
                                 bool used_in_ep_low_latency,
                                 OutT *out,
                                 cudaStream_t stream);
