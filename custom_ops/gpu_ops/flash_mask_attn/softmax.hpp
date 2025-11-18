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

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.hpp"

using namespace cute;

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

template <bool zero_init = true,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1,
          typename Operator>
__device__ __forceinline__ void thread_reduce_(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &summary,
    Operator &op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template <typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1,
          typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst,
                                                Tensor<Engine1, Layout1> &src,
                                                Operator &op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template <bool zero_init = true,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1,
          typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const &tensor,
                                        Tensor<Engine1, Layout1> &summary,
                                        Operator &op) {
  thread_reduce_<zero_init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}

template <bool zero_init = true,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1>
__device__ __forceinline__ void reduce_max(
    Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &max) {
  MaxOp<float> max_op;
  reduce_<zero_init>(tensor, max, max_op);
}

template <bool zero_init = true,
          bool warp_reduce = true,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1>
__device__ __forceinline__ void reduce_sum(
    Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &sum) {
  SumOp<float> sum_op;
  thread_reduce_<zero_init>(tensor, sum, sum_op);
  if constexpr (warp_reduce) {
    quad_allreduce_(sum, sum, sum_op);
  }
}

__forceinline__ __device__ __half2 half_exp(__half2 x) {
  uint32_t tmp_out, tmp_in;
  tmp_in = reinterpret_cast<uint32_t &>(x);
  asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(tmp_out) : "r"(tmp_in));
  __half2 out = reinterpret_cast<__half2 &>(tmp_out);
  return out;
}

// Apply the exp to all the elements.
template <bool zero_init = false,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> &max,
    Tensor<Engine1, Layout1> &sum,
    const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    MaxOp<float> max_op;
    max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      max(mi) = max_op(max(mi), tensor(mi, ni));
    }
    max(mi) = Allreduce<4>::run(max(mi), max_op);
    const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
    sum(mi) = 0;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
      sum(mi) += tensor(mi, ni);
    }
  }
}

template <typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max,
    const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled = max(mi) * scale;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

template <int kNRows>
struct Softmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;

  CUTLASS_DEVICE Softmax(){};

  template <bool Is_first, bool Check_inf = false, typename Tensor0>
  __forceinline__ __device__ TensorT max(Tensor0 &acc_s,
                                         float softmax_scale_log2) {
    Tensor scores =
        make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);
    TensorT scores_scale;
    if constexpr (Is_first) {
      reduce_max</*zero_init=*/true>(scores, row_max);
      cute::fill(scores_scale, 1.f);
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      reduce_max</*zero_init=*/false>(scores, row_max);
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = row_max(mi);
        scores_scale(mi) =
            exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
    }
    return scores_scale;
  };

  template <bool Is_first, typename Tensor0>
  __forceinline__ __device__ TensorT online_softmax(Tensor0 &acc_s,
                                                    float softmax_scale_log2) {
    Tensor scores =
        make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);
    TensorT scores_scale;
    if constexpr (Is_first) {
      reduce_max</*zero_init=*/true>(scores, row_max);
      scale_apply_exp2(scores, row_max, softmax_scale_log2);
      reduce_sum</*zero_init=*/true, /*warp_reduce=*/false>(scores, row_sum);
      cute::fill(scores_scale, 1.f);
    } else {
      scale_apply_exp2(scores, row_max, softmax_scale_log2);
      reduce_sum</*zero_init=*/false, /*warp_reduce=*/false>(scores, row_sum);
    }
    return scores_scale;
  };

  __forceinline__ __device__ TensorT finalize(float softmax_scale_log2) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT scores_scale;
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
      float sum = row_sum(mi);
      float inv_sum = 1.0f / sum;
      row_sum(mi) =
          row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
      scores_scale(mi) = inv_sum;
    }
    return scores_scale;
  };

  template <typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1 &acc_o,
                                            TensorT const &scores_scale) {
    Tensor acc_o_rowcol =
        make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale(mi);
      }
    }
  };
};
