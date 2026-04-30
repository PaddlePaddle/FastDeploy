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

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {
template <typename T>
__attribute__((global)) void top_k_renorm_probs(const T* probs,
                                                T* renorm_probs,
                                                const int64_t* top_k,
                                                int batch_size,
                                                int vocab_size);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

template <typename T>
static int cpu_wrapper(api::Context* ctx,
                       const T* probs,
                       T* renorm_probs,
                       const int64_t* top_k,
                       int batch_size,
                       int vocab_size) {
  for (int i = 0; i < batch_size; i++) {
    const T* row = probs + i * vocab_size;
    T* out = renorm_probs + i * vocab_size;
    int k =
        (top_k[i] <= 0 || top_k[i] >= vocab_size) ? vocab_size : (int)top_k[i];
    if (k >= vocab_size) {
      for (int j = 0; j < vocab_size; j++) out[j] = row[j];
      continue;
    }
    // Phase 1: find max_val
    float max_val = 0.f;
    for (int j = 0; j < vocab_size; j++) {
      float p = (float)row[j];
      if (p > max_val) max_val = p;
    }
    // Phase 2: ternary search for pivot (kth largest value)
    double low = 0.0, high = (double)max_val;
    float sum_low = 1.f;
    float min_gt_low, max_le_high;
    do {
      double p0 = (high + 2.0 * low) / 3.0;
      double p1 = (2.0 * high + low) / 3.0;
      float sg0 = 0.f, sg1 = 0.f;
      int cg0 = 0, cg1 = 0;
      min_gt_low = (float)high;
      max_le_high = (float)low;
      for (int j = 0; j < vocab_size; j++) {
        float p = (float)row[j];
        if (p > (float)p0) {
          sg0 += p;
          cg0++;
        }
        if (p > (float)p1) {
          sg1 += p;
          cg1++;
        }
        if (p > (float)low && p < min_gt_low) min_gt_low = p;
        if (p <= (float)high && p > max_le_high) max_le_high = p;
      }
      if (cg1 >= k) {
        low = p1;
        sum_low = sg1;
      } else if (cg0 >= k) {
        low = p0;
        double h_cand = (p1 < (double)max_le_high) ? p1 : (double)max_le_high;
        high = h_cand;
        sum_low = sg0;
      } else {
        double h_cand = (p0 < (double)max_le_high) ? p0 : (double)max_le_high;
        high = h_cand;
      }
    } while (min_gt_low != max_le_high);
    float normalizer = 1.f / (sum_low > 1e-8f ? sum_low : 1e-8f);
    float pivot = (float)low;
    // Phase 3: write output
    for (int j = 0; j < vocab_size; j++) {
      float p = (float)row[j];
      out[j] = (T)(p > pivot ? p * normalizer : 0.f);
    }
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(api::Context* ctx,
                        const T* probs,
                        T* renorm_probs,
                        const int64_t* top_k,
                        int batch_size,
                        int vocab_size) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      fd_xpu3::top_k_renorm_probs<T><<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          probs,
          renorm_probs,
          reinterpret_cast<const XPU_INT64*>(top_k),
          batch_size,
          vocab_size);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

template <typename T>
int top_k_renorm_probs(api::Context* ctx,
                       const T* probs,
                       T* renorm_probs,
                       const int64_t* top_k,
                       int batch_size,
                       int vocab_size) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "top_k_renorm_probs", T);
  WRAPPER_DUMP_PARAM5(ctx, probs, renorm_probs, top_k, batch_size, vocab_size);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, batch_size, 0);
  WRAPPER_ASSERT_GT(ctx, vocab_size, 0);
  WRAPPER_CHECK_PTR(ctx, T, batch_size * vocab_size, probs);
  WRAPPER_CHECK_PTR(ctx, T, batch_size * vocab_size, renorm_probs);
  WRAPPER_CHECK_PTR(ctx, int64_t, batch_size, top_k);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(
        ctx, probs, renorm_probs, top_k, batch_size, vocab_size);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(
        ctx, probs, renorm_probs, top_k, batch_size, vocab_size);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int top_k_renorm_probs<float>(
    api::Context*, const float*, float*, const int64_t*, int, int);
template int top_k_renorm_probs<float16>(
    api::Context*, const float16*, float16*, const int64_t*, int, int);
template int top_k_renorm_probs<bfloat16>(
    api::Context*, const bfloat16*, bfloat16*, const int64_t*, int, int);

}  // namespace plugin
}  // namespace fastdeploy
