// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
__attribute__((global)) void build_sampling_params_kernel(
    float* top_p_padding,
    int64_t* top_k_padding,
    int64_t* topp_seed,
    const float* top_p,
    const int64_t* top_k,
    int64_t* infer_seed,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    int bs,
    int64_t token_num,
    int64_t increment_value);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

constexpr int64_t BUILD_SAMPLING_MAX_INFER_SEED = 2147483646LL;

static int cpu_wrapper(api::Context* ctx,
                       float* top_p_padding,
                       int64_t* top_k_padding,
                       int64_t* topp_seed,
                       const float* top_p,
                       const int64_t* top_k,
                       int64_t* infer_seed,
                       const int* seq_lens_this_time,
                       const int* seq_lens_encoder,
                       int bs,
                       int64_t token_num,
                       int64_t increment_value) {
  int64_t pad_idx = 0;
  for (int bi = 0; bi < bs; bi++) {
    bool is_decoder = (seq_lens_encoder[bi] == 0);
    int repeat = is_decoder ? seq_lens_this_time[bi] : 1;
    int64_t bi_seed = infer_seed[bi];
    for (int local_pos = 0; local_pos < repeat; local_pos++) {
      int64_t offset = is_decoder ? static_cast<int64_t>(local_pos) * 4 : 0LL;
      top_p_padding[pad_idx] = top_p[bi];
      top_k_padding[pad_idx] = top_k[bi];
      topp_seed[pad_idx] = (bi_seed + offset) % BUILD_SAMPLING_MAX_INFER_SEED;
      pad_idx++;
    }
    infer_seed[bi] =
        (infer_seed[bi] + increment_value) % BUILD_SAMPLING_MAX_INFER_SEED;
  }
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        float* top_p_padding,
                        int64_t* top_k_padding,
                        int64_t* topp_seed,
                        const float* top_p,
                        const int64_t* top_k,
                        int64_t* infer_seed,
                        const int* seq_lens_this_time,
                        const int* seq_lens_encoder,
                        int bs,
                        int64_t token_num,
                        int64_t increment_value) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre = fd_xpu3::
      build_sampling_params_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          top_p_padding,
          reinterpret_cast<XPU_INT64*>(top_k_padding),
          reinterpret_cast<XPU_INT64*>(topp_seed),
          top_p,
          reinterpret_cast<const XPU_INT64*>(top_k),
          reinterpret_cast<XPU_INT64*>(infer_seed),
          seq_lens_this_time,
          seq_lens_encoder,
          bs,
          token_num,
          increment_value);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int build_sampling_params(api::Context* ctx,
                          float* top_p_padding,
                          int64_t* top_k_padding,
                          int64_t* topp_seed,
                          const float* top_p,
                          const int64_t* top_k,
                          int64_t* infer_seed,
                          const int* seq_lens_this_time,
                          const int* seq_lens_encoder,
                          int bs,
                          int64_t token_num,
                          int64_t increment_value) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "build_sampling_params", float);
  WRAPPER_DUMP_PARAM5(
      ctx, top_p_padding, top_k_padding, topp_seed, top_p, top_k);
  WRAPPER_DUMP_PARAM5(
      ctx, infer_seed, seq_lens_this_time, seq_lens_encoder, bs, token_num);
  WRAPPER_DUMP_PARAM1(ctx, increment_value);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, float, token_num, top_p_padding);
  WRAPPER_CHECK_PTR(ctx, int64_t, token_num, top_k_padding);
  WRAPPER_CHECK_PTR(ctx, int64_t, token_num, topp_seed);
  WRAPPER_CHECK_PTR(ctx, float, bs, top_p);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs, top_k);
  WRAPPER_CHECK_PTR(ctx, int64_t, bs, infer_seed);
  WRAPPER_CHECK_PTR(ctx, int, bs, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bs, seq_lens_encoder);

  WRAPPER_ASSERT_GT(ctx, bs, 0);
  WRAPPER_ASSERT_GT(ctx, token_num, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       top_p_padding,
                       top_k_padding,
                       topp_seed,
                       top_p,
                       top_k,
                       infer_seed,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       bs,
                       token_num,
                       increment_value);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        top_p_padding,
                        top_k_padding,
                        topp_seed,
                        top_p,
                        top_k,
                        infer_seed,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        bs,
                        token_num,
                        increment_value);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
