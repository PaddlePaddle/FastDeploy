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
#include "xpu/refactor/core/quant.h"
#include "xpu/refactor/impl/xdnn_impl.h"
#include "xpu/xdnn.h"

namespace xpu3 {
namespace plugin {
template <typename TX, typename TSCALE, typename TY>
__attribute__((global)) void quant2d_per_channel_cluster(
    const TX *x, const TSCALE *scale, TY *y, int64_t m, int64_t n);

template <typename TX, typename TSCALE, typename TY, int MAX_N>
__attribute__((global)) void quant2d_per_channel_cached(
    const TX *input, TY *output, TSCALE *scale, int64_t m, int64_t n);

template <typename TX, typename TSCALE, typename TY>
__attribute__((global)) void quant2d_per_channel_bign(
    const TX *input, TY *output, TSCALE *scale, int64_t m, int64_t n);
}  // namespace plugin
}  // namespace xpu3

namespace api = baidu::xpu::api;

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename TX,
          typename TSCALE,
          typename TY,
          typename std::enable_if<std::is_same<TY, int8_t>::value, TY>::type
              *ptr = nullptr>
int cpu_wrapper_input_scale(api::Context *ctx,
                            const TX *x,
                            const TSCALE *scale,
                            TY *y,
                            int64_t m,
                            int64_t n) {
  float absmax = 1e-30f;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      absmax = static_cast<float>(scale[j]);
      y[i * n + j] = baidu::xpu::api::__fp32_to_intx<int8_t, 127>(
          static_cast<float>(x[i * n + j]), absmax);
    }
  }
  return api::SUCCESS;
}

static inline int64_t __round_half_to_even(const float16 src) {
  int64_t ret = llround(src);
  if (fabs(fabs(std::round(src) - src) - 0.5) > 0) {
    return ret;
  } else {
    if (std::abs(ret) % 2 == 0) {
      return ret;
    } else {
      return ret + (ret > 0 ? -1 : 1);
    }
  }
}

static float16 quant_int4(float x, float scale) {
  auto r = x * scale;
  r = std::max(static_cast<float>(r), -8.f);
  return (float16)std::min(static_cast<float>(r), 7.f);
}

template <typename TX,
          typename TSCALE,
          typename TY,
          typename std::enable_if<std::is_same<TY, int4_t>::value, TY>::type
              *ptr = nullptr>
int cpu_wrapper_input_scale(
    api::Context *ctx, const TX *x, const TSCALE *scale, TY *y, int m, int n) {
  int8_t *y_ptr = reinterpret_cast<int8_t *>(y);
  float t1, t2;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j += 2) {
      float absmax = static_cast<float>(scale[j]);
      float act_scale = 7.0 / absmax;
      t1 = static_cast<float>(x[i * n + j]);
      y_ptr[i * ((n - 1) / 2 + 1) + j / 2] =
          (int8_t)(__round_half_to_even(quant_int4(t1, act_scale)) & 0x0f);
      if (j + 1 < n) {
        absmax = static_cast<float>(scale[j + 1]);
        act_scale = 7.0 / absmax;
        t2 = static_cast<float>(x[i * n + j + 1]);
        y_ptr[i * ((n - 1) / 2 + 1) + j / 2] =
            (int8_t)(__round_half_to_even(quant_int4(t2, act_scale)) << 4) |
            y_ptr[i * ((n - 1) / 2 + 1) + j / 2];
      } else {
        y_ptr[i * ((n - 1) / 2 + 1) + j / 2] =
            (int8_t)(__round_half_to_even(quant_int4(0.f, act_scale)) << 4) |
            y_ptr[i * ((n - 1) / 2 + 1) + j / 2];
      }
    }
  }
  return api::SUCCESS;
}

template <typename TX,
          typename TSCALE,
          typename TY,
          typename std::enable_if<!std::is_same<TY, int4_t>::value, TY>::type
              *ptr = nullptr>
int cpu_wrapper_output_scale(api::Context *ctx,
                             const TX *x,
                             TSCALE *scale,
                             TY *y,
                             int64_t m,
                             int64_t n) {
  int64_t i, j;
  for (j = 0; j < n; ++j) {
    float absmax = 1e-30f;
    for (i = 0; i < m; ++i) {
      absmax = std::max<float>(absmax, fabs(static_cast<float>(x[i * n + j])));
    }
    scale[j] = absmax;
    for (i = 0; i < m; ++i) {
      y[i * n + j] = baidu::xpu::api::__fp32_to_intx<int8_t, 127>(
          static_cast<float>(x[i * n + j]), absmax);
    }
  }
  return api::SUCCESS;
}

template <typename TX,
          typename TSCALE,
          typename TY,
          typename std::enable_if<std::is_same<TY, int4_t>::value, TY>::type
              *ptr = nullptr>
int cpu_wrapper_output_scale(
    api::Context *ctx, const TX *x, TSCALE *scale, TY *y, int m, int n) {
  int8_t *y_ptr = reinterpret_cast<int8_t *>(y);
  float t1, t2, absmax_1, absmax_2, act_scale_1, act_scale_2;
  for (int j = 0; j < n; j += 2) {
    absmax_1 = 1e-30f;
    absmax_2 = 1e-30f;
    for (int i = 0; i < m; ++i) {
      absmax_1 =
          std::max<float>(absmax_1, fabs(static_cast<float>(x[i * n + j])));
    }
    scale[j] = static_cast<float>(absmax_1);
    if (j + 1 < n) {
      for (int i = 0; i < m; ++i) {
        absmax_2 = std::max<float>(absmax_2,
                                   fabs(static_cast<float>(x[i * n + j + 1])));
      }
      scale[j + 1] = static_cast<float>(absmax_2);
    }
    act_scale_1 = 7.0 / absmax_1;
    act_scale_2 = 7.0 / absmax_2;
    for (int i = 0; i < m; i++) {
      t1 = static_cast<float>(x[i * n + j]);
      y_ptr[i * ((n + 1) / 2) + j / 2] =
          (int8_t)(__round_half_to_even(quant_int4(t1, act_scale_1)) & 0x0f);
      if (j + 1 < n) {
        t2 = static_cast<float>(x[i * n + j + 1]);
        y_ptr[i * ((n - 1) / 2 + 1) + j / 2] =
            (int8_t)(__round_half_to_even(quant_int4(t2, act_scale_2)) << 4) |
            y_ptr[i * ((n - 1) / 2 + 1) + j / 2];
      } else {
        y_ptr[i * ((n - 1) / 2 + 1) + j / 2] =
            (int8_t)(__round_half_to_even(quant_int4(0.f, act_scale_2)) << 4) |
            y_ptr[i * ((n - 1) / 2 + 1) + j / 2];
      }
    }
  }
  return api::SUCCESS;
}

template <typename TX, typename TSCALE, typename TY>
int xpu3_wrapper_input_scale(api::Context *ctx,
                             const TX *x,
                             const TSCALE *scale,
                             TY *y,
                             int64_t m,
                             int64_t n) {
  auto func = xpu3::plugin::quant2d_per_channel_cluster<TX, TSCALE, TY>;
  func<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, scale, y, m, n);
  return api::SUCCESS;
}

template <typename TX,
          typename TSCALE,
          typename TY,
          typename std::enable_if<!std::is_same<TY, int4_t>::value, TY>::type
              * = nullptr>
int xpu3_wrapper_output_scale(api::Context *ctx,
                              const TX *x,
                              TSCALE *scale,
                              TY *y,
                              int64_t m,
                              int64_t n) {
  int64_t channel_size = m * sizeof(TX);
  int64_t cluster_n = (n + ctx->ncluster() - 1) / ctx->ncluster();
  auto func = xpu3::plugin::quant2d_per_channel_bign<TX, TSCALE, TY>;
  if (n < 1536) {
    if (channel_size <= 2048) {
      if (cluster_n <= 64) {
        func = xpu3::plugin::quant2d_per_channel_cached<TX, TSCALE, TY, 64>;
      } else if (cluster_n <= 32) {
        func = xpu3::plugin::quant2d_per_channel_cached<TX, TSCALE, TY, 32>;
      } else {
        func = xpu3::plugin::quant2d_per_channel_cached<TX, TSCALE, TY, 128>;
      }
    } else if (channel_size <= 4096) {
      if (cluster_n <= 32) {
        func = xpu3::plugin::quant2d_per_channel_cached<TX, TSCALE, TY, 32>;
      } else {
        func = xpu3::plugin::quant2d_per_channel_cached<TX, TSCALE, TY, 64>;
      }
    } else if (channel_size <= 8192) {
      func = xpu3::plugin::quant2d_per_channel_cached<TX, TSCALE, TY, 32>;
    }
  }
  func<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, scale, m, n);
  return api::SUCCESS;
}
template <typename TX,
          typename TSCALE,
          typename TY,
          typename std::enable_if<std::is_same<TY, int4_t>::value, TY>::type * =
              nullptr>
int xpu3_wrapper_output_scale(api::Context *ctx,
                              const TX *x,
                              TSCALE *scale,
                              TY *y,
                              int64_t m,
                              int64_t n) {
  auto func = xpu3::plugin::quant2d_per_channel_bign<TX, TSCALE, TY>;
  func<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, scale, m, n);
  return api::SUCCESS;
}

template <typename TX, typename TSCALE, typename TY>
int quant2d_per_channel(api::Context *ctx,
                        const TX *x,
                        const TSCALE *scale_in,
                        TY *y,
                        TSCALE *scale_out,
                        int64_t m,
                        int64_t n) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T3(ctx, "quant2d_per_channel", TX, TSCALE, TY);
  WRAPPER_DUMP_PARAM4(ctx, x, scale_in, y, scale_out);
  WRAPPER_DUMP_PARAM2(ctx, m, n);
  WRAPPER_DUMP(ctx);
  // shape check
  WRAPPER_ASSERT_GT(ctx, m, 0);
  WRAPPER_ASSERT_GT(ctx, n, 0);
  // for now, hidden_size should be less then 32768(128KB) for sm usage.
  WRAPPER_ASSERT_LT(ctx, n, 32768);
  // Dump & check input/scale/output data.
  WRAPPER_CHECK_PTR(ctx, TX, m * n, x);
  if (scale_in != nullptr) {
    WRAPPER_CHECK_PTR(ctx, TSCALE, n, scale_in);
  } else {
    WRAPPER_CHECK_PTR(ctx, TSCALE, n, scale_out);
  }
  WRAPPER_CHECK_PTR(ctx, TY, m * n, y);

  if (ctx->dev().type() == api::kCPU) {
    // Add cpu wrapper
    if (scale_in != nullptr) {
      return cpu_wrapper_input_scale<TX, TSCALE, TY>(ctx, x, scale_in, y, m, n);
    }
    return cpu_wrapper_output_scale<TX, TSCALE, TY>(ctx, x, scale_out, y, m, n);
  }
  if (ctx->dev().type() == api::kXPU3) {
    if (scale_in != nullptr) {
      return xpu3_wrapper_input_scale<TX, TSCALE, TY>(
          ctx, x, scale_in, y, m, n);
    }
    return xpu3_wrapper_output_scale<TX, TSCALE, TY>(
        ctx, x, scale_out, y, m, n);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
  return 0;
}

#define INSTANTIATION_QUANT2D_PER_CHANNEL(TX, TSCALE, TY)          \
  template int quant2d_per_channel<TX, TSCALE, TY>(api::Context *, \
                                                   const TX *,     \
                                                   const TSCALE *, \
                                                   TY *,           \
                                                   TSCALE *,       \
                                                   int64_t,        \
                                                   int64_t);

INSTANTIATION_QUANT2D_PER_CHANNEL(float16, float, int8_t);
INSTANTIATION_QUANT2D_PER_CHANNEL(bfloat16, float, int8_t);
INSTANTIATION_QUANT2D_PER_CHANNEL(float, float, int8_t);
// further support fp32 input/scale
INSTANTIATION_QUANT2D_PER_CHANNEL(float16, float16, int4_t);
INSTANTIATION_QUANT2D_PER_CHANNEL(float16, float, int4_t);
INSTANTIATION_QUANT2D_PER_CHANNEL(float, float, int4_t);
INSTANTIATION_QUANT2D_PER_CHANNEL(bfloat16, float, int4_t);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
