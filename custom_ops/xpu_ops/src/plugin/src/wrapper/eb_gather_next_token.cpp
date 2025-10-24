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
#include "xpu/refactor/impl/launch_strategy.h"
#include "xpu/refactor/impl_public/wrapper_check.h"
#include "xpu/xdnn.h"

namespace xpu3 {
namespace plugin {
template <typename TX, typename TY>
__attribute__((global)) void eb_gather_next_token(TX *src,
                                                  TY *dst,
                                                  int *encoder_seqs_lods,
                                                  int *encoder_batch_map,
                                                  int *decoder_batch_map,
                                                  int en_batch,
                                                  int de_batch,
                                                  int64_t copy_size);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {
template <typename TX, typename TY>
static int cpu_wrapper(api::Context *ctx,
                       const TX *x,
                       TY *y,
                       const int *encoder_seqs_lods,
                       const int *encoder_batch_map,
                       const int *decoder_batch_map,
                       int en_batch,
                       int de_batch,
                       int64_t hidden_dim) {
  int ret = 0;
  int encoder_len_total = encoder_seqs_lods[en_batch];
  for (int i = 0; i < en_batch; i++) {
    ret = api::cast<TX, TY>(ctx,
                            x + (encoder_seqs_lods[i + 1] - 1) * hidden_dim,
                            y + encoder_batch_map[i] * hidden_dim,
                            hidden_dim);
    WRAPPER_ASSERT_SUCCESS(ctx, ret);
  }
  for (int i = 0; i < de_batch; i++) {
    ret = api::cast<TX, TY>(ctx,
                            x + (encoder_len_total + i) * hidden_dim,
                            y + decoder_batch_map[i] * hidden_dim,
                            hidden_dim);
    WRAPPER_ASSERT_SUCCESS(ctx, ret);
  }

  return api::SUCCESS;
}
template <typename TX, typename TY>
static int xpu3_wrapper(api::Context *ctx,
                        const TX *x,
                        TY *y,
                        api::VectorParam<int32_t> &encoder_seqs_lods,  // NOLINT
                        api::VectorParam<int32_t> &encoder_batch_map,  // NOLINT
                        api::VectorParam<int32_t> &decoder_batch_map,  // NOLINT
                        int en_batch,
                        int de_batch,
                        int64_t hidden_dim) {
  auto eb_gather_next_token_kernel = xpu3::plugin::eb_gather_next_token<TX, TY>;
  // NOTE: Don't change 16 to 64, because kernel use gsm
  eb_gather_next_token_kernel<<<ctx->ncluster(), 16, ctx->xpu_stream>>>(
      const_cast<TX *>(x),
      y,
      encoder_seqs_lods.xpu,
      encoder_batch_map.xpu,
      decoder_batch_map.xpu,
      en_batch,
      de_batch,
      hidden_dim);
  return api::SUCCESS;
}

template <typename TX, typename TY>
int eb_gather_next_token(
    api::Context *ctx,
    const TX *x,
    TY *y,
    api::VectorParam<int32_t> &encoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t> &encoder_batch_map,  // NOLINT
    api::VectorParam<int32_t> &decoder_batch_map,  // NOLINT
    int64_t hidden_dim) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T2(ctx, "eb_gather_next_token", TX, TY);
  WRAPPER_DUMP_PARAM6(ctx,
                      x,
                      y,
                      encoder_seqs_lods,
                      encoder_batch_map,
                      decoder_batch_map,
                      hidden_dim);
  WRAPPER_DUMP(ctx);
  int encoder_batch = encoder_batch_map.len;
  int batch = encoder_batch + decoder_batch_map.len;
  int max_encoder_lod = encoder_seqs_lods.cpu[encoder_batch];
  int m = encoder_seqs_lods.cpu[encoder_batch] + decoder_batch_map.len;
  WRAPPER_CHECK_PTR(ctx, TX, m * hidden_dim, x);
  WRAPPER_CHECK_PTR(ctx, TY, batch * hidden_dim, y);
  WRAPPER_ASSERT_GT(ctx, hidden_dim, 0);
  // check VectorParam
  WRAPPER_ASSERT_EQ(ctx, encoder_seqs_lods.len, encoder_batch_map.len + 1);
  WRAPPER_ASSERT_GE(ctx, encoder_seqs_lods.cpu[0], 0);
  WRAPPER_ASSERT_LE(ctx, encoder_seqs_lods.cpu[0], max_encoder_lod);
  // 注意: encoder/decoder的batch
  // map数值上有可能大于batch，因为复原后的batch排布有可能是稀疏的，所以这里只做非负检查
  for (int i = 0; i < encoder_batch_map.len; ++i) {
    WRAPPER_ASSERT_GE(ctx, encoder_batch_map.cpu[i], 0);
    WRAPPER_ASSERT_GE(ctx, encoder_seqs_lods.cpu[i + 1], 0);
    WRAPPER_ASSERT_LE(ctx, encoder_seqs_lods.cpu[i + 1], max_encoder_lod);
  }
  for (int i = 0; i < decoder_batch_map.len; ++i) {
    WRAPPER_ASSERT_GE(ctx, decoder_batch_map.cpu[i], 0);
  }
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<TX, TY>(ctx,
                               x,
                               y,
                               encoder_seqs_lods.cpu,
                               encoder_batch_map.cpu,
                               decoder_batch_map.cpu,
                               encoder_batch_map.len,
                               decoder_batch_map.len,
                               hidden_dim);
  }
  if (ctx->dev().type() == api::kXPU3) {
    api::ctx_guard RAII_GUARD(ctx);
    api::VectorParam<int32_t> encoder_seqs_lods_xpu =
        encoder_seqs_lods.to_xpu(RAII_GUARD);
    api::VectorParam<int32_t> encoder_batch_map_xpu =
        encoder_batch_map.to_xpu(RAII_GUARD);
    api::VectorParam<int32_t> decoder_batch_map_xpu =
        decoder_batch_map.to_xpu(RAII_GUARD);
    return xpu3_wrapper<TX, TY>(ctx,
                                x,
                                y,
                                encoder_seqs_lods_xpu,
                                encoder_batch_map_xpu,
                                decoder_batch_map_xpu,
                                encoder_batch_map.len,
                                decoder_batch_map.len,
                                hidden_dim);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}
#define INSTANTIATION_EB_GATHER_NEXT_TOKEN(TX, TY)                       \
  template int eb_gather_next_token<TX, TY>(api::Context *,              \
                                            const TX *,                  \
                                            TY *,                        \
                                            api::VectorParam<int32_t> &, \
                                            api::VectorParam<int32_t> &, \
                                            api::VectorParam<int32_t> &, \
                                            int64_t);

INSTANTIATION_EB_GATHER_NEXT_TOKEN(float16, float16);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(bfloat16, bfloat16);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(float, float);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(float16, float);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(float, float16);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(bfloat16, float16);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(float16, bfloat16);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(bfloat16, float);
INSTANTIATION_EB_GATHER_NEXT_TOKEN(float, bfloat16);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
