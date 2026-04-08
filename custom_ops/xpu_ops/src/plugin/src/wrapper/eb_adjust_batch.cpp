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
__attribute__((global)) void eb_adjust_batch(TX *src,
                                             TY *dst,
                                             int *encoder_seqs_lods,
                                             int *decoder_seqs_lods,
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
                       const int *decoder_seqs_lods,
                       const int *encoder_batch_map,
                       const int *decoder_batch_map,
                       int en_batch,
                       int de_batch,
                       int64_t hidden_dim) {
  int ret = 0;
  int cur_offset = 0;
  int en_idx = 0;
  int de_idx = 0;
  int cur_bs = en_batch + de_batch;
  int encoder_len_total = encoder_seqs_lods[en_batch];
  for (int i = 0; i < cur_bs; i++) {
    // get copy size && src_offset
    int cpy_m = 0;
    if (de_batch > 0 && decoder_batch_map[de_idx] == i) {
      cpy_m = decoder_seqs_lods[de_idx + 1] - decoder_seqs_lods[de_idx];
      ret = api::cast<TX, TY>(
          ctx,
          x + cur_offset * hidden_dim,
          y + (encoder_len_total + decoder_seqs_lods[de_idx]) * hidden_dim,
          cpy_m * hidden_dim);
      WRAPPER_ASSERT_SUCCESS(ctx, ret);
      de_idx++;
    }
    if (en_batch > 0 && encoder_batch_map[en_idx] == i) {
      cpy_m = encoder_seqs_lods[en_idx + 1] - encoder_seqs_lods[en_idx];
      ret = api::cast<TX, TY>(ctx,
                              x + cur_offset * hidden_dim,
                              y + encoder_seqs_lods[en_idx] * hidden_dim,
                              cpy_m * hidden_dim);
      WRAPPER_ASSERT_SUCCESS(ctx, ret);
      en_idx++;
    }
    cur_offset += cpy_m;
  }
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  return api::SUCCESS;
}

template <typename TX, typename TY>
static int xpu3_wrapper(api::Context *ctx,
                        const TX *x,
                        TY *y,
                        api::VectorParam<int32_t> &encoder_seqs_lods,  // NOLINT
                        api::VectorParam<int32_t> &decoder_seqs_lods,  // NOLINT
                        api::VectorParam<int32_t> &encoder_batch_map,  // NOLINT
                        api::VectorParam<int32_t> &decoder_batch_map,  // NOLINT
                        int en_batch,
                        int de_batch,
                        int64_t hidden_dim) {
  using XPU_INDEX_TYPE_TX = typename XPUIndexType<TX>::type;
  using XPU_INDEX_TYPE_TY = typename XPUIndexType<TY>::type;
  auto eb_adjust_batch_kernel =
      xpu3::plugin::eb_adjust_batch<XPU_INDEX_TYPE_TX, XPU_INDEX_TYPE_TY>;
  // NOTE: Don't change 16 to 64, because kernel use gsm
  eb_adjust_batch_kernel<<<ctx->ncluster(), 16, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INDEX_TYPE_TX *>(const_cast<TX *>(x)),
      reinterpret_cast<XPU_INDEX_TYPE_TY *>(y),
      encoder_seqs_lods.xpu,
      decoder_seqs_lods.xpu,
      encoder_batch_map.xpu,
      decoder_batch_map.xpu,
      en_batch,
      de_batch,
      hidden_dim);
  return api::SUCCESS;
}

template <typename TX, typename TY>
int eb_adjust_batch(api::Context *ctx,
                    const TX *x,
                    TY *y,
                    api::VectorParam<int32_t> &encoder_seqs_lods,  // NOLINT
                    api::VectorParam<int32_t> &decoder_seqs_lods,  // NOLINT
                    api::VectorParam<int32_t> &encoder_batch_map,  // NOLINT
                    api::VectorParam<int32_t> &decoder_batch_map,  // NOLINT
                    int64_t hidden_dim) {
  // int dev_id = -1;
  // xpu_current_device(&dev_id);
  // if (dev_id ==0) {
  //     ctx->set_debug_level(0xA1);
  // }
  // std::cout << decoder_seqs_lods.cpu[0] << " " << decoder_seqs_lods.cpu[1] <<
  // std::endl;
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T2(ctx, "eb_adjust_batch", TX, TY);
  WRAPPER_DUMP_PARAM6(ctx,
                      x,
                      y,
                      encoder_seqs_lods,
                      decoder_seqs_lods,
                      encoder_batch_map,
                      decoder_batch_map);
  WRAPPER_DUMP_PARAM1(ctx, hidden_dim);
  WRAPPER_DUMP(ctx);
  int encoder_batch = encoder_batch_map.len;
  int decoder_batch = decoder_batch_map.len;
  int total_batch = encoder_batch + decoder_batch;
  int max_encoder_lod = encoder_seqs_lods.cpu[encoder_batch];
  int max_decoder_lod = decoder_seqs_lods.cpu[decoder_batch];
  int m = max_encoder_lod + max_decoder_lod;
  WRAPPER_CHECK_PTR(ctx, TX, m * hidden_dim, x);
  WRAPPER_CHECK_PTR(ctx, TY, m * hidden_dim, y);
  WRAPPER_ASSERT_GT(ctx, hidden_dim, 0);
  // check VectorParam
  WRAPPER_ASSERT_EQ(ctx, encoder_seqs_lods.len, encoder_batch_map.len + 1);
  WRAPPER_ASSERT_EQ(ctx, decoder_seqs_lods.len, decoder_batch_map.len + 1);
  WRAPPER_ASSERT_GE(ctx, encoder_seqs_lods.cpu[0], 0);
  WRAPPER_ASSERT_LE(ctx, encoder_seqs_lods.cpu[0], max_encoder_lod);
  WRAPPER_ASSERT_GE(ctx, decoder_seqs_lods.cpu[0], 0);
  WRAPPER_ASSERT_LE(ctx, decoder_seqs_lods.cpu[0], max_decoder_lod);
  for (int i = 0; i < encoder_batch_map.len; ++i) {
    WRAPPER_ASSERT_GE(ctx, encoder_batch_map.cpu[i], 0);
    WRAPPER_ASSERT_LT(ctx, encoder_batch_map.cpu[i], total_batch)
    WRAPPER_ASSERT_GE(ctx, encoder_seqs_lods.cpu[i + 1], 0);
    WRAPPER_ASSERT_LE(ctx, encoder_seqs_lods.cpu[i + 1], max_encoder_lod);
  }
  for (int i = 0; i < decoder_batch_map.len; ++i) {
    WRAPPER_ASSERT_GE(ctx, decoder_batch_map.cpu[i], 0);
    WRAPPER_ASSERT_LT(ctx, decoder_batch_map.cpu[i], total_batch)
    WRAPPER_ASSERT_GE(ctx, decoder_seqs_lods.cpu[i + 1], 0);
    WRAPPER_ASSERT_LE(ctx, decoder_seqs_lods.cpu[i + 1], max_decoder_lod);
  }
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<TX, TY>(ctx,
                               x,
                               y,
                               encoder_seqs_lods.cpu,
                               decoder_seqs_lods.cpu,
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
    api::VectorParam<int32_t> decoder_seqs_lods_xpu =
        decoder_seqs_lods.to_xpu(RAII_GUARD);
    api::VectorParam<int32_t> encoder_batch_map_xpu =
        encoder_batch_map.to_xpu(RAII_GUARD);
    api::VectorParam<int32_t> decoder_batch_map_xpu =
        decoder_batch_map.to_xpu(RAII_GUARD);
    return xpu3_wrapper<TX, TY>(ctx,
                                x,
                                y,
                                encoder_seqs_lods_xpu,
                                decoder_seqs_lods_xpu,
                                encoder_batch_map_xpu,
                                decoder_batch_map_xpu,
                                encoder_batch_map.len,
                                decoder_batch_map.len,
                                hidden_dim);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

#define INSTANTIATION_EB_ADJUST_BATCH(TX, TY)                       \
  template int eb_adjust_batch<TX, TY>(api::Context *,              \
                                       const TX *,                  \
                                       TY *,                        \
                                       api::VectorParam<int32_t> &, \
                                       api::VectorParam<int32_t> &, \
                                       api::VectorParam<int32_t> &, \
                                       api::VectorParam<int32_t> &, \
                                       int64_t);

INSTANTIATION_EB_ADJUST_BATCH(float16, float16);
INSTANTIATION_EB_ADJUST_BATCH(bfloat16, bfloat16);
INSTANTIATION_EB_ADJUST_BATCH(float, float);
INSTANTIATION_EB_ADJUST_BATCH(float16, float);
INSTANTIATION_EB_ADJUST_BATCH(float, float16);
INSTANTIATION_EB_ADJUST_BATCH(bfloat16, float16);
INSTANTIATION_EB_ADJUST_BATCH(float16, bfloat16);
INSTANTIATION_EB_ADJUST_BATCH(bfloat16, float);
INSTANTIATION_EB_ADJUST_BATCH(float, bfloat16);
INSTANTIATION_EB_ADJUST_BATCH(int32_t, int32_t);
INSTANTIATION_EB_ADJUST_BATCH(int64_t, int64_t);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
