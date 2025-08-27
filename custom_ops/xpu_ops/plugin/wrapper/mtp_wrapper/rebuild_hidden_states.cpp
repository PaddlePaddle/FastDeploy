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

namespace xpu3 {
namespace plugin {
template <typename T>
__attribute__((global)) void rebuildHiddenStatesKernel(const T* input,
                                                       const int* position_map,
                                                       T* output,
                                                       int dim_embed,
                                                       int elem_cnt);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(Context* ctx,
                       const T* input,
                       const int* position_map,
                       T* output,
                       int dim_embed,
                       int elem_cnt) {
  for (int elem_id = 0; elem_id < elem_cnt; elem_id++) {
    int ori_token_idx = elem_id / dim_embed;
    int token_idx = position_map[ori_token_idx];
    int offset = elem_id % dim_embed;
    if (token_idx >= 0) {
      output[token_idx * dim_embed + offset] =
          input[ori_token_idx * dim_embed + offset];
    }
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(Context* ctx,
                        const T* input,
                        const int* position_map,
                        T* output,
                        int dim_embed,
                        int elem_cnt) {
  xpu3::plugin::rebuildHiddenStatesKernel<T>
      <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          input, position_map, output, dim_embed, elem_cnt);
  return api::SUCCESS;
}

template <typename T>
int rebuild_hidden_states(Context* ctx,
                          const T* input,
                          const int* position_map,
                          T* output,
                          int dim_embed,
                          int elem_cnt) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "rebuild_hidden_states", T);
  WRAPPER_DUMP_PARAM5(ctx, input, position_map, output, dim_embed, elem_cnt);
  WRAPPER_DUMP(ctx);

  WRAPPER_ASSERT_GT(ctx, dim_embed, 0);
  WRAPPER_ASSERT_GT(ctx, elem_cnt, 0);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(
        ctx, input, position_map, output, dim_embed, elem_cnt);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(
        ctx, input, position_map, output, dim_embed, elem_cnt);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
  return api::SUCCESS;
}

template int rebuild_hidden_states(
    Context*, const bfloat16*, const int*, bfloat16*, int, int);
template int rebuild_hidden_states(
    Context*, const float*, const int*, float*, int, int);
template int rebuild_hidden_states(
    Context*, const float16*, const int*, float16*, int, int);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
