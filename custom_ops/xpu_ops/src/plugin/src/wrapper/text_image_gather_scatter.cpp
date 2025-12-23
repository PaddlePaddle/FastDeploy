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
__attribute__((global)) void text_image_gather_scatter(T* input,
                                                       T* text_input,
                                                       T* image_input,
                                                       int* token_type_ids,
                                                       int* text_index,
                                                       int* image_index,
                                                       int64_t token_num,
                                                       int64_t text_token_num,
                                                       int64_t image_token_num,
                                                       int64_t hidden_size,
                                                       bool is_scatter);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(
    Context* ctx,
    T* input,             // shape [token_num, hidden_size]
    T* text_input,        // shape [text_token_num, hidden_size]
    T* image_input,       // shape [image_token_num, hidden_size]
    int* token_type_ids,  // shape [token_num], 0 for text, 1 for image
    int* text_index,      // shape [token_num], mapping from input to text_input
    int* image_index,  // shape [token_num], mapping from input to image_input
    int64_t token_num,
    int64_t text_token_num,
    int64_t image_token_num,
    int64_t hidden_size,
    bool is_scatter) {
  if (is_scatter) {
    // Scatter mode: input -> text_input/image_input
    for (int64_t i = 0; i < token_num; i++) {
      int token_type = token_type_ids[i];

      T* text_image_input = nullptr;
      int* text_image_index = nullptr;
      if (token_type == 0) {
        text_image_input = text_input;
        text_image_index = text_index;
      } else {  // token_type == 1
        text_image_input = image_input;
        text_image_index = image_index;
      }

      int text_image_token_idx = text_image_index[i];
      int input_offset = i * hidden_size;
      int text_image_offset = text_image_token_idx * hidden_size;

      for (int64_t j = 0; j < hidden_size; j++) {
        T value = input[input_offset + j];
        text_image_input[text_image_offset + j] = value;
      }
    }
  } else {
    // Gather mode: text_input/image_input -> input
    for (int64_t i = 0; i < token_num; i++) {
      int token_type = token_type_ids[i];

      T* text_image_input = nullptr;
      int* text_image_index = nullptr;
      if (token_type == 0) {
        text_image_input = text_input;
        text_image_index = text_index;
      } else {  // token_type == 1
        text_image_input = image_input;
        text_image_index = image_index;
      }

      int text_image_token_idx = text_image_index[i];
      int input_offset = i * hidden_size;
      int text_image_offset = text_image_token_idx * hidden_size;

      for (int64_t j = 0; j < hidden_size; j++) {
        T value = text_image_input[text_image_offset + j];
        input[input_offset + j] = value;
      }
    }
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(Context* ctx,
                        T* input,
                        T* text_input,
                        T* image_input,
                        int* token_type_ids,
                        int* text_index,
                        int* image_index,
                        int64_t token_num,
                        int64_t text_token_num,
                        int64_t image_token_num,
                        int64_t hidden_size,
                        bool is_scatter) {
  xpu3::plugin::text_image_gather_scatter<T>
      <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(input,
                                                 text_input,
                                                 image_input,
                                                 token_type_ids,
                                                 text_index,
                                                 image_index,
                                                 token_num,
                                                 text_token_num,
                                                 image_token_num,
                                                 hidden_size,
                                                 is_scatter);
  return api::SUCCESS;
}

template <typename T>
int text_image_gather_scatter(
    Context* ctx,
    T* input,             // shape [token_num, hidden_size]
    T* text_input,        // shape [text_token_num, hidden_size]
    T* image_input,       // shape [image_token_num, hidden_size]
    int* token_type_ids,  // shape [token_num], 0 for text, 1 for image
    int* text_index,      // shape [token_num], mapping from input to text_input
    int* image_index,  // shape [token_num], mapping from input to image_input
    int64_t token_num,
    int64_t text_token_num,
    int64_t image_token_num,
    int64_t hidden_size,
    bool is_scatter) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "text_image_gather_scatter", T);
  WRAPPER_DUMP_PARAM6(ctx,
                      input,
                      text_input,
                      image_input,
                      token_type_ids,
                      text_index,
                      image_index);
  WRAPPER_DUMP_PARAM5(
      ctx, token_num, text_token_num, image_token_num, hidden_size, is_scatter);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, T, token_num * hidden_size, input);
  if (text_token_num !=
      0) {  // avoiding text_input tensor with shape [0, hidden_size]
    WRAPPER_CHECK_PTR(ctx, T, text_token_num * hidden_size, text_input);
  }
  if (image_token_num !=
      0) {  // avoiding image_input tensor with shape [0, hidden_size]
    WRAPPER_CHECK_PTR(ctx, T, image_token_num * hidden_size, image_input);
  }
  WRAPPER_CHECK_PTR(ctx, int, token_num, token_type_ids);
  WRAPPER_CHECK_PTR(ctx, int, token_num, text_index);
  WRAPPER_CHECK_PTR(ctx, int, token_num, image_index);
  // When all tokens are image type, text_token_num will be set to 1 in model.py
  WRAPPER_ASSERT_EQ(ctx,
                    true,
                    (text_token_num + image_token_num == token_num) ||
                        (image_token_num == token_num && text_token_num == 1));

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          input,
                          text_input,
                          image_input,
                          token_type_ids,
                          text_index,
                          image_index,
                          token_num,
                          text_token_num,
                          image_token_num,
                          hidden_size,
                          is_scatter);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(ctx,
                           input,
                           text_input,
                           image_input,
                           token_type_ids,
                           text_index,
                           image_index,
                           token_num,
                           text_token_num,
                           image_token_num,
                           hidden_size,
                           is_scatter);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int text_image_gather_scatter(Context*,
                                       bfloat16*,
                                       bfloat16*,
                                       bfloat16*,
                                       int*,
                                       int*,
                                       int*,
                                       const int64_t,
                                       const int64_t,
                                       const int64_t,
                                       const int64_t,
                                       bool);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
