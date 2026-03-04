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

#include "helper.h"
#include "iluvatar_context.h"

template <paddle::DataType T>
void FlashAttnUnpaddedKernel(const paddle::Tensor& q,
                             const paddle::Tensor& k,
                             const paddle::Tensor& v,
                             const paddle::Tensor& cu_seqlens_q,
                             const paddle::Tensor& cu_seqlens_k,
                             int num_heads,
                             int head_dim,
                             int num_kv_heads,
                             int max_seqlens_q,
                             int max_seqlens_k,
                             bool causal,
                             float scale,
                             paddle::Tensor& out) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(q.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());

  // check dtype and contiguous
  const auto& dtype = q.dtype();
  cuinferDataType_t data_type;
  if (dtype == paddle::DataType::FLOAT16) {
    data_type = CUINFER_DATA_HALF;

  } else if (dtype == paddle::DataType::BFLOAT16) {
    data_type = CUINFER_DATA_BFLOAT16;
  } else {
    common::errors::InvalidArgument(
        "flash_attn_unpadded support half and bfloat16 now");
  }

  PADDLE_ENFORCE_EQ(
      cu_seqlens_q.dtype(),
      paddle::DataType::INT32,
      common::errors::InvalidArgument("cu_seqlens_q dtype must be int32"));
  PADDLE_ENFORCE_EQ(
      cu_seqlens_q.is_contiguous(),
      true,
      common::errors::InvalidArgument(
          "flash_attn_unpadded expects cu_seqlens_k is contiguous"));
  // check dim and shape
  // q: [num_tokens, num_heads, head_dim]
  // out: [num_tokens, num_heads, head_dim]

  const auto& q_dims = q.dims();
  PADDLE_ENFORCE_EQ(q_dims.size(),
                    3,
                    common::errors::InvalidArgument(
                        "flash_attn_unpadded receive query dims is "
                        "[num_tokens, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      out.dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_unpadded receive out dims is "
                                      "[num_tokens, num_heads, head_dim]"));

  const auto& cu_seqlens_q_dims = cu_seqlens_q.dims();
  PADDLE_ENFORCE_EQ(
      cu_seqlens_q_dims.size(),
      1,
      common::errors::InvalidArgument(
          "flash_attn_unpadded receive cu_seqlens_q dims is [batch_size]"));

  int batch_size = cu_seqlens_q_dims[0] - 1;
  int num_tokens = q_dims[0];

  cuinferHandle_t cuinfer_handle =
      iluvatar::getContextInstance()->getIxInferHandle();
  CUINFER_CHECK(cuinferSetStream(cuinfer_handle, stream));

  cuinferTensorDescriptor_t q_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&q_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      q_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_heads, head_dim}).data(),
      std::vector<int>({num_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t k_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&k_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      k_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_kv_heads, head_dim}).data(),
      std::vector<int>({num_kv_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t v_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&v_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      v_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_kv_heads, head_dim}).data(),
      std::vector<int>({num_kv_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t q_seqlens_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&q_seqlens_desc));
  CUINFER_CHECK(
      cuinferSetTensorNdDescriptor(q_seqlens_desc,
                                   CUINFER_DATA_INT32,
                                   1,
                                   std::vector<int>({batch_size + 1}).data(),
                                   std::vector<int>({1}).data()));

  cuinferTensorDescriptor_t k_seqlens_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&k_seqlens_desc));
  CUINFER_CHECK(
      cuinferSetTensorNdDescriptor(k_seqlens_desc,
                                   CUINFER_DATA_INT32,
                                   1,
                                   std::vector<int>({batch_size + 1}).data(),
                                   std::vector<int>({1}).data()));

  cuinferTensorDescriptor_t o_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&o_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      o_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_heads, head_dim}).data(),
      std::vector<int>({num_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t block_table_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&block_table_desc));

  cuinferTensorDescriptor_t alibi_slope_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&alibi_slope_desc));

  cuinferTensorDescriptor_t lse_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&lse_desc));

  FmhaFwdFuncArguments args;
  args.batch = batch_size;
  args.max_seqlen_q = max_seqlens_q;
  args.max_seqlen_k = max_seqlens_k;
  args.is_causal = causal;
  args.scaling = scale;
  args.window_size_left = -1;
  args.window_size_right = -1;
  args.softcap = 0;
  args.is_persistent = false;
  args.alibi_mode = CUINFER_FATTN_ALIBI_MODE_NONE;
  CUINFER_CHECK(cuinferFmhaFwdLseFunc(cuinfer_handle,
                                      q_desc,
                                      q.data(),
                                      k_desc,
                                      k.data(),
                                      v_desc,
                                      v.data(),
                                      q_seqlens_desc,
                                      cu_seqlens_q.data(),
                                      k_seqlens_desc,
                                      cu_seqlens_k.data(),
                                      block_table_desc,
                                      nullptr,
                                      alibi_slope_desc,
                                      nullptr,
                                      o_desc,
                                      out.data(),
                                      lse_desc,
                                      nullptr,
                                      args));

  CUINFER_CHECK(cuinferDestroyTensorDescriptor(q_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(k_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(v_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(q_seqlens_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(k_seqlens_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(o_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(block_table_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(alibi_slope_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(lse_desc));
}

std::vector<paddle::Tensor> FlashAttnUnpadded(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    int max_seqlens_q,
    int max_seqlens_k,
    bool causal,
    float scale,
    bool training) {
  const auto dtype = q.dtype();
  const auto& q_dims = q.dims();
  int num_tokens = q_dims[0];
  int num_heads = q_dims[1];
  int head_dim = q_dims[2];
  int num_kv_heads = k.dims()[1];
  auto out = paddle::empty({num_tokens, num_heads, head_dim}, dtype, q.place());

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      FlashAttnUnpaddedKernel<paddle::DataType::BFLOAT16>(q,
                                                          k,
                                                          v,
                                                          cu_seqlens_q,
                                                          cu_seqlens_k,
                                                          num_heads,
                                                          head_dim,
                                                          num_kv_heads,
                                                          max_seqlens_q,
                                                          max_seqlens_k,
                                                          causal,
                                                          scale,
                                                          out);
      break;
    case paddle::DataType::FLOAT16:
      FlashAttnUnpaddedKernel<paddle::DataType::FLOAT16>(q,
                                                         k,
                                                         v,
                                                         cu_seqlens_q,
                                                         cu_seqlens_k,
                                                         num_heads,
                                                         head_dim,
                                                         num_kv_heads,
                                                         max_seqlens_q,
                                                         max_seqlens_k,
                                                         causal,
                                                         scale,
                                                         out);
      break;
    default:
      PD_THROW("Unsupported data type for Paged attn");
  }
  return {out};
}

std::vector<std::vector<int64_t>> FlashAttnUnpaddedInferShape(
    const std::vector<int64_t>& q_shape) {
  return {{q_shape[0], q_shape[1], q_shape[2]}};
}

std::vector<paddle::DataType> FlashAttnUnpaddedInferDtype(
    const paddle::DataType& q_dtype) {
  return {q_dtype};
}

PD_BUILD_STATIC_OP(cuinfer_flash_attn_unpadded)
    .Inputs({"q", "k", "v", "cu_seqlens_q", "cu_seqlens_k"})
    .Outputs({"out"})
    .Attrs({"max_seqlens_q:int",
            "max_seqlens_k:int",
            "causal:bool",
            "scale:float",
            "training:bool"})
    .SetKernelFn(PD_KERNEL(FlashAttnUnpadded))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashAttnUnpaddedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashAttnUnpaddedInferDtype));
