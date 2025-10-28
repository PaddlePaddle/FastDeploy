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
void PrefillFusedPagedAttnKernel(
    const paddle::Tensor& qkv,
    paddle::Tensor& k_cache,
    paddle::Tensor& v_cache,
    const paddle::Tensor& block_table,
    const paddle::Tensor& cu_seqlens_qkv,
    const paddle::optional<paddle::Tensor>& rope_sin,
    const paddle::optional<paddle::Tensor>& rope_cos,
    int num_heads,
    int head_dim,
    int num_kv_heads,
    int block_size,
    int max_seq_len,
    float scale,
    bool causal,
    bool q_rope,
    bool k_rope,
    bool v_rope,
    paddle::Tensor& out) {
  // check dtype and contiguous
  const auto& dtype = qkv.dtype();
  cuinferDataType_t data_type;
  if (dtype == paddle::DataType::FLOAT16) {
    data_type = CUINFER_DATA_HALF;

  } else if (dtype == paddle::DataType::BFLOAT16) {
    data_type = CUINFER_DATA_BFLOAT16;
  } else {
    common::errors::InvalidArgument(
        "paged_attention support half and bfloat16 now");
  }

  PADDLE_ENFORCE_EQ(k_cache.dtype(),
                    dtype,
                    common::errors::InvalidArgument(
                        "k_cache dtype must be the same as query dtype"));
  PADDLE_ENFORCE_EQ(k_cache.is_contiguous(),
                    true,
                    common::errors::InvalidArgument(
                        "paged_attention expects k_cache is contiguous"));
  PADDLE_ENFORCE_EQ(
      block_table.dtype(),
      paddle::DataType::INT32,
      common::errors::InvalidArgument("block_table dtype must be int32"));
  PADDLE_ENFORCE_EQ(block_table.is_contiguous(),
                    true,
                    common::errors::InvalidArgument(
                        "paged_attention expects block_table is contiguous"));
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv.dtype(),
      paddle::DataType::INT32,
      common::errors::InvalidArgument("cu_seqlens_qkv dtype must be int32"));
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv.is_contiguous(),
      true,
      common::errors::InvalidArgument(
          "paged_attention expects cu_seqlens_qkv is contiguous"));
  // check dim and shape
  // k_cache: [num_blocks, kv_num_heads, block_size, head_dim]
  // v_cache: [num_blocks, kv_num_heads, block_size, head_dim]
  // block_table: [batch_size, max_num_blocks_per_seq]
  // seq_lens: [batch_size]
  // qkv: [num_tokens, (num_heads+2*num_kv_heads)*head_dim]
  // out: [num_tokens, hidden_size]

  const auto& qkv_dims = qkv.dims();
  PADDLE_ENFORCE_EQ(qkv_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "paged_attn receive query dims is "
                        "[num_tokens, (num_heads+2*num_kv_heads)*head_dim]"));
  PADDLE_ENFORCE_EQ(
      out.dims().size(),
      2,
      common::errors::InvalidArgument("paged_attn receive out dims is "
                                      "[num_tokens, hidden_size]"));

  const auto& kv_cache_dims = k_cache.dims();
  PADDLE_ENFORCE_EQ(kv_cache_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "paged_attn receive kv cache dims is "
                        "[num_blocks, kv_num_heads, block_size, head_dim]"));

  const auto& block_table_dims = block_table.dims();
  PADDLE_ENFORCE_EQ(
      block_table_dims.size(),
      2,
      common::errors::InvalidArgument("paged_attn receive block_table dims is "
                                      "[batch_size, max_num_blocks_per_seq]"));

  const auto& cu_seqlens_qkv_dims = cu_seqlens_qkv.dims();
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv_dims.size(),
      1,
      common::errors::InvalidArgument(
          "paged_attn receive cu_seqlens_qkv dims is [batch_size]"));

  int batch_size = block_table_dims[0];
  int num_tokens = qkv_dims[0];
  int num_total_heads = num_heads + 2 * num_kv_heads;
  int qkv_stride = qkv.strides()[0];
  int num_blocks = kv_cache_dims[0];

  PADDLE_ENFORCE_EQ(kv_cache_dims[1],
                    num_kv_heads,
                    common::errors::InvalidArgument(
                        "kv_cache_dims[1] must be equal to num_kv_head"));
  PADDLE_ENFORCE_EQ(kv_cache_dims[2],
                    block_size,
                    common::errors::InvalidArgument(
                        "kv_cache_dims[2] must be equal to block_size"));
  PADDLE_ENFORCE_EQ(kv_cache_dims[3],
                    head_dim,
                    common::errors::InvalidArgument(
                        "kv_cache_dims[3] must be equal to head_dim"));
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv_dims[0],
      batch_size + 1,
      common::errors::InvalidArgument(
          "cu_seqlens_qkv_dims[0] must be equal to batch_size + 1"));

  int block_table_stride = block_table.strides()[0];
  const float* rope_sin_ptr = rope_sin ? rope_sin.get().data<float>() : nullptr;
  const float* rope_cos_ptr = rope_cos ? rope_cos.get().data<float>() : nullptr;

  cuinferHandle_t cuinfer_handle =
      iluvatar::getContextInstance()->getIxInferHandle();

  size_t workspace_size = 0;
  CUINFER_CHECK(cuinferGetFmhaFwdMergedFuseRopeWorkspaceSize(num_tokens,
                                                             num_heads,
                                                             num_kv_heads,
                                                             head_dim,
                                                             q_rope,
                                                             k_rope,
                                                             v_rope,
                                                             data_type,
                                                             data_type,
                                                             data_type,
                                                             &workspace_size));
  auto* allocator = paddle::GetAllocator(qkv.place());
  phi::Allocator::AllocationPtr tmp_workspace =
      allocator->Allocate(workspace_size);
  void* workspace_ptr = tmp_workspace->ptr();

  cuinferTensorDescriptor_t qkv_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&qkv_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      qkv_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_total_heads, head_dim}).data(),
      std::vector<int>({num_total_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t qkv_seqlens_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&qkv_seqlens_desc));
  CUINFER_CHECK(
      cuinferSetTensorNdDescriptor(qkv_seqlens_desc,
                                   CUINFER_DATA_INT32,
                                   1,
                                   std::vector<int>({batch_size + 1}).data(),
                                   std::vector<int>({1}).data()));

  cuinferTensorDescriptor_t block_table_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&block_table_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      block_table_desc,
      CUINFER_DATA_INT32,
      2,
      std::vector<int>({batch_size, block_table_stride}).data(),
      std::vector<int>({block_table_stride, 1}).data()));

  cuinferTensorDescriptor_t o_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&o_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      o_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_heads, head_dim}).data(),
      std::vector<int>({num_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t k_cache_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&k_cache_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      k_cache_desc,
      data_type,
      4,
      std::vector<int>({num_blocks, num_kv_heads, block_size, head_dim}).data(),
      std::vector<int>({num_kv_heads * block_size * head_dim,
                        block_size * head_dim,
                        head_dim,
                        1})
          .data()));

  cuinferTensorDescriptor_t v_cache_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&v_cache_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      v_cache_desc,
      data_type,
      4,
      std::vector<int>({num_blocks, num_kv_heads, block_size, head_dim}).data(),
      std::vector<int>({num_kv_heads * block_size * head_dim,
                        block_size * head_dim,
                        head_dim,
                        1})
          .data()));

  cuinferTensorDescriptor_t cos_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&cos_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      cos_desc,
      CUINFER_DATA_FLOAT,
      2,
      std::vector<int>({max_seq_len, head_dim}).data(),
      std::vector<int>({head_dim, 1}).data()));

  cuinferTensorDescriptor_t sin_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&sin_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      sin_desc,
      CUINFER_DATA_FLOAT,
      2,
      std::vector<int>({max_seq_len, head_dim}).data(),
      std::vector<int>({head_dim, 1}).data()));

  CUINFER_CHECK(cuinferFmhaFwdMergedFuseRopeFunc(cuinfer_handle,
                                                 qkv_desc,
                                                 qkv.data(),
                                                 qkv_seqlens_desc,
                                                 cu_seqlens_qkv.data<int32_t>(),
                                                 block_table_desc,
                                                 block_table.data<int32_t>(),
                                                 o_desc,
                                                 out.data(),
                                                 k_cache_desc,
                                                 k_cache.data(),
                                                 v_cache_desc,
                                                 v_cache.data(),
                                                 workspace_ptr,
                                                 workspace_size,
                                                 cos_desc,
                                                 rope_cos_ptr,
                                                 sin_desc,
                                                 rope_sin_ptr,
                                                 batch_size,
                                                 num_heads,
                                                 num_kv_heads,
                                                 head_dim,
                                                 causal,
                                                 scale,
                                                 q_rope,
                                                 k_rope,
                                                 v_rope));

  CUINFER_CHECK(cuinferDestroyTensorDescriptor(qkv_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(qkv_seqlens_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(block_table_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(o_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(k_cache_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(v_cache_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(cos_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(sin_desc));
}

std::vector<paddle::Tensor> PrefillFusedPagedAttn(
    const paddle::Tensor& qkv,
    paddle::Tensor& k_cache,
    paddle::Tensor& v_cache,
    const paddle::Tensor& block_table,
    const paddle::Tensor& cu_seqlens_qkv,
    const paddle::optional<paddle::Tensor>& rope_sin,
    const paddle::optional<paddle::Tensor>& rope_cos,
    int num_heads,
    int head_dim,
    int num_kv_heads,
    int block_size,
    int max_seq_len,
    float scale,
    bool causal,
    bool q_rope,
    bool k_rope,
    bool v_rope) {
  const auto dtype = qkv.dtype();
  auto out =
      paddle::empty({qkv.shape()[0], num_heads * head_dim}, dtype, qkv.place());

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      PrefillFusedPagedAttnKernel<paddle::DataType::BFLOAT16>(qkv,
                                                              k_cache,
                                                              v_cache,
                                                              block_table,
                                                              cu_seqlens_qkv,
                                                              rope_sin,
                                                              rope_cos,
                                                              num_heads,
                                                              head_dim,
                                                              num_kv_heads,
                                                              block_size,
                                                              max_seq_len,
                                                              scale,
                                                              causal,
                                                              q_rope,
                                                              k_rope,
                                                              v_rope,
                                                              out);
      break;
    case paddle::DataType::FLOAT16:
      PrefillFusedPagedAttnKernel<paddle::DataType::FLOAT16>(qkv,
                                                             k_cache,
                                                             v_cache,
                                                             block_table,
                                                             cu_seqlens_qkv,
                                                             rope_sin,
                                                             rope_cos,
                                                             num_heads,
                                                             head_dim,
                                                             num_kv_heads,
                                                             block_size,
                                                             max_seq_len,
                                                             scale,
                                                             causal,
                                                             q_rope,
                                                             k_rope,
                                                             v_rope,
                                                             out);
      break;
    default:
      PD_THROW("Unsupported data type for Paged attn");
  }
  return {out};
}

std::vector<std::vector<int64_t>> PrefillFusedPagedAttnInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& k_cache_shape,
    const std::vector<int64_t>& v_cache_shape,
    const std::vector<int64_t>& block_table_shape,
    const std::vector<int64_t>& cu_seqlens_qkv_shape,
    const std::vector<int64_t>& rope_sin_shape,
    const std::vector<int64_t>& rope_cos_shape,
    int num_heads,
    int head_dim,
    int num_kv_heads,
    int block_size,
    int max_seq_len,
    float scale,
    bool causal,
    bool q_rope,
    bool k_rope,
    bool v_rope) {
  return {{qkv_shape[0], num_heads * head_dim}};
}

std::vector<paddle::DataType> PrefillFusedPagedAttnInferDtype(
    const paddle::DataType& qkv_dtype) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(prefill_fused_paged_attn)
    .Inputs({"qkv",
             "k_cache",
             "v_cache",
             "block_table",
             "cu_seqlens_qkv",
             paddle::Optional("rope_sin"),
             paddle::Optional("rope_cos")})
    .Outputs({"out"})
    .Attrs({"num_heads:int",
            "head_dim:int",
            "num_kv_heads:int",
            "block_size:int",
            "max_seq_len:int",
            "scale:float",
            "causal:bool",
            "q_rope:bool",
            "k_rope:bool",
            "v_rope:bool"})
    .SetKernelFn(PD_KERNEL(PrefillFusedPagedAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(PrefillFusedPagedAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PrefillFusedPagedAttnInferDtype));
