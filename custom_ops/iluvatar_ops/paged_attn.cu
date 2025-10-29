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
void PagedAttnKernel(const paddle::Tensor& q,
                     const paddle::Tensor& k_cache,
                     const paddle::Tensor& v_cache,
                     const paddle::Tensor& block_table,
                     const paddle::Tensor& seq_lens,
                     const paddle::optional<paddle::Tensor>& alibi_slopes,
                     const paddle::optional<paddle::Tensor>& k,
                     const paddle::optional<paddle::Tensor>& v,
                     const paddle::optional<paddle::Tensor>& rope_sin,
                     const paddle::optional<paddle::Tensor>& rope_cos,
                     int num_heads,
                     int head_dim,
                     int num_kv_heads,
                     float scale,
                     int block_size,
                     int max_context_len,
                     bool causal,
                     int window_left,
                     int window_right,
                     float softcap,
                     bool enable_cuda_graph,
                     bool use_sqrt_alibi,
                     bool merged_qkv,
                     paddle::Tensor& out) {
  if (alibi_slopes) {
    PADDLE_ENFORCE_EQ(alibi_slopes.get().dtype(),
                      paddle::DataType::FLOAT32,
                      common::errors::InvalidArgument(
                          "paged_attention expects alibi_slopes float tensor"));
    PADDLE_ENFORCE_EQ(
        alibi_slopes.get().is_contiguous(),
        true,
        common::errors::InvalidArgument(
            "paged_attention expects alibi_slopes is contiguous"));
  }

  // check dtype and contiguous
  const auto& dtype = q.dtype();
  cudaDataType_t data_type;
  if (dtype == paddle::DataType::FLOAT16) {
    data_type = CUDA_R_16F;
  } else if (dtype == paddle::DataType::BFLOAT16) {
    data_type = CUDA_R_16BF;
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
      seq_lens.dtype(),
      paddle::DataType::INT32,
      common::errors::InvalidArgument("seq_lens dtype must be int32"));
  PADDLE_ENFORCE_EQ(seq_lens.is_contiguous(),
                    true,
                    common::errors::InvalidArgument(
                        "paged_attention expects seq_lens is contiguous"));
  // check dim and shape
  // k_cache: [num_blocks, kv_num_heads, block_size, head_dim]
  // v_cache: [num_blocks, kv_num_heads, block_size, head_dim]
  // block_table: [num_seqs, max_num_blocks_per_seq]
  // seq_lens: [num_seqs]
  // q and out:
  // if merged_qkv = false:
  // q:[num_seqs, hidden_size]
  // out:[num_seqs, hidden_size]
  // if merged_qkv = true:
  // q: [num_seqs, (num_heads+2*num_kv_heads)*head_dim]
  // out: [num_seqs, hidden_size]

  const auto& q_dims = q.dims();
  PADDLE_ENFORCE_EQ(q_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "paged_attn receive query dims is "
                        "[num_seqs, (num_heads+2*num_kv_heads)*head_dim]"));
  PADDLE_ENFORCE_EQ(
      out.dims().size(),
      2,
      common::errors::InvalidArgument("paged_attn receive out dims is "
                                      "[num_seqs, hidden_size]"));

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
                                      "[num_seqs, max_num_blocks_per_seq]"));

  const auto& seq_lens_dims = seq_lens.dims();
  PADDLE_ENFORCE_EQ(seq_lens_dims.size(),
                    1,
                    common::errors::InvalidArgument(
                        "paged_attn receive seq_lens dims is [num_seqs]"));

  int num_seqs = q_dims[0];
  int max_num_blocks_per_seq = block_table_dims[1];
  int q_stride = q.strides()[0];
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
  PADDLE_ENFORCE_EQ(block_table_dims[0],
                    num_seqs,
                    common::errors::InvalidArgument(
                        "block_table_dims[0] must be equal to num_seqs"));
  PADDLE_ENFORCE_EQ(seq_lens_dims[0],
                    num_seqs,
                    common::errors::InvalidArgument(
                        "seq_lens_dims[0] must be equal to num_seqs"));

  int kv_block_stride = k_cache.strides()[0];
  int kv_head_stride = k_cache.strides()[1];
  const float* alibi_slopes_ptr =
      alibi_slopes ? alibi_slopes.get().data<float>() : nullptr;
  const void* key_ptr = k ? k.get().data() : nullptr;
  const void* value_ptr = v ? v.get().data() : nullptr;
  const float* rope_sin_ptr =
      merged_qkv ? rope_sin.get().data<float>() : nullptr;
  const float* rope_cos_ptr =
      merged_qkv ? rope_cos.get().data<float>() : nullptr;

  cuinferHandle_t cuinfer_handle =
      iluvatar::getContextInstance()->getIxInferHandle();

  size_t workspace_size = 0;
  CUINFER_CHECK(cuInferPageAttentionGetWorkspaceV7(num_seqs,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_dim,
                                                   block_size,
                                                   max_context_len,
                                                   &workspace_size));
  auto* allocator = paddle::GetAllocator(q.place());
  phi::Allocator::AllocationPtr tmp_workspace =
      allocator->Allocate(workspace_size);
  void* workspace_ptr = tmp_workspace->ptr();

  PageAttentionWithKVCacheArguments args{static_cast<float>(scale),
                                         1.0,
                                         1.0,
                                         static_cast<float>(softcap),
                                         window_left,
                                         window_right,
                                         causal,
                                         use_sqrt_alibi,
                                         enable_cuda_graph,
                                         false,
                                         alibi_slopes_ptr,
                                         key_ptr,
                                         value_ptr,
                                         workspace_ptr,
                                         merged_qkv,
                                         rope_sin_ptr,
                                         rope_cos_ptr};
  CUINFER_CHECK(cuInferPageAttentionV7(cuinfer_handle,
                                       out.data(),
                                       data_type,
                                       q.data(),
                                       data_type,
                                       num_seqs,
                                       num_heads,
                                       num_kv_heads,
                                       head_dim,
                                       q_stride,
                                       kv_block_stride,
                                       kv_head_stride,
                                       k_cache.data(),
                                       data_type,
                                       v_cache.data(),
                                       data_type,
                                       block_size,
                                       max_num_blocks_per_seq,
                                       max_context_len,
                                       block_table.data<int32_t>(),
                                       seq_lens.data<int32_t>(),
                                       args));
}

std::vector<paddle::Tensor> PagedAttn(
    const paddle::Tensor& q,
    const paddle::Tensor& k_cache,
    const paddle::Tensor& v_cache,
    const paddle::Tensor& block_table,
    const paddle::Tensor& seq_lens,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    const paddle::optional<paddle::Tensor>& k,
    const paddle::optional<paddle::Tensor>& v,
    const paddle::optional<paddle::Tensor>& rope_sin,
    const paddle::optional<paddle::Tensor>& rope_cos,
    int num_heads,
    int head_dim,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_context_len,
    bool causal,
    int window_left,
    int window_right,
    float softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    bool merged_qkv) {
  const auto dtype = q.dtype();
  auto out =
      paddle::empty({q.shape()[0], num_heads * head_dim}, dtype, q.place());

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      PagedAttnKernel<paddle::DataType::BFLOAT16>(q,
                                                  k_cache,
                                                  v_cache,
                                                  block_table,
                                                  seq_lens,
                                                  alibi_slopes,
                                                  k,
                                                  v,
                                                  rope_sin,
                                                  rope_cos,
                                                  num_heads,
                                                  head_dim,
                                                  num_kv_heads,
                                                  scale,
                                                  block_size,
                                                  max_context_len,
                                                  causal,
                                                  window_left,
                                                  window_right,
                                                  softcap,
                                                  enable_cuda_graph,
                                                  use_sqrt_alibi,
                                                  merged_qkv,
                                                  out);
      break;
    case paddle::DataType::FLOAT16:
      PagedAttnKernel<paddle::DataType::FLOAT16>(q,
                                                 k_cache,
                                                 v_cache,
                                                 block_table,
                                                 seq_lens,
                                                 alibi_slopes,
                                                 k,
                                                 v,
                                                 rope_sin,
                                                 rope_cos,
                                                 num_heads,
                                                 head_dim,
                                                 num_kv_heads,
                                                 scale,
                                                 block_size,
                                                 max_context_len,
                                                 causal,
                                                 window_left,
                                                 window_right,
                                                 softcap,
                                                 enable_cuda_graph,
                                                 use_sqrt_alibi,
                                                 merged_qkv,
                                                 out);
      break;
    default:
      PD_THROW("Unsupported data type for Paged attn");
  }
  return {out};
}

std::vector<std::vector<int64_t>> PagedAttnInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_cache_shape,
    const std::vector<int64_t>& v_cache_shape,
    const std::vector<int64_t>& block_table_shape,
    const std::vector<int64_t>& seq_lens_shape,
    const std::vector<int64_t>& alibi_slopes_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& v_shape,
    const std::vector<int64_t>& rope_sin_shape,
    const std::vector<int64_t>& rope_cos_shape,
    int num_heads,
    int head_dim,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_context_len,
    bool causal,
    int window_left,
    int window_right,
    float softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    bool merged_qkv) {
  if (merged_qkv) {
    return {{q_shape[0], num_heads * head_dim}};
  } else {
    return {q_shape};
  }
}

std::vector<paddle::DataType> PagedAttnInferDtype(
    const paddle::DataType& q_dtype) {
  return {q_dtype};
}

PD_BUILD_STATIC_OP(paged_attn)
    .Inputs({"q",
             "k_cache",
             "v_cache",
             "block_table",
             "seq_lens",
             paddle::Optional("alibi_slopes"),
             paddle::Optional("k"),
             paddle::Optional("v"),
             paddle::Optional("rope_sin"),
             paddle::Optional("rope_cos")})
    .Outputs({"out"})
    .Attrs({"num_heads:int",
            "head_dim:int",
            "num_kv_heads:int",
            "scale:float",
            "block_size:int",
            "max_context_len:int",
            "causal:bool",
            "window_left:int",
            "window_right:int",
            "softcap:float",
            "enable_cuda_graph:bool",
            "use_sqrt_alibi:bool",
            "merged_qkv:bool"})
    .SetKernelFn(PD_KERNEL(PagedAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(PagedAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PagedAttnInferDtype));
