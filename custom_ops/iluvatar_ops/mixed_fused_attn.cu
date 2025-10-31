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
void MixedFusedPagedAttnKernel(const paddle::Tensor& qkv,
                               paddle::Tensor& k_cache,
                               paddle::Tensor& v_cache,
                               const paddle::Tensor& prefill_block_table,
                               const paddle::Tensor& decode_block_table,
                               const paddle::Tensor& cu_seqlens_qkv,
                               const paddle::Tensor& seq_lens,
                               const paddle::optional<paddle::Tensor>& rope_sin,
                               const paddle::optional<paddle::Tensor>& rope_cos,
                               int prefill_num_tokens,
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
                               int window_left,
                               int window_right,
                               float softcap,
                               bool enable_cuda_graph,
                               bool use_sqrt_alibi,
                               paddle::Tensor& out) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::data_t data_t;

  const auto& dtype = qkv.dtype();
  cuinferDataType_t cuinfer_data_type;
  cudaDataType_t cu_data_type;
  if (dtype == paddle::DataType::FLOAT16) {
    cuinfer_data_type = CUINFER_DATA_HALF;
    cu_data_type = CUDA_R_16F;
  } else {
    cuinfer_data_type = CUINFER_DATA_BFLOAT16;
    cu_data_type = CUDA_R_16BF;
  }

  const auto& qkv_dims = qkv.dims();
  const auto& kv_cache_dims = k_cache.dims();
  const auto& prefill_block_table_dims = prefill_block_table.dims();
  const auto& cu_seqlens_qkv_dims = cu_seqlens_qkv.dims();

  int prefill_batch_size = prefill_block_table_dims[0];
  int num_tokens = qkv_dims[0];
  int decode_num_tokens = num_tokens - prefill_num_tokens;
  int num_total_heads = num_heads + 2 * num_kv_heads;
  int max_num_blocks_per_seq = prefill_block_table_dims[1];
  int qkv_stride = qkv.strides()[0];
  int num_blocks = kv_cache_dims[0];

  int kv_block_stride = k_cache.strides()[0];
  int kv_head_stride = k_cache.strides()[1];
  int block_table_stride = prefill_block_table.strides()[0];
  const float* rope_sin_ptr = rope_sin ? rope_sin.get().data<float>() : nullptr;
  const float* rope_cos_ptr = rope_cos ? rope_cos.get().data<float>() : nullptr;

  cuinferTensorDescriptor_t qkv_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&qkv_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      qkv_desc,
      cuinfer_data_type,
      3,
      std::vector<int>({prefill_num_tokens, num_total_heads, head_dim}).data(),
      std::vector<int>({num_total_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t qkv_seqlens_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&qkv_seqlens_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      qkv_seqlens_desc,
      CUINFER_DATA_INT32,
      1,
      std::vector<int>({prefill_batch_size + 1}).data(),
      std::vector<int>({1}).data()));

  cuinferTensorDescriptor_t block_table_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&block_table_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      block_table_desc,
      CUINFER_DATA_INT32,
      2,
      std::vector<int>({prefill_batch_size, block_table_stride}).data(),
      std::vector<int>({block_table_stride, 1}).data()));

  cuinferTensorDescriptor_t o_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&o_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      o_desc,
      cuinfer_data_type,
      3,
      std::vector<int>({prefill_num_tokens, num_heads, head_dim}).data(),
      std::vector<int>({num_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t k_cache_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&k_cache_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      k_cache_desc,
      cuinfer_data_type,
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
      cuinfer_data_type,
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

  cuinferHandle_t cuinfer_handle =
      iluvatar::getContextInstance()->getIxInferHandle();

  size_t prefill_workspace_size = 0;
  CUINFER_CHECK(
      cuinferGetFmhaFwdMergedFuseRopeWorkspaceSize(prefill_num_tokens,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_dim,
                                                   q_rope,
                                                   k_rope,
                                                   v_rope,
                                                   cuinfer_data_type,
                                                   cuinfer_data_type,
                                                   cuinfer_data_type,
                                                   &prefill_workspace_size));

  auto* allocator = paddle::GetAllocator(qkv.place());

  phi::Allocator::AllocationPtr prefill_tmp_workspace =
      allocator->Allocate(prefill_workspace_size);
  void* prefill_workspace_ptr = prefill_tmp_workspace->ptr();

  CUINFER_CHECK(
      cuinferFmhaFwdMergedFuseRopeFunc(cuinfer_handle,
                                       qkv_desc,
                                       qkv.data(),
                                       qkv_seqlens_desc,
                                       cu_seqlens_qkv.data<int32_t>(),
                                       block_table_desc,
                                       prefill_block_table.data<int32_t>(),
                                       o_desc,
                                       out.data(),
                                       k_cache_desc,
                                       k_cache.data(),
                                       v_cache_desc,
                                       v_cache.data(),
                                       prefill_workspace_ptr,
                                       prefill_workspace_size,
                                       cos_desc,
                                       rope_cos_ptr,
                                       sin_desc,
                                       rope_sin_ptr,
                                       prefill_batch_size,
                                       num_heads,
                                       num_kv_heads,
                                       head_dim,
                                       causal,
                                       scale,
                                       q_rope,
                                       k_rope,
                                       v_rope));

  size_t decode_workspace_size = 0;
  CUINFER_CHECK(cuInferPageAttentionGetWorkspaceV7(decode_num_tokens,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_dim,
                                                   block_size,
                                                   max_seq_len,
                                                   &decode_workspace_size));

  phi::Allocator::AllocationPtr decode_tmp_workspace =
      allocator->Allocate(decode_workspace_size);
  void* decode_workspace_ptr = decode_tmp_workspace->ptr();

  void* decode_qkv_ptr =
      (void*)(qkv.data<data_t>() + prefill_num_tokens * qkv_stride);
  void* decode_out_ptr =
      (void*)(out.data<data_t>() + prefill_num_tokens * out.strides()[0]);

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
                                         nullptr,
                                         decode_qkv_ptr,
                                         decode_qkv_ptr,
                                         decode_workspace_ptr,
                                         true,
                                         rope_sin_ptr,
                                         rope_cos_ptr};

  CUINFER_CHECK(cuInferPageAttentionV7(cuinfer_handle,
                                       decode_out_ptr,
                                       cu_data_type,
                                       decode_qkv_ptr,
                                       cu_data_type,
                                       decode_num_tokens,
                                       num_heads,
                                       num_kv_heads,
                                       head_dim,
                                       qkv_stride,
                                       kv_block_stride,
                                       kv_head_stride,
                                       k_cache.data(),
                                       cu_data_type,
                                       v_cache.data(),
                                       cu_data_type,
                                       block_size,
                                       max_num_blocks_per_seq,
                                       max_seq_len,
                                       decode_block_table.data<int32_t>(),
                                       seq_lens.data<int32_t>(),
                                       args));

  CUINFER_CHECK(cuinferDestroyTensorDescriptor(qkv_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(qkv_seqlens_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(block_table_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(o_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(k_cache_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(v_cache_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(cos_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(sin_desc));
}

std::vector<paddle::Tensor> MixedFusedPagedAttn(
    const paddle::Tensor& qkv,
    paddle::Tensor& k_cache,
    paddle::Tensor& v_cache,
    const paddle::Tensor& prefill_block_table,
    const paddle::Tensor& decode_block_table,
    const paddle::Tensor& cu_seqlens_qkv,
    const paddle::Tensor& seq_lens,
    const paddle::optional<paddle::Tensor>& rope_sin,
    const paddle::optional<paddle::Tensor>& rope_cos,
    int prefill_num_tokens,
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
    int window_left,
    int window_right,
    float softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi) {
  const auto dtype = qkv.dtype();
  auto out =
      paddle::empty({qkv.shape()[0], num_heads * head_dim}, dtype, qkv.place());

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      MixedFusedPagedAttnKernel<paddle::DataType::BFLOAT16>(qkv,
                                                            k_cache,
                                                            v_cache,
                                                            prefill_block_table,
                                                            decode_block_table,
                                                            cu_seqlens_qkv,
                                                            seq_lens,
                                                            rope_sin,
                                                            rope_cos,
                                                            prefill_num_tokens,
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
                                                            window_left,
                                                            window_right,
                                                            softcap,
                                                            enable_cuda_graph,
                                                            use_sqrt_alibi,
                                                            out);
      break;
    case paddle::DataType::FLOAT16:
      MixedFusedPagedAttnKernel<paddle::DataType::FLOAT16>(qkv,
                                                           k_cache,
                                                           v_cache,
                                                           prefill_block_table,
                                                           decode_block_table,
                                                           cu_seqlens_qkv,
                                                           seq_lens,
                                                           rope_sin,
                                                           rope_cos,
                                                           prefill_num_tokens,
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
                                                           window_left,
                                                           window_right,
                                                           softcap,
                                                           enable_cuda_graph,
                                                           use_sqrt_alibi,
                                                           out);
      break;
    default:
      PD_THROW("Unsupported data type for mixed paged attn");
  }
  return {out};
}

std::vector<std::vector<int64_t>> MixedFusedPagedAttnInferShape(
    const std::vector<int64_t>& qkv_shape, int num_heads, int head_dim) {
  return {{qkv_shape[0], num_heads * head_dim}};
}

std::vector<paddle::DataType> MixedFusedPagedAttnInferDtype(
    const paddle::DataType& qkv_dtype) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(mixed_fused_paged_attn)
    .Inputs({"qkv",
             "k_cache",
             "v_cache",
             "prefill_block_table",
             "decode_block_table",
             "cu_seqlens_qkv",
             "seq_lens",
             paddle::Optional("rope_sin"),
             paddle::Optional("rope_cos")})
    .Outputs({"out"})
    .Attrs({"prefill_num_tokens:int",
            "num_heads: int",
            "head_dim:int",
            "num_kv_heads:int",
            "block_size:int",
            "max_seq_len:int",
            "scale:float",
            "causal:bool",
            "q_rope:bool",
            "k_rope:bool",
            "v_rope:bool",
            "window_left:int",
            "window_right:int",
            "softcap:float",
            "enable_cuda_graph:bool",
            "use_sqrt_alibi:bool"})
    .SetKernelFn(PD_KERNEL(MixedFusedPagedAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(MixedFusedPagedAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MixedFusedPagedAttnInferDtype));
