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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "utility/helper.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <paddle::DataType T>
std::vector<paddle::Tensor> AdjustBatchKernel(
    const paddle::Tensor &x,            // [token_num, dim_embed]
    const paddle::Tensor &cum_offsets,  // [bsz, 1]
    const paddle::Tensor &encoder_seq_lod,
    const paddle::Tensor &decoder_seq_lod,
    const paddle::Tensor &encoder_batch_idx,
    const paddle::Tensor &decoder_batch_idx,
    const paddle::Tensor &encoder_seq_lod_cpu,
    const paddle::Tensor &decoder_seq_lod_cpu,
    const paddle::Tensor &encoder_batch_idx_cpu,
    const paddle::Tensor &decoder_batch_idx_cpu,
    const paddle::Tensor &len_info_cpu,
    const paddle::optional<paddle::Tensor> &output_padding_offset,
    int max_input_length) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto ctx = static_cast<const phi::XPUContext *>(dev_ctx)->x_context();
  PD_CHECK(x.dtype() == T);
  PD_CHECK(x.dims().size() == 2);
  if (x.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }
  using XPUType = typename XPUTypeTrait<typename PDTraits<T>::DataType>::Type;
  using data_t = typename PDTraits<T>::data_t;
  const int token_num = x.dims()[0];
  const int dim = x.dims()[1];
  const int bsz = cum_offsets.shape()[0];
  int enc_batch = len_info_cpu.data<int32_t>()[0];
  int dec_batch = len_info_cpu.data<int32_t>()[1];

  baidu::xpu::api::VectorParam<int32_t> encoder_seqs_lods_vp{
      const_cast<int32_t *>(encoder_seq_lod_cpu.data<int32_t>()),
      enc_batch + 1,
      const_cast<int32_t *>(encoder_seq_lod.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> decoder_seqs_lods_vp{
      const_cast<int32_t *>(decoder_seq_lod_cpu.data<int32_t>()),
      dec_batch + 1,
      const_cast<int32_t *>(decoder_seq_lod.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> encoder_batch_map_vp{
      const_cast<int32_t *>(encoder_batch_idx_cpu.data<int32_t>()),
      enc_batch,
      const_cast<int32_t *>(encoder_batch_idx.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> decoder_batch_map_vp{
      const_cast<int32_t *>(decoder_batch_idx_cpu.data<int32_t>()),
      dec_batch,
      const_cast<int32_t *>(decoder_batch_idx.data<int32_t>())};

  auto out = paddle::empty({token_num, dim}, x.type(), x.place());

  int r = baidu::xpu::api::plugin::eb_adjust_batch<XPUType, XPUType>(
      ctx,
      reinterpret_cast<const XPUType *>(x.data<data_t>()),
      reinterpret_cast<XPUType *>(out.data<data_t>()),
      encoder_seqs_lods_vp,
      decoder_seqs_lods_vp,
      encoder_batch_map_vp,
      decoder_batch_map_vp,
      dim);
  return {out};
}

using AdjustBatchKernelFuncPtr = std::vector<paddle::Tensor> (*)(
    const paddle::Tensor &x,            // [token_num, dim_embed]
    const paddle::Tensor &cum_offsets,  // [bsz, 1]
    const paddle::Tensor &encoder_seq_lod,
    const paddle::Tensor &decoder_seq_lod,
    const paddle::Tensor &encoder_batch_idx,
    const paddle::Tensor &decoder_batch_idx,
    const paddle::Tensor &encoder_seq_lod_cpu,
    const paddle::Tensor &decoder_seq_lod_cpu,
    const paddle::Tensor &encoder_batch_idx_cpu,
    const paddle::Tensor &decoder_batch_idx_cpu,
    const paddle::Tensor &len_info_cpu,
    const paddle::optional<paddle::Tensor> &output_padding_offset,
    int max_input_length);

std::vector<paddle::Tensor> AdjustBatch(
    const paddle::Tensor &x,            // [token_num, dim_embed]
    const paddle::Tensor &cum_offsets,  // [bsz, 1]
    const paddle::Tensor &encoder_seq_lod,
    const paddle::Tensor &decoder_seq_lod,
    const paddle::Tensor &encoder_batch_idx,
    const paddle::Tensor &decoder_batch_idx,
    const paddle::Tensor &encoder_seq_lod_cpu,
    const paddle::Tensor &decoder_seq_lod_cpu,
    const paddle::Tensor &encoder_batch_idx_cpu,
    const paddle::Tensor &decoder_batch_idx_cpu,
    const paddle::Tensor &len_info_cpu,
    const paddle::optional<paddle::Tensor> &output_padding_offset,
    int max_input_length) {
  AdjustBatchKernelFuncPtr func = nullptr;

  switch (x.dtype()) {
    case paddle::DataType::BFLOAT16:
      func = &AdjustBatchKernel<paddle::DataType::BFLOAT16>;
      break;
    case paddle::DataType::FLOAT16:
      func = &AdjustBatchKernel<paddle::DataType::FLOAT16>;
      break;
    case paddle::DataType::INT64:
      func = &AdjustBatchKernel<paddle::DataType::INT64>;
      break;
    case paddle::DataType::FLOAT32:
      func = &AdjustBatchKernel<paddle::DataType::FLOAT32>;
      break;
    default:
      PD_THROW("Unsupported data type: ", x.dtype());
  }

  return func(x,
              cum_offsets,
              encoder_seq_lod,
              decoder_seq_lod,
              encoder_batch_idx,
              decoder_batch_idx,
              encoder_seq_lod_cpu,
              decoder_seq_lod_cpu,
              encoder_batch_idx_cpu,
              decoder_batch_idx_cpu,
              len_info_cpu,
              output_padding_offset,
              max_input_length);
}

std::vector<std::vector<int64_t>> AdjustBatchInferShape(
    const std::vector<int64_t> &x_shape,
    const std::vector<int64_t> &cum_offsets_shape,
    const std::vector<int64_t> &encoder_seq_lod_shape,
    const std::vector<int64_t> &decoder_seq_lod_shape,
    const std::vector<int64_t> &encoder_batch_idx_shape,
    const std::vector<int64_t> &decoder_batch_idx_shape,
    const std::vector<int64_t> &encoder_seq_lod_cpu_shape,
    const std::vector<int64_t> &decoder_seq_lod_cpu_shape,
    const std::vector<int64_t> &encoder_batch_idx_cpu_shape,
    const std::vector<int64_t> &decoder_batch_idx_cpu_shape,
    const std::vector<int64_t> &len_info_cpu_shape,
    const paddle::optional<std::vector<int64_t>> &output_padding_offset_shape) {
  if (output_padding_offset_shape) {
    PD_THROW("speculative decoding is not supported in XPU.");
  }
  int64_t token_num = x_shape[0];
  int64_t dim_embed = x_shape[1];
  return {{token_num, dim_embed}};
}

std::vector<paddle::DataType> AdjustBatchInferDtype(
    const paddle::DataType &x_dtype,
    const paddle::DataType &cum_offsets_dtype,
    const paddle::DataType &encoder_seq_lod_dtype,
    const paddle::DataType &decoder_seq_lod_dtype,
    const paddle::DataType &encoder_batch_idx_dtype,
    const paddle::DataType &decoder_batch_idx_dtype,
    const paddle::DataType &encoder_seq_lod_cpu_dtype,
    const paddle::DataType &decoder_seq_lod_cpu_dtype,
    const paddle::DataType &encoder_batch_idx_cpu_dtype,
    const paddle::DataType &decoder_batch_idx_cpu_dtype,
    const paddle::DataType &len_info_cpu_dtype,
    const paddle::optional<paddle::DataType> &output_padding_offset_dtype) {
  return {x_dtype};
}

PD_BUILD_STATIC_OP(adjust_batch)
    .Inputs({"x",
             "cum_offsets",
             "encoder_seq_lod",
             "decoder_seq_lod",
             "encoder_batch_idx",
             "decoder_batch_idx",
             "encoder_seq_lod_cpu",
             "decoder_seq_lod_cpu",
             "encoder_batch_idx_cpu",
             "decoder_batch_idx_cpu",
             "len_info_cpu",
             paddle::Optional("output_padding_offset")})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(AdjustBatch))
    .SetInferShapeFn(PD_INFER_SHAPE(AdjustBatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AdjustBatchInferDtype));
