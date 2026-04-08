// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include <xft/xdnn_plugin.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif
template <typename TX>
std::vector<paddle::Tensor> RecoverBatchSequenceKernel(
    const paddle::Tensor& x,            // [token_num, dim_embed]
    const paddle::Tensor& cum_offsets,  // [bsz, 1]
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& len_info_cpu) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto ctx = static_cast<const phi::XPUContext*>(dev_ctx)->x_context();
  if (x.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }
  using XPUType = typename XPUTypeTrait<TX>::Type;
  typedef TX data_t;
  const int dim = x.dims()[1];
  const int token_num = x.shape()[0];
  int bsz = cum_offsets.shape()[0];
  int enc_batch = len_info_cpu.data<int32_t>()[0];
  int dec_batch = len_info_cpu.data<int32_t>()[1];
  baidu::xpu::api::VectorParam<int32_t> encoder_seqs_lods_vp{
      const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()),
      enc_batch + 1,
      const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> decoder_seqs_lods_vp{
      const_cast<int32_t*>(decoder_seq_lod_cpu.data<int32_t>()),
      dec_batch + 1,
      const_cast<int32_t*>(decoder_seq_lod.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> encoder_batch_map_vp{
      const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()),
      enc_batch,
      const_cast<int32_t*>(encoder_batch_map.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> decoder_batch_map_vp{
      const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()),
      dec_batch,
      const_cast<int32_t*>(decoder_batch_map.data<int32_t>())};

  paddle::Tensor out;
  out = paddle::empty({token_num, dim}, x.type(), x.place());

  int r = baidu::xpu::api::plugin::eb_recover_batch_sequence<XPUType, XPUType>(
      ctx,
      reinterpret_cast<const XPUType*>(x.data<data_t>()),
      reinterpret_cast<XPUType*>(out.data<data_t>()),
      encoder_seqs_lods_vp,
      decoder_seqs_lods_vp,
      encoder_batch_map_vp,
      decoder_batch_map_vp,
      dim);
  PD_CHECK(r == 0, "xpu::plugin::eb_recover_batch_sequence failed.");
  return {out};
}

#define APPLY_KERNEL(TX)                                       \
  return RecoverBatchSequenceKernel<TX>(x,                     \
                                        cum_offsets,           \
                                        encoder_seq_lod,       \
                                        decoder_seq_lod,       \
                                        encoder_batch_map,     \
                                        decoder_batch_map,     \
                                        encoder_seq_lod_cpu,   \
                                        decoder_seq_lod_cpu,   \
                                        encoder_batch_map_cpu, \
                                        decoder_batch_map_cpu, \
                                        len_info_cpu);

std::vector<paddle::Tensor> RecoverBatchSequence(
    const paddle::Tensor& x,            // [token_num, dim_embed]
    const paddle::Tensor& cum_offsets,  // [bsz, 1]
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& len_info_cpu) {
  const auto x_dtype = x.dtype();
  if (x_dtype == paddle::DataType::BFLOAT16) {
    APPLY_KERNEL(paddle::bfloat16);
  } else if (x_dtype == paddle::DataType::FLOAT16) {
    APPLY_KERNEL(paddle::float16);
  } else if (x_dtype == paddle::DataType::FLOAT32) {
    APPLY_KERNEL(float);
  } else {
    PD_THROW("recover_batch_sequence not support x_dtype=",
             static_cast<int>(x_dtype));
    return {};
  }
}

std::vector<std::vector<int64_t>> RecoverBatchSequenceInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& encoder_seq_lod_shape,
    const std::vector<int64_t>& decoder_seq_lod_shape,
    const std::vector<int64_t>& encoder_batch_map_shape,
    const std::vector<int64_t>& decoder_batch_map_shape,
    const std::vector<int64_t>& encoder_seq_lod_cpu_shape,
    const std::vector<int64_t>& decoder_seq_lod_cpu_shape,
    const std::vector<int64_t>& encoder_batch_map_cpu_shape,
    const std::vector<int64_t>& decoder_batch_map_cpu_shape,
    const std::vector<int64_t>& len_info_cpu_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> RecoverBatchSequenceInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& encoder_seq_lod_dtype,
    const paddle::DataType& decoder_seq_lod_dtype,
    const paddle::DataType& encoder_batch_map_dtype,
    const paddle::DataType& decoder_batch_map_dtype,
    const paddle::DataType& encoder_seq_lod_cpu_dtype,
    const paddle::DataType& decoder_seq_lod_cpu_dtype,
    const paddle::DataType& encoder_batch_map_cpu_dtype,
    const paddle::DataType& decoder_batch_map_cpu_dtype,
    const paddle::DataType& len_info_cpu_dtype) {
  return {x_dtype};
}

PD_BUILD_STATIC_OP(recover_batch_sequence)
    .Inputs({"x",
             "cum_offsets",
             "encoder_seq_lod",
             "decoder_seq_lod",
             "encoder_batch_map",
             "decoder_batch_map",
             "encoder_seq_lod_cpu",
             "decoder_seq_lod_cpu",
             "encoder_batch_map_cpu",
             "decoder_batch_map_cpu",
             "len_info_cpu"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(RecoverBatchSequence))
    .SetInferShapeFn(PD_INFER_SHAPE(RecoverBatchSequenceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RecoverBatchSequenceInferDtype));
