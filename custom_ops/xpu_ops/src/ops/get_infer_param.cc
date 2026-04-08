// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/internal/infra_op.h"
#include "xpu/plugin.h"
namespace api = baidu::xpu::api;

std::vector<paddle::Tensor> GetInferParam(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& block_tables,
    int block_size) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  const int bsz = seq_lens_encoder.dims()[0];
  const int block_bs = block_tables.dims()[0];
  const int block_num_per_seq = block_tables.dims()[1];
  auto all_param = paddle::empty(
      {bsz * 3}, seq_lens_encoder.type(), seq_lens_encoder.place());
  int ret = api::copy<int32_t>(xpu_ctx->x_context(),
                               seq_lens_encoder.data<int32_t>(),
                               reinterpret_cast<int32_t*>(all_param.data()),
                               bsz);
  ret = api::copy<int32_t>(xpu_ctx->x_context(),
                           seq_lens_decoder.data<int32_t>(),
                           reinterpret_cast<int32_t*>(all_param.data()) + bsz,
                           bsz);
  ret =
      api::copy<int32_t>(xpu_ctx->x_context(),
                         seq_lens_this_time.data<int32_t>(),
                         reinterpret_cast<int32_t*>(all_param.data()) + 2 * bsz,
                         bsz);
  std::unique_ptr<int32_t[]> all_param_cpu(new int32_t[bsz * 3]);
  // input ex: [100, 0,  0, 0,  300]
  int32_t* seq_lens_encoder_vec = all_param_cpu.get();
  // input ex: [0,   5,  0, 25, 64] (64 means prefix len)
  int32_t* seq_lens_decoder_vec = all_param_cpu.get() + bsz;
  int32_t* seq_lens_this_time_vec = all_param_cpu.get() + 2 * bsz;

  std::vector<int32_t> encoder_batch_map_vec(bsz, 0);
  std::vector<int32_t> decoder_batch_map_vec(
      bsz, 0);  // ex : [1, 3]
                // 去除空隙的batch map ex : [0, 3]
  std::vector<int32_t> encoder_batch_idx_vec(bsz, 0);
  // 去除空隙的batch map ex : [1, 2]
  std::vector<int32_t> decoder_batch_idx_vec(bsz, 0);
  std::vector<int32_t> encoder_seq_lod_vec(bsz + 1, 0);  // ex : [0, 100, 400]
  std::vector<int32_t> decoder_seq_lod_vec(bsz + 1, 0);
  std::vector<int32_t> encoder_kv_lod_vec(bsz + 1, 0);   // ex : [0, 100, 464]
  std::vector<int32_t> prefix_len_vec(bsz, 0);           // ex : [0, 64]
  std::vector<int32_t> decoder_context_len_vec(bsz, 0);  // ex : [6, 26]
  std::vector<int32_t> decoder_context_len_cache_vec(bsz, 0);  // ex : [5, 25]
  xpu_wait(xpu_ctx->x_context()->xpu_stream);
  int r = xpu_memcpy(all_param_cpu.get(),
                     all_param.data<int32_t>(),
                     sizeof(int32_t) * 3 * bsz,
                     XPUMemcpyKind::XPU_DEVICE_TO_HOST);

  int enc_batch = 0, dec_batch = 0;
  int total_enc_len = 0;
  int batch_offset = 0;
  int max_seq_len = 0;
  int max_prefix_len = 0;
  int max_kv_len = 0;
  for (int i = 0; i < bsz; ++i) {
    if (seq_lens_encoder_vec[i] > 0) {
      enc_batch++;
      int seq_len = seq_lens_encoder_vec[i];
      int prefix_len = seq_lens_decoder_vec[i];
      total_enc_len += seq_len;
      max_seq_len = std::max(max_seq_len, seq_len);
      max_prefix_len = std::max(max_prefix_len, prefix_len);
      max_kv_len = std::max(max_kv_len, seq_len + prefix_len);
      encoder_batch_map_vec[enc_batch - 1] = i;
      encoder_batch_idx_vec[enc_batch - 1] = i - batch_offset;
      encoder_seq_lod_vec[enc_batch] =
          seq_len + encoder_seq_lod_vec[enc_batch - 1];
      encoder_kv_lod_vec[enc_batch] =
          seq_len + prefix_len + encoder_kv_lod_vec[enc_batch - 1];
      prefix_len_vec[enc_batch - 1] = prefix_len;
    } else if (seq_lens_decoder_vec[i] > 0) {
      dec_batch++;
      decoder_batch_map_vec[dec_batch - 1] = i;
      decoder_batch_idx_vec[dec_batch - 1] = i - batch_offset;
      decoder_context_len_vec[dec_batch - 1] =
          seq_lens_decoder_vec[i] + seq_lens_this_time_vec[i];
      decoder_context_len_cache_vec[dec_batch - 1] = seq_lens_decoder_vec[i];
      decoder_seq_lod_vec[dec_batch] =
          seq_lens_this_time_vec[i] +
          decoder_seq_lod_vec[dec_batch - 1];  // use for mtp
    } else {
      batch_offset++;
    }
  }
  int prefix_block_num_per_seq = (max_kv_len + block_size - 1) / block_size;
  std::vector<int32_t> prefix_block_tables_vec(
      enc_batch * prefix_block_num_per_seq, -1);
  if (max_prefix_len > 0) {
    std::vector<int> block_tables_vec(block_bs * block_num_per_seq, -1);
    r = xpu_memcpy(block_tables_vec.data(),
                   block_tables.data<int32_t>(),
                   sizeof(int32_t) * block_bs * block_num_per_seq,
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    for (int i = 0; i < enc_batch; i++) {
      int src_bs = encoder_batch_map_vec[i];
      int copy_len =
          (encoder_kv_lod_vec[i + 1] - encoder_kv_lod_vec[i] + block_size - 1) /
          block_size;
      std::memcpy(prefix_block_tables_vec.data() + i * prefix_block_num_per_seq,
                  block_tables_vec.data() + src_bs * block_num_per_seq,
                  copy_len * sizeof(int32_t));
    }
  } else {
    prefix_block_num_per_seq = -1;
  }

  auto encoder_batch_map = paddle::empty({encoder_batch_map_vec.size()},
                                         seq_lens_encoder.type(),
                                         seq_lens_encoder.place());
  auto decoder_batch_map = paddle::empty({decoder_batch_map_vec.size()},
                                         seq_lens_encoder.type(),
                                         seq_lens_encoder.place());
  auto encoder_batch_idx = paddle::empty({encoder_batch_idx_vec.size()},
                                         seq_lens_encoder.type(),
                                         seq_lens_encoder.place());
  auto decoder_batch_idx = paddle::empty({decoder_batch_idx_vec.size()},
                                         seq_lens_encoder.type(),
                                         seq_lens_encoder.place());
  auto encoder_seq_lod = paddle::empty({encoder_seq_lod_vec.size()},
                                       seq_lens_encoder.type(),
                                       seq_lens_encoder.place());
  auto decoder_seq_lod = paddle::empty({decoder_seq_lod_vec.size()},
                                       seq_lens_encoder.type(),
                                       seq_lens_encoder.place());
  auto encoder_kv_lod = paddle::empty({encoder_kv_lod_vec.size()},
                                      seq_lens_encoder.type(),
                                      seq_lens_encoder.place());
  auto prefix_len = paddle::empty({prefix_len_vec.size()},
                                  seq_lens_encoder.type(),
                                  seq_lens_encoder.place());
  auto decoder_context_len = paddle::empty({decoder_context_len_vec.size()},
                                           seq_lens_encoder.type(),
                                           seq_lens_encoder.place());
  auto decoder_context_len_cache =
      paddle::empty({decoder_context_len_cache_vec.size()},
                    seq_lens_encoder.type(),
                    seq_lens_encoder.place());
  auto prefix_block_tables =
      paddle::empty({block_bs, block_num_per_seq},  // full size
                    seq_lens_encoder.type(),
                    seq_lens_encoder.place());

  auto encoder_batch_map_cpu = paddle::empty({encoder_batch_map_vec.size()},
                                             seq_lens_encoder.type(),
                                             paddle::CPUPlace());
  auto decoder_batch_map_cpu = paddle::empty({decoder_batch_map_vec.size()},
                                             seq_lens_encoder.type(),
                                             paddle::CPUPlace());
  auto encoder_batch_idx_cpu = paddle::empty({encoder_batch_idx_vec.size()},
                                             seq_lens_encoder.type(),
                                             paddle::CPUPlace());
  auto decoder_batch_idx_cpu = paddle::empty({decoder_batch_idx_vec.size()},
                                             seq_lens_encoder.type(),
                                             paddle::CPUPlace());
  auto encoder_seq_lod_cpu = paddle::empty({encoder_seq_lod_vec.size()},
                                           seq_lens_encoder.type(),
                                           paddle::CPUPlace());
  auto decoder_seq_lod_cpu = paddle::empty({decoder_seq_lod_vec.size()},
                                           seq_lens_encoder.type(),
                                           paddle::CPUPlace());

  auto encoder_kv_lod_cpu = paddle::empty(
      {encoder_kv_lod_vec.size()}, seq_lens_encoder.type(), paddle::CPUPlace());
  auto prefix_len_cpu = paddle::empty(
      {prefix_len_vec.size()}, seq_lens_encoder.type(), paddle::CPUPlace());
  auto decoder_context_len_cpu = paddle::empty({decoder_context_len_vec.size()},
                                               seq_lens_encoder.type(),
                                               paddle::CPUPlace());
  auto decoder_context_len_cache_cpu =
      paddle::empty({decoder_context_len_cache_vec.size()},
                    seq_lens_encoder.type(),
                    paddle::CPUPlace());

  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(encoder_batch_map_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(encoder_batch_map.data<int32_t>())),
      sizeof(int32_t) * encoder_batch_map_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(decoder_batch_map_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(decoder_batch_map.data<int32_t>())),
      sizeof(int32_t) * decoder_batch_map_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(encoder_batch_idx_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(encoder_batch_idx.data<int32_t>())),
      sizeof(int32_t) * encoder_batch_idx_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(decoder_batch_idx_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(decoder_batch_idx.data<int32_t>())),
      sizeof(int32_t) * decoder_batch_idx_vec.size());
  ret = api::do_host2device(xpu_ctx->x_context(),
                            reinterpret_cast<void*>(encoder_seq_lod_vec.data()),
                            reinterpret_cast<void*>(const_cast<int32_t*>(
                                encoder_seq_lod.data<int32_t>())),
                            sizeof(int32_t) * encoder_seq_lod_vec.size());
  ret = api::do_host2device(xpu_ctx->x_context(),
                            reinterpret_cast<void*>(decoder_seq_lod_vec.data()),
                            reinterpret_cast<void*>(const_cast<int32_t*>(
                                decoder_seq_lod.data<int32_t>())),
                            sizeof(int32_t) * decoder_seq_lod_vec.size());
  ret = api::do_host2device(xpu_ctx->x_context(),
                            reinterpret_cast<void*>(encoder_kv_lod_vec.data()),
                            reinterpret_cast<void*>(const_cast<int32_t*>(
                                encoder_kv_lod.data<int32_t>())),
                            sizeof(int32_t) * encoder_kv_lod_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(prefix_len_vec.data()),
      reinterpret_cast<void*>(const_cast<int32_t*>(prefix_len.data<int32_t>())),
      sizeof(int32_t) * prefix_len_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(decoder_context_len_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(decoder_context_len.data<int32_t>())),
      sizeof(int32_t) * decoder_context_len_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(decoder_context_len_cache_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(decoder_context_len_cache.data<int32_t>())),
      sizeof(int32_t) * decoder_context_len_cache_vec.size());
  ret = api::do_host2device(
      xpu_ctx->x_context(),
      reinterpret_cast<void*>(prefix_block_tables_vec.data()),
      reinterpret_cast<void*>(
          const_cast<int32_t*>(prefix_block_tables.data<int32_t>())),
      sizeof(int32_t) * prefix_block_tables_vec.size());

  std::memcpy(encoder_batch_map_cpu.data<int32_t>(),
              encoder_batch_map_vec.data(),
              sizeof(int32_t) * encoder_batch_map_vec.size());
  std::memcpy(decoder_batch_map_cpu.data<int32_t>(),
              decoder_batch_map_vec.data(),
              sizeof(int32_t) * decoder_batch_map_vec.size());
  std::memcpy(encoder_batch_idx_cpu.data<int32_t>(),
              encoder_batch_idx_vec.data(),
              sizeof(int32_t) * encoder_batch_idx_vec.size());
  std::memcpy(decoder_batch_idx_cpu.data<int32_t>(),
              decoder_batch_idx_vec.data(),
              sizeof(int32_t) * decoder_batch_idx_vec.size());
  std::memcpy(encoder_seq_lod_cpu.data<int32_t>(),
              encoder_seq_lod_vec.data(),
              sizeof(int32_t) * encoder_seq_lod_vec.size());
  std::memcpy(decoder_seq_lod_cpu.data<int32_t>(),
              decoder_seq_lod_vec.data(),
              sizeof(int32_t) * decoder_seq_lod_vec.size());
  std::memcpy(encoder_kv_lod_cpu.data<int32_t>(),
              encoder_kv_lod_vec.data(),
              sizeof(int32_t) * encoder_kv_lod_vec.size());
  std::memcpy(prefix_len_cpu.data<int32_t>(),
              prefix_len_vec.data(),
              sizeof(int32_t) * prefix_len_vec.size());
  std::memcpy(decoder_context_len_cpu.data<int32_t>(),
              decoder_context_len_vec.data(),
              sizeof(int32_t) * decoder_context_len_vec.size());
  std::memcpy(decoder_context_len_cache_cpu.data<int32_t>(),
              decoder_context_len_cache_vec.data(),
              sizeof(int32_t) * decoder_context_len_cache_vec.size());

  std::vector<int> len_info_vec = {enc_batch,
                                   dec_batch,
                                   total_enc_len,
                                   max_seq_len,
                                   max_kv_len,
                                   prefix_block_num_per_seq};
  auto len_info_cpu =
      paddle::empty({6}, seq_lens_encoder.type(), paddle::CPUPlace());
  std::memcpy(len_info_cpu.data<int32_t>(),
              len_info_vec.data(),
              sizeof(int32_t) * len_info_vec.size());

  return {encoder_batch_map,
          decoder_batch_map,
          encoder_batch_idx,
          decoder_batch_idx,
          encoder_seq_lod,
          decoder_seq_lod,
          encoder_kv_lod,
          prefix_len,
          decoder_context_len,
          decoder_context_len_cache,
          prefix_block_tables,
          encoder_batch_map_cpu,
          decoder_batch_map_cpu,
          encoder_batch_idx_cpu,
          decoder_batch_idx_cpu,
          encoder_seq_lod_cpu,
          decoder_seq_lod_cpu,
          encoder_kv_lod_cpu,
          prefix_len_cpu,
          decoder_context_len_cpu,
          decoder_context_len_cache_cpu,
          len_info_cpu};
}

std::vector<std::vector<int64_t>> GetInferParamInferShape(
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& block_tables_shape) {
  return {seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          {seq_lens_encoder_shape[0] + 1},
          {seq_lens_encoder_shape[0] + 1},
          {seq_lens_encoder_shape[0] + 1},
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          block_tables_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          {seq_lens_encoder_shape[0] + 1},
          {seq_lens_encoder_shape[0] + 1},
          {seq_lens_encoder_shape[0] + 1},
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          seq_lens_encoder_shape,
          {6}};
}

std::vector<paddle::DataType> GetInferParamInferDtype(
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& block_tables_dtype) {
  return {
      seq_lens_encoder_dtype, seq_lens_encoder_dtype, seq_lens_encoder_dtype,
      seq_lens_encoder_dtype, seq_lens_encoder_dtype, seq_lens_encoder_dtype,
      seq_lens_encoder_dtype, seq_lens_encoder_dtype, seq_lens_encoder_dtype,
      seq_lens_encoder_dtype, block_tables_dtype,     seq_lens_encoder_dtype,
      seq_lens_encoder_dtype, seq_lens_encoder_dtype, seq_lens_encoder_dtype,
      seq_lens_encoder_dtype, seq_lens_encoder_dtype, seq_lens_encoder_dtype,
      seq_lens_encoder_dtype, seq_lens_encoder_dtype, seq_lens_encoder_dtype,
      seq_lens_encoder_dtype};
}

PD_BUILD_OP(get_infer_param)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "block_tables"})
    .Outputs({"encoder_batch_map",
              "decoder_batch_map",
              "encoder_batch_idx",
              "decoder_batch_idx",
              "encoder_seq_lod",
              "decoder_seq_lod",
              "encoder_kv_lod",
              "prefix_len",
              "decoder_context_len",
              "decoder_context_len_cache",
              "prefix_block_tables",
              "encoder_batch_map_cpu",
              "decoder_batch_map_cpu",
              "encoder_batch_idx_cpu",
              "decoder_batch_idx_cpu",
              "encoder_seq_lod_cpu",
              "decoder_seq_lod_cpu",
              "encoder_kv_lod_cpu",
              "prefix_len_cpu",
              "decoder_context_len_cpu",
              "decoder_context_len_cache_cpu",
              "len_info_cpu"})
    .SetKernelFn(PD_KERNEL(GetInferParam))
    .Attrs({"block_size: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(GetInferParamInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetInferParamInferDtype));
