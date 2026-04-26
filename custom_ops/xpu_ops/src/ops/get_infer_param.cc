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
#include "ops/utility/env.h"

XPU_DECLARE_BOOL(encoder_splice, false);
XPU_DECLARE_BOOL(decoder_splice, false);

namespace api = baidu::xpu::api;

void lod_to_slot_mapping(api::Context* xpu_ctx,
                         paddle::Place place,
                         const paddle::Tensor& block_table_xpu,
                         const std::vector<int32_t>& kv_seq_lod,
                         const std::vector<int32_t>& start_tokens,
                         const std::vector<int32_t>& real_batch,
                         int32_t* slot_mapping,
                         int32_t token_num,
                         int32_t block_size,
                         int32_t batch_size,
                         int32_t max_num_blocks_per_seq,
                         int32_t num_speculative_tokens) {
  int32_t actual_token_num = kv_seq_lod[batch_size];
  if (token_num <= 0 || actual_token_num <= 0) {
    return;
  }

  int ret;

  std::vector<int32_t> block_table_idx_vec(actual_token_num);
  std::vector<int32_t> seq_offset_vec(actual_token_num);

  int32_t idx = 0;
  // For each Batch
  for (auto batch_ = 0; batch_ < batch_size; batch_++) {
    int32_t seq_len = kv_seq_lod[batch_ + 1] - kv_seq_lod[batch_];
    int32_t seq_start = start_tokens[batch_];
    int32_t dst_batch_id = real_batch[batch_];
    // for each token
    for (auto seq_ = seq_start; seq_ < seq_start + seq_len; seq_++) {
      block_table_idx_vec[idx] =
          seq_ / block_size + dst_batch_id * max_num_blocks_per_seq;
      seq_offset_vec[idx] = seq_ % block_size;
      idx++;
    }
  }
  auto block_table_idx =
      paddle::empty({actual_token_num}, paddle::DataType::INT32, place);
  auto seq_offset =
      paddle::empty({actual_token_num}, paddle::DataType::INT32, place);
  ret = api::do_host2device(xpu_ctx,
                            block_table_idx_vec.data(),
                            block_table_idx.data<int32_t>(),
                            actual_token_num * sizeof(int32_t));
  PD_CHECK(ret == api::SUCCESS, "api::do_host2device failed.");
  ret = api::do_host2device(xpu_ctx,
                            seq_offset_vec.data(),
                            seq_offset.data<int32_t>(),
                            actual_token_num * sizeof(int32_t));
  PD_CHECK(ret == api::SUCCESS, "api::do_host2device failed.");

  // int32_t block_id =
  //     block_table[dst_batch_id * max_num_blocks_per_seq + table_id];
  auto block_id =
      paddle::empty({actual_token_num}, paddle::DataType::INT32, place);
  auto block_size_tensor =
      paddle::full({1}, block_size, paddle::DataType::INT32, place);
  ret = api::index_select<int32_t, int32_t>(xpu_ctx,
                                            block_table_xpu.data<int32_t>(),
                                            block_table_idx.data<int32_t>(),
                                            block_id.data<int32_t>(),
                                            {block_table_xpu.numel()},
                                            actual_token_num,
                                            0);
  PD_CHECK(ret == api::SUCCESS, "api::index_select failed.");
  // int32_t dst_token_offset = block_id * block_size + seq_offset;
  ret = api::broadcast_mul<int32_t>(xpu_ctx,
                                    block_id.data<int32_t>(),
                                    block_size_tensor.data<int32_t>(),
                                    block_id.data<int32_t>(),
                                    {actual_token_num},
                                    {1});
  PD_CHECK(ret == api::SUCCESS, "api::broadcast_mul failed.");
  ret = api::broadcast_add<int32_t>(xpu_ctx,
                                    block_id.data<int32_t>(),
                                    seq_offset.data<int32_t>(),
                                    slot_mapping,
                                    {actual_token_num},
                                    {actual_token_num});
  PD_CHECK(ret == api::SUCCESS, "api::broadcast_add failed.");
}

std::vector<paddle::Tensor> GetInferParam(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& block_tables,
    paddle::Tensor& encoder_batch_map,
    paddle::Tensor& decoder_batch_map,
    paddle::Tensor& encoder_batch_idx,
    paddle::Tensor& decoder_batch_idx,
    paddle::Tensor& encoder_seq_lod,
    paddle::Tensor& decoder_seq_lod,
    paddle::Tensor& encoder_kv_lod,
    paddle::Tensor& prefix_len,
    paddle::Tensor& decoder_context_len,
    paddle::Tensor& decoder_context_len_cache,
    paddle::Tensor& prefix_block_tables,
    paddle::Tensor& encoder_batch_map_cpu,
    paddle::Tensor& decoder_batch_map_cpu,
    paddle::Tensor& encoder_batch_idx_cpu,
    paddle::Tensor& decoder_batch_idx_cpu,
    paddle::Tensor& encoder_seq_lod_cpu,
    paddle::Tensor& decoder_seq_lod_cpu,
    paddle::Tensor& encoder_kv_lod_cpu,
    paddle::Tensor& prefix_len_cpu,
    paddle::Tensor& decoder_context_len_cpu,
    paddle::Tensor& decoder_context_len_cache_cpu,
    paddle::Tensor& len_info_cpu,
    int block_size,
    int num_speculative_tokens) {
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
  int max_dec_len = 0;
  for (int i = 0; i < bsz; ++i) {
    if (seq_lens_encoder_vec[i] > 0) {
      enc_batch++;
      int seq_len = seq_lens_encoder_vec[i];
      int prefix_len_int = seq_lens_decoder_vec[i];
      total_enc_len += seq_len;
      max_seq_len = std::max(max_seq_len, seq_len);
      max_prefix_len = std::max(max_prefix_len, prefix_len_int);
      max_kv_len = std::max(max_kv_len, seq_len + prefix_len_int);
      encoder_batch_map_vec[enc_batch - 1] = i;
      encoder_batch_idx_vec[enc_batch - 1] = i - batch_offset;
      encoder_seq_lod_vec[enc_batch] =
          seq_len + encoder_seq_lod_vec[enc_batch - 1];
      encoder_kv_lod_vec[enc_batch] =
          seq_len + prefix_len_int + encoder_kv_lod_vec[enc_batch - 1];
      prefix_len_vec[enc_batch - 1] = prefix_len_int;
    } else if (seq_lens_decoder_vec[i] > 0 && seq_lens_this_time_vec[i] > 0) {
      dec_batch++;
      max_dec_len = std::max(max_dec_len, seq_lens_this_time_vec[i]);
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
  // for vsl_rotary_embedding_gptj of cudagraph mode
  int prev_val = 0;
  for (int i = 0; i < bsz + 1; i++) {
    if (decoder_seq_lod_vec[i] > prev_val) {
      prev_val = decoder_seq_lod_vec[i];
    } else if (decoder_seq_lod_vec[i] < prev_val) {
      decoder_seq_lod_vec[i] = prev_val;
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

  // for store_paged_kv_cache of cudagraph mode
  // if slot_mapping is -1, store_paged_kv_cache will not write to kv cache
  paddle::Tensor slot_mapping_enc = paddle::full(
      {total_enc_len}, -1, paddle::DataType::INT32, seq_lens_encoder.place());
  // TODO: mtp mode not verified yet, need further adaption
  paddle::Tensor slot_mapping_dec =
      paddle::full({bsz * (1 + num_speculative_tokens)},
                   -1,
                   paddle::DataType::INT32,
                   seq_lens_decoder.place());
  if (FLAGS_encoder_splice) {
    lod_to_slot_mapping(xpu_ctx->x_context(),
                        seq_lens_encoder.place(),
                        block_tables,
                        encoder_seq_lod_vec,
                        prefix_len_vec,
                        encoder_batch_map_vec,
                        slot_mapping_enc.data<int32_t>(),
                        slot_mapping_enc.numel(),
                        block_size,
                        enc_batch,
                        block_num_per_seq,
                        0);
  }
  if (FLAGS_decoder_splice) {
    lod_to_slot_mapping(xpu_ctx->x_context(),
                        seq_lens_decoder.place(),
                        block_tables,
                        decoder_seq_lod_vec,
                        decoder_context_len_cache_vec,
                        decoder_batch_map_vec,
                        slot_mapping_dec.data<int32_t>(),
                        slot_mapping_dec.numel(),
                        block_size,
                        dec_batch,
                        block_num_per_seq,
                        num_speculative_tokens);
  }

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
                                   prefix_block_num_per_seq,
                                   max_dec_len};

  std::memcpy(len_info_cpu.data<int32_t>(),
              len_info_vec.data(),
              sizeof(int32_t) * len_info_vec.size());

  return {slot_mapping_enc, slot_mapping_dec};
}

std::vector<std::vector<int64_t>> GetInferParamInferShape(
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& block_tables_shape,
    int num_speculative_tokens) {
  // Return shapes for slot_mapping_enc and slot_mapping_dec
  // slot_mapping_enc shape depends on encoder token count (unknown at shape
  // inference time) slot_mapping_dec shape depends on batch size and
  // speculative token count
  return {{-1}, {seq_lens_encoder_shape[0] * (1 + num_speculative_tokens)}};
}

std::vector<paddle::DataType> GetInferParamInferDtype(
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& block_tables_dtype) {
  // Return dtypes for slot_mapping_enc and slot_mapping_dec (both INT32)
  return {paddle::DataType::INT32, paddle::DataType::INT32};
}

PD_BUILD_OP(get_infer_param)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "block_tables",
             "encoder_batch_map",
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
    .Outputs({"slot_mapping_enc", "slot_mapping_dec"})
    .SetKernelFn(PD_KERNEL(GetInferParam))
    .Attrs({"block_size: int", "num_speculative_tokens: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(GetInferParamInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetInferParamInferDtype));
