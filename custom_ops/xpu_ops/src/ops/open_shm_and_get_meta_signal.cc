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

#include "ops/pybind/pybind.h"
#include "ops/remote_cache_kv_ipc.h"
#include "ops/utility/env.h"
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

XPU_DECLARE_BOOL(fmt_write_cache_completed_signal, false);

using cache_write_complete_signal_type =
    RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data;

paddle::Tensor OpenShmAndGetMetaSignalFunc(const int rank,
                                           const int device_id,
                                           const bool keep_pd_step_flag) {
  cache_write_complete_signal_type kv_signal_metadata;
  const char* fmt_write_cache_completed_signal_str =
      std::getenv("FLAGS_fmt_write_cache_completed_signal");
  if (fmt_write_cache_completed_signal_str &&
      (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
       std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
    kv_signal_metadata =
        RemoteCacheKvIpc::open_shm_and_get_complete_signal_meta_data(
            rank, device_id, keep_pd_step_flag);
  }

  auto kv_signal_metadata_out =
      paddle::full({3}, -1, paddle::DataType::INT64, paddle::CPUPlace());
  kv_signal_metadata_out.data<int64_t>()[0] =
      static_cast<int64_t>(kv_signal_metadata.layer_id);
  kv_signal_metadata_out.data<int64_t>()[1] =
      reinterpret_cast<int64_t>(kv_signal_metadata.shm_ptr);
  kv_signal_metadata_out.data<int64_t>()[2] =
      static_cast<int64_t>(kv_signal_metadata.shm_fd);
  return kv_signal_metadata_out;
}

void InitKVSignalPerQuery(const paddle::Tensor& seq_lens_encoder_tensor,
                          const paddle::Tensor& seq_lens_this_time_tensor,
                          const paddle::Tensor& seq_lens_decoder_tensor,
                          const int rank,
                          const int num_layers) {
  if (FLAGS_fmt_write_cache_completed_signal) {
    int real_bsz = seq_lens_this_time_tensor.dims()[0];
    // GPU init, cp to cpu?
    auto seq_lens_encoder_cpu =
        seq_lens_encoder_tensor.copy_to(paddle::CPUPlace(), false);
    auto seq_lens_decoder_cpu =
        seq_lens_decoder_tensor.copy_to(paddle::CPUPlace(), false);
    RemoteCacheKvIpc::kv_complete_signal_meta_data_per_query.init(
        seq_lens_encoder_cpu.data<int>(),
        seq_lens_decoder_cpu.data<int>(),
        rank,
        num_layers,
        real_bsz);
  }
}

std::vector<paddle::Tensor> OpenShmAndGetMetaSignal(
    const int rank, const int device_id, const bool keep_pd_step_flag) {
  return {OpenShmAndGetMetaSignalFunc(rank, device_id, keep_pd_step_flag)};
}

std::vector<std::vector<int64_t>> OpenShmAndGetMetaSignalShape(
    const int rank, const int device_id, const bool keep_pd_step_flag) {
  return {{3}};
}

std::vector<paddle::DataType> OpenShmAndGetMetaSignalDtype(
    const int rank, const int device_id, const bool keep_pd_step_flag) {
  return {paddle::DataType::INT64};
}

PD_BUILD_STATIC_OP(open_shm_and_get_meta_signal)
    .Inputs({})
    .Outputs({"kv_signal_metadata"})
    .Attrs({"rank: int", "device_id: int", "keep_pd_step_flag: bool"})
    .SetKernelFn(PD_KERNEL(OpenShmAndGetMetaSignal))
    .SetInferShapeFn(PD_INFER_SHAPE(OpenShmAndGetMetaSignalShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(OpenShmAndGetMetaSignalDtype));
