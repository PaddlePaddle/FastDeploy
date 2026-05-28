// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "remote_cache_kv_ipc.h"

void SendCacheFunc(const paddle::Tensor& qkv,
                   const paddle::optional<paddle::Tensor>& kv_signal_data) {
  const char* fmt_write_cache_completed_signal_str =
      std::getenv("FLAGS_fmt_write_cache_completed_signal");
  const char* FLAGS_use_pd_disaggregation_per_chunk =
      std::getenv("FLAGS_use_pd_disaggregation_per_chunk");
  if (fmt_write_cache_completed_signal_str &&
      (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
       std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
    if (FLAGS_use_pd_disaggregation_per_chunk &&
        (std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "true") == 0 ||
         std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "1") == 0)) {
      cudaLaunchHostFunc(
          qkv.stream(),
          &(RemoteCacheKvIpc::
                save_cache_kv_complete_signal_layerwise_per_query),
          (void*)nullptr);
    } else {
      if (kv_signal_data) {
        cudaLaunchHostFunc(
            qkv.stream(),
            &RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise,
            (void*)(const_cast<int64_t*>(
                kv_signal_data.get().data<int64_t>())));
      }
    }
  }
}

PD_BUILD_STATIC_OP(send_cache)
    .Inputs({"qkv", paddle::Optional("kv_signal_data")})
    .Outputs({"qkv_out"})
    .SetInplaceMap({{"qkv", "qkv_out"}})
    .SetKernelFn(PD_KERNEL(SendCacheFunc));
