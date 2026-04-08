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

#include "paddle/extension.h"
#include "remote_cache_kv_ipc.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

using cache_write_complete_signal_type = RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data;

paddle::Tensor InitSignalLayerwiseFunc(const paddle::Tensor& kv_signal_metadata, const int layer_id) {
    auto kv_signal_metadata_out = kv_signal_metadata.copy_to(paddle::CPUPlace(), false);
    kv_signal_metadata_out.data<int64_t>()[0] = static_cast<int64_t>(layer_id);
    return kv_signal_metadata_out;
}

std::vector<paddle::Tensor> InitSignalLayerwise(const paddle::Tensor& kv_signal_metadata, const int layer_id) {
    return {InitSignalLayerwiseFunc(kv_signal_metadata, layer_id)};
}

std::vector<std::vector<int64_t>> InitSignalLayerwiseShape(
    const std::vector<int64_t>& kv_signal_metadata_shape,
    const int layer_id) {
    return {kv_signal_metadata_shape};
}

std::vector<paddle::DataType> InitSignalLayerwiseDtype(
    const paddle::DataType& kv_signal_metadata_dtype,
    const int layer_id) {
    return {paddle::DataType::INT64};
}

PD_BUILD_STATIC_OP(init_signal_layerwise)
    .Inputs({"kv_signal_metadata"})
    .Outputs({"kv_signal_metadata_out"})
    .Attrs({"layer_id: int"})
    .SetKernelFn(PD_KERNEL(InitSignalLayerwise))
    .SetInferShapeFn(PD_INFER_SHAPE(InitSignalLayerwiseShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InitSignalLayerwiseDtype));
