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

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

__device__ bool is_in_list(const int64_t id, const int64_t *ids, int bs_id) {
    bool is_in_list = false;
    for (int i = 0; i < bs_id; i++) {
        if (ids[i] == id) {
            return true;
        }
    }
    return is_in_list;
}

__global__ void set_value_by_id(const bool *stop_flags,
                                const int64_t *ids,
                                bool *stop_flags_out,
                                int bs,
                                int bs_id) {
    int tid = threadIdx.x;
    if (tid < bs && !is_in_list(tid, ids, bs_id)) {
        stop_flags_out[tid] = true;
    }
}

std::vector<paddle::Tensor> SetFlags(const paddle::Tensor &stop_flags,
                                     const paddle::Tensor &gather_id) {
    PD_CHECK(gather_id.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
    auto cu_stream = stop_flags.stream();
    std::vector<int64_t> flag_shape = stop_flags.shape();
    std::vector<int64_t> id_shape = gather_id.shape();
    auto stop_flags_out =
        stop_flags.copy_to(stop_flags.place(), false);  // gpu -> gpu
    if (flag_shape[0] == id_shape[0]) {
        return {stop_flags_out};
    }
    int flag_bs = flag_shape[0];
    int id_bs = id_shape[0];
    int block_size = (flag_bs + 32 - 1) / 32 * 32;
    set_value_by_id<<<1, block_size, 0, cu_stream>>>(
        stop_flags.data<bool>(),
        gather_id.data<int64_t>(),
        stop_flags_out.data<bool>(),
        flag_bs,
        id_bs);
    return {stop_flags_out};
}

std::vector<std::vector<int64_t>> SetFlagsInferShape(
    const std::vector<int64_t> &stop_flags_shape,
    const std::vector<int64_t> &gather_id_shape) {
    return {stop_flags_shape};
}

std::vector<paddle::DataType> SetFlagsInferDtype(
    const paddle::DataType &stop_flags_dtype,
    const paddle::DataType &gather_id_dtype) {
    return {stop_flags_dtype};
}

PD_BUILD_STATIC_OP(set_flags)
    .Inputs({"stop_flags", "gather_id"})
    .Outputs({"stop_flags_out"})
    .SetKernelFn(PD_KERNEL(SetFlags))
    .SetInferShapeFn(PD_INFER_SHAPE(SetFlagsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetFlagsInferDtype));
