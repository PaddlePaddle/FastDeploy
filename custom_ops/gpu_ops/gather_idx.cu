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

#ifdef PADDLE_WITH_HIP
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
    public:
    typedef float DataType;
    typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
    public:
    typedef half DataType;
    typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
    public:
#ifdef PADDLE_WITH_HIP
    typedef hip_bfloat16 DataType;
#else
    typedef __nv_bfloat16 DataType;
#endif
    typedef paddle::bfloat16 data_t;
};

template <typename T>
__global__ void get_value_by_id(const T *logits,
                                const int *ids,
                                T *logits_out,
                                int bs,
                                int seq_len,
                                int length) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;
    for (int i = idx; i < bs * length; i += gridDim.x * blockDim.x) {
        int bi = i / length;
        int lane = i % length;
        int si = ids[bi];
        if (si == -1) {
            si = 0;
        }
        const T *logits_now = logits + bi * seq_len * length + si * length;
        T *logits_out_now = logits_out + bi * length;
        logits_out_now[lane] = logits_now[lane];
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> gather_idx(const paddle::Tensor &logits,
                                       const paddle::Tensor &gather_id) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    PD_CHECK(gather_id.dtype() == paddle::DataType::INT32);
    auto cu_stream = logits.stream();
    std::vector<int64_t> logits_shape = logits.shape();
    std::vector<int64_t> id_shape = gather_id.shape();
    int logits_bs = logits_shape[0];
    int seq_len = logits_shape[1];
    int logits_len = logits_shape[2];
    auto logits_out =
        paddle::empty({logits_bs, logits_len}, logits.type(), logits.place());
    int id_bs = id_shape[0];
    int64_t numels = logits_bs * logits_len;
    int block_size = 128;
    int grid_size = (numels + block_size - 1) / block_size;
    get_value_by_id<<<grid_size, block_size, 0, cu_stream>>>(
        reinterpret_cast<DataType_ *>(
            const_cast<data_t *>(logits.data<data_t>())),
        gather_id.data<int>(),
        reinterpret_cast<DataType_ *>(
            const_cast<data_t *>(logits_out.data<data_t>())),
        logits_bs,
        seq_len,
        logits_len);
    return {logits_out};
}

std::vector<paddle::Tensor> GatherIdx(const paddle::Tensor &logits,
                                      const paddle::Tensor &gather_id) {
    switch (logits.type()) {
        case paddle::DataType::BFLOAT16: {
            return gather_idx<paddle::DataType::BFLOAT16>(logits, gather_id);
        }
        case paddle::DataType::FLOAT16: {
            return gather_idx<paddle::DataType::FLOAT16>(logits, gather_id);
        }
        case paddle::DataType::FLOAT32: {
            return gather_idx<paddle::DataType::FLOAT32>(logits, gather_id);
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> GatherIdxInferShape(
    const std::vector<int64_t> &logits_shape,
    const std::vector<int64_t> &gather_id_shape) {
    std::vector<int64_t> out_shape = {logits_shape[0], logits_shape[2]};
    return {out_shape};
}

std::vector<paddle::DataType> GatherIdxInferDtype(
    const paddle::DataType &logits_dtype,
    const paddle::DataType &gather_id_dtype) {
    return {logits_dtype};
}

PD_BUILD_STATIC_OP(gather_idx)
    .Inputs({"logits", "gather_id"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(GatherIdx))
    .SetInferShapeFn(PD_INFER_SHAPE(GatherIdxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GatherIdxInferDtype));
