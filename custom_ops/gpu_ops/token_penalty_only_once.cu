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

__global__ void update_id_flags(const int64_t *pre_ids,
                                bool *id_flags,
                                const int64_t bs,
                                const int64_t length,
                                const int64_t length_id) {
    int bi = blockIdx.x;
    int tid = threadIdx.x;
    const int64_t *pre_ids_now = pre_ids + bi * length_id;
    bool *id_flags_now = id_flags + bi * length;
    for (int i = tid; i < length_id; i += blockDim.x) {
        int64_t id = pre_ids_now[i];
        if (id < 0) break;
        id_flags_now[id] = true;
    }
}

template <typename T>
__global__ void update_value_by_id_flags(const bool *id_flags,
                                         const T *penalty_scores,
                                         T *logits,
                                         const int64_t bs,
                                         const int64_t length) {
    int bi = blockIdx.x;
    int tid = threadIdx.x;
    T *logits_now = logits + bi * length;
    const bool *id_flags_now = id_flags + bi * length;
    float alpha = static_cast<float>(penalty_scores[bi]);
    for (int i = tid; i < length; i += blockDim.x) {
        bool flag = id_flags_now[i];
        if (!flag) continue;
        float logit_now = static_cast<float>(logits_now[i]);
        logits_now[i] = static_cast<T>(logit_now < 0 ? logit_now * alpha
                                                     : logit_now / alpha);
        // printf("bi: %d, i: %d, length: %d, logit: %f, alpha: %f, res: %f\n",
        // bi, i, length, logit_now, alpha, (float)logits_now[id]);
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> token_penalty_once_kernel(
    const paddle::Tensor &pre_ids,
    const paddle::Tensor &logits,
    const paddle::Tensor &penalty_scores) {
    // print_shape(pre_ids, "pre_ids");
    // print_shape(logits, "logits");
    // print_shape(penalty_scores, "penalty_scores");
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto cu_stream = logits.stream();
    std::vector<int64_t> shape = logits.shape();
    auto id_flags =
        paddle::full(shape, false, paddle::DataType::BOOL, pre_ids.place());
    int64_t bs = shape[0];
    int64_t length = shape[1];
    int64_t length_id = pre_ids.shape()[1];
    auto logits_out = logits.copy_to(logits.place(), false);  // gpu -> gpu
    int block_size_1 = (length_id + 32 - 1) / 32 * 32;
    block_size_1 = min(block_size_1, 512);
    update_id_flags<<<bs, block_size_1, 0, cu_stream>>>(
        pre_ids.data<int64_t>(), id_flags.data<bool>(), bs, length, length_id);
    int block_size_2 = (length + 32 - 1) / 32 * 32;
    block_size_2 = min(block_size_2, 512);
    update_value_by_id_flags<DataType_><<<bs, block_size_2, 0, cu_stream>>>(
        id_flags.data<bool>(),
        reinterpret_cast<DataType_ *>(
            const_cast<data_t *>(penalty_scores.data<data_t>())),
        reinterpret_cast<DataType_ *>(
            const_cast<data_t *>(logits_out.data<data_t>())),
        bs,
        length);
    return {logits_out};
}

std::vector<paddle::Tensor> TokenPenaltyOnce(
    const paddle::Tensor &pre_ids,
    const paddle::Tensor &logits,
    const paddle::Tensor &penalty_scores) {
    switch (logits.type()) {
        case paddle::DataType::BFLOAT16: {
            // printf("bf16\n");
            return token_penalty_once_kernel<paddle::DataType::BFLOAT16>(
                pre_ids, logits, penalty_scores);
        }
        case paddle::DataType::FLOAT16: {
            // printf("fp16\n");
            return token_penalty_once_kernel<paddle::DataType::FLOAT16>(
                pre_ids, logits, penalty_scores);
        }
        case paddle::DataType::FLOAT32: {
            // printf("fp32\n");
            return token_penalty_once_kernel<paddle::DataType::FLOAT32>(
                pre_ids, logits, penalty_scores);
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> TokenPenaltyOnceInferShape(
    const std::vector<int64_t> &pre_ids_shape,
    const std::vector<int64_t> &logits_shape,
    const std::vector<int64_t> &penalty_scores_shape) {
    return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyOnceInferDtype(
    const paddle::DataType &pre_ids_dtype,
    const paddle::DataType &logits_dtype,
    const paddle::DataType &penalty_scores_dtype) {
    return {logits_dtype};
}

PD_BUILD_STATIC_OP(get_token_penalty_once)
    .Inputs({"pre_ids", "logits", "penalty_scores"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyOnce))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyOnceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyOnceInferDtype));
