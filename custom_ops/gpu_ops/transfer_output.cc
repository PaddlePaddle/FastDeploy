// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <dlfcn.h>  // dladdr
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "paddle/extension.h"

#include "token_transfer.hpp"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// #define DEBUG_TRANSFER_OUTPUT

#ifdef DEBUG_TRANSFER_OUTPUT
void PrintVec(std::vector<int64_t> &vec) {
    std::cout << "std::vector vec_size: " << vec.size();
    for (int i{0}; i < vec.size(); i++) {
        std::cout << " " << vec[i];
    }
    std::cout << std::endl;
}

void PrintVec(int64_t *arr) {
    std::cout << "READ vec_size: " << arr[0];
    for (int i{1}; i < arr[0] + 1; i++) {
        std::cout << " " << arr[i];
    }
    std::cout << std::endl;
}

void PrintVec(int64_t bs, int64_t *arr) {
    std::cout << "WRITE vec_size: " << bs;
    for (int i{0}; i < bs; i++) {
        std::cout << " " << arr[i];
    }
    std::cout << std::endl;
}
#endif

std::vector<paddle::Tensor> TransferOutput(const paddle::Tensor &x,
                                           int64_t rank_id) {
    using namespace paddle::inference::transfer;

    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    if (rank_id != 0) {
        return {x_cpu};
    }
    std::vector<int64_t> x_shape = x_cpu.shape();
    int64_t token_num = x_cpu.numel();
    // only support int64_t
    assert(x_cpu.type() == paddle::DataType::INT64);

    auto &token_transfer = TokenTransfer::Instance();
    if (token_transfer.stream_cb_fn_) {
        auto data_ptr = x_cpu.data<int64_t>();
        std::vector<int64_t> tokens(data_ptr, data_ptr + token_num);
        token_transfer.stream_cb_fn_(tokens, token_transfer.stream_cb_data_);
    }
#ifdef DEBUG_TRANSFER_OUTPUT
    else {
        token_transfer.PushBatchToken(token_num, x_cpu.data<int64_t>());
    }
#endif

    return {x_cpu};
}

std::vector<std::vector<int64_t>> TransferOutputInferShape(
    const std::vector<int64_t> &x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> TransferOutputInferDtype(
    const paddle::DataType &x_dtype) {
    return {x_dtype};
}

PD_BUILD_STATIC_OP(transfer_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(TransferOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(TransferOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TransferOutputInferDtype));
