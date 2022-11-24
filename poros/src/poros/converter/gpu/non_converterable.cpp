// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
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

/**
* @file non_converterable.cpp
* @author tianjinjin@baidu.com
* @date Thu Aug 26 10:24:14 CST 2021
* @brief 
**/

#include "poros/converter/gpu/non_converterable.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/weight.h"
#include "poros/context/poros_global.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

/*aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)*/
bool ContiguousConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::TensorType::get()), 
        "input[0] for ContiguousConverter is not Tensor as expected");

    //extract tensors
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    //need to do nothing, update the map directly.
    engine->context().set_tensor(node->outputs()[0], in_tensor);
    LOG(INFO) << "Output tensor shape: " << in_tensor->getDimensions();
    return true;
}

/*aten::dropout(Tensor input, float p, bool train) -> Tensor*/
bool DropoutConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for DropoutConverter");
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::TensorType::get()), 
        "input[0] for DropoutConverter is not Tensor as expected");

    //extract tensors
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    //need to do nothing, update the map directly.
    engine->context().set_tensor(node->outputs()[0], in_tensor);
    LOG(INFO) << "Output tensor shape: " << in_tensor->getDimensions();
    return true;
}

// aten::IntImplicit(Tensor a) -> (int)
bool IntimplicitConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::TensorType::get()), 
        "input[0] for ContiguousConverter is not Tensor as expected");

    //extract tensors
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    //need to do nothing, update the map directly.
    engine->context().set_tensor(node->outputs()[0], in_tensor);
    LOG(INFO) << "Output tensor shape: " << in_tensor->getDimensions();
    return true;
}

// prim::tolist(Tensor a) -> (int[])
bool TolistConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::TensorType::get()), 
        "input[0] for ContiguousConverter is not Tensor as expected");

    //extract tensors
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    //need to do nothing, update the map directly.
    engine->context().set_tensor(node->outputs()[0], in_tensor);
    LOG(INFO) << "Output tensor shape: " << in_tensor->getDimensions();
    return true;
}

// aten::detach(Tensor(a) self) -> Tensor(a)
bool DetachConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::TensorType::get()), 
        "input[0] for DetachConverter is not Tensor as expected");

    //extract tensors
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    //need to do nothing, update the map directly.
    engine->context().set_tensor(node->outputs()[0], in_tensor);
    LOG(INFO) << "Output tensor shape: " << in_tensor->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ContiguousConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, DropoutConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, IntimplicitConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, TolistConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, DetachConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
