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
* @file einsum.cpp
* @author tianshaoqing@baidu.com
* @date Wed Jul 06 11:24:51 CST 2022
* @brief 
**/

#include "poros/converter/gpu/einsum.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

// aten::einsum(str equation, Tensor[] tensors) -> (Tensor)
bool EinsumConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for EinsumConverter");
    POROS_CHECK_TRUE(inputs[1]->type()->isSubtypeOf(c10::ListType::ofTensors()), 
        "input[1] for EinsumConverter is not TensorList as expected.");

    // extract equation string
    torch::jit::IValue equation_ivalue = engine->context().get_constant(inputs[0]);
    POROS_CHECK_TRUE(equation_ivalue.isString(), "EinsumConverter input[0] is not constant string as expected.");
    std::string equation_str = equation_ivalue.toStringRef();

    // 大写转小写，nvinfer1::IEinsumLayer不支持equation中包含大写字母
    for (auto it = equation_str.begin(); it != equation_str.end(); it++) {
        if ((*it) >= 'A' && (*it) <= 'Z') {
            *it = *it + 32;
        }                
    }

    // extract tensorlist
    // mark：单测时输入2个以上的tensor trt会报错 nbInputs > 0 && nbInputs <= MAX_EINSUM_NB_INPUTS
    // 不确定MAX_EINSUM_NB_INPUTS是固定=2还是根据equation来定，暂时不加判断。
    std::vector<nvinfer1::ITensor*> tensorlist;
    POROS_CHECK_TRUE(engine->context().get_tensorlist(inputs[1], tensorlist), "EinsumConverter "
    "extract tensor list error.");

    nvinfer1::IEinsumLayer* einsum_layer = engine->network()->addEinsum(tensorlist.data(), 
                                                                        tensorlist.size(), 
                                                                        equation_str.c_str());
    einsum_layer->setName((layer_info(node) + "_IEinsumLayer").c_str());

    nvinfer1::ITensor* output = einsum_layer->getOutput(0);

    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output shape: " << output->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, EinsumConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
