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
* @file concat.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/concat.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/weight.h"
#include "poros/context/poros_global.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

/*"aten::cat(Tensor[] tensors, int dim=0) -> Tensor*/
bool ConcatConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();

    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for ConcatConverter");
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::ListType::ofTensors()), 
        "input[0] for ConcatConverter is not TensorList as expected");
    POROS_CHECK_TRUE(inputs[1]->type()->isSubtypeOf(c10::NumberType::get()), 
        "input[1] for ConcatConverter is not int64_t as expected");

    std::vector<nvinfer1::ITensor*> tensorlist;
    POROS_CHECK_TRUE((engine->context().get_tensorlist(inputs[0], tensorlist)), "extract tensor list err")

    //extract dims
    auto dim = (engine->context().get_constant(inputs[1])).toInt();
    if (dim < 0) {
        dim = tensorlist[0]->getDimensions().nbDims + dim;
    }
    
    auto cat_layer = engine->network()->addConcatenation(tensorlist.data(), tensorlist.size());
    cat_layer->setAxis(static_cast<int>(dim));
    cat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
    engine->context().set_tensor(node->outputs()[0], cat_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << cat_layer->getOutput(0)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ConcatConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
