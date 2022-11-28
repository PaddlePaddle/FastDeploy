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
* @file coercion.cpp
* @author wangrui39@baidu.com
* @date Fri May 13 11:36:11 CST 2022
* @brief 
**/

#include "poros/converter/gpu/coercion.h"
#include "poros/converter/gpu/weight.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

/*"aten::Int.float(float a) -> (int)"
"aten::Int.Tensor(Tensor a) -> (int)*/
bool CoercionConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for CoercionConverter");
    nvinfer1::ITensor *tensor_a = engine->context().get_tensor(inputs[0]);

    // int to tensor
    if (nullptr != tensor_a) {
        auto id_layer = engine->network()->addIdentity(*tensor_a);
        id_layer->setName((layer_info(node) + "_IIdentityLayer").c_str());
        id_layer->setOutputType(0, nvinfer1::DataType::kINT32);
        engine->context().set_tensor(node->outputs()[0], id_layer->getOutput(0));
    } else {
        int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
        engine->context().set_constant(node->outputs()[0], a);
    }
    return true;
}  

POROS_REGISTER_CONVERTER(TensorrtEngine, CoercionConverter);


}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
