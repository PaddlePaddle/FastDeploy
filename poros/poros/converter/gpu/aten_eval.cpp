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
* @file aten_eval.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/aten_eval.h"
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

/*
"aten::append.t(t[](a!) self, t(c -> *) el) -> (t[](a!))"*/
bool AppendConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for AppendConverter");

    //extract self
    std::vector<nvinfer1::ITensor*> tensorlist;
    POROS_CHECK_TRUE((engine->context().get_tensorlist(inputs[0], tensorlist)), "extract tensor list err");
    // 防止输入的tensorlist没有经过trtengine的情况
    //（不加的话 build trtengine 时会报 Unused Input 或者 Tensor xxx cannot be both input and output 的错误）
    for (size_t i = 0; i < tensorlist.size(); i++) {
        tensorlist[i] = engine->network()->addIdentity(*tensorlist[i])->getOutput(0);
    }
    //extract element
    auto element = engine->context().get_tensor(inputs[1]);
    //element is an already changed nvtensor
    if (element != nullptr) {
        element = engine->network()->addIdentity(*element)->getOutput(0);
        tensorlist.emplace_back(element);
        engine->context().set_tensorlist(node->outputs()[0], tensorlist);
        engine->context().set_tensorlist(node->inputs()[0], tensorlist);
        return true;
    } else {
        LOG(WARNING) << "non tensor kind element append is currently not support in AppendConverter";
        return false;
    }
}

/*
"aten::__getitem__.t(t[](a) list, int idx) -> (t(*))"*/
bool GetitemConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for GetitemConverter");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "inputs[1] for GetitemConverter is not come from prim::Constant as expected");

    if (node->outputs()[0]->type()->str() == "Tensor") {
        //extract list
        std::vector<nvinfer1::ITensor*> tensorlist;
        POROS_CHECK_TRUE((engine->context().get_tensorlist(inputs[0], tensorlist)), "extract tensor list err")
        
        const int64_t list_size = tensorlist.size();
        auto index = (engine->context().get_constant(inputs[1])).toInt();
        const int64_t normalized_index = index < 0 ? list_size + index : index;
        nvinfer1::ITensor* out_tensor = tensorlist[normalized_index];
        engine->context().set_tensor(node->outputs()[0], out_tensor);
        return true;
    } else {
        // extract nvtensor intlist
        if (check_inputs_tensor_scalar(engine, node)) {
            nvinfer1::ITensor* list_itensor = this->get_tensor_scalar(inputs[0]);
            POROS_CHECK_TRUE((list_itensor != nullptr), 
                                node_info(node) + std::string("get int nvtensor false."));

            auto index = (engine->context().get_constant(inputs[1])).toInt();
            auto list_len = (list_itensor->getDimensions()).d[0];
            POROS_CHECK_TRUE((index >= -list_len && index <= list_len - 1), 
                                node_info(node) + std::string(" idx is out of range."));

            // 倒序改正序
            index = index < 0 ? index + list_len : index;

            nvinfer1::ITensor* index_itensor = tensor_to_const(engine, torch::tensor({index}, torch::kInt));

            //extract the specific dynamic dim as a 1D-1value tensor
            std::vector<int64_t> start_vec{0}, size_vec{1}, stride_vec{1};
            nvinfer1::ISliceLayer* slice_layer = engine->network()->addSlice(*list_itensor,
                                                    sizes_to_nvdim(start_vec),
                                                    sizes_to_nvdim(size_vec),
                                                    sizes_to_nvdim(stride_vec));
            POROS_CHECK(slice_layer, "Unable to given dim info from node: " << *node);
            slice_layer->setInput(1, *index_itensor);
            slice_layer->setName((layer_info(node) + "_ISliceLayer").c_str());
            nvinfer1::ITensor* slice_out = slice_layer->getOutput(0);
            engine->context().set_tensor(node->outputs()[0], slice_out);
            LOG(INFO) << "Output tensor shape: " << slice_out->getDimensions();
        } else {
            torch::jit::IValue ts_ivalue = engine->context().get_constant(inputs[0]);
            POROS_CHECK_TRUE((ts_ivalue.isList()), "Unable to init input tensor for node: " << *node);
            auto list = ts_ivalue.toListRef();
            const int64_t list_size = list.size();
            int64_t index = (engine->context().get_constant(inputs[1])).toInt();
            const int64_t normalized_index = index < 0 ? list_size + index : index;
            auto value_item = list[normalized_index];
            engine->context().set_constant(node->outputs()[0], value_item);
        }
        return true;
    }
}

/*
"aten::_set_item.t(t[](a!) l, int idx, t(b -> *) el) -> (t[](a!))"*/
bool SetitemConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for SetitemConverter");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "inputs[1] for SetitemConverter is not come from prim::Constant as expected");

    size_t idx = engine->context().get_constant(inputs[1]).toInt();

    if (node->outputs()[0]->type()->str() == "Tensor[]") {
        std::vector<nvinfer1::ITensor*> tensorlist;
        POROS_CHECK_TRUE((engine->context().get_tensorlist(inputs[0], tensorlist)), "extract tensor list err");
        POROS_CHECK_TRUE((tensorlist.size() > idx), "Tensorlist index out of range: " << *node);
        // 防止输入的tensorlist没有经过trtengine的情况
        //（不加的话 build trtengine 时会报 Unused Input 或者 Tensor xxx cannot be both input and output 的错误）
        for (size_t i = 0; i < tensorlist.size(); i++) {
            tensorlist[i] = engine->network()->addIdentity(*tensorlist[i])->getOutput(0);
        }
        nvinfer1::ITensor *input_tensor = engine->context().get_tensor(inputs[2]);
        input_tensor = engine->network()->addIdentity(*input_tensor)->getOutput(0);
        tensorlist[idx] = input_tensor;
        engine->context().set_tensorlist(node->outputs()[0], tensorlist);
        engine->context().set_tensorlist(node->inputs()[0], tensorlist);
        return true;
    }
    else{
        LOG(WARNING) << "non tensor kind element _set_item is currently not support in SetitemConverter";
        return true;
    }
}

POROS_REGISTER_CONVERTER(TensorrtEngine, AppendConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, GetitemConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, SetitemConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
