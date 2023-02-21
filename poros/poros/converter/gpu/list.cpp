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
* @file list.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/
#include "torch/script.h"

#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/list.h"
#include "poros/converter/gpu/weight.h"
#include "poros/context/poros_global.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

bool ListConstructConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    const torch::jit::Value* output = node->outputs()[0];
    const auto num_inputs = inputs.size();
    //typical situation: Construct a TensorList
    if (output->type()->isSubtypeOf(c10::ListType::ofTensors()) || 
        output->type()->str().find("Tensor?[]") != std::string::npos) {
        std::vector<nvinfer1::ITensor*> tensorlist;
        tensorlist.reserve(num_inputs);
        for (auto in : inputs) {
            auto in_tensor = engine->context().get_tensor(in);
            POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to extract in_tensor for node: " << *node);
            tensorlist.emplace_back(in_tensor);
        }
        engine->context().set_tensorlist(node->outputs()[0], tensorlist);

    // IntList
    } else if (output->type()->isSubtypeOf(c10::ListType::ofInts())) {
        // 检查int是否以nvtensor的形式输入
        if (check_inputs_tensor_scalar(engine, node)) {
            std::vector<nvinfer1::ITensor*> inputs_nvtensor;
            // 将所有int对应的nvtensor加入vector, 最后cat起来
            for (auto in : inputs) {
                nvinfer1::ITensor* temp_tensor = this->get_tensor_scalar(in);
                POROS_CHECK_TRUE((temp_tensor != nullptr), node_info(node) + std::string("get int nvtensor false."));
                inputs_nvtensor.push_back(temp_tensor);
            }
            nvinfer1::IConcatenationLayer* concat_layer = 
                    engine->network()->addConcatenation(inputs_nvtensor.data(), inputs_nvtensor.size());
            // 这里确保输出类型是int
            concat_layer->setOutputType(0, nvinfer1::DataType::kINT32);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
            concat_layer->setAxis(0);
            engine->context().set_tensor(node->outputs()[0], concat_layer->getOutput(0));  
        }
        else {
            // 输入是正常情况
            c10::List<int64_t> list;
            list.reserve(num_inputs);
            for (auto in : inputs) {
                auto in_const = engine->context().get_constant(in);
                list.emplace_back(std::move(in_const.toInt()));
            }
            auto output_ivalue = c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            engine->context().set_constant(node->outputs()[0], output_ivalue);   
        }

    // FloatList
    } else if (output->type()->isSubtypeOf(c10::ListType::ofFloats())) {
        c10::List<double> list;
        list.reserve(num_inputs);
        for (auto in : inputs) {
            auto in_const = engine->context().get_constant(in);
            list.emplace_back(std::move(in_const.toDouble()));
        }
        auto output_ivalue = c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
        engine->context().set_constant(node->outputs()[0], output_ivalue);

    // BoolList
    } else if (output->type()->isSubtypeOf(c10::ListType::ofBools())) {
        c10::List<bool> list;
        list.reserve(num_inputs);
        for (auto in : inputs) {
            auto in_const = engine->context().get_constant(in);
            list.emplace_back(std::move(in_const.toBool()));
        }
        auto output_ivalue = c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
        engine->context().set_constant(node->outputs()[0], output_ivalue);

    //TODO: meet some unsupported type
    } else {
        POROS_THROW_ERROR("Meet some unsupported output value type in ListConstructConverter" << *node);
    }
    return true;
}

bool ListUnpackConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    at::ArrayRef<const torch::jit::Value*> outputs = node->outputs();
    // 检查int[]是否以nvtensor的形式输入
    if (check_inputs_tensor_scalar(engine, node)) {
        nvinfer1::ITensor* input_int_nvtensor = get_tensor_scalar(inputs[0]);
        POROS_CHECK_TRUE((input_int_nvtensor != nullptr), node_info(node) + std::string("get int nvtensor false."));

        nvinfer1::Dims input_dims = input_int_nvtensor->getDimensions();
        // int[]只有一维数据, 获得要unpack的int数量
        int64_t dim_rank = input_dims.d[0];
        POROS_CHECK_TRUE((outputs.size() == (size_t)dim_rank), 
                "the input list size do not equal output size for ListUnpackConverter as expected");
        // int[]
        for (int64_t i = 0; i < dim_rank; i++) {
            std::vector<int64_t> start_vec{i}, size_vec{1}, stride_vec{1};
            auto slice_layer = engine->network()->addSlice(*input_int_nvtensor,
                                                    sizes_to_nvdim(start_vec),
                                                    sizes_to_nvdim(size_vec),
                                                    sizes_to_nvdim(stride_vec));
            POROS_CHECK(slice_layer, "Unable to given dim info from node: " << *node);
            slice_layer->setName((layer_info(node) + "_ISliceLayer" + std::to_string(i)).c_str());
            nvinfer1::ITensor* slice_output = slice_layer->getOutput(0);
            engine->context().set_tensor(outputs[i], slice_output);
        }
        return true;
    }
    
    std::vector<nvinfer1::ITensor*> output_vec;
    engine->context().get_tensorlist(inputs[0], output_vec);
    POROS_CHECK_TRUE((outputs.size() == output_vec.size()),
        "the input list size do not equal output size for ListUnpackConverter as expected");   
    
    //TODO: check if this implement is right, check output is tuple or mulituple ivalues.
    for (size_t index = 0; index < outputs.size(); index++) {
        auto out = outputs[index];
        //Tensor situation
        if (out->type()->isSubtypeOf(c10::TensorType::get())) {
            engine->context().set_tensor(out, output_vec[index]);
        } else {
            POROS_THROW_ERROR("Meet some unsupported output value type in ListUnpackConverter" << *node);
        }
    }
    return true;
}

// OP aten::list, original python code looks like: "for a_shape in list(data.shape): ......"
bool ListConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    const torch::jit::Value* output = node->outputs()[0];
    const auto num_inputs = inputs.size();
    POROS_CHECK_TRUE((num_inputs == 1),"More than 1 input is not supported for node:" << *node)
    auto input = inputs[0];
    POROS_CHECK_TRUE((input->type()->str() == output->type()->str()),"Input and Output are in different types")
    auto input_tensor = engine->context().get_tensor(input);
    if (!input_tensor) {
        std::vector<nvinfer1::ITensor*> tensor_list;
        POROS_CHECK_TRUE(engine->context().get_tensorlist(input, tensor_list), "extract tensor list err");
        engine->context().set_tensorlist(node->outputs()[0], tensor_list);
    }
    else {
        engine->context().set_tensor(node->outputs()[0],input_tensor);
    }
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ListConstructConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ListUnpackConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ListConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
