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
* @file pooling.cpp
* @author tianjinjin@baidu.com
* @date Wed Aug 18 11:25:13 CST 2021
* @brief 
**/

#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/pooling.h"
#include "poros/converter/gpu/weight.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

//note1: max_pool?d 输入参数都是6个，各个参数的含义是一致的，差异在于 int[] 的维度不一样。
//note2: avg_pool1d 的输入参数是6个，avg_pool2d 和 3d 的输入参数是7个，多了最后一个参数 divisor_override。
//note3: max 与 avg 从第5个参数开始，出现了定义上的差异，需要注意。
bool PoolingConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 6 || inputs.size() == 7), 
        "invaid inputs size for PoolingConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for PoolingConverter is not Tensor as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    
    // Max Pool needs at least 4D input
    auto orig_dims = in->getDimensions();
    bool expandDims = (orig_dims.nbDims < 4);
    if (expandDims) {
        in = add_padding(engine, node, in, 4, false, true);
    }

    auto kernel_size = sizes_to_nvdim((engine->context().get_constant(inputs[1])).toIntList());
    auto stride = sizes_to_nvdim((engine->context().get_constant(inputs[2])).toIntList());
    auto padding = sizes_to_nvdim((engine->context().get_constant(inputs[3])).toIntList());
    if (stride.nbDims == 0) {
        LOG(INFO) << "Stride not provided, using kernel_size as stride";
        stride = sizes_to_nvdim((engine->context().get_constant(inputs[1])).toIntList());
    }
    if (kernel_size.nbDims == 1) {
        kernel_size = unsqueeze_dims(kernel_size, 0, 1);
    }
    if (padding.nbDims == 1) {
        padding = unsqueeze_dims(padding, 0, 0);
    }
    if (stride.nbDims == 1) {
        stride = unsqueeze_dims(stride, 0, 1);
    }
    LOG(INFO) << "kernel_size: " << kernel_size << ", padding: " << padding << ", stride: " << stride;
    

    bool ceil_mode = false;
    nvinfer1::IPoolingLayer* new_layer;

    //when it's max pooling
    if (node->kind() == torch::jit::aten::max_pool1d ||
        node->kind() == torch::jit::aten::max_pool2d ||
        node->kind() == torch::jit::aten::max_pool3d) { 
        auto dilation = sizes_to_nvdim((engine->context().get_constant(inputs[4])).toIntList());
        POROS_CHECK(dilation == sizes_to_nvdim(std::vector<int64_t>(dilation.nbDims, 1)),
            "Pooling dilation is not supported in TensorRT");
        
        LOG(INFO) << "dilation: " << dilation;
        LOG(WARNING) << "Dilation not used in Max pooling converter";

        ceil_mode = (engine->context().get_constant(inputs[5])).toBool();
        
        new_layer = engine->network()->addPoolingNd(*in, nvinfer1::PoolingType::kMAX, kernel_size);
        POROS_CHECK(new_layer, "Unable to create Max Pooling layer from node: " << *node);
        new_layer->setName((layer_info(node) + "_IPoolingLayer_max").c_str());

    //when it's avg pooling
    } else if (node->kind() == torch::jit::aten::avg_pool1d ||
            node->kind() == torch::jit::aten::avg_pool2d ||
            node->kind() == torch::jit::aten::avg_pool3d) {

        ceil_mode = (engine->context().get_constant(inputs[4])).toBool();
        bool count_inlcude_pad = (engine->context().get_constant(inputs[5])).toBool();
        
        new_layer = engine->network()->addPoolingNd(*in, nvinfer1::PoolingType::kAVERAGE, kernel_size);
        POROS_CHECK(new_layer, "Unable to create Avg Pooling layer from node: " << *node);
        new_layer->setAverageCountExcludesPadding(!count_inlcude_pad);
        new_layer->setName((layer_info(node) + "_IPoolingLayer_average").c_str());

    //we should never reach here  
    } else {
        POROS_THROW_ERROR("Unsupported pool mode!");
    }
    
    auto padding_mode = 
        ceil_mode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    
    new_layer->setPaddingMode(padding_mode);
    new_layer->setPaddingNd(padding);
    new_layer->setStrideNd(stride);
    
    auto out_tensor = add_unpadding(engine, node, new_layer->getOutput(0), orig_dims.nbDims, false, true);
    
    // avg_pool2d or avg_pool3d divisor_override
    if (node->kind() == torch::jit::aten::avg_pool2d ||
        node->kind() == torch::jit::aten::avg_pool3d) {
        auto maybe_divisor = engine->context().get_constant(inputs[6]);
        if (maybe_divisor.isScalar()) {
            auto divisor = maybe_divisor.toScalar().to<int>();
            if (divisor != 0){
                auto kernel_size_list = sizes_to_nvdim((engine->context().get_constant(inputs[1])).toIntList());
                int64_t kernel_area = 1;
                for(auto i = 0; i < kernel_size_list.nbDims; i++) {
                    kernel_area *= kernel_size_list.d[i];
                }
                auto actual_divisor = tensor_to_const(engine, torch::tensor({(float)kernel_area / (float)divisor}));
                auto mul = add_elementwise(engine, nvinfer1::ElementWiseOperation::kPROD, 
                                            out_tensor, actual_divisor, layer_info(node) + "_prod");
                POROS_CHECK(mul, "Unable to create mul layer from node: " << *node);
                out_tensor = mul->getOutput(0);
            } else {
                LOG(INFO) << "Invalid parameter: divisor_override";
                return false;
            }
        }
    }

    engine->context().set_tensor(node->outputs()[0], out_tensor);
    LOG(INFO) << "Output tensor shape: " << out_tensor->getDimensions();
    return true;
}

//note1: adaptive_avg_pool?d，各个参数的含义是一致的，差异在于 int[] 的维度不一样。
//note2: adaptive_max_pool?d，同note1, 各个参数含义一样，差异在于 int[] 的维度不一样。
//note3: avg 与 max的 输入参数都是2个，avg的输出参数是1个，max的输出参数是2个。
//note4: 这个家族的6个op，没有全部实现，需要注意！！！
bool AdaptivePoolingConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for AdaptivePoolingConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for AdaptivePoolingConverter is not Tensor as expected");
        
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    nvinfer1::Dims orig_dims = in->getDimensions();
    auto out_size = sizes_to_nvdim((engine->context().get_constant(inputs[1])).toIntList());
    LOG(INFO) << "get out_size: " << out_size << " in AdaptivePoolingConverter";

    nvinfer1::PoolingType pool_type;
    if (node->kind() == torch::jit::aten::adaptive_avg_pool1d ||
        node->kind() == torch::jit::aten::adaptive_avg_pool2d) {
        pool_type = nvinfer1::PoolingType::kAVERAGE;
    } else if (node->kind() == torch::jit::aten::adaptive_max_pool2d) {
        pool_type = nvinfer1::PoolingType::kMAX;
    } else {
        POROS_THROW_ERROR("Unsupported Adaptive pool mode!");
    }
    
    // Corner case: when out dimension is all ones, replace with simpler operation
    if (out_size.d[0] == 1 && (out_size.nbDims < 2 || out_size.d[1] == 1) &&
    (out_size.nbDims < 3 || out_size.d[2] == 1)) {
        LOG(INFO) << "Matched corner case in AdaptivePoolingConverter";
        // Generate a bitmask of all 1s except the last 2 bits (N and C axes)
        uint32_t reduceAxes = ((1 << orig_dims.nbDims) - 1) & ~0b11;
        auto* new_layer = engine->network()->addReduce(
                *in,
                pool_type == nvinfer1::PoolingType::kMAX ? nvinfer1::ReduceOperation::kMAX : nvinfer1::ReduceOperation::kAVG,
                reduceAxes,
                /*keepDimensions=*/true);
        new_layer->setName((layer_info(node) + "_IReduceLayer").c_str());

        engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
        LOG(INFO) << "AdaptivePoolingConverter: Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
        return true;
    }
    
    bool expandDims = (orig_dims.nbDims < 4);
    POROS_CHECK(orig_dims.nbDims > 2, "Unable to create pooling layer from node: " << *node);
    if (expandDims) {
        in = add_padding(engine, node, in, 4, false, false);
    }
    
    if (out_size.nbDims == 1) {
        out_size = unsqueeze_dims(out_size, 0, 1);
    }
    
    auto in_shape = nvdim_to_sizes(in->getDimensions());
    nvinfer1::ILayer* new_layer = nullptr;
    
    nvinfer1::PluginFieldCollection fc;
    std::vector<nvinfer1::PluginField> f;
    auto out_shape = in_shape;
    auto out_size_vec = nvdim_to_sizes(out_size);
    
    std::copy(out_size_vec.begin(), out_size_vec.end(), out_shape.begin() + (in_shape.size() - out_size_vec.size()));
    std::vector<int32_t> in_shape_casted(in_shape.begin(), in_shape.end());
    f.emplace_back(
        nvinfer1::PluginField("in_shape", in_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, in_shape.size()));
    std::vector<int32_t> out_shape_casted(out_shape.begin(), out_shape.end());
    f.emplace_back(
        nvinfer1::PluginField("out_shape", out_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, out_shape.size()));
    std::vector<int32_t> out_size_casted(out_size_vec.begin(), out_size_vec.end());
    f.emplace_back(
        nvinfer1::PluginField("out_size", out_size_casted.data(), nvinfer1::PluginFieldType::kINT32, out_size_vec.size()));
    f.emplace_back(
        nvinfer1::PluginField("scales", nullptr, nvinfer1::PluginFieldType::kFLOAT64, 0));
    
    int32_t align_corners_casted = 0;
    f.emplace_back(
        nvinfer1::PluginField("align_corners", &align_corners_casted, nvinfer1::PluginFieldType::kINT32, 1));
    int32_t use_scales_casted = 0;
    f.emplace_back(
        nvinfer1::PluginField("use_scales", &use_scales_casted, nvinfer1::PluginFieldType::kINT32, 1));
        
    std::string mode = "adaptive_avg_pool2d";
    if (pool_type == nvinfer1::PoolingType::kMAX) {
        mode = "adaptive_max_pool2d";
    }
    f.emplace_back(
        nvinfer1::PluginField("mode", &mode, nvinfer1::PluginFieldType::kCHAR, 1));
        
    fc.nbFields = f.size();
    fc.fields = f.data();
     
    auto creator = getPluginRegistry()->getPluginCreator("Interpolate", "1", "");
    auto interpolate_plugin = creator->createPlugin(mode.c_str(), &fc);
    LOG(INFO) << "create Interpolate plugin done";
    
    new_layer = engine->network()->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *interpolate_plugin);
    POROS_CHECK(new_layer, "Unable to create pooling (interpolation) plugin from node" << *node);
    new_layer->setName((layer_info(node) + "_plugin_Interpolate").c_str());
    auto layer_output = add_unpadding(engine, node, new_layer->getOutput(0), orig_dims.nbDims, false, false);

    engine->context().set_tensor(node->outputs()[0], layer_output);
    LOG(INFO) << "Output tensor shape: " << layer_output->getDimensions();
    //attention: 对于adaptive_max_pool2d, 映射第二个output
    if (mode == "adaptive_max_pool2d") {
        engine->context().set_tensor(node->outputs()[1], new_layer->getOutput(1));
        LOG(INFO) << "Output tensor2 shape: " << new_layer->getOutput(1)->getDimensions();
    }
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, PoolingConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, AdaptivePoolingConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu