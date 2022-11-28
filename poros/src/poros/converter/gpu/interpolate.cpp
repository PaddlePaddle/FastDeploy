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

// Part of the following code in this file refs to
// https://github.com/pytorch/TensorRT/blob/master/core/conversion/converters/impl/interpolate.cpp
//
// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the 3-Clause BSD License

/**
* @file interpolate.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/interpolate.h"
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
 * Helper functions
 */
void create_plugin(TensorrtEngine* engine,
                    const torch::jit::Node* node,
                    nvinfer1::ITensor* in,
                    const char* name,
                    std::vector<int64_t> in_shape,
                    std::vector<int64_t> out_shape,
                    std::vector<int64_t> out_size,
                    std::vector<double> scales,
                    std::string mode,
                    bool align_corners,
                    bool use_scales = false) {
    LOG(WARNING) << "Interpolation layer will be run through ATen, not TensorRT. "
                 << "Performance may be lower than expected";
    nvinfer1::PluginFieldCollection fc;
    std::vector<nvinfer1::PluginField> f;
    std::vector<int32_t> in_shape_casted(in_shape.begin(), in_shape.end());
    f.emplace_back(nvinfer1::PluginField(
        "in_shape", in_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, in_shape.size()));
        
    std::vector<int32_t> out_shape_casted(out_shape.begin(), out_shape.end());
    f.emplace_back(nvinfer1::PluginField(
        "out_shape", out_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, out_shape.size()));
        
    std::vector<int32_t> out_size_casted(out_size.begin(), out_size.end());
    f.emplace_back(nvinfer1::PluginField(
        "out_size", out_size_casted.data(), nvinfer1::PluginFieldType::kINT32, out_size.size()));
        
    f.emplace_back(nvinfer1::PluginField(
        "scales", scales.data(), nvinfer1::PluginFieldType::kFLOAT64, scales.size()));
    f.emplace_back(nvinfer1::PluginField(
        "mode", &mode, nvinfer1::PluginFieldType::kCHAR, 1));
    
    int32_t align_corners_casted = static_cast<int32_t>(align_corners);
    f.emplace_back(nvinfer1::PluginField(
        "align_corners", &align_corners_casted, nvinfer1::PluginFieldType::kINT32, 1));
    
    int32_t use_scales_casted = static_cast<int32_t>(use_scales);
    f.emplace_back(nvinfer1::PluginField(
        "use_scales", &use_scales_casted, nvinfer1::PluginFieldType::kINT32, 1));
    
    fc.nbFields = f.size();
    fc.fields = f.data();
    auto creator = getPluginRegistry()->getPluginCreator("Interpolate", "1", "");
    auto interpolate_plugin = creator->createPlugin(name, &fc);
    
    auto resize_layer = engine->network()->addPluginV2(
        reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *interpolate_plugin);
    POROS_CHECK(resize_layer, "Unable to create interpolation plugin from node" << *node);
    resize_layer->setName((layer_info(node) + "_plugin_Interpolate").c_str());

    engine->context().set_tensor(node->outputs()[0], resize_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << resize_layer->getOutput(0)->getDimensions();
}

void resize_layer_size(TensorrtEngine* engine,
                        const torch::jit::Node* node,
                        nvinfer1::ITensor* in,
                        std::vector<int64_t> out_shape,
                        std::vector<float> scales,
                        nvinfer1::ResizeMode mode,
                        bool align_corners = false) {
    POROS_CHECK((out_shape.size() > 0) ^ (scales.size() > 0), "only one of out_shape or scales should be defined");
    auto resize_layer = engine->network()->addResize(*in);
    POROS_CHECK(resize_layer, "Unable to create interpolation (resizing) layer from node" << *node);
    
    if (out_shape.size() > 0) {
        auto th_dynamic_shape_mask = torch::zeros(out_shape.size(), torch::kInt32);
        auto th_static_shape_mask = torch::zeros(out_shape.size(), torch::kInt32);
        for (size_t idx = 0; idx < out_shape.size(); ++idx) {
            if (out_shape[idx] == -1) {
                th_dynamic_shape_mask[idx] = 1;
            } else {
                th_static_shape_mask[idx] = out_shape[idx];
            }
        }

    auto dynamic_shape_mask = tensor_to_const(engine, th_dynamic_shape_mask);
    auto static_shape_mask = tensor_to_const(engine, th_static_shape_mask);
    auto input_shape = engine->network()->addShape(*in)->getOutput(0);
    auto dynamic_shape = engine->network()->addElementWise(
        *input_shape, *dynamic_shape_mask, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto target_output_shape = engine->network()->addElementWise(
        *dynamic_shape, *static_shape_mask, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    resize_layer->setInput(1, *target_output_shape);
    } else {
        resize_layer->setScales(scales.data(), scales.size());
        if (align_corners) {
            LOG(WARNING) << "interpolate with align_corners and scale_factor works differently in TensorRT and PyTorch.";
        }
    }
    
    resize_layer->setResizeMode(mode);
    resize_layer->setName((layer_info(node) + "_IResizeLayer").c_str());
#if NV_TENSORRT_MAJOR < 8
    resize_layer->setAlignCorners(align_corners);
#else
    if (align_corners) {
        resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS);
    }
#endif
    engine->context().set_tensor(node->outputs()[0], resize_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << resize_layer->getOutput(0)->getDimensions();
}

/*
"aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor",
"aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor",
*/
bool UnsampleNearest1DConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnsampleNearest1DConverter is not Tensor as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());

    auto maybe_outsize = engine->context().get_constant(inputs[1]);
    auto maybe_scales = engine->context().get_constant(inputs[2]);
    if (maybe_outsize.isNone() &&  maybe_scales.isNone()) {
        POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                << "\nOne of output_size or scale_factors should be defined");
    }

    if (!maybe_scales.isNone()) {
        float scale = 0.0f;
        // Case 1: user uses scales
        if (maybe_scales.isDouble()) {
            scale = maybe_scales.toDouble();
        } else { // maybe_scales.isDoubleList()
            auto scale_factors = maybe_scales.toDoubleList();
            POROS_ASSERT(scale_factors.size() == 1, "Number of scale factors should match the input size");
            scale = scale_factors[0];
        }
        std::vector<float> padded_scales(in_shape.size(), 1);
        padded_scales[padded_scales.size() - 1] = scale;
        resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);
    } else {
        // Case 2: user uses output size
        auto output_size = maybe_outsize.toIntList();
        auto out_size = nvdim_to_sizes(sizes_to_nvdim(output_size));
        POROS_ASSERT(out_size.size() == 1, "aten::upsample_nearest1d input Tensor and output size dimension mismatch");
        auto out_shape = in_shape;
        std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
        resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
    }
    return true;
}

/*
"aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor",
"aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor",
*/
bool UnsampleNearest2DConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnsampleNearest2DConverter is not Tensor as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());

    auto maybe_outsize = engine->context().get_constant(inputs[1]);
    float scale_h = 0.0f;
    float scale_w = 0.0f;

    if (inputs.size() == 4) {
        auto maybe_scales_h = engine->context().get_constant(inputs[2]);
        auto maybe_scales_w = engine->context().get_constant(inputs[3]);
        if (maybe_outsize.isNone() &&  (maybe_scales_h.isNone() || maybe_scales_w.isNone())) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scales should be defined");
        }
        if (!maybe_scales_h.isNone() && !maybe_scales_w.isNone()) {
            // Case 1: user uses scales
            scale_h = maybe_scales_h.toDouble();
            scale_w = maybe_scales_w.toDouble();
        }
    } else {  //(inputs_size() == 3)
        auto maybe_scale_factors = engine->context().get_constant(inputs[2]);
        if (maybe_outsize.isNone() &&  maybe_scale_factors.isNone()) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scale_factors should be defined");
        }

        if (!maybe_scale_factors.isNone()) {
            // Case 1: user uses scales
            auto scale_factors = maybe_scale_factors.toDoubleList();
            POROS_ASSERT(scale_factors.size() == 2, "Number of scale factors should match the input size");
            scale_h = scale_factors[0];
            scale_w = scale_factors[1];
        }
    }

    if (!engine->context().get_constant(inputs[2]).isNone()) {
        std::vector<float> padded_scales(in_shape.size(), 1);
        padded_scales[padded_scales.size() - 2] = scale_h;
        padded_scales[padded_scales.size() - 1] = scale_w;
        resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);  
    } else {
        // Case 2: user uses output size
        auto output_size = maybe_outsize.toIntList();
        auto out_size = nvdim_to_sizes(sizes_to_nvdim(output_size));
        POROS_ASSERT(out_size.size() == 2, "aten::upsample_nearest2d input Tensor and output size dimension mismatch");
        auto out_shape = in_shape;
        std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
        resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
    }
    return true;
}

/*
"aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor",
"aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor",
*/
bool UnsampleNearest3DConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnsampleNearest3DConverter is not Tensor as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());

    auto maybe_outsize = engine->context().get_constant(inputs[1]);
    float scale_d = 0.0f;
    float scale_h = 0.0f;
    float scale_w = 0.0f;

    if (inputs.size() == 5) {
        auto maybe_scales_d = engine->context().get_constant(inputs[2]);
        auto maybe_scales_h = engine->context().get_constant(inputs[3]);
        auto maybe_scales_w = engine->context().get_constant(inputs[4]);
        if (maybe_outsize.isNone() && (maybe_scales_d.isNone() || 
            maybe_scales_h.isNone() || maybe_scales_w.isNone())) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scales should be defined");
        }
        if (!maybe_scales_d.isNone() && !maybe_scales_h.isNone() && !maybe_scales_w.isNone()) {
            // Case 1: user uses scales
            scale_d = maybe_scales_d.toDouble();
            scale_h = maybe_scales_h.toDouble();
            scale_w = maybe_scales_w.toDouble();
        }
    } else {  //(inputs_size() == 3)
        auto maybe_scale_factors = engine->context().get_constant(inputs[2]);
        if (maybe_outsize.isNone() &&  maybe_scale_factors.isNone()) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scale_factors should be defined");
        }

        if (!maybe_scale_factors.isNone()) {
            // Case 1: user uses scales
            auto scale_factors = maybe_scale_factors.toDoubleList();
            POROS_ASSERT(scale_factors.size() == 3, "Number of scale factors should match the input size");
            scale_d = scale_factors[0];
            scale_h = scale_factors[1];
            scale_w = scale_factors[2];
        }
    }

    if (!engine->context().get_constant(inputs[2]).isNone()) {
        std::vector<float> padded_scales(in_shape.size(), 1);
        padded_scales[padded_scales.size() - 3] = scale_d;
        padded_scales[padded_scales.size() - 2] = scale_h;
        padded_scales[padded_scales.size() - 1] = scale_w;
        resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);  
    } else {
        // Case 2: user uses output size
        auto output_size = maybe_outsize.toIntList();
        auto out_size = nvdim_to_sizes(sizes_to_nvdim(output_size));
        POROS_ASSERT(out_size.size() == 3, "aten::upsample_nearest3d input Tensor and output size dimension mismatch");
        auto out_shape = in_shape;
        std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
        resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
    }
    return true;
}

/*
"aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor",
"aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor",
*/
bool UnsampleLinear1DConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnsampleLinear1DConverter is not Tensor as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());
    //extract align_corners
    auto align_corners = engine->context().get_constant(inputs[2]).toBool();

    auto maybe_outsize = engine->context().get_constant(inputs[1]);
    auto maybe_scales = engine->context().get_constant(inputs[3]);
    if (maybe_outsize.isNone() && maybe_scales.isNone()) {
        POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                << "\nOne of output_size or scales should be defined");
    }
    
    if (!maybe_scales.isNone()) {
        // Case 1: user uses scales
        float scale = 0.0f;
        if (maybe_scales.isDouble()) {
            scale = maybe_scales.toDouble();
        } else { //maybe_scales.isDoubleList()
            auto scale_factors = maybe_scales.toDoubleList();
            POROS_ASSERT(scale_factors.size() == 1, "Number of scale factors should match the input size");
            scale = scale_factors[0];
        }
        std::vector<float> padded_scales(in_shape.size(), 1);
        padded_scales[padded_scales.size() - 1] = scale;
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1) // IF TRT VERSION <= 7.0
        if (!align_corners) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node) 
                << "\nupsample_linear1d only supports align_corner with TensorRT <= 7.0.");
        } else {
            resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, true);
        }
#else
        auto is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;
        POROS_CHECK(!(align_corners && is_dynamic_shape), "Poros currently does not support the compilation of dynamc engines"
           << "from code using using PyTorch [bi/tri]linear interpolation via scale factor and align_corners=True");
        if (align_corners) {
        // Align corners and scale factor behave slightly different together in TRT and PyTorch so run the
        // layer in ATen to maintain consistancy between TRTorch and PyTorch
        // https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
            create_plugin(engine, node, in, "linear1d", in_shape, {}, {}, {scale}, std::string("linear"), align_corners, true);
        } else {
            resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, align_corners);
        }
#endif 
    } else {
        // Case 2: user uses output size
        auto output_size = maybe_outsize.toIntList();
        auto out_size = nvdim_to_sizes(sizes_to_nvdim(output_size));
        POROS_ASSERT(out_size.size() == 1, "aten::upsample_linear1d input Tensor and output size dimension mismatch");
        auto out_shape = in_shape;
        std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1) // IF TRT VERSION <= 7.0
        if (!align_corners) {
            create_plugin(engine, node, in, "linear1d", in_shape, out_shape, out_size, {}, std::string("linear"), align_corners);            
        } else {
            resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, true);
        }
#else
        resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
    }
    return true;
}

/*
"aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor",
"aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor",
*/
bool UnsampleBilinear2DConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnsampleBilinear2DConverter is not Tensor as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());
    //extract align_corners
    auto align_corners = engine->context().get_constant(inputs[2]).toBool();

    auto maybe_outsize = engine->context().get_constant(inputs[1]);
    float scale_h = 0.0f;
    float scale_w = 0.0f;

    if (inputs.size() == 5) {
        auto maybe_scales_h = engine->context().get_constant(inputs[3]);
        auto maybe_scales_w = engine->context().get_constant(inputs[4]);
        if (maybe_outsize.isNone() &&  (maybe_scales_h.isNone() || maybe_scales_w.isNone())) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scales should be defined");
        }
        if (!maybe_scales_h.isNone() && !maybe_scales_w.isNone()) {
            // Case 1: user uses scales
            scale_h = maybe_scales_h.toDouble();
            scale_w = maybe_scales_w.toDouble();
        }
    } else {  //(inputs_size() == 4)
        auto maybe_scale_factors = engine->context().get_constant(inputs[3]);
        if (maybe_outsize.isNone() &&  maybe_scale_factors.isNone()) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scale_factors should be defined");
        }
        if (!maybe_scale_factors.isNone()) {
            // Case 1: user uses scales
            auto scale_factors = maybe_scale_factors.toDoubleList();
            POROS_ASSERT(scale_factors.size() == 2, "Number of scale factors should match the input size");
            scale_h = scale_factors[0];
            scale_w = scale_factors[1];
        }
    }

    if (!engine->context().get_constant(inputs[3]).isNone()) {
        std::vector<float> padded_scales(in_shape.size(), 1);
        padded_scales[padded_scales.size() - 2] = scale_h;
        padded_scales[padded_scales.size() - 1] = scale_w;
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1) // IF TRT VERSION <= 7.0
        if (!align_corners) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node) 
                << "\nupsample_linear1d only supports align_corner with TensorRT <= 7.0.");
        } else {
            resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, true);
        }
#else
        auto is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;
        POROS_CHECK(!(align_corners && is_dynamic_shape), "Poros currently does not support the compilation of dynamc engines"
           << "from code using using PyTorch [bi/tri]linear interpolation via scale factor and align_corners=True");
        if (align_corners) {
        // Align corners and scale factor behave slightly different together in TRT and PyTorch so run the
        // layer in ATen to maintain consistancy between TRTorch and PyTorch
        // https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
            create_plugin(engine, node, in, "bilinear2d", in_shape, {}, {}, {scale_h, scale_w}, std::string("bilinear"), align_corners, true);
        } else {
            resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, align_corners);
        }
#endif 
    } else {
        // Case 2: user uses output size
        auto output_size = maybe_outsize.toIntList();
        auto out_size = nvdim_to_sizes(sizes_to_nvdim(output_size));
        POROS_ASSERT(out_size.size() == 2, "aten::upsample_bilinear2d input Tensor and output size dimension mismatch");
        auto out_shape = in_shape;
        std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1) // IF TRT VERSION <= 7.0
        if (!align_corners) {
            create_plugin(engine, node, in, "bilinear2d", in_shape, out_shape, out_size, {}, std::string("bilinear"), align_corners);            
        } else {
            resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, true);
        }
#else
        resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
    }
    return true;
}

/*
"aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor",
"aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor",
*/
bool UnsampleTrilinear3DConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnsampleTrilinear3DConverter is not Tensor as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());
    //extract align_corners
    auto align_corners = engine->context().get_constant(inputs[2]).toBool();

    auto maybe_outsize = engine->context().get_constant(inputs[1]);
    float scale_d = 0.0f;
    float scale_h = 0.0f;
    float scale_w = 0.0f;

    if (inputs.size() == 6) {
        auto maybe_scales_d = engine->context().get_constant(inputs[3]);
        auto maybe_scales_h = engine->context().get_constant(inputs[4]);
        auto maybe_scales_w = engine->context().get_constant(inputs[5]);
        if (maybe_outsize.isNone() && (maybe_scales_h.isNone() 
            || maybe_scales_w.isNone() || maybe_scales_d.isNone())) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scales should be defined");
        }
        if (!maybe_scales_h.isNone() && !maybe_scales_w.isNone() && maybe_scales_d.isNone()) {
            // Case 1: user uses scales
            scale_d = maybe_scales_d.toDouble();
            scale_h = maybe_scales_h.toDouble();
            scale_w = maybe_scales_w.toDouble();
        }
    } else {  //(inputs_size() == 4)
        auto maybe_scale_factors = engine->context().get_constant(inputs[3]);
        if (maybe_outsize.isNone() &&  maybe_scale_factors.isNone()) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node)
                    << "\nOne of output_size or scale_factors should be defined");
        }
        if (!maybe_scale_factors.isNone()) {
            // Case 1: user uses scales
            auto scale_factors = maybe_scale_factors.toDoubleList();
            POROS_ASSERT(scale_factors.size() == 3, "Number of scale factors should match the input size");
            scale_d = scale_factors[0];
            scale_h = scale_factors[1];
            scale_w = scale_factors[2];
        }
    }

    if (!engine->context().get_constant(inputs[3]).isNone()) {
        std::vector<float> padded_scales(in_shape.size(), 1);
        padded_scales[padded_scales.size() - 3] = scale_d;
        padded_scales[padded_scales.size() - 2] = scale_h;
        padded_scales[padded_scales.size() - 1] = scale_w;
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1) // IF TRT VERSION <= 7.0
        if (!align_corners) {
            POROS_THROW_ERROR("Unable to convert node: " << node_info(node) 
                << "\nupsample_linear1d only supports align_corner with TensorRT <= 7.0.");
        } else {
            resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, true);
        }
#else
        auto is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;
        POROS_CHECK(!(align_corners && is_dynamic_shape), "Poros currently does not support the compilation of dynamc engines"
           << "from code using using PyTorch [bi/tri]linear interpolation via scale factor and align_corners=True");
        if (align_corners) {
        // Align corners and scale factor behave slightly different together in TRT and PyTorch so run the
        // layer in ATen to maintain consistancy between TRTorch and PyTorch
        // https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
            create_plugin(engine, node, in, "trilinear3d", in_shape, {}, {}, {scale_d, scale_h, scale_w}, std::string("trilinear"), align_corners, true);
        } else {
            resize_layer_size(engine, node, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, align_corners);
        }
#endif 
    } else {
        // Case 2: user uses output size
        auto output_size = maybe_outsize.toIntList();
        auto out_size = nvdim_to_sizes(sizes_to_nvdim(output_size));
        POROS_ASSERT(out_size.size() == 3, "aten::upsample_trilinear3d input Tensor and output size dimension mismatch");
        auto out_shape = in_shape;
        std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1) // IF TRT VERSION <= 7.0
        if (!align_corners) {
            create_plugin(engine, node, in, "trilinear3d", in_shape, out_shape, out_size, {}, std::string("trilinear"), align_corners);            
        } else {
            resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, true);
        }
#else
        resize_layer_size(engine, node, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
    }
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, UnsampleNearest1DConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, UnsampleNearest2DConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, UnsampleNearest3DConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, UnsampleLinear1DConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, UnsampleBilinear2DConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, UnsampleTrilinear3DConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
