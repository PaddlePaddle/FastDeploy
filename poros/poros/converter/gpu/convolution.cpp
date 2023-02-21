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
* @file convolution.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/convolution.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/weight.h"
#include "poros/context/poros_global.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

//note: conv?d 与 _convolution 相比，前者入参有7个，后者入参有12个or13个。
//note2: conv?d 与 _convolution 相比，缺了 transposed 参数(bool类型)，默认补零即可。
//note3: conv?d 与 _convolution 相比，缺了 output_padding 参数(int[]类型)，默认也补零即可。
//note4: conv?d 之间的差异，在于 int[] 的维度不一样。
bool ConvolutionConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 12 || inputs.size() == 13 || inputs.size() == 7), 
        "invaid inputs size for ConvolutionConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ConvolutionConverter is not Tensor as expected");
    //ATTENTION HERE: assumes weight inputs are all come from prim::Constant and is TensorType.
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get()) && 
        inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for ConvolutionConverter is not Tensor or not come from prim::Constant as expected");
    //ATTENTION HERE: assumes int[] inputs are all come from prim::Constant.
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
        "input[3] for ConvolutionConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[4]->node()->kind() == torch::jit::prim::Constant),
        "input[4] for ConvolutionConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[5]->node()->kind() == torch::jit::prim::Constant),
        "input[5] for ConvolutionConverter is not come from prim::Constant as expected");
    if (inputs.size() == 12 || inputs.size() == 13) {
        POROS_CHECK_TRUE((inputs[7]->node()->kind() == torch::jit::prim::Constant),
            "input[7] for ConvolutionConverter is not come from prim::Constant as expected");
    }
    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node); 

    //extract dims settings
    auto stride = sizes_to_nvdim((engine->context().get_constant(inputs[3])).toIntList());
    auto padding =  sizes_to_nvdim((engine->context().get_constant(inputs[4])).toIntList());
    auto dilation = sizes_to_nvdim((engine->context().get_constant(inputs[5])).toIntList());

    //handle the difference between _convolution and conv?d.
    bool transposed = false;
    nvinfer1::Dims out_padding;
    int64_t groups = 1;
    if (inputs.size() == 12 || inputs.size() == 13) {
        transposed = (engine->context().get_constant(inputs[6])).toBool();
        out_padding = sizes_to_nvdim((engine->context().get_constant(inputs[7])).toIntList());
        groups = (engine->context().get_constant(inputs[8])).toInt();
    //situation when conv1d & conv2d & conv3d has no transposed and out_padding paragrams.
    } else {
        out_padding.nbDims = padding.nbDims;
        for (int i = 0; i < padding.nbDims; i++) {
            out_padding.d[i] = 0;
        }
        groups = (engine->context().get_constant(inputs[6])).toInt();
    }
  
    //handle stride & dilation & padding & out_apdding
    if (stride.nbDims == 1) {
        stride = unsqueeze_dims(stride, 1, stride.d[0]);
        LOG(INFO) << "Reshaped stride for ConvolutionConverter: " << stride;
    }
    if (dilation.nbDims == 1) {
        dilation = unsqueeze_dims(dilation, 1, dilation.d[0]);
        LOG(INFO) << "Reshaped dilation for ConvolutionConverter: " << dilation;
    }
    if (padding.nbDims == 1) {
        padding = unsqueeze_dims(padding, 1, 0);
        LOG(INFO) << "Reshaped padding for ConvolutionConverter: " << padding;
    }
    if (out_padding.nbDims == 1) {
        out_padding = unsqueeze_dims(out_padding, 1, 0);
        LOG(INFO) << "Reshaped out_padding for ConvolutionConverter: " << out_padding;
    }

    // According to our tests, when the GPU architecture is Ampere and nvidia tf32 is enabled, 
    // IConvolutionLayer set PostPadding explicitly even if it is the default value will cause 
    // tensorrt choose the slow Conv+BN+Relu kernel in some case.
    // So we try not to set PostPadding when it is the default value.
    bool out_padding_need_set = false;
    for (int32_t i = 0; i < out_padding.nbDims; i++) {
        if (out_padding.d[i] != 0) {
            out_padding_need_set = true;
            break;
        }
    }

    //extract bias
    auto maybe_bias = engine->context().get_constant(inputs[2]);
    Weights bias;
    if (maybe_bias.isTensor()) {
        bias = Weights(maybe_bias.toTensor());
    } else {//when bias is None
        bias = Weights();
    }

    //extract weight
    //the situation can be complex. 
    //sometimes this params is come from constant. sometimes it's come from another tensor.
    //because of the handle strategy we set in prim::Constant.
    //we'd better check if it is come from constant first.
    auto maybe_weight = engine->context().get_constant(inputs[1]);
    /*---------------------------------------------------------------------------
    *          when weight is come from constant
    ---------------------------------------------------------------------------*/
    if (maybe_weight.isTensor()) {
        auto weight = Weights(maybe_weight.toTensor());

        //first: handle input
        auto dims = in->getDimensions();
        auto orig_dims = dims;
        POROS_CHECK(orig_dims.nbDims > 2, "Unable to create convolution layer from node: " << *node);

        bool expandDims = (orig_dims.nbDims < 4);
        if (expandDims) {
            in = add_padding(engine, node, in, 4);
            dims = in->getDimensions();
            LOG(INFO) << "Reshaped Input dims: " << dims;
        }

        //second: handle dims
        if (weight.shape.nbDims < 4) {
            for (int i = weight.shape.nbDims; i < 4; ++i) {
                weight.shape.d[i] = 1;
            }
            weight.shape.nbDims = 4;
            weight.kernel_shape.nbDims = 2;
            weight.kernel_shape.d[1] = 1;
            LOG(INFO) << "Reshaped Weights for ConvolutionConverter: " << weight;
        }

        //fifth: try to add new layer
        nvinfer1::ILayer* new_layer;
        if (transposed) {   
            // shape of deconvolution's weight: [in, out/groups, ...]
            auto deconv = engine->network()->addDeconvolutionNd(*in,
                                weight.shape.d[1] * groups, weight.kernel_shape, weight.data, bias.data);
            POROS_CHECK(deconv, "Unable to create deconvolution layer from node: " << *node);
            
            deconv->setStrideNd(stride);
            deconv->setPaddingNd(padding);
            deconv->setName((layer_info(node) + "_IDeconvolutionLayer").c_str());
#if NV_TENSORRT_MAJOR > 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR >= 1)
            deconv->setDilationNd(dilation);
            deconv->setNbGroups(groups);
#else
            POROS_CHECK(groups == 1, "for deconv with groups > 1, require TensorRT version >= 7.1");
            for (int idx = 0; idx < dilation.nbDims; idx++) {
                POROS_CHECK(dilation.d[idx] == 1, "for deconv with dilation > 1, require TensorRT version >= 7.1");
            }
#endif
            new_layer = deconv;
        // when transposed == false
        } else {
            // shape of convolution's weight: [out, in/groups, ...]
            auto conv = engine->network()->addConvolutionNd(*in, 
                                weight.shape.d[0], weight.kernel_shape, weight.data, bias.data);
            POROS_CHECK(conv, "Unable to create convolution layer from node: " << *node);
            
            conv->setStrideNd(stride);
            conv->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN);
            conv->setPaddingNd(padding);
            if (out_padding_need_set) {
                conv->setPostPadding(out_padding);
            }
            conv->setDilationNd(dilation);
            conv->setNbGroups(groups);
            conv->setName((layer_info(node) + "_IConvolutionLayer").c_str());
            new_layer = conv;
        }

        auto out = add_unpadding(engine, node, new_layer->getOutput(0), orig_dims.nbDims);
        engine->context().set_tensor(node->outputs()[0], out);
        LOG(INFO) << "Output tensor shape: " << out->getDimensions();

        return true;
    /*---------------------------------------------------------------------------
    * when weight is come from other layer's (especial Dequantize layer) output
    ---------------------------------------------------------------------------*/
    } else {
        auto kernel = engine->context().get_tensor(inputs[1]);
        POROS_CHECK_TRUE((kernel != nullptr), "Unable to init input tensor for node: " << *node);
        auto kernel_dims = kernel->getDimensions();
        
        // Make a new Dims with only the spatial dimensions.
        nvinfer1::Dims filter_dim;
        int64_t nbSpatialDims = in->getDimensions().nbDims - 2;
        POROS_CHECK(nbSpatialDims == (kernel_dims.nbDims - 2),
        "Number of input spatial dimensions should match the kernel spatial dimensions");
        filter_dim.nbDims = nbSpatialDims;
        filter_dim.d[0] = kernel_dims.d[2];
        filter_dim.d[1] = kernel_dims.d[3];
        
        // Initialize a dummy constant kernel to pass it to INetwork->addConvolutionNd/addDeconvolutionNd API.
        auto kernel_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
        
        nvinfer1::ILayer* layer = nullptr;
        if (transposed) {
            nvinfer1::IDeconvolutionLayer* deconv = engine->network()->addDeconvolutionNd(*in,
                        kernel_dims.d[0],
                        filter_dim,
                        kernel_weights,
                        bias.data);
            deconv->setStrideNd(stride);
            deconv->setDilationNd(dilation);
            deconv->setNbGroups(groups);
            deconv->setPaddingNd(padding);
            // Set deconv kernel weights
            deconv->setInput(1, *kernel);
            deconv->setName((layer_info(node) + "_IDeconvolutionLayer").c_str());
            POROS_CHECK(deconv, "Unable to create deconv layer with non-const weights from node: " << *node);
            layer = deconv;
        } else {
            nvinfer1::IConvolutionLayer* conv = engine->network()->addConvolutionNd(*in,
                        kernel_dims.d[0],
                        filter_dim,
                        kernel_weights,
                        bias.data);
            conv->setStrideNd(stride);
            conv->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN);
            conv->setPaddingNd(padding);
            if (out_padding_need_set) {
                conv->setPostPadding(out_padding);
            }
            conv->setDilationNd(dilation);
            conv->setNbGroups(groups);
            // Set conv kernel weights
            conv->setInput(1, *kernel);
            conv->setName((layer_info(node) + "_IConvolutionLayer").c_str());
            layer = conv;
        }
        engine->context().set_tensor(node->outputs()[0], layer->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << layer->getOutput(0)->getDimensions();
        return true;
    }
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ConvolutionConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
