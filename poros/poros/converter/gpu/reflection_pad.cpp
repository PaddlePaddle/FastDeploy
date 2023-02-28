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
* @file reflection_pad.cpp
* @author tianshaoqing@baidu.com
* @date Tue Aug 16 16:54:20 CST 2022
* @brief 
**/

#include "poros/converter/gpu/reflection_pad.h"
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

/**
 * @brief 翻转input的dim维，支持dynamic
 *
 * @param [in] engine : trtengine
 * @param [in] node : 当前节点
 * @param [in] input : 要翻转的tensor
 * @param [in] is_dynamic : 输入是否是dynamic的
 * @param [in] dim : 指定要翻转的维度
 * 
 * @return nvinfer1::ITensor*
 * @retval 返回翻转后的tensor
**/
static nvinfer1::ITensor* flip_nvtensor(TensorrtEngine* engine, 
                                        const torch::jit::Node *node, 
                                        nvinfer1::ITensor* input, 
                                        bool is_dynamic, 
                                        int dim) {

    auto in_dims = input->getDimensions();
    int64_t in_rank = in_dims.nbDims;
    dim = dim < 0 ? in_rank + dim : dim;

    POROS_ASSERT(dim >= 0 && dim < in_rank, "flip dim is out of range. expect range is [" + 
            std::to_string(-in_rank) + ", " + std::to_string(in_rank - 1) + "].");

    if (!is_dynamic) {
        std::vector<int64_t> start_vec, size_vec, stride_vec;
        for (int32_t r = 0; r < in_rank; r++) {
            start_vec.push_back(0);
            size_vec.push_back(in_dims.d[r]);
            stride_vec.push_back(1);
        }
        start_vec[dim] = size_vec[dim] - 1;
        stride_vec[dim] = -1;

        auto slice_layer = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(start_vec), 
                                                sizes_to_nvdim(size_vec), 
                                                sizes_to_nvdim(stride_vec));
        slice_layer->setName((layer_info(node) + "_ISliceLayer_flip_for_dim_" + std::to_string(dim)).c_str());
        return slice_layer->getOutput(0);
    } else {
        nvinfer1::ITensor* input_shape = engine->network()->addShape(*input)->getOutput(0);
        std::vector<int64_t> stride_vec(in_rank, 1), dim_mask_vec(in_rank, 0), tmp_vec(in_rank, 0);
        stride_vec[dim] = -1;
        dim_mask_vec[dim] = 1;

        nvinfer1::ITensor* dim_mask_tensor = tensor_to_const(engine, torch::tensor(dim_mask_vec, torch::kInt32));
        nvinfer1::ITensor* stride_tensor = tensor_to_const(engine, torch::tensor(stride_vec, torch::kInt32));

        nvinfer1::ITensor* start_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD, 
                                            input_shape, 
                                            dim_mask_tensor,
                                            layer_info(node) + "_prod_flip_for_dim_" +
                                            std::to_string(dim))->getOutput(0);

        start_tensor = add_elementwise(engine, 
                                    nvinfer1::ElementWiseOperation::kSUB, 
                                    start_tensor, 
                                    dim_mask_tensor,
                                    layer_info(node) + "_sub_flip_for_dim_" +
                                    std::to_string(dim))->getOutput(0);

        auto slice_layer = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(tmp_vec), 
                                                sizes_to_nvdim(tmp_vec), 
                                                sizes_to_nvdim(tmp_vec));
        slice_layer->setInput(0, *input);
        slice_layer->setInput(1, *start_tensor);
        slice_layer->setInput(2, *input_shape);
        slice_layer->setInput(3, *stride_tensor);
        slice_layer->setName((layer_info(node) + "_ISliceLayer_flip_for_dim_" + std::to_string(dim)).c_str());

        return slice_layer->getOutput(0);
    }
}

/**
 * @brief 根据padding的大小，计算left slice 的 start 和 size，以及 righit slice 的 size，
 * 具体作用见ReflectionPadConverter::converter注释
 *
 * @param [in] engine : trtengine
 * @param [in] node : 当前节点
 * @param [in] input : 要 reflection padding 的 tensor
 * @param [in] padding_is_nvtensor : padding参数是否是以nvscalar形式输入的
 * @param [in] padding_tensor : padding_is_nvtensor 为 true 时，读取内部 padding 数据
 * @param [in] padding_size : padding_is_nvtensor 为 false 时，读取内部 padding 数据
 * @param [in] axis : 当前 padding 的维度
 * 
 * @return std::tuple<nvinfer1::ITensor*, nvinfer1::ITensor*, nvinfer1::ITensor*>
 * @retval 返回left slice 的 start 和 size，以及 righit slice 的 size
**/
static std::tuple<nvinfer1::ITensor*, nvinfer1::ITensor*, nvinfer1::ITensor*> gen_slice_start_size(TensorrtEngine* engine, 
                                                                                const torch::jit::Node *node, 
                                                                                nvinfer1::ITensor* input,
                                                                                bool padding_is_nvtensor, 
                                                                                std::vector<nvinfer1::ITensor*> padding_tensor, 
                                                                                std::vector<int32_t> padding_size,
                                                                                int32_t axis) {
    nvinfer1::ITensor* left_start_tensor = nullptr;
    nvinfer1::ITensor* left_size_tensor = nullptr;
    nvinfer1::ITensor* right_size_tensor = nullptr;
    // start_vec[axis] = inDims.d[axis] - padding[padding_index] - 1; 基础是0
    // size_vec[axis] = padding[padding_index]; 基础是size
    nvinfer1::ITensor* input_shape = engine->network()->addShape(*input)->getOutput(0);

    auto in_dims = input->getDimensions();
    int64_t in_rank = in_dims.nbDims;
    std::vector<int64_t> dim_mask_vec(in_rank, 0), dim_remask_vec(in_rank, 1);
    dim_mask_vec[axis] = 1;
    dim_remask_vec[axis] = 0;
    nvinfer1::ITensor* dim_mask_tensor = tensor_to_const(engine, torch::tensor(dim_mask_vec, torch::kInt32));
    nvinfer1::ITensor* dim_remask_tensor = tensor_to_const(engine, torch::tensor(dim_remask_vec, torch::kInt32));
    
    nvinfer1::ITensor* shape_mask_axis = add_elementwise(engine, 
                                        nvinfer1::ElementWiseOperation::kPROD, 
                                        input_shape, 
                                        dim_mask_tensor,
                                        layer_info(node) + std::string("_left_shape_mask_axis_") + 
                                        std::to_string(axis))->getOutput(0);

    nvinfer1::ITensor* left_padding_size_tensor = nullptr;
    nvinfer1::ITensor* right_padding_size_tensor = nullptr;
    if (!padding_is_nvtensor) {
        left_padding_size_tensor = tensor_to_const(engine, torch::tensor({padding_size[0]}, torch::kInt32));
        right_padding_size_tensor = tensor_to_const(engine, torch::tensor({padding_size[1]}, torch::kInt32));
    } else {
        left_padding_size_tensor = padding_tensor[0];
        right_padding_size_tensor = padding_tensor[1];
    }

    nvinfer1::ITensor* left_padding_mask_axis = add_elementwise(engine, 
                                        nvinfer1::ElementWiseOperation::kPROD, 
                                        dim_mask_tensor,
                                        left_padding_size_tensor, 
                                        layer_info(node) + std::string("_left_padding_mask_axis_") + 
                                        std::to_string(axis))->getOutput(0);

    nvinfer1::ITensor* shape_sub_left_padding_mask_axis = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUB, 
                                            shape_mask_axis, 
                                            left_padding_mask_axis,
                                            layer_info(node) + std::string("_left_shape_sub_padding_mask_axis_") + 
                                            std::to_string(axis))->getOutput(0);

    left_start_tensor = add_elementwise(engine, 
                                nvinfer1::ElementWiseOperation::kSUB, 
                                shape_sub_left_padding_mask_axis, 
                                dim_mask_tensor,
                                layer_info(node) + std::string("_left_shape_sub_padding_sub_one_mask_axis_") + 
                                std::to_string(axis))->getOutput(0);
    
    nvinfer1::ITensor* shape_remask_axis = add_elementwise(engine, 
                                    nvinfer1::ElementWiseOperation::kPROD, 
                                    input_shape, 
                                    dim_remask_tensor,
                                    layer_info(node) + std::string("_left_shape_remask_axis_") + 
                                    std::to_string(axis))->getOutput(0);

    left_size_tensor = add_elementwise(engine, 
                                nvinfer1::ElementWiseOperation::kSUM, 
                                shape_remask_axis, 
                                left_padding_mask_axis,
                                layer_info(node) + std::string("_left_shape_remask_sum_padding_axis_") + 
                                std::to_string(axis))->getOutput(0);
    
    nvinfer1::ITensor* right_padding_mask_axis = add_elementwise(engine, 
                                        nvinfer1::ElementWiseOperation::kPROD, 
                                        dim_mask_tensor,
                                        right_padding_size_tensor, 
                                        layer_info(node) + std::string("_right_padding_mask_axis_") + 
                                        std::to_string(axis))->getOutput(0);

    right_size_tensor = add_elementwise(engine, 
                                nvinfer1::ElementWiseOperation::kSUM, 
                                shape_remask_axis, 
                                right_padding_mask_axis,
                                layer_info(node) + std::string("_right_shape_remask_sum_padding_axis_") + 
                                std::to_string(axis))->getOutput(0);

    return std::make_tuple(left_start_tensor, left_size_tensor, right_size_tensor);
}

/**
 * @brief ReflectionPad功能：镜像填充，pad规则类似constant_pad_nd，只不过pad值换成边缘的镜像
 * 例如：输入 x = torch.arange(8).reshape(2, 4) =
 * [[0,1,2,3],
 *  [4,5,6,7]]
 * 那么 ReflectionPad1d(x, [1,2]) = 
 * [[1,0,1,2,3,2,1],
 *  [5,4,5,6,7,6,5]]
 * 
 * converter实现思路
 * 先将x整体按照pad维度翻转x_flip =
 * [[3,2,1,0],
 *  [7,6,5,4]]
 * 左边padding size = 1
 * x' = cat([x_flip[:, -2], x], dim = 1)
 * [[3,2,|1|,0],        [[|1|,0,1,2,3],
 *  [7,6,|5|,4]]         [|5|,4,5,6,7]]
 *        ^                ^
 * 右边padding size = 2
 * x'' = cat([x', x_flip[:, 1:2]], dim = 1)
 * [[3,|2,1|,0],        [[1,0,1,2,3,|2,1|],
 *  [7,|6,5|,4]]         [5,4,5,6,7,|6,5|]]
 *        ^                            ^
 * ReflectionPad2d同理
 *
 * @param [in] engine : trtengine
 * @param [in] node : 当前节点
 * 
 * @return bool
 * @retval true convert 成功，false convert 失败
**/
bool ReflectionPadConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    // "aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor"
    // "aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor"
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for ReflectionPadConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ReflectionPadConverter is not Tensor as expected");

    //extract self
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto inDims = in->getDimensions();
    int64_t inRank = inDims.nbDims;
    
    std::vector<nvinfer1::ITensor*> tensors_vec;

    bool has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
    bool input0_is_dynamic = check_nvtensor_is_dynamic(in);

    if (!has_tensor_scalar) {

        //extract padding
        auto padding = (engine->context().get_constant(inputs[1])).toIntList().vec();
        
        for (int64_t i = 0; i < int(padding.size() / 2); i++) {
            int64_t axis = inRank - (i + 1); // axis = {inRank - 1, inRank - 2}
            int64_t padding_index = i * 2;

            nvinfer1::ITensor* in_flip = flip_nvtensor(engine, node, in, input0_is_dynamic, axis);

            std::tuple<nvinfer1::ITensor*, nvinfer1::ITensor*, nvinfer1::ITensor*> left_start_size_right_size;

            if (inDims.d[axis] < 0) {
                std::vector<nvinfer1::ITensor*> tmp_itensor_vec;
                std::vector<int32_t> padding_size_vec = {(int32_t)padding[padding_index], (int32_t)padding[padding_index + 1]};
                left_start_size_right_size = gen_slice_start_size(engine, node, in, false, tmp_itensor_vec, padding_size_vec, axis);
            }

            if (padding[padding_index] > 0) { // left padding value
                tensors_vec.clear();
                std::vector<int64_t> start_vec, size_vec, stride_vec;
                for (int32_t r = 0; r < inRank; r++) {
                    start_vec.push_back(0);
                    size_vec.push_back(inDims.d[r]);
                    stride_vec.push_back(1);
                }
                start_vec[axis] = inDims.d[axis] - padding[padding_index] - 1;
                size_vec[axis] = padding[padding_index];

                auto slice_layer = engine->network()->addSlice(*in_flip,
                                                    sizes_to_nvdim(start_vec), 
                                                    sizes_to_nvdim(size_vec), 
                                                    sizes_to_nvdim(stride_vec));
                slice_layer->setName((layer_info(node) + "_ISliceLayer_for_leftpadding_" + std::to_string(axis)).c_str());
                if (inDims.d[axis] < 0) {
                    slice_layer->setInput(1, *(std::get<0>(left_start_size_right_size)));
                    slice_layer->setInput(2, *(std::get<1>(left_start_size_right_size)));
                }
                
                tensors_vec.push_back(slice_layer->getOutput(0));
                tensors_vec.push_back(in);

                auto concat_layer = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
                concat_layer->setAxis(axis);
                concat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_leftpadding_" + std::to_string(axis)).c_str());
                in = concat_layer->getOutput(0);
                inDims = in->getDimensions();
            }

            if (padding[padding_index + 1] > 0) { // right padding value
                tensors_vec.clear();
                tensors_vec.push_back(in);

                std::vector<int64_t> start_vec, size_vec, stride_vec;
                for (int32_t r = 0; r < inRank; r++) {
                    start_vec.push_back(0);
                    size_vec.push_back(inDims.d[r]);
                    stride_vec.push_back(1);
                }
                start_vec[axis] = 1;
                size_vec[axis] = padding[padding_index + 1];

                auto slice_layer = engine->network()->addSlice(*in_flip,
                                                    sizes_to_nvdim(start_vec), 
                                                    sizes_to_nvdim(size_vec), 
                                                    sizes_to_nvdim(stride_vec));
                slice_layer->setName((layer_info(node) + "_ISliceLayer_for_rightpadding_"+ std::to_string(axis)).c_str());
                if (inDims.d[axis] < 0) {
                    slice_layer->setInput(2, *(std::get<2>(left_start_size_right_size)));
                }
                
                tensors_vec.push_back(slice_layer->getOutput(0));

                auto concat_layer = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
                concat_layer->setAxis(axis);
                concat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_rightpadding_" + std::to_string(axis)).c_str());
                in = concat_layer->getOutput(0);
                inDims = in->getDimensions();

            }
        }
    } else {
        // 先分开
        std::vector<nvinfer1::ITensor*> padding_tensor_vec;
        nvinfer1::ITensor* padding_tensor = this->get_tensor_scalar(inputs[1]);
        nvinfer1::Dims padding_tensor_dims = padding_tensor->getDimensions();

        for (int i = 0; i < padding_tensor_dims.d[0]; i++) {
            std::vector<int64_t> start_vec(1, i), size_vec(1, 1), stride_vec(1, 1);
            auto slice_layer = engine->network()->addSlice(*padding_tensor,
                                                    sizes_to_nvdim(start_vec), 
                                                    sizes_to_nvdim(size_vec), 
                                                    sizes_to_nvdim(stride_vec));
            slice_layer->setName((layer_info(node) + "_ISliceLayer_for_padding_tensor_" + std::to_string(i)).c_str());
            padding_tensor_vec.push_back(slice_layer->getOutput(0));
        }

        for (size_t i = 0; i < padding_tensor_vec.size() / 2; i++) {
            int64_t axis = inRank - (i + 1); // axis = {inRank - 1, inRank - 2}
            int64_t padding_index = i * 2;

            nvinfer1::ITensor* in_flip = flip_nvtensor(engine, node, in, input0_is_dynamic, axis);

            std::vector<nvinfer1::ITensor*> itensor_vec = {padding_tensor_vec[padding_index], padding_tensor_vec[padding_index + 1]};
            std::vector<int32_t> tmp_vec;
            auto left_start_size_right_size = gen_slice_start_size(engine, node, in, true, itensor_vec, tmp_vec, axis);

            // left
            tensors_vec.clear();
            std::vector<int64_t> start_vec, size_vec, stride_vec(inRank, 1); 

            auto slice_layer = engine->network()->addSlice(*in_flip,
                                                sizes_to_nvdim(stride_vec), 
                                                sizes_to_nvdim(stride_vec), 
                                                sizes_to_nvdim(stride_vec));
            
            slice_layer->setInput(1, *(std::get<0>(left_start_size_right_size)));
            slice_layer->setInput(2, *(std::get<1>(left_start_size_right_size)));
            slice_layer->setName((layer_info(node) + "_ISliceLayer_for_leftpadding_" + std::to_string(axis)).c_str());
            
            tensors_vec.push_back(slice_layer->getOutput(0));
            tensors_vec.push_back(in);

            auto concat_layer = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
            concat_layer->setAxis(axis);
            in = concat_layer->getOutput(0);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_leftpadding_" + std::to_string(axis)).c_str());
            inDims = in->getDimensions();

            // right
            tensors_vec.clear();
            tensors_vec.push_back(in);

            std::vector<int64_t> start_vec2, stride_vec2;
            for (int32_t r = 0; r < inRank; r++) {
                start_vec2.push_back(0);
                stride_vec2.push_back(1);
            }
            start_vec2[axis] = 1;

            auto slice_layer2 = engine->network()->addSlice(*in_flip,
                                                sizes_to_nvdim(start_vec2), 
                                                sizes_to_nvdim(stride_vec2), 
                                                sizes_to_nvdim(stride_vec2));
            
            slice_layer2->setInput(2, *(std::get<2>(left_start_size_right_size)));
            slice_layer2->setName((layer_info(node) + "_ISliceLayer_for_rightpadding_" + std::to_string(axis)).c_str());
            tensors_vec.push_back(slice_layer2->getOutput(0));

            auto concat_layer2 = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
            concat_layer2->setAxis(axis);
            concat_layer2->setName((layer_info(node) + "_IConcatenationLayer_for_rightpadding_" + std::to_string(axis)).c_str());
            in = concat_layer2->getOutput(0);
            inDims = in->getDimensions();
        }
    }
    
    engine->context().set_tensor(node->outputs()[0], in);
    LOG(INFO) << "Output tensor shape: " << in->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ReflectionPadConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
