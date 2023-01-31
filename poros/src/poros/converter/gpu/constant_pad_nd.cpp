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
* @file constant_pad_nd.cpp
* @author tianshaoqing@baidu.com
* @date Thur Dec 2 14:29:20 CST 2021
* @brief 
**/

#include "poros/converter/gpu/constant_pad_nd.h"
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

//DEPRECATED: 该实现方式内部采用的contat，在trt的profile阶段，会额外引入一些copy节点，导致性能变差。
// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
bool ConstantPadNdConverter::converter_old_version(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for ReplicationPadConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ConstantPadNdConverter is not Tensor as expected");

    // extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto self_dims = self->getDimensions();
    int64_t self_rank = self_dims.nbDims;

    // check input is dynamic or not
    std::vector<int64_t> self_dims_vec = nvdim_to_sizes(self_dims);
    // extract self
    torch::jit::IValue maybe_pad = engine->context().get_constant(inputs[1]);
    POROS_CHECK_TRUE((!maybe_pad.isNone()), "invaid inputs[1] for ConstantPadNdConverter");
    std::vector<int64_t> padding = maybe_pad.toIntList().vec();
    int64_t pad_size = padding.size();

    // pad_size must be an integer multiple of 2
    POROS_CHECK_TRUE((pad_size % 2 == 0), "Length of pad must be even but instead it equals: " << pad_size);
    int64_t l_pad = pad_size / 2;
    POROS_CHECK_TRUE((self_rank >= l_pad), "Length of pad should be no more than twice the number of "
            "dimensions of the input. Pad length is " << pad_size << "while the input has " << self_rank << "dimensions.");

    // extract value
    torch::jit::IValue maybe_value = engine->context().get_constant(inputs[2]);
    POROS_CHECK_TRUE((!maybe_value.isNone()), "invaid inputs[2] for ConstantPadNdConverter");
    float value = maybe_value.toScalar().to<float>();

    // prepare for dynamic
    const bool is_dynamic = check_nvtensor_is_dynamic(self);

    // dynamic下trt的Ifilllayer无法构建bool类型，所以先返回false（虽然constant_pad_nd bool的非常少）
    if (is_dynamic && maybe_value.isBool()) {
        LOG(WARNING) << "ConstantPadNdConverter is not support padding value is type of bool when dynamic.";
        return false;
    }

    nvinfer1::ITensor* self_shape = nullptr;
    nvinfer1::ITensor* rev_mask_shape_tensor = nullptr;
    if (is_dynamic) {
        self_shape = engine->network()->addShape(*self)->getOutput(0);
    }

    // create itensors vector
    std::vector<nvinfer1::ITensor*> itensors_vec;
    
    for (int64_t i = 0; i < l_pad; i++) {
        int64_t axis = self_rank - (i + 1);
        int64_t padding_index = i * 2;   
        // dynamic情况，需要使用mask完成padding shape的构造
        // 首先，使用rev_mask_tensor使self_shape[axis] = 0
        // 例如：self_shape = [2, 3, 4, 5]，axis = 3，则rev_mask_shape_tensor = [2, 3, 4, 0]
        if (is_dynamic && (padding[padding_index] > 0 || padding[padding_index + 1] > 0)) {
            at::Tensor rev_mask_tensor = at::ones({self_rank}, torch::kInt);
            rev_mask_tensor[axis] = 0;
            nvinfer1::ITensor* nv_rev_mask_tensor = tensor_to_const(engine, rev_mask_tensor);
            rev_mask_shape_tensor = add_elementwise(engine, 
                                                    nvinfer1::ElementWiseOperation::kPROD, 
                                                    self_shape, 
                                                    nv_rev_mask_tensor,
                                                    layer_info(node) + "_prod(axis_dim_to_zero)_" + std::to_string(i))->getOutput(0);
        }

        if (padding[padding_index] > 0) {
            itensors_vec.clear();
            // 非dynamic情况
            if (!is_dynamic) {
                // create pad tensor
                self_dims_vec[axis] = padding[padding_index];
                at::Tensor pad_tenosr = at::full(self_dims_vec, value, torch::kFloat32);
                // 默认是float32类型，如果self是int32的需转换类型
                if (self->getType() == nvinfer1::DataType::kINT32) {
                    pad_tenosr = pad_tenosr.to(at::ScalarType::Int);
                }
                // 默认是float32类型，如果self是bool的需转换类型（bool情况很少）
                if (self->getType() == nvinfer1::DataType::kBOOL && maybe_value.isBool()) {
                    pad_tenosr = pad_tenosr.to(at::ScalarType::Bool);
                }
                itensors_vec.push_back(tensor_to_const(engine, pad_tenosr));
            } else {
            // dynamic情况
                // 然后，使用mask_tensor构造只有axis下标是padding[padding_index]，其余数据都是0的tensor
                // 例如：self_shape = [2, 3, 4, 5]，则self_rank = 4，
                // 若当前axis = 3，padding[padding_index] = 2，则构造出来的nv_mask_tensor = [0, 0, 0, 2]
                at::Tensor mask_tensor = at::zeros({self_rank}, torch::kInt);
                mask_tensor[axis] = padding[padding_index];
                nvinfer1::ITensor* nv_mask_tensor = tensor_to_const(engine, mask_tensor);
                // 最后，nv_mask_tensor与之前得到的rev_mask_shape_tensor相加，就得到padding shape
                // 例如：刚才rev_mask_shape_tensor = [2, 3, 4, 0]， nv_mask_tensor = [0, 0, 0, 2]
                // 则pad_shape_tensor = [2, 3, 4, 2]
                nvinfer1::ITensor* pad_shape_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUM, 
                                            rev_mask_shape_tensor, 
                                            nv_mask_tensor,
                                            layer_info(node) + "_sum(gen_left_pad_shape)_" + std::to_string(i))->getOutput(0);
                // 根据padding shape和value创建nvtensor
                auto fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, nvinfer1::FillOperation::kLINSPACE);
                fill_layer->setInput(0, *pad_shape_tensor);
                fill_layer->setName((layer_info(node) + "_IFillLayer_" + std::to_string(padding_index)).c_str());

                at::Tensor value_tensor = torch::tensor(value, torch::kFloat32);
                at::Tensor delta_tensor = torch::zeros(self_rank, torch::kFloat32);
                // 默认是float32类型，如果self是int32的需转换类型
                if (self->getType() == nvinfer1::DataType::kINT32) {
                    value_tensor = value_tensor.to(at::ScalarType::Int);
                    delta_tensor = delta_tensor.to(at::ScalarType::Int);
                }
                auto value_itensor = tensor_to_const(engine, value_tensor);
                fill_layer->setInput(1, *value_itensor); // 初始值
                auto delta_itensor = tensor_to_const(engine, delta_tensor);
                fill_layer->setInput(2, *delta_itensor); // delta值
                
                itensors_vec.push_back(fill_layer->getOutput(0));
            }

            itensors_vec.push_back(self);
            // concat
            nvinfer1::IConcatenationLayer* concat_layer = 
                        engine->network()->addConcatenation(itensors_vec.data(), itensors_vec.size());
            concat_layer->setAxis(axis);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer_" + std::to_string(padding_index)).c_str());
            self = concat_layer->getOutput(0);
            // 非dynamic更新维度信息
            self_dims = self->getDimensions();
            self_dims_vec = nvdim_to_sizes(self_dims);
            // dynamic更新维度信息
            if (is_dynamic) {
                self_shape = engine->network()->addShape(*self)->getOutput(0);
            }
        }

        if (padding[padding_index + 1] > 0) {
            itensors_vec.clear();
            // padding self dim=axis的另一边，
            // 与上面的代码只有self加入itensors_vec先后顺序的区别，这里是先push_back
            itensors_vec.push_back(self);

            // create pad tensor
            if (!is_dynamic) {
                self_dims_vec[axis] = padding[padding_index + 1];
                at::Tensor pad_tenosr = at::full(self_dims_vec, value, torch::kFloat32);
                // 默认是float32类型，如果self是int32的需转换类型
                if (self->getType() == nvinfer1::DataType::kINT32) {
                    pad_tenosr = pad_tenosr.to(at::ScalarType::Int);
                }
                // 默认是float32类型，如果self是bool的需转换类型（bool情况很少）
                if (self->getType() == nvinfer1::DataType::kBOOL && maybe_value.isBool()) {
                    pad_tenosr = pad_tenosr.to(at::ScalarType::Bool);
                }
                itensors_vec.push_back(tensor_to_const(engine, pad_tenosr));
            } else {
                // 与上面代码类似
                at::Tensor mask_tensor = at::zeros({self_rank}, torch::kInt);
                mask_tensor[axis] = padding[padding_index + 1];
                nvinfer1::ITensor* nv_mask_tensor = tensor_to_const(engine, mask_tensor);
                nvinfer1::ITensor* pad_shape_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUM, 
                                            rev_mask_shape_tensor, 
                                            nv_mask_tensor,
                                            layer_info(node) + "_sum(gen_right_pad_shape)_" + std::to_string(i))->getOutput(0);

                auto fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, nvinfer1::FillOperation::kLINSPACE);
                fill_layer->setInput(0, *pad_shape_tensor);  // 设置output shape
                fill_layer->setName((layer_info(node) + "_IFillLayer_more_" + std::to_string(padding_index)).c_str());
                at::Tensor value_tensor = torch::tensor(value, torch::kFloat32);
                at::Tensor delta_tensor = torch::zeros(self_rank, torch::kFloat32); // 只有1个维度
                // 默认是float32类型，如果self是int32的需转换类型
                if (self->getType() == nvinfer1::DataType::kINT32) {
                    value_tensor = value_tensor.to(at::ScalarType::Int);
                    delta_tensor = delta_tensor.to(at::ScalarType::Int);
                }
                auto value_itensor = tensor_to_const(engine, value_tensor);
                fill_layer->setInput(1, *value_itensor); // 初始值
                auto delta_itensor = tensor_to_const(engine, delta_tensor);
                fill_layer->setInput(2, *delta_itensor);
                
                itensors_vec.push_back(fill_layer->getOutput(0));
            }
            
            // concat
            nvinfer1::IConcatenationLayer* concat_layer = 
                        engine->network()->addConcatenation(itensors_vec.data(), itensors_vec.size());
            concat_layer->setAxis(axis);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer_" + std::to_string(padding_index + 1)).c_str());
            self = concat_layer->getOutput(0);
            // 非dynamic更新维度信息
            self_dims = self->getDimensions();
            self_dims_vec = nvdim_to_sizes(self_dims);
            // dynamic更新维度信息
            if (is_dynamic) {
                self_shape = engine->network()->addShape(*self)->getOutput(0);
            }
        }
    }

    engine->context().set_tensor(node->outputs()[0], self);
    LOG(INFO) << "Output tensor shape: " << self->getDimensions();

    return true;
}

/** 
 * @brief 将pytorch组织的padding信息，转化成tensorrt可以接受的padding。
 * pytorch 下的padding order：
 *      The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ..., dim_m_begin, dim_m_end,
 *      where m is in range [0, n].
 * 期望被转变成的padding order：
 *      dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
 *      while n is the dimension of input.
 * 当前的转化逻辑，基于padding 本身是constant 这个前提（被padding的tensor是否dynamic不影响）。
 * 目前不支持padding本身是动态的，如果遇到相应的场景，再添加。
 * padding本身dynamic下的转化思路是：
 *      padding后面先补0，再reshape成(-1,2)维度，然后flip+transpose，最后reshape成1维即可。
 *      torch::reshape(torch::transpose(aten::flip(torch::reshape(padding_tensor, {-1, 2}), [0]), 1, 0),{-1});
 * **/
bool ConstantPadNdConverter::converter_padding(TensorrtEngine* engine,
                    int64_t rank, 
                    const std::vector<int64_t>& padding,
                    nvinfer1::ITensor*& start_tensor,
                    nvinfer1::ITensor*& total_padding_tensor) {
    
    std::vector<int64_t> start;
    std::vector<int64_t> total_padding;
    if (padding.size() % 2U != 0) {
        LOG(WARNING) << "padding size should be even but instead it equals: " << padding.size();
        return false;
    }

    const int64_t pad_dim_len = static_cast<int64_t>(padding.size() / 2U);
    const int64_t diff = rank - pad_dim_len;
    if (diff < 0) {
        LOG(WARNING) << "padding size should be no more than twice the number of dimensions of the input"
                    << " , but given padding size is: " << padding.size() 
                    << " , given input dimensions is: " << rank << ".";
        return false;
    }

    start.resize(rank, 0);
    total_padding.resize(rank, 0);

    for (int64_t i = diff; i < rank; i++) {
        const int64_t idx = i - diff;
        const int64_t reverse_idx = pad_dim_len - idx - 1;
        const int64_t before = padding[reverse_idx * 2];
        const int64_t after = padding[reverse_idx * 2 + 1];
        if (before < 0 || after < 0) {
            return false;
        }
        start[i] = -before;
        total_padding[i] = before + after;
    }

    at::Tensor at_start_tensor = torch::from_blob(start.data(), start.size(), 
                                                torch::TensorOptions().dtype(torch::kInt64));
    start_tensor = tensor_to_const(engine, at_start_tensor);

    at::Tensor at_total_padding_tensor = torch::from_blob(total_padding.data(), total_padding.size(), 
                                                        torch::TensorOptions().dtype(torch::kInt64));
    total_padding_tensor = tensor_to_const(engine, at_total_padding_tensor);
    return start_tensor && total_padding_tensor;
}

// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
bool ConstantPadNdConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for ReplicationPadConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ConstantPadNdConverter is not Tensor as expected");

    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    nvinfer1::Dims self_dims = self->getDimensions();
    const int64_t self_rank = self_dims.nbDims;

    // extract pad
    torch::jit::IValue maybe_pad = engine->context().get_constant(inputs[1]);
    POROS_CHECK_TRUE((!maybe_pad.isNone()), "invaid inputs[1] for ConstantPadNdConverter");
    std::vector<int64_t> padding = maybe_pad.toIntList().vec();
   
    // extract value
    torch::jit::IValue maybe_value = engine->context().get_constant(inputs[2]);
    POROS_CHECK_TRUE((!maybe_value.isNone()), "invaid inputs[2] for ConstantPadNdConverter");
    nvinfer1::ITensor* value_tensor = nullptr;
    float value = maybe_value.toScalar().to<float>();
    // value的类型与self对齐
    if (self->getType() == nvinfer1::DataType::kINT32) {
        value_tensor = tensor_to_const(engine, torch::tensor({value}).to(at::ScalarType::Int));
    } else if (self->getType() == nvinfer1::DataType::kBOOL && maybe_value.isBool()) {
        value_tensor = tensor_to_const(engine, torch::tensor({value}).to(at::ScalarType::Bool));
    } else {
        value_tensor = tensor_to_const(engine, torch::tensor({value}).to(at::ScalarType::Float));
    }

    // const bool is_dynamic = check_nvtensor_is_dynamic(self);
    // // dynamic下trt的Ifilllayer无法构建bool类型，所以先返回false（虽然constant_pad_nd bool的非常少）
    // if (is_dynamic && maybe_value.isBool()) {
    //     LOG(WARNING) << "ConstantPadNdConverter is not support padding value is type of bool when dynamic.";
    //     return false;
    // }

    nvinfer1::ITensor* start = nullptr;
    nvinfer1::ITensor* total_padding = nullptr;
    if (converter_padding(engine, self_rank, padding, start, total_padding) == false) {
        return false;
    }
    nvinfer1::ITensor* self_shape = engine->network()->addShape(*self)->getOutput(0);
    nvinfer1::ITensor* size = add_elementwise(engine, 
                                nvinfer1::ElementWiseOperation::kSUM, 
                                self_shape, 
                                total_padding,
                                layer_info(node) + "_sum(for_padding)")->getOutput(0);

    //fix stride setting
    nvinfer1::Dims stride;
    stride.nbDims = self_rank;
    std::fill_n(stride.d, self_rank, 1);
    const nvinfer1::Dims& dummy = stride;
    nvinfer1::ISliceLayer* layer = engine->network()->addSlice(*self, dummy, dummy, stride);
    layer->setInput(1, *start);
    layer->setInput(2, *size);
    layer->setMode(nvinfer1::SliceMode::kFILL);
    layer->setInput(4, *value_tensor);
    layer->setName((layer_info(node) + "_ISliceLayer").c_str());

    engine->context().set_tensor(node->outputs()[0], layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << layer->getOutput(0)->getDimensions();

    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ConstantPadNdConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu