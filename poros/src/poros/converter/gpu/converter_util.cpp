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
// https://github.com/pytorch/TensorRT/blob/master/core/conversion/converters/converter_util.cpp
//
// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the 3-Clause BSD License

/**
* @file converter_util.cpp
* @author tianjinjin@baidu.com
* @date Thu Aug 12 14:50:37 CST 2021
* @brief 
**/

#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/weight.h"
#include "poros/engine/trtengine_util.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {


nvinfer1::ITensor* add_padding(TensorrtEngine* engine,
                            const torch::jit::Node* n, 
                            nvinfer1::ITensor* tensor,
                            int nDim,
                            bool trailing,
                            bool use_zeros) {
    const auto dims = tensor->getDimensions();
    if (dims.nbDims < nDim) {
        auto newDims = dims;
        for (int dim = dims.nbDims; dim < nDim; ++dim) {
            newDims = unsqueeze_dims(newDims, trailing ? dim : 0, 1, use_zeros);
        }
        LOG(INFO) << "Original shape: " << dims << ", reshaping to: " << newDims;
        auto shuffle_layer = engine->network()->addShuffle(*tensor);
        POROS_CHECK(shuffle_layer, "Unable to create shuffle layer");
        shuffle_layer->setReshapeDimensions(newDims);
        shuffle_layer->setZeroIsPlaceholder(use_zeros);
        shuffle_layer->setName((layer_info(n) + " [Reshape to " + nvdim_to_str(newDims) + ']').c_str());
        return shuffle_layer->getOutput(0);
    } else {
        return tensor;
    }
}

nvinfer1::ITensor* add_unpadding(TensorrtEngine* engine,
                            const torch::jit::Node* n,
                            nvinfer1::ITensor* tensor,
                            int nDim,
                            bool trailing,
                            bool use_zeros) {
    const auto dims = tensor->getDimensions();
    if (dims.nbDims > nDim) {
        auto newDims = dims;
        for (int dim = dims.nbDims; dim > nDim; --dim) {
            newDims = squeeze_dims(newDims, trailing ? dim - 1 : 0);
        }
        LOG(INFO) << "Original shape: " << dims << ", reshaping to: " << newDims;
        auto shuffle_layer = engine->network()->addShuffle(*tensor);
        POROS_CHECK(shuffle_layer, "Unable to create shuffle layer");
        shuffle_layer->setReshapeDimensions(newDims);
        shuffle_layer->setZeroIsPlaceholder(use_zeros);
        shuffle_layer->setName((layer_info(n) + " [Reshape to " + nvdim_to_str(newDims) + "]").c_str());
        return shuffle_layer->getOutput(0);
    } else {
        return tensor;
    }
}

bool check_tensor_type(nvinfer1::ITensor* &self, nvinfer1::ITensor* &other) {
    // 保证二元操作的两个tensor类型相同 float32 > harf > int32 > int8 
    if (self->getType() != nvinfer1::DataType::kBOOL){
        if (self->getType() < other->getType()) {
            if (self->getType() == nvinfer1::DataType::kINT8){
                self->setType(other->getType());
            }
            else {
                other->setType(self->getType());
            }
        }
        else if(other->getType() < self->getType()) {
            if (other->getType() == nvinfer1::DataType::kINT8){
                other->setType(self->getType());
            }
            else {
                self->setType(other->getType());
            }
        }
    }
    return true;
}

nvinfer1::ILayer* add_elementwise(TensorrtEngine* engine,
                            nvinfer1::ElementWiseOperation op,
                            nvinfer1::ITensor* self,
                            nvinfer1::ITensor* other,
                            const std::string& name) {
    // ensure self to have larger number of dimension
    bool swapSelfOther = false;
    check_tensor_type(self, other);
    if (self->getDimensions().nbDims < other->getDimensions().nbDims) {
        std::swap(self, other);
        swapSelfOther = true;
    }
    auto selfDim = nvdim_to_sizes(self->getDimensions());
    auto otherDim = nvdim_to_sizes(other->getDimensions());
    if (selfDim.size() != otherDim.size()) {
        // other is with dynamic shape, need to expand its dimension now and get its
        // shape at runtime
        // 对other而言，如果其dim是-1，则需要保持原维度，如果其dim是1，则需要等于self相应的维度。
        if (otherDim.end() != std::find(otherDim.begin(), otherDim.end(), -1)) {
            auto thOtherStaticShapeMask = torch::ones(selfDim.size(), torch::kInt32);
            auto thOtherDynamicShapeMask = torch::zeros(selfDim.size(), torch::kInt32);
            for (size_t start = selfDim.size() - otherDim.size(), idx = 0; idx < otherDim.size(); ++idx) {
                if (-1 != otherDim[idx]) {
                    thOtherStaticShapeMask[start + idx] = otherDim[idx];
                } else {
                    thOtherStaticShapeMask[start + idx] = 0;
                    if (selfDim[start + idx] == 1) {
                        thOtherDynamicShapeMask[start + idx] = -1;
                    } else {
                        thOtherDynamicShapeMask[start + idx] = 1;
                    }
                }
            }
            auto otherStaticShapeMask = tensor_to_const(engine, thOtherStaticShapeMask);
            auto otherDynamicShapeMask = tensor_to_const(engine, thOtherDynamicShapeMask);
            auto selfShape = engine->network()->addShape(*self)->getOutput(0);
            
            // size of dynamic dimension of other need to the same as that of
            // corresponding dimension of self
            auto otherDynamicShape = engine->network()->addElementWise(*selfShape,
                            *otherDynamicShapeMask, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
            auto targetOtherShape = engine->network()->addElementWise(*otherDynamicShape, 
                            *otherStaticShapeMask, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);

            auto otherShuffle = engine->network()->addShuffle(*other);
            otherShuffle->setName((name + "_IShuffleLayer").c_str());
            otherShuffle->setInput(1, *targetOtherShape);
            other = otherShuffle->getOutput(0);
        } else {
            // other is with static shape, expand dimension to make tow tensor have
            // the same number of dimension
            auto otherShuffle = engine->network()->addShuffle(*other);
            otherShuffle->setReshapeDimensions(sizes_to_nvdim_with_pad(otherDim, selfDim.size()));
            otherShuffle->setName((name + "_IShuffleLayer").c_str());
            other = otherShuffle->getOutput(0);
        }
    }
    if (swapSelfOther) {
        // swap back
        std::swap(self, other);
        swapSelfOther = false;
    }
    auto ele = engine->network()->addElementWise(*self, *other, op);
    ele->setName(name.c_str());
    return ele;
}

nvinfer1::ITensor* broadcast_itensor(TensorrtEngine* engine,
                                const torch::jit::Node* n,
                                nvinfer1::ITensor* tensor,
                                const int new_rank,
                                std::string name) {
    int current_rank = tensor->getDimensions().nbDims;
    POROS_CHECK((current_rank <= new_rank), "Cannot broadcast a higher rank tensor to a lower rank tensor.");
    if (current_rank < new_rank) {
        //1. get shape tensor
        nvinfer1::ITensor* shape_tensor = engine->network()->addShape(*tensor)->getOutput(0);

        //2.padding the missing rank part with value 1.
        std::vector<int64_t> padding_vec(new_rank - current_rank, 1);
        nvinfer1::Dims padding_dim = sizes_to_nvdim(c10::IntArrayRef(padding_vec));
        at::Tensor the_padding = torch::tensor(nvdim_to_sizes(padding_dim), torch::kInt32);
        nvinfer1::ITensor* padding_shape = tensor_to_const(engine, the_padding);

        //3. concat the shape tensor
        std::vector<nvinfer1::ITensor*> to_concat_tensors = {padding_shape, shape_tensor};
        nvinfer1::IConcatenationLayer* shape_cat_layer = engine->network()->addConcatenation(to_concat_tensors.data(), to_concat_tensors.size());
        shape_cat_layer->setName((layer_info(n) + "_IConcatenationLayer_for_" + name).c_str());
        auto new_shape = shape_cat_layer->getOutput(0);

        //4. shuffle given tensor to the new shape
        nvinfer1::IShuffleLayer* reshape_layer = engine->network()->addShuffle(*tensor);
        reshape_layer->setInput(1, *new_shape);
        reshape_layer->setName((layer_info(n) + "_IShuffleLayer_for_" + name).c_str());
        nvinfer1::ITensor* new_tensor = reshape_layer->getOutput(0);
        return new_tensor;
    }
    return tensor;               
}

nvinfer1::ITensor* cast_itensor(TensorrtEngine* engine, 
                                nvinfer1::ITensor* tensor, 
                                nvinfer1::DataType dtype) {
    if (tensor->getType() != dtype) {
        std::ostringstream tensor_id;
        tensor_id << reinterpret_cast<int*>(tensor);
        
        auto id_layer = engine->network()->addIdentity(*tensor);
        POROS_CHECK(id_layer, "Unable to create identity layer for ITensor: " << tensor_id.str());
        auto casted_tensor = id_layer->getOutput(0);
        casted_tensor->setType(dtype);
        
        LOG(INFO) << "Casting ITensor " << tensor_id.str() << " from " << tensor->getType() << " to " << dtype;
        std::stringstream ss;
        ss << "[Cast ITensor " << tensor_id.str() << " from " << tensor->getType() << " to " << dtype << "]";
        id_layer->setName(ss.str().c_str());
        return casted_tensor;
    } else {
        return tensor;
    }
}

// 对nv shape tensor进行unsqueeze操作, 支持dim倒序
nvinfer1::ITensor* unsqueeze_nv_shapetensor(TensorrtEngine* engine, 
                                    nvinfer1::ITensor* input, int dim) {
    nvinfer1::Dims input_dims = input->getDimensions();

    if (input_dims.nbDims != 1 || input->getType() != nvinfer1::DataType::kINT32) {
        LOG(INFO) << "input is not shape tensor";
        return nullptr;
    }
    // dim must be in range of [-input_dims.d[0] - 1, input_dims.d[0]].
    if (dim < -input_dims.d[0] - 1 || dim > input_dims.d[0]) {
        LOG(INFO) << "expected to be in range of [" << -input_dims.d[0] - 1 << "," 
        << input_dims.d[0] << "], but got " << dim;
        return nullptr;
    }
    if (dim < 0) {
        dim = input_dims.d[0] + dim + 1;
    }
    std::vector<nvinfer1::ITensor*> inputs_nvtensor;
    nvinfer1::ITensor* insert_tensor = tensor_to_const(engine, torch::tensor({1}, torch::kInt));
    // if dim == 0 or dim == input_dims.d[0], concat origin tensor and insert tensor directly.
    if (dim == 0) {
        inputs_nvtensor.push_back(insert_tensor);
        inputs_nvtensor.push_back(input);
        
    } else if (dim == input_dims.d[0]) {
        inputs_nvtensor.push_back(input);
        inputs_nvtensor.push_back(insert_tensor);
    } else {
        // divide origin tensor into two parts, then insert the unsqueeze tensor.
        std::vector<int64_t> start_vec{0}, size_vec{dim}, stride_vec{1};
        nvinfer1::ISliceLayer* slice_front = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(start_vec),
                                                sizes_to_nvdim(size_vec),
                                                sizes_to_nvdim(stride_vec));
        inputs_nvtensor.push_back(slice_front->getOutput(0));
        inputs_nvtensor.push_back(insert_tensor);
        start_vec[0] = dim;
        size_vec[0] = input_dims.d[0] - dim;
        nvinfer1::ISliceLayer* slice_back = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(start_vec),
                                                sizes_to_nvdim(size_vec),
                                                sizes_to_nvdim(stride_vec));
        inputs_nvtensor.push_back(slice_back->getOutput(0));
    }
    nvinfer1::IConcatenationLayer* concat_layer = 
                    engine->network()->addConcatenation(inputs_nvtensor.data(), inputs_nvtensor.size());
    concat_layer->setAxis(0);
    return concat_layer->getOutput(0);
}

// 对nv shape tensor进行squeeze操作, 支持dim倒序
// note: 使用前须检查 input[dim] == 1
nvinfer1::ITensor* squeeze_nv_shapetensor(TensorrtEngine* engine, 
                                    nvinfer1::ITensor* input, int dim) {
    nvinfer1::Dims input_dims = input->getDimensions();

    if (input_dims.nbDims != 1 || input->getType() != nvinfer1::DataType::kINT32) {
        LOG(INFO) << "input is not shape tensor";
        return nullptr;
    }
    // dim must be in range of [-input_dims.d[0], input_dims.d[0] - 1].
    if (dim < -input_dims.d[0] || dim > input_dims.d[0] - 1) {
        LOG(INFO) << "expected to be in range of [" << -input_dims.d[0] << "," 
        << input_dims.d[0] - 1 << "], but got " << dim;
        return nullptr;
    }
    if (dim < 0) {
        dim = input_dims.d[0] + dim;
    }
    std::vector<nvinfer1::ITensor*> inputs_nvtensor;
    //nvinfer1::ITensor* insert_tensor = tensor_to_const(engine, torch::tensor({1}, torch::kInt));
    tensor_to_const(engine, torch::tensor({1}, torch::kInt));
    // if dim == 0 or dim == input_dims.d[0] - 1, slice squeeze dimension directly.
    std::vector<int64_t> start_vec{0}, size_vec{input_dims.d[0] - 1}, stride_vec{1};
    if (dim == 0 || dim == input_dims.d[0] - 1) {
        if (dim == 0) {
            start_vec[0] = 1;
        }
        nvinfer1::ISliceLayer* slice_l = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(start_vec),
                                                sizes_to_nvdim(size_vec),
                                                sizes_to_nvdim(stride_vec));
        return slice_l->getOutput(0);

    } else {
        // divide origin tensor into two parts (skip the squeeze dim), and concat them.
        std::vector<int64_t> start_vec{0}, size_vec{dim}, stride_vec{1};
        nvinfer1::ISliceLayer* slice_front = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(start_vec),
                                                sizes_to_nvdim(size_vec),
                                                sizes_to_nvdim(stride_vec));
        inputs_nvtensor.push_back(slice_front->getOutput(0));
        start_vec[0] = dim + 1;
        size_vec[0] = input_dims.d[0] - dim - 1;
        nvinfer1::ISliceLayer* slice_back = engine->network()->addSlice(*input,
                                                sizes_to_nvdim(start_vec),
                                                sizes_to_nvdim(size_vec),
                                                sizes_to_nvdim(stride_vec));
        inputs_nvtensor.push_back(slice_back->getOutput(0));
    }
    nvinfer1::IConcatenationLayer* concat_layer = 
                    engine->network()->addConcatenation(inputs_nvtensor.data(), inputs_nvtensor.size());
    concat_layer->setAxis(0);
    return concat_layer->getOutput(0);
}

nvinfer1::ITensor* unsqueeze_itensor(TensorrtEngine* engine, 
                                    nvinfer1::ITensor* input,
                                    const std::vector<int>& axes) {
    nvinfer1::ITensor* input_shape_tensor = engine->network()->addShape(*input)->getOutput(0);
    int input_rank = input->getDimensions().nbDims;

    const std::set<int> axes_set(axes.begin(), axes.end());
    if (input_rank + axes_set.size() > nvinfer1::Dims::MAX_DIMS)
    {
        return nullptr;
    }

    // compute interlacing subscripts.
    std::vector<int64_t> subscripts(input_rank);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (const auto& axis : axes_set)
    {
        subscripts.insert(subscripts.begin() + axis, input_rank);
    }
    at::Tensor indices = torch::tensor(subscripts, torch::kInt32);
    auto indices_tensor = tensor_to_const(engine, indices);

    //calculate gather(concat(input_shape_tensor, {1}), indices_tensor)
    torch::Tensor the_one = torch::tensor(std::vector<int32_t>({1}), torch::kInt32);
    nvinfer1::ITensor* one_tensor = tensor_to_const(engine, the_one);
    nvinfer1::ITensor* const args[2] = {input_shape_tensor, one_tensor};
    nvinfer1::ITensor* tmp_concat_tensor =  engine->network()->addConcatenation(args, 2)->getOutput(0);
    nvinfer1::ITensor* new_shape_tensor =  engine->network()->addGather(*tmp_concat_tensor, *indices_tensor, 0)->getOutput(0);

    nvinfer1::IShuffleLayer* reshape_layer = engine->network()->addShuffle(*input);
    reshape_layer->setInput(1, *new_shape_tensor);
    return reshape_layer->getOutput(0);
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
