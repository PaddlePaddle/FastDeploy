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
* @file select.cpp
* @author tianjinjin@baidu.com
* @date Tue Aug 24 16:31:28 CST 2021
* @brief
**/

#include "poros/converter/gpu/select.h"
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

/*aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)*/
bool SelectConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for SelectConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SelectConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant), 
        "input[1] for SelectConverter is not come from prim::Constant as expected");
    // POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
    //     "input[2] for SelectConverter is not come from prim::Constant as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto maxDim = static_cast<int64_t>(in->getDimensions().nbDims);

    //extract dim
    auto dim = (engine->context().get_constant(inputs[1])).toInt();
    dim = dim < 0 ? dim + maxDim : dim;

    nvinfer1::ITensor* index_tensor = engine->context().get_tensor(inputs[2]);
    //extract index
    if (index_tensor == nullptr) {
        auto ind = (int32_t)((engine->context().get_constant(inputs[2])).toInt());
        // dynamic情况下 dim这一维是动态的-1，且index为倒序，需要转正
        if (in->getDimensions().d[dim] < 0 && ind < 0) {
            nvinfer1::ITensor* in_shape_tensor = engine->network()->addShape(*in)->getOutput(0);
            std::vector<int64_t> start_vec = {dim}, size_vec = {1}, stride_vec = {1}; 
            nvinfer1::ISliceLayer* slice_layer = engine->network()->addSlice(*in_shape_tensor,
                                                        sizes_to_nvdim(start_vec),
                                                        sizes_to_nvdim(size_vec),
                                                        sizes_to_nvdim(stride_vec));
            nvinfer1::ITensor* in_dim_val = slice_layer->getOutput(0);
            nvinfer1::ITensor* ind_tensor = tensor_to_const(engine, torch::tensor({ind}).to(torch::kI32));
            index_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUM,  
                                            in_dim_val,
                                            ind_tensor,
                                            layer_info(node) + std::string("_neg_index_to_pos"))->getOutput(0);

        } else {
            ind = ind < 0 ? ind + in->getDimensions().d[dim] : ind;
            // index to access needs to be an at::Tensor
            at::Tensor indices = torch::tensor({ind}).to(torch::kI32);
            index_tensor = tensor_to_const(engine, indices);
        }
    } else {
        POROS_CHECK_TRUE((in->getDimensions().d[dim] >= 0), "When index(input[2]) of aten::select is not from prim::Constant,"
        " the selected " + std::to_string(dim) + "th dim of input must be fixed (not dynamic)." << node_info(node));
    }

    // IGatherLayer takes in input tensor, the indices, and the axis
    // of input tensor to take indices from
    auto gather_layer = engine->network()->addGather(*in, *index_tensor, dim);
    POROS_CHECK(gather_layer, "Unable to create gather layer from node: " << *node);
    gather_layer->setName((layer_info(node) + "_gathier").c_str());
    auto out = gather_layer->getOutput(0);
    LOG(INFO) << "Gather tensor shape: " << out->getDimensions();

    if (out->getDimensions().nbDims != 1) {
        // IShuffleLayer removes redundant dimensions
        auto shuffle_layer = engine->network()->addShuffle(*out);
        POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
        // when input is dynamic
        if (check_nvtensor_is_dynamic(out)) {
            nvinfer1::ITensor* gather_out_shape_tensor = engine->network()->addShape(*out)->getOutput(0);
            gather_out_shape_tensor = squeeze_nv_shapetensor(engine, gather_out_shape_tensor, dim);
            shuffle_layer->setInput(1, *gather_out_shape_tensor);
        } else {
            // when input is not dynamic
            shuffle_layer->setReshapeDimensions(squeeze_dims(out->getDimensions(), dim, false));
        }
        shuffle_layer->setName(layer_info(node).c_str());
        out = shuffle_layer->getOutput(0);
    } 
    
    engine->context().set_tensor(node->outputs()[0], out);
    LOG(INFO) << "Output tensor shape: " << out->getDimensions();
    return true;
}

// aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
bool SliceConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    
    if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[1]).operator_name()) {
        // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
        POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for SliceConverter");
        
        nvinfer1::ITensor* self_nvtensor = nullptr;
        std::vector<int64_t> self_vec = {};
        int32_t dim_rank = 0;
        std::vector<nvinfer1::ITensor*> itensor_vec = {};
        bool has_tensor_scalar = false;
        // input[0] is int[]
        if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofInts())) {
            has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
            if (has_tensor_scalar) {
                self_nvtensor = this->get_tensor_scalar(inputs[0]);
                POROS_CHECK_TRUE((self_nvtensor != nullptr), node_info(node) + std::string("get int nvtensor false."));
                dim_rank = (self_nvtensor->getDimensions()).d[0];
            } else {
                self_vec = (engine->context().get_constant(inputs[0])).toIntList().vec();
                dim_rank = self_vec.size();
            }   
        // tensor[]
        } else if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofTensors())) {
            POROS_CHECK_TRUE(engine->context().get_tensorlist(inputs[0], itensor_vec), "extract tensor list error.");
            dim_rank = itensor_vec.size();
        } else {
            LOG(WARNING) << node->schema().operator_name() << " converter input[0] meets unsupported type.";
            return false;
        }
        // extract start, end and step
        torch::jit::IValue maybe_start = engine->context().get_constant(inputs[1]);
        int64_t startIdx = maybe_start.isNone() ? 0 : maybe_start.toInt();
        startIdx = (startIdx < 0) ? (dim_rank + startIdx) : startIdx;

        torch::jit::IValue maybe_end = engine->context().get_constant(inputs[2]);
        int64_t endIdx = maybe_end.isNone() ? dim_rank : maybe_end.toInt();
        endIdx = (endIdx < 0) ? (dim_rank + endIdx) : endIdx;

        int64_t step = (engine->context().get_constant(inputs[3])).toInt();

        POROS_CHECK_TRUE((startIdx <= endIdx && endIdx <= dim_rank), 
                                node_info(node) + std::string("start > end or end > self_size"));
        // input[0] is int[]                        
        if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofInts())) {
            if (has_tensor_scalar) {
                int64_t size = ceil(float(endIdx - startIdx) / float(step));
                std::vector<int64_t> start_vec{startIdx}, size_vec{size}, stride_vec{step};
                auto slice_layer = engine->network()->addSlice(*self_nvtensor,
                                                        sizes_to_nvdim(start_vec),
                                                        sizes_to_nvdim(size_vec),
                                                        sizes_to_nvdim(stride_vec));
                POROS_CHECK(slice_layer, "Unable to given dim info from node: " << *node);
                slice_layer->setName(layer_info(node).c_str());
                nvinfer1::ITensor* slice_output = slice_layer->getOutput(0);
                engine->context().set_tensor(node->outputs()[0], slice_output);
            } else {
                c10::List<int64_t> list;
                int index = startIdx;
                while (index <= endIdx - 1) {
                    list.push_back(std::move(self_vec[index]));
                    index += step;
                }
                auto output_ivalue = c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
                engine->context().set_constant(node->outputs()[0], output_ivalue);
            }
        } else if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofTensors())) {
            std::vector<nvinfer1::ITensor*> output_itensor_vec = {};
            int index = startIdx;
            while (index <= endIdx - 1) {
                output_itensor_vec.push_back(itensor_vec[index]);
                index += step;
            }
            engine->context().set_tensorlist(node->outputs()[0], output_itensor_vec);
        } else {
            LOG(WARNING) << node->schema().operator_name() << " converter input[0] meets unsupported type.";
            return false;
        }
        
        return true;
    }

    POROS_CHECK_TRUE((inputs.size() == 5), "invaid inputs size for SliceConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SliceConverter is not Tensor as expected");
    for (int32_t i = 1; i < 5; i++) {
        if (i == 2 || i == 3) {
            continue;
        }
        POROS_CHECK_TRUE((inputs[i]->node()->kind() == torch::jit::prim::Constant), 
        std::string("input[") + std::to_string(i) + std::string("] for SliceConverter is not come from prim::Constant as expected"));
    }

    nvinfer1::ITensor* in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);

    int64_t dim = (engine->context().get_constant(inputs[1])).toInt();
    nvinfer1::Dims in_dims = in->getDimensions();
    int64_t axis = c10::maybe_wrap_dim(dim, in_dims.nbDims);
    
    torch::jit::IValue maybe_start = engine->context().get_constant(inputs[2]);
    int64_t startIdx = maybe_start.isNone() ? 0 : maybe_start.toInt();
    torch::jit::IValue maybe_end = engine->context().get_constant(inputs[3]);
    int64_t endIdx = maybe_end.isNone() ? INT64_MAX : maybe_end.toInt();
    int64_t step =(engine->context().get_constant(inputs[4])).toInt();
    POROS_CHECK_TRUE((step > 0), "step for SliceConverter must be postive");

    int64_t maxDim = static_cast<int64_t>(in_dims.d[axis]);
    int64_t start = 0, end = INT64_MAX;

    // not dynamic or axis dim is not negtive
    // make sure start and end are postive
    if (maxDim >= 0) {
        //extract start
        start = (startIdx < 0) ? (maxDim + startIdx) : startIdx;
        POROS_CHECK_TRUE((start >= 0 && start <= maxDim), "invalid start for SliceConverter");
        //extract end
        endIdx = std::min(endIdx, maxDim);
        end = (endIdx < 0) ? (maxDim + endIdx) : endIdx;
        POROS_CHECK_TRUE((end >= start && end <= maxDim), "invalid end for SliceConverter or end less than start");
        POROS_CHECK_TRUE((step <= maxDim), "invalid step for SliceConverter");
    }

    std::vector<int64_t> start_vec, size_vec, stride_vec;
    bool is_dynamic = check_nvtensor_is_dynamic(in);
    bool has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
    for (int32_t i = 0; i < in_dims.nbDims; i++) {
        start_vec.push_back(0);
        size_vec.push_back(in_dims.d[i]);
        stride_vec.push_back(1);
    }
    stride_vec[axis] = step;
    start_vec[axis] = start;

    nvinfer1::ILayer* slice_layer = nullptr;

    // no dynamic and ints don't have nvtensor inputs.
    if (!is_dynamic && !has_tensor_scalar) { 
        int64_t size = ceil(float(end - start) / float(step));
        size_vec[axis] = size;
        slice_layer = engine->network()->addSlice(*in, 
                                                sizes_to_nvdim(start_vec), 
                                                sizes_to_nvdim(size_vec), 
                                                sizes_to_nvdim(stride_vec));
        slice_layer->setName(layer_info(node).c_str());
    } else { // dynamic
        nvinfer1::IShapeLayer* shape_layer = engine->network()->addShape(*in);
        nvinfer1::ITensor* in_shape_tensor = shape_layer->getOutput(0);
        nvinfer1::ITensor* start_tensor = nullptr, *size_tensor = nullptr,  *end_tensor = nullptr;
        
        std::vector<int64_t> dy_mask_vec, dy_rev_mask_vec;
        
        for (int32_t i = 0; i < in_dims.nbDims; i++) {
            dy_mask_vec.push_back(0);
            dy_rev_mask_vec.push_back(1);
        }

        // Prepare for following calculations. 
        // Such as, get dynamic input dims is [4, 5, *, 7] (runtime input dims is [4, 5, 6, 7]), and axis dim is 2.
        // Then, mask_tensor is [0, 0, 1, 0], rev_mask_tensor is [1, 1, 0, 1],
        // mask_shape_tensor is [0, 0, 6, 0], rev_mask_shape_tensor is [4, 5, 0, 7].
        at::Tensor mask_tensor = torch::tensor(dy_mask_vec, torch::kInt);
        at::Tensor rev_mask_tensor = torch::tensor(dy_rev_mask_vec, torch::kInt);

        rev_mask_tensor[axis] = 0;
        nvinfer1::ITensor* const_rev_mask_tensor = tensor_to_const(engine, rev_mask_tensor);
        nvinfer1::ITensor* rev_mask_shape_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD, 
                                            in_shape_tensor, 
                                            const_rev_mask_tensor,
                                            layer_info(node) + std::string("_axis_dim_to_zero"))->getOutput(0);
        mask_tensor[axis] = 1;
        nvinfer1::ITensor* const_mask_tensor = tensor_to_const(engine, mask_tensor);
        nvinfer1::ITensor* mask_shape_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD, 
                                            in_shape_tensor, 
                                            const_mask_tensor,
                                            layer_info(node) + std::string("_other_dims_to_zero"))->getOutput(0);
        bool has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
        if (has_tensor_scalar) {
            // Generally, only start and end come from nvtensor
            // nvinfer1::ITensor* dim_int_nvtensor = this->get_tensor_scalar(inputs[1]);
            nvinfer1::ITensor* start_int_nvtensor = this->get_tensor_scalar(inputs[2]);
            nvinfer1::ITensor* end_int_nvtensor = this->get_tensor_scalar(inputs[3]);
            // nvinfer1::ITensor* stride_int_nvtensor = this->get_tensor_scalar(inputs[4]);

            // only end from nvtensor (start is none)
            if (end_int_nvtensor != nullptr && start_int_nvtensor == nullptr) {
                LOG(INFO) << "Slice only end from nvtensor";
                mask_tensor[axis] = 0;
                start_tensor = tensor_to_const(engine, mask_tensor);
                nvinfer1::ITensor* end_tensor_temp = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD,  
                                            const_mask_tensor,
                                            end_int_nvtensor,
                                            layer_info(node) + std::string("_end_prod_mask_shape_tensor"))->getOutput(0);
                end_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUM,  
                                            end_tensor_temp,
                                            rev_mask_shape_tensor,
                                            layer_info(node) + std::string("_end_tmp_sum_rev_mask_shape_tensor"))->getOutput(0);
            // only start from nvtensor (end is none)
            } else if (end_int_nvtensor == nullptr && start_int_nvtensor != nullptr) {
                LOG(INFO) << "Slice only start from nvtensor";
                start_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD,  
                                            const_mask_tensor,
                                            start_int_nvtensor,
                                            layer_info(node) + std::string("_start_prod_mask_shape_tensor"))->getOutput(0);
                end_tensor = in_shape_tensor; 
            // start and end both from nvtensor
            } else {
                LOG(INFO) << "Slice start and end both from nvtensor";
                // make sure that start or end which not from nvtensor is postive when maxDims >= 0
                if (maxDim >= 0) {
                    if (!maybe_start.isNone()) {
                        LOG(INFO) << "Slice start can be from constant";
                        start_int_nvtensor = tensor_to_const(engine, torch::tensor({start}, torch::kInt));
                    }
                    if (!maybe_end.isNone()) {
                        LOG(INFO) << "Slice end can be from constant";
                        end_int_nvtensor = tensor_to_const(engine, torch::tensor({end}, torch::kInt));
                    }
                }
                start_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD,  
                                            const_mask_tensor,
                                            start_int_nvtensor,
                                            layer_info(node) + std::string("_start_prod_mask_shape_tensor"))->getOutput(0);
                nvinfer1::ITensor* end_tensor_temp = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kPROD,  
                                            const_mask_tensor,
                                            end_int_nvtensor,
                                            layer_info(node) + std::string("_end_prod_mask_shape_tensor"))->getOutput(0);
                end_tensor = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUM,  
                                            end_tensor_temp,
                                            rev_mask_shape_tensor,
                                            layer_info(node) + std::string("_end_tmp_sum_rev_mask_shape_tensor"))->getOutput(0);                              
            }
            nvinfer1::ITensor* sub_tensor = add_elementwise(engine, 
                                                    nvinfer1::ElementWiseOperation::kSUB, 
                                                    end_tensor, 
                                                    start_tensor,
                                                    layer_info(node) + std::string("_end_sub_start"))->getOutput(0);
            // Equivalent to ceil((end - start) / step) -> size
            if (step > 1) {
                mask_tensor[axis] = step - 1;
                nvinfer1::ITensor* sum_step_tensor = add_elementwise(engine, 
                                                        nvinfer1::ElementWiseOperation::kSUM, 
                                                        sub_tensor, 
                                                        tensor_to_const(engine, mask_tensor),
                                                        layer_info(node) + std::string("_sum_step_sub_one"))->getOutput(0);
                rev_mask_tensor[axis] = step;
                size_tensor = add_elementwise(engine, 
                                nvinfer1::ElementWiseOperation::kFLOOR_DIV, 
                                sum_step_tensor, 
                                tensor_to_const(engine, rev_mask_tensor),
                                layer_info(node) + std::string("_div_get_size"))->getOutput(0);
            } else {
                size_tensor = sub_tensor;
            }

        } else {                                           
            if (maxDim < 0) {
                // start
                mask_tensor[axis] = startIdx;
                if (startIdx < 0) {
                    start_tensor = add_elementwise(engine, 
                                        nvinfer1::ElementWiseOperation::kSUM, 
                                        mask_shape_tensor, 
                                        tensor_to_const(engine, mask_tensor),
                                        layer_info(node) + std::string("_start_tensor"))->getOutput(0);
                } else {
                    start_tensor = tensor_to_const(engine, mask_tensor);
                }
                // end
                if (maybe_end.isNone()){
                    end_tensor = in_shape_tensor;
                } else {
                    mask_tensor[axis] = endIdx;
                    if (endIdx < 0) {
                        end_tensor = add_elementwise(engine, 
                                        nvinfer1::ElementWiseOperation::kSUM, 
                                        in_shape_tensor, 
                                        tensor_to_const(engine, mask_tensor),
                                        layer_info(node) + std::string("_end_tensor_to_pos"))->getOutput(0);
                    } else {
                        end_tensor = add_elementwise(engine, 
                                        nvinfer1::ElementWiseOperation::kSUM, 
                                        rev_mask_shape_tensor, 
                                        tensor_to_const(engine, mask_tensor),
                                        layer_info(node) + std::string("_end_tensor"))->getOutput(0);
                    }
                }
                nvinfer1::ITensor* sub_tensor = add_elementwise(engine, 
                                                    nvinfer1::ElementWiseOperation::kSUB, 
                                                    end_tensor, 
                                                    start_tensor,
                                                    layer_info(node) + std::string("_end_sub_start"))->getOutput(0);
                // Equivalent to ceil((end - start) / step) -> size
                if (step > 1) {
                    mask_tensor[axis] = step - 1;
                    nvinfer1::ITensor* sum_step_tensor = add_elementwise(engine, 
                                                            nvinfer1::ElementWiseOperation::kSUM, 
                                                            sub_tensor, 
                                                            tensor_to_const(engine, mask_tensor),
                                                            layer_info(node) + std::string("_sum_step_sub_one"))->getOutput(0);
                    rev_mask_tensor[axis] = step;
                    size_tensor = add_elementwise(engine, 
                                    nvinfer1::ElementWiseOperation::kFLOOR_DIV, 
                                    sum_step_tensor, 
                                    tensor_to_const(engine, rev_mask_tensor),
                                    layer_info(node) + std::string("_div_get_size"))->getOutput(0);
                } else {
                    size_tensor = sub_tensor;
                }
            } else {
                mask_tensor[axis] = start;
                start_tensor = tensor_to_const(engine, mask_tensor);

                mask_tensor[axis] = ceil(float(end - start) / float(step));
                size_tensor = add_elementwise(engine, 
                                nvinfer1::ElementWiseOperation::kSUM, 
                                rev_mask_shape_tensor,
                                tensor_to_const(engine, mask_tensor),
                                layer_info(node) + std::string("_sum_get_size"))->getOutput(0);
            }
        }

        std::vector<int64_t> temp_vec = {0, 0};
        slice_layer = engine->network()->addSlice(*in, sizes_to_nvdim(temp_vec), 
                                                    sizes_to_nvdim(temp_vec), 
                                                    sizes_to_nvdim(stride_vec));
        slice_layer->setInput(0, *in);
        slice_layer->setInput(1, *start_tensor);
        slice_layer->setInput(2, *size_tensor);
        // slice_layer->setInput(3, *stride_tensor);
        slice_layer->setName(layer_info(node).c_str());
    }

    nvinfer1::ITensor* slice_out = slice_layer->getOutput(0);
    engine->context().set_tensor(node->outputs()[0], slice_out);
    LOG(INFO) << "Output tensor shape: " << slice_out->getDimensions();
    return true;
}

/*aten::embedding(Tensor weight, 
Tensor indices, 
int padding_idx=-1, 
bool scale_grad_by_freq=False, 
bool sparse=False) -> Tensor*/
bool EmbeddingConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5), "invaid inputs size for EmbeddingConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for EmbeddingConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[1] for EmbeddingConverter is not Tensor as expected");

    auto embedding = engine->context().get_tensor(inputs[0]);
    auto indices  = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE(((embedding != nullptr) && (indices != nullptr)), 
        "Unable to init input tensor for node: " << *node);
        
    // Set datatype for indices tensor to INT32
    auto identity = engine->network()->addIdentity(*indices);
    identity->setOutputType(0, nvinfer1::DataType::kINT32);
    identity->setName((layer_info(node) + "_identify").c_str());
    indices = identity->getOutput(0);
    
    // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
    auto gather_layer = engine->network()->addGather(*embedding, *indices, 0);
    POROS_CHECK(gather_layer, "Unable to create gather layer from node: " << *node);
    gather_layer->setName(layer_info(node).c_str());
    auto gather_out = gather_layer->getOutput(0);
    
    engine->context().set_tensor(node->outputs()[0], gather_out);
    LOG(INFO) << "Output tensor shape: " << gather_out->getDimensions();
    return true;
}

/*
aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)
*/
bool NarrowConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for NarrowConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for NarrowConverter is not Tensor as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);

    //extract dim & length
    auto maxDim = static_cast<int64_t>(in->getDimensions().nbDims);
    auto axis  = (engine->context().get_constant(inputs[1])).toInt();
    axis = (axis < 0) ? (axis + maxDim) : axis;
    auto length  = (int32_t)(engine->context().get_constant(inputs[3])).toInt();

    //extract start
    int32_t start = 0;
    auto maybe_start = engine->context().get_constant(inputs[2]);
    if (maybe_start.isInt()) {
        start = (int32_t)maybe_start.toInt();
        start = (start < 0) ? (maxDim + start) : start;
    } else if (maybe_start.isTensor()) {
        auto start_tensor = maybe_start.toTensor().to(torch::kI32);
        start = start_tensor.item().to<int32_t>();
    }

    // index to access needs to be an at::Tensor
    at::Tensor indices = torch::arange(start, start + length, 1).to(torch::kI32);
    auto weights = Weights(indices);

    // IConstantLayer to convert indices from Weights to ITensor
    auto const_layer = engine->network()->addConstant(weights.shape, weights.data);
    POROS_CHECK(const_layer, "Unable to create constant layer from node: " << *node);
    auto const_out = const_layer->getOutput(0);

    // IGatherLayer takes in input tensor, the indices, and the axis
    // of input tensor to take indices from
    auto gather_layer = engine->network()->addGather(*in, *const_out, axis);
    POROS_CHECK(gather_layer, "Unable to create gather layer from node: " << *node);
    auto gather_out = gather_layer->getOutput(0);

    // IShuffleLayer removes redundant dimensions
    auto shuffle_layer = engine->network()->addShuffle(*gather_out);
    POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
    shuffle_layer->setReshapeDimensions(unpad_nvdim(gather_out->getDimensions()));
    shuffle_layer->setName(layer_info(node).c_str());
    auto shuffle_out = shuffle_layer->getOutput(0);    
    engine->context().set_tensor(node->outputs()[0], shuffle_out);
    LOG(INFO) << "Output tensor shape: " << shuffle_out->getDimensions();
    return true;
}

/*
aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
*/
bool SplitConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3 || inputs.size() == 2), "invaid inputs size for SplitConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SplitConverter is not Tensor as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);

    int axis = 0;
    //extract dim
    if (inputs.size() == 3) {
        axis = (engine->context().get_constant(inputs[2])).toInt();
    } else {
        // node->kind() == torch::jit::aten::unbind
        // aten::unbind 和 split_size=1时的aten::split 非常像。但aten::unbind最后会做一次squeeze。
        // 例如：输入shape（2，3，4），dim=1，split_size=1，
        // 那么aten::unbind出来的就是3个（2，4），aten::split出来的就是3个（2，1，4）
        axis = (engine->context().get_constant(inputs[1])).toInt();
    }
    
    auto in_dim_size = in->getDimensions().d[axis];
    
    //extract split_size
    auto num_outputs = 1;
    auto num_remainder = 0;
    std::vector<int64_t> sizes;
    auto maybe_split_size = engine->context().get_constant(inputs[1]);
    if (node->kind() == torch::jit::aten::split_with_sizes) {
        sizes = maybe_split_size.toIntList().vec();
        num_outputs = sizes.size();
    } else { // node->kind() == torch::jit::aten::split
        auto split_size = maybe_split_size.toInt();
        // node->kind() == torch::jit::aten::unbind 时设置 split_size 为 1
        if (inputs.size() == 2) {
            split_size = 1;
        }
        num_outputs = in_dim_size / split_size;
        num_remainder = in_dim_size % split_size;
        for (int64_t i = 0; i < num_outputs; i++) {
            sizes.push_back(split_size);
        }
        if (num_remainder) {
            num_outputs += 1;
            sizes.push_back(num_remainder);
        }
    }

    LOG(INFO) << "Number of split outputs: " << num_outputs;

    std::vector<nvinfer1::ITensor*> tensorlist;
    tensorlist.reserve(num_outputs);

    int start_idx = 0;
    for (int64_t i = 0; i < num_outputs; i++) {
        at::Tensor indices = torch::arange(start_idx, start_idx + sizes[i], 1).to(torch::kI32);
        auto indices_tensor = tensor_to_const(engine, indices);

        auto gather_layer = engine->network()->addGather(*in, *indices_tensor, axis);
        auto gather_out = gather_layer->getOutput(0);
        // 为 aten::unbind axis维度做一次 squeeze
        if (inputs.size() == 2) {
            nvinfer1::IShuffleLayer* shuffle_l = engine->network()->addShuffle(*gather_out);
            std::vector<int64_t> in_shape_vec = nvdim_to_sizes(in->getDimensions());
            in_shape_vec.erase(in_shape_vec.begin() + axis);
            shuffle_l->setReshapeDimensions(sizes_to_nvdim(in_shape_vec));
            gather_out = shuffle_l->getOutput(0);
        }

        tensorlist.emplace_back(gather_out);
        start_idx = start_idx + sizes[i];
    }
    
    engine->context().set_tensorlist(node->outputs()[0], tensorlist);
    return true;
}

/*
aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
*/
bool MaskedFillConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for MaskedFillConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for MaskedFillConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[1] for MaskedFillConverter is not Tensor as expected");

    //extract self & mask
    auto self = engine->context().get_tensor(inputs[0]);
    auto mask = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE((self != nullptr && mask != nullptr), "Unable to init input tensor for node: " << *node);
    int max_rank = std::max({self->getDimensions().nbDims, mask->getDimensions().nbDims});

    bool is_dynamic = check_nvtensor_is_dynamic(self) || check_nvtensor_is_dynamic(mask);
    if (is_dynamic) {
        self = broadcast_itensor(engine, node, self, max_rank, "self");
        mask = broadcast_itensor(engine, node, mask, max_rank, "mask");
    } else {
        mask = add_padding(engine, node, mask, max_rank, false, true);
        self = add_padding(engine, node, self, max_rank, false, true);
    }

    //extract value
    nvinfer1::ITensor* val_t = engine->context().get_tensor(inputs[2]);
    //situation1: val is a scalar and is_dynamic == false
    if (val_t == nullptr && !is_dynamic) {
        auto val = (engine->context().get_constant(inputs[2])).toScalar().to<float>();
        val_t = tensor_to_const(engine, torch::full(nvdim_to_sizes(self->getDimensions()), val));
    //situation2: val is a scalar and is_dynamic == true
    } else if (val_t == nullptr && is_dynamic) {
            //change scalar to tensor and broadcast it
            auto val = (engine->context().get_constant(inputs[2])).toScalar().to<float>();
            at::Tensor val_at_tensor = torch::tensor({val});
            nvinfer1::ITensor* val_nv_tensor = tensor_to_const(engine, val_at_tensor);
            val_t = broadcast_itensor(engine, node, val_nv_tensor, max_rank, "value");
    //situation3: val is a tensor
    } else {
        int32_t value_rank = val_t->getDimensions().nbDims;
        POROS_CHECK(value_rank == 0, "masked_fill only supports a 0-dimensional value tensor");
        //let's expand value
        int32_t new_value_rank = self->getDimensions().nbDims;
        //nvinfer1::ITensor*  new_value_shape = engine->network()->addShape(*self)->getOutput(0);

        //先给value把维度补起来,补成[1, 1, 1, ...], 用shuffle实现 
        std::vector<int64_t> new_dim(new_value_rank, 1);
        auto reshape_layer = engine->network()->addShuffle(*val_t);
        reshape_layer->setReshapeDimensions(sizes_to_nvdim(c10::IntArrayRef(new_dim)));
        reshape_layer->setName((layer_info(node) + "_IShuffleLayer_for_value").c_str());
        val_t = reshape_layer->getOutput(0);

        //无需专门调用slice把维度对齐，addSelect接口要求rank对齐就行，rank对齐的情况下，接口内部自己会broadcast。
        /*
        //再slice一下, 因为是从rank 0 expand到其他的dim，
        //所以此处start_dim 设置为全0，stride_dim 也设置为全0，
        //sizes信息先用start_dim 作为dummy input, 后面用setInput 接口设置真是的output_dim 信息。
        std::vector<int64_t> start_vec_new(new_value_rank, 0);
        auto offset = sizes_to_nvdim(c10::IntArrayRef(start_vec_new));
        
        // Slice layer does the expansion in TRT. Desired output size is specified by new_value_shape
        auto slice_layer = engine->network()->addSlice(*val_t, offset, offset, offset);
        slice_layer->setInput(2, *new_value_shape);
        slice_layer->setName((layer_info(node) + "_ISliceLayer_for_value").c_str());
        val_t = slice_layer->getOutput(0);
        */
    }

    //no need anymore
    // POROS_CHECK(broadcastable(self->getDimensions(), mask->getDimensions(), /*multidirectional=*/false),
    //     "Self and mask tensors are not broadcastable");

    nvinfer1::ISelectLayer* new_layer = engine->network()->addSelect(*mask, *val_t, *self);
    POROS_CHECK(new_layer, "Unable to create layer for aten::masked_fill");

    new_layer->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}

// aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
bool GatherConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for GatherConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for GatherConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant), 
        "input[1] for GatherConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[2]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[2] for GatherConverter is not Tensor as expected");
    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    auto maxDim = static_cast<int64_t>(self->getDimensions().nbDims);
    // extract index
    nvinfer1::ITensor* index  = engine->context().get_tensor(inputs[2]);
    POROS_CHECK_TRUE(((self != nullptr) && (index != nullptr)), 
        "Unable to init input tensor for node: " << *node);
    //extract dim
    int64_t dim = engine->context().get_constant(inputs[1]).toInt();
    // make sure dim >= 0
    dim = dim < 0 ? dim + maxDim : dim;
    
    // Set datatype for indices tensor to INT32
    nvinfer1::IIdentityLayer* identity = engine->network()->addIdentity(*index);
    identity->setOutputType(0, nvinfer1::DataType::kINT32);
    identity->setName((layer_info(node) + "_identify").c_str());
    index = identity->getOutput(0);
    
    // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
    nvinfer1::IGatherLayer* gather_layer = engine->network()->addGather(*self, *index, dim);
    POROS_CHECK(gather_layer, "Unable to create gather layer from node: " << *node);
    gather_layer->setName(layer_info(node).c_str());
    gather_layer->setMode(nvinfer1::GatherMode::kELEMENT);
    nvinfer1::ITensor* gather_out = gather_layer->getOutput(0);
    
    engine->context().set_tensor(node->outputs()[0], gather_out);
    LOG(INFO) << "Output tensor shape: " << gather_out->getDimensions();
    return true;
}

/*
aten::index含义：用indices指定下标，选取self指定维度。
（实际上是用tensor将多个indices分dims打包起来，能够一起选取）
例如：输入x，其shape = {3, 4, 5}
输入两个indices tensors，
indices_1 = [0, 2]
indices_2 = [1, 3]
这组输入表示用indices_1选取x dim=0 的 0和2下标，用indices_2选取x dim=1 的 1和3下标
则结果为
output = [x[0][1], x[2][3]]
由于剩余x dim=2 的维度是5，则
output.shape = {1, 2, 5}
这样就实现了同时选取 x[0][1] 和 x[2][3] 的功能了。
规则：
1、输入的indices数量不能超过self rank数。（也就是说本例子中输入的indices tensor数量不能大于3）
2、输入的indices中的值不能超过自己对应维度范围。（例如：indices_1对应x的dim=0，则其最大值必须小于3；indices_2对应x的dim=1，则其最大值必须小于4。）
3、输入的indices shape必须一致或可以broadcast。（为的是相应位置能够同时选取。）
---------------------------------
如果继续上面再输入一个indices tensor
indices_3 = [2, 4]
则结果为
output = [x[0][1][2], x[2][3][4]]
output.shape = {1, 2}
*/
// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor 
bool IndexConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for IndexConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for IndexConverter is not Tensor as expected");
    // torch对于Tensor?[]类型的解释：
    // the index of aten::index should be a type of List[Optional[Tensor]],
    // this is to support the case like t[:, :, 1] where : here indicates a
    // None/undefined tensor(optional tensor)
    POROS_CHECK_TRUE((inputs[1]->type()->str().find("Tensor?[]") != std::string::npos), 
        "input[1] for IndexConverter is not List[Optional[Tensor]] (Tensor?[]) as expected");
    
    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    // extract indices
    std::vector<nvinfer1::ITensor*> indices_tensors;
    engine->context().get_tensorlist(inputs[1], indices_tensors);

    // ps: 目前能支持在self的0 dim上选取，也就是说indices_tensors只能输入一个。
    // 而下面注释掉这段代码实现了更全面的功能，支持多个indices_tensors（见下方介绍），但是实测中会使模型速度更慢（由于使用了更多的gather），先注释掉。解掉注释不支持dynamic
    POROS_CHECK_TRUE((indices_tensors.size() == 1), 
        "aten::Index of torchscript implements the selection of multiple dimensions with several indices. "
        "But due to the functional limitations of trt gatherlayer, in this version of poros, "
        "aten::Index only support one indice input, which means only 0 dim of self can be indexed.");

    /*
    // 设self.rank = r，indices_tensors.size() = q （根据规则1 q <= r），则该组输入会选取self的前q维。
    // 现由于trt gatherlayer功能限制，现只能实现self前q-1维的每一维只能选取一个值。
    // 例如：上面例子 output = [x[0][1][2], x[2][3][4]] 是不能支持的（因为选取的第1维同时有0和2，第2维同时有1和3），
    // 而output = [x[0][1][2], x[0][1][4]]是能够支持的。（因为选取的第1维只有0，第2维只有1）
    // 换句话说，第1至q-1的indices_tensors中的每个值都必须相等。
    // 为便于判断，先设定只有前q-1的indices_tensors所有维度都是1才能支持（因为这样broadcast过去能保证indices_tensor中的每个值都相等）
    for (size_t i = 0; i < indices_tensors.size() - 1; i++) {
        std::vector<int64_t> input_index_shape_vec = nvdim_to_sizes(indices_tensors[i]->getDimensions());
        size_t shape_prod = 1;
        for (size_t j = 0; j < input_index_shape_vec.size(); j++) {
            shape_prod *= input_index_shape_vec[j];
        }
        if (shape_prod > 1) {
            LOG(WARNING) << "Torchscript could have implemented aten::Index with several indices. But due to the functional limitations of trt gatherlayer, "
            "in this version of poros, aten::Index only support that every dimension of indices is equal to 1 except the last one.";
            return false;
        }
    }
    // 前q - 1维选取
    for (size_t i = 0; i < indices_tensors.size() - 1; i++) {
        // Set datatype for indices tensor to INT32
        nvinfer1::IIdentityLayer* identity_layer = engine->network()->addIdentity(*indices_tensors[i]);
        identity_layer->setOutputType(0, nvinfer1::DataType::kINT32);
        identity_layer->setName((layer_info(node) + "_identify" + std::to_string(i)).c_str());
        indices_tensors[i] = identity_layer->getOutput(0);

        // 由于前q-1的indices_tensors所有维度都是1，可以将indices reshape到1维
        nvinfer1::IShuffleLayer* shuffle_layer = engine->network()->addShuffle(*indices_tensors[i]);
        POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
        shuffle_layer->setName((layer_info(node) + "_shuffle" + std::to_string(i)).c_str());
        std::vector<int64_t> one_vec = {1};
        shuffle_layer->setReshapeDimensions(sizes_to_nvdim(one_vec));
        indices_tensors[i] = shuffle_layer->getOutput(0);

        // 用1维的indices 去gather self的第0维
        nvinfer1::IGatherLayer* gather_layer = engine->network()->addGather(*self, *indices_tensors[i], 0);
        POROS_CHECK(gather_layer, "Unable to create gather layer from node: " << *node);
        gather_layer->setName((layer_info(node) + "_gather" + std::to_string(i)).c_str());
        self = gather_layer->getOutput(0);

        // 由于gather出的结果第0维是1，可以将gather出的第0维抹掉
        auto self_shape_vec = nvdim_to_sizes(self->getDimensions());
        self_shape_vec.erase(self_shape_vec.begin());
        nvinfer1::IShuffleLayer* shuffle_layer2 = engine->network()->addShuffle(*self);
        POROS_CHECK(shuffle_layer2, "Unable to create shuffle layer from node: " << *node);
        shuffle_layer->setName((layer_info(node) + "_shuffle2_" + std::to_string(i)).c_str());
        shuffle_layer2->setReshapeDimensions(sizes_to_nvdim(self_shape_vec));
        self = shuffle_layer2->getOutput(0);
    }*/

    // 最后一维选取，支持indices中包含多个不同值
    nvinfer1::ITensor* final_index = *(--indices_tensors.end());
    // Set datatype for indices tensor to INT32
    nvinfer1::IIdentityLayer* identity_layer = engine->network()->addIdentity(*final_index);
    identity_layer->setOutputType(0, nvinfer1::DataType::kINT32);
    identity_layer->setName((layer_info(node) + "_identify").c_str());
    final_index = identity_layer->getOutput(0);
    
    // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
    nvinfer1::IGatherLayer* gather_layer = engine->network()->addGather(*self, *final_index, 0);
    POROS_CHECK(gather_layer, "Unable to create gather layer from node: " << *node);
    gather_layer->setName((layer_info(node) + "_gather").c_str());
    nvinfer1::ITensor* gather_out = gather_layer->getOutput(0);
    
    engine->context().set_tensor(node->outputs()[0], gather_out);
    LOG(INFO) << "Output tensor shape: " << gather_out->getDimensions();
    return true;
}

//aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
//TODO: when meet accumulate == True  situation. not support yet.
//TODO: when indices element type is Bool, not support yet.
bool IndexPutConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for IndexPutConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for IndexPutConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->str().find("Tensor?[]") != std::string::npos), 
        "input[1] for IndexPutConverter is not List[Optional[Tensor]] (Tensor?[]) as expected");
    
    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    // extract indices
    std::vector<nvinfer1::ITensor*> indices_tensors;
    engine->context().get_tensorlist(inputs[1], indices_tensors);
    //extract values
    nvinfer1::ITensor* values = engine->context().get_tensor(inputs[2]);
    //extract accumulate
    bool accumulate = (engine->context().get_constant(inputs[3])).toBool();

    //situation 1/3: ---------- when indices_tensors.size() == 0  -------------   
    if (indices_tensors.size() == 0) {
        engine->context().set_tensor(node->outputs()[0], values);
        LOG(WARNING) << "meet the situation when indices_tensors(the second input value) for index_put is empty.";
        LOG(INFO) << "Output tensor shape: " << values->getDimensions();
        return true;
    }

    if (accumulate == true) {
        LOG(WARNING) << "accumulate equal true situation is not supported yet";
        return false;
    }

    LOG(INFO) << "handle node info: " << node_info(node)
            << ", self tensor shape: " << self->getDimensions()
            << ", value tensor shape: " << values->getDimensions()
            << ", indices_tensors.size(): " << indices_tensors.size();

    auto is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;
    nvinfer1::ITensor* index_tensor = nullptr;
    nvinfer1::ITensor* broadcast_index_shape = nullptr;
    //situation 2/3: ---------- when indices_tensors.size() > 1  -------------   
    if (indices_tensors.size() > 1) {
        //TODO: check the indices type, if scalartype is bool. we should add NonZero handler

        nvinfer1::ITensor* broadcast_index = indices_tensors[0];
        //add the element in tensor_list to get broadcast index_tensor
        for (size_t index = 1; index < indices_tensors.size(); index++) {
            auto add = add_elementwise(engine, nvinfer1::ElementWiseOperation::kSUM, 
                                        broadcast_index, 
                                        indices_tensors[index],
                                        layer_info(node) + "select_add_" + std::to_string(index));
            broadcast_index = add->getOutput(0);
        }
        //get broadcast_index shape.
        LOG(INFO) << "broadcast_index dim is : " << broadcast_index->getDimensions();
        broadcast_index_shape = engine->network()->addShape(*broadcast_index)->getOutput(0);  //shape tensor
        auto target_dims = broadcast_index->getDimensions();
        auto output_rank = target_dims.nbDims;

        std::vector<nvinfer1::ITensor*> new_indices_tensors;

        nvinfer1::ITensor* new_input_shape_tensor = nullptr;
        nvinfer1::ITensor* in = nullptr;  //current handle indice tensor
        for (size_t index = 0; index < indices_tensors.size(); index++) {
            //step 2.0: expand the indices
            in = indices_tensors[index];
            auto input_dims = in->getDimensions();
            auto input_rank = in->getDimensions().nbDims;
            LOG(INFO) << "try to expand tensor shape: " << in->getDimensions()
                    << " to new shape: " << broadcast_index->getDimensions()
                    << ", input rank: " << input_rank  << ", output rank: " << output_rank;
            //situation1: ---------- when input is dynamic shape -------------
            if (is_dynamic_shape == true) {
                size_t max_rank = std::max(input_rank, output_rank);
                // Dimensions are right alignment. Eg: an input of [3, 1] and max_rank = 4, the result of concat is [1, 1, 3, 1]
                if (max_rank - input_rank > 0) { //need shuffle
                    torch::Tensor the_one = torch::tensor(std::vector<int32_t>(max_rank - input_rank, 1), torch::kInt32);
                    auto one_tensor = tensor_to_const(engine, the_one);
                    auto in_shape_tensor = engine->network()->addShape(*in)->getOutput(0);
                    nvinfer1::ITensor* const args[2] = {one_tensor, in_shape_tensor};
                    new_input_shape_tensor =  engine->network()->addConcatenation(args, 2)->getOutput(0);
                } else { //max_rank - input_rank == 0
                    new_input_shape_tensor =  engine->network()->addShape(*in)->getOutput(0);
                }
                auto shuffle = engine->network()->addShuffle(*in);
                shuffle->setInput(1, *new_input_shape_tensor);
                //LOG(INFO) << "input shuffle to shape: " << shuffle->getOutput(0)->getDimensions();

                // Start the slicing from beginning of tensor since this is an expand layer
                std::vector<int64_t> start_vec(max_rank, 0);
                nvinfer1::Dims starts_dim = sizes_to_nvdim(c10::IntArrayRef(start_vec));
                at::Tensor th_start = torch::tensor(nvdim_to_sizes(starts_dim), torch::kInt32);
                auto starts = tensor_to_const(engine, th_start);
        
                // compute sizes = max(x,y).
                auto sizes = engine->network()->addElementWise(*new_input_shape_tensor, 
                                            *broadcast_index_shape, 
                                            nvinfer1::ElementWiseOperation::kMAX)->getOutput(0);
                nvinfer1::Dims sizes_dim{-1, {}};
                sizes_dim.nbDims = max_rank;
        
                // Compute (x > 1 ? 1 : 0) for x in newDims, assuming positive x, using only TensorRT operations.
                // min(1, sub(input_shape, 1))
                torch::Tensor thOne = torch::tensor({1}, torch::kInt32);
                auto thone_tensor = tensor_to_const(engine, thOne);
                auto x_sub_one = engine->network()->addElementWise(*new_input_shape_tensor,
                                                *thone_tensor,
                                                nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
                auto strides = engine->network()->addElementWise(*thone_tensor,
                                                *x_sub_one,
                                                nvinfer1::ElementWiseOperation::kMIN)->getOutput(0);
                nvinfer1::Dims strides_dim{-1, {}};
                strides_dim.nbDims = max_rank;
        
                // Slice layer does the expansion in TRT. Desired output size is specified by sizes input at index 2.
                auto slice = engine->network()->addSlice(*shuffle->getOutput(0), starts_dim, sizes_dim, strides_dim);
                slice->setInput(1, *starts);
                slice->setInput(2, *sizes);
                slice->setInput(3, *strides);
                auto new_indice = slice->getOutput(0);
                //LOG(INFO) << "new indice tensor shape: " << new_indice->getDimensions();

                //unsqueeze it.
                auto dim = nvdim_to_sizes(new_indice->getDimensions()).size();  //this is ok
                auto shuffle_layer = engine->network()->addShuffle(*new_indice);
                nvinfer1::ITensor* input_shape_tensor = (engine->network()->addShape(*new_indice))->getOutput(0);
                nvinfer1::ITensor* reshape_tensor = unsqueeze_nv_shapetensor(engine, input_shape_tensor, dim);
                shuffle_layer->setInput(1, *reshape_tensor);
                //LOG(INFO) << "unsqueeze new indice tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();

                new_indices_tensors.push_back(shuffle_layer->getOutput(0));

            //situation2: ---------- when input is NOT dynamic shape -------------   
            }  else {  // is_dynamic_shape == false
                // Validate the expansion. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]
                for (int64_t i = target_dims.nbDims - 1; i >= 0; --i) {
                    int64_t offset = target_dims.nbDims - 1 - i;
                    int64_t dim = input_dims.nbDims - 1 - offset;
                    int64_t targetSize = target_dims.d[i];
                    // In expand layer passing -1 as the size for a dimension means not changing the size of that dimension.
                    if (targetSize == -1) {
                        // in(3, 1), expand(3, -1, 4) -> expand(3, 3, 4)
                        target_dims.d[i] = input_dims.d[dim];
                    }
                }

                auto num_expand_dims = target_dims.nbDims - input_dims.nbDims;
                if (num_expand_dims > 0) {
                    nvinfer1::Dims reshape_dims;
                    reshape_dims.nbDims = target_dims.nbDims;
                    for (int64_t i = 0; i < num_expand_dims; i++) {
                        reshape_dims.d[i] = 1;
                    }
                    for (int64_t i = 0; i < input_dims.nbDims; i++) {
                        reshape_dims.d[num_expand_dims + i] = input_dims.d[i];
                    }
            
                    // Add a reshape layer to expand dims
                    auto reshape_layer = engine->network()->addShuffle(*in);
                    reshape_layer->setReshapeDimensions(reshape_dims);
                    in = reshape_layer->getOutput(0);
                    //LOG(INFO) << "Input reshaped to : " << in->getDimensions() << " from " << input_dims;
                }
        
                // Start the slicing from beginning of tensor since this is an expand layer
                std::vector<int64_t> start_vec(target_dims.nbDims, 0);
                auto start_offset = sizes_to_nvdim(c10::IntArrayRef(start_vec));
        
                // Set the stride of non singleton dimension to 1
                std::vector<int64_t> strides_vec(target_dims.nbDims, 0);
                for (int64_t i = 0; i < target_dims.nbDims; i++) {
                    strides_vec[i] = (in->getDimensions().d[i] != 1);
                }
        
                auto strides = sizes_to_nvdim(c10::IntArrayRef(strides_vec));
                // Slice layer does the expansion in TRT. Desired output size is specified by target_dims
                auto slice_layer = engine->network()->addSlice(*in, start_offset, target_dims, strides);
                auto new_indice = slice_layer->getOutput(0);
                //LOG(INFO) << "new indice tensor shape: " << new_indice->getDimensions();

                //unsqueeze it.
                auto dim = nvdim_to_sizes(new_indice->getDimensions()).size();  //this is ok
                auto shuffle_layer = engine->network()->addShuffle(*new_indice);
                shuffle_layer->setReshapeDimensions(unsqueeze_dims(new_indice->getDimensions(), dim));
                //LOG(INFO) << "unsqueeze new indice tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();

                new_indices_tensors.push_back(shuffle_layer->getOutput(0));
            }
        }

        auto dim = new_indices_tensors[0]->getDimensions().nbDims - 1;
        auto cat_layer = engine->network()->addConcatenation(new_indices_tensors.data(), new_indices_tensors.size());
        cat_layer->setAxis(static_cast<int>(dim));
        cat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_indices").c_str());
        index_tensor = cat_layer->getOutput(0);
    
    //situation 3/3: ---------- when indices_tensors.size() ==  1  -------------   
    } else {  
        auto indices_tensor = indices_tensors[0];
        broadcast_index_shape =  engine->network()->addShape(*indices_tensor)->getOutput(0);
        auto dim = nvdim_to_sizes(indices_tensor->getDimensions()).size();  //this is ok
        auto shuffle_layer = engine->network()->addShuffle(*indices_tensor);
        shuffle_layer->setReshapeDimensions(unsqueeze_dims(indices_tensor->getDimensions(), dim));
        LOG(INFO) << "unsqueeze indice tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();
        index_tensor = shuffle_layer->getOutput(0);
    }
   
    /********************************************************************
     *               values handle begin 
     * ******************************************************************/
    auto value_rank = values->getDimensions().nbDims;
    //1 get self shape  self_shape is a 1D tensor
    nvinfer1::ITensor*  self_shape = engine->network()->addShape(*self)->getOutput(0);
    nvinfer1::Dims self_shape_dim = self_shape->getDimensions();

    //2 sub_data_shape = slice(self_shape, axes=[0], starts=[indices_tensors.size()], ends=[INT64_MAX])
    int64_t start = indices_tensors.size();
    int64_t end = static_cast<int64_t>(self_shape_dim.d[0]);
    int64_t size = ceil(float(end - start) / float(1));

    std::vector<int64_t> start_vec = {start};
    std::vector<int64_t> size_vec = {size};
    std::vector<int64_t> stride_vec = {1};

    nvinfer1::ITensor* sub_data_shape = engine->network()->addSlice(*self_shape, 
                                                sizes_to_nvdim(start_vec), 
                                                sizes_to_nvdim(size_vec), 
                                                sizes_to_nvdim(stride_vec))->getOutput(0);

    //3 values_shape = g.op("Concat", broadcast_index_shape, sub_data_shape, axis_i=0)
    std::vector<nvinfer1::ITensor*> to_concat_tensors = {broadcast_index_shape, sub_data_shape};
    auto shape_cat_layer = engine->network()->addConcatenation(to_concat_tensors.data(), to_concat_tensors.size());
    shape_cat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_values").c_str());
    auto values_shape = shape_cat_layer->getOutput(0);

    //4. we should expand values when it is a singular value 
    //values = g.op("Expand", values, values_shape)
    if (value_rank == 0) {
        LOG(INFO) << "given value is rank == 0, expand it now";
        auto new_value_rank = values_shape->getDimensions().d[0];

        //先给value把维度补起来,补成[1, 1, 1, ...], 用shuffle实现 
        std::vector<int64_t> new_dim(new_value_rank, 1);
        auto reshape_layer = engine->network()->addShuffle(*values);
        reshape_layer->setReshapeDimensions(sizes_to_nvdim(c10::IntArrayRef(new_dim)));
        reshape_layer->setName((layer_info(node) + "_IShuffleLayer_for_rank0_values").c_str());
        values = reshape_layer->getOutput(0);

        //再slice一下, 因为是从rank 0 expand到其他的dim，
        //所以此处start_dim 设置为全0，stride_dim 也设置为全0，
        //sizes信息先用start_dim 作为dummy input, 后面用setInput 接口设置真是的output_dim 信息。
        std::vector<int64_t> start_vec_new(new_value_rank, 0);
        auto offset = sizes_to_nvdim(c10::IntArrayRef(start_vec_new));
        
        // Slice layer does the expansion in TRT. Desired output size is specified by values_shape
        auto slice_layer = engine->network()->addSlice(*values, offset, offset, offset);
        slice_layer->setInput(2, *values_shape);
        slice_layer->setName((layer_info(node) + "_ISliceLayer_for_rank0_values").c_str());
        values = slice_layer->getOutput(0);
    }

    auto reshape_layer_final = engine->network()->addShuffle(*values);
    reshape_layer_final->setInput(1, *values_shape);
    reshape_layer_final->setName((layer_info(node) + "_IShuffleLayer_for_values").c_str());
    values = reshape_layer_final->getOutput(0);
    LOG(INFO) << "new_values tensor shape: " << values->getDimensions();
    /********************************************************************
     *               values handle ends
     * ******************************************************************/

    nvinfer1::IScatterLayer* scatter_layer = engine->network()->addScatter(*self, *index_tensor, *values, nvinfer1::ScatterMode::kND);
    //scatter_layer->setAxis(0);  // no need
    scatter_layer->setName((layer_info(node) + "_scatterND").c_str());
    nvinfer1::ITensor* output = scatter_layer->getOutput(0);
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
// For a 3-D tensor, self is updated as:
// self[index[i][j][k]][j][k] = value  # if dim == 0
// self[i][index[i][j][k]][k] = value  # if dim == 1
// self[i][j][index[i][j][k]] = value  # if dim == 2
// ps: self和index的shape不一定一样，所以只遍历index的所有下标。index中不存在的下标self不更新还用原来的值。
bool ScatterConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
        at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for ScatterConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ScatterConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[2] for ScatterConverter is not Tensor as expected");

    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    // extract dim
    int64_t dim = engine->context().get_constant(inputs[1]).toInt();
    auto maxDim = static_cast<int64_t>(self->getDimensions().nbDims);
    dim = dim < 0 ? dim + maxDim : dim;
    // extract indices
    nvinfer1::ITensor* index_tensor = engine->context().get_tensor(inputs[2]);
    // extract scalar
    auto ivalue_scalar = engine->context().get_constant(inputs[3]);
    float scalar = ivalue_scalar.toScalar().to<float>();

    nvinfer1::DataType self_data_type = self->getType();

    // IScatterLayer要求输入self必须是float类型
    if (self_data_type != nvinfer1::DataType::kFLOAT) {
        auto identity_layer = engine->network()->addIdentity(*self);
        identity_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        identity_layer->setName((layer_info(node) + "_self_identify_float").c_str());
        self = identity_layer->getOutput(0);
    }

    // 当value和self类型不一致时，向self对齐。这里手动做一次类型转换对齐精度。
    if (ivalue_scalar.isDouble() && self_data_type == nvinfer1::DataType::kINT32) {
        scalar = (float)(int)scalar;
    }

    nvinfer1::ITensor* updates_tensor = nullptr;
    bool is_dynamic = check_nvtensor_is_dynamic(index_tensor);
    
    // 输入nvinfer1::IScatterLayer的index和updates的shape必须相同
    if (!is_dynamic) {
        std::vector<int64_t> index_dims_vec = nvdim_to_sizes(index_tensor->getDimensions());
        updates_tensor = tensor_to_const(engine, at::full(index_dims_vec, scalar, torch::kFloat32));
    } else {
        nvinfer1::ITensor* index_shape_tensor = engine->network()->addShape(*index_tensor)->getOutput(0);
        auto fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, nvinfer1::FillOperation::kLINSPACE);
        fill_layer->setInput(0, *index_shape_tensor);
        at::Tensor alpha_tensor = torch::tensor(scalar, torch::kFloat32);
        fill_layer->setInput(1, *tensor_to_const(engine, alpha_tensor)); // 初始值
        at::Tensor delta_tensor = torch::zeros(index_tensor->getDimensions().nbDims, torch::kFloat32);
        fill_layer->setInput(2, *tensor_to_const(engine, delta_tensor)); // delta值
        fill_layer->setName((layer_info(node) + "_fill_index_shape_value").c_str());
        updates_tensor = fill_layer->getOutput(0);
    }

    // self tensor data type must be DataType::kFLOAT.
    // index tensor data type must be DataType::kINT32.
    // updates tensor data type must be DataType::kFLOAT.
    nvinfer1::IScatterLayer* scatter_layer = engine->network()->addScatter(*self, *index_tensor, *updates_tensor, nvinfer1::ScatterMode::kELEMENT);
    scatter_layer->setAxis(dim);
    scatter_layer->setName((layer_info(node) + "_scatter").c_str());

    nvinfer1::ITensor* output = scatter_layer->getOutput(0);
    // 输出不是原来的type需要还原回去，一般是int
    if (output->getType() != self_data_type) {
        auto identity_layer = engine->network()->addIdentity(*output);
        identity_layer->setOutputType(0, self_data_type);
        identity_layer->setName((layer_info(node) + "_output_identify_original_type").c_str());
        output = identity_layer->getOutput(0);
    }

    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

// prim::ConstantChunk
bool ChunkConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for ChunkConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ChunkConverter is not Tensor as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);

    // In IR, the prim::ConstantChunk always appears in the form of "prim::ConstantChunk[chunks=xx, dim=xx]()".
    // And the way to extract its parameters is a little different.
    int32_t raw_dim = (int32_t)node->i(torch::jit::attr::dim);
    int32_t chunks = (int32_t)node->i(torch::jit::attr::chunks);

    int32_t in_rank = in->getDimensions().nbDims;
    // When dim < 0
    raw_dim = raw_dim < 0 ? in_rank + raw_dim : raw_dim;
    int32_t in_dim_size = in->getDimensions().d[raw_dim];

    int32_t every_chunk_size = (int32_t)ceil((double)in_dim_size / (double)chunks);
    int32_t remainder_size = in_dim_size % every_chunk_size;
    int32_t chunk_num = (int32_t)ceil((double)in_dim_size / (double)every_chunk_size);
    
    // Check whether the calculated chunk_num is equal to the output_num of the node.
    POROS_CHECK_TRUE((chunk_num == (int32_t)(node->outputs().size())), "The caulated chunk_num (" + std::to_string(chunk_num) + 
    ") is not equal to the node outputs size (" + std::to_string(node->outputs().size()) + ").");
    
    std::vector<int32_t> chunk_sizes_vec;
    for (int i = 0; i < chunk_num - 1; i++) {
        chunk_sizes_vec.push_back(every_chunk_size);
    }
    if (remainder_size != 0) {
        chunk_sizes_vec.push_back(remainder_size);
    } else {
        chunk_sizes_vec.push_back(every_chunk_size);
    }

    std::vector<nvinfer1::ITensor*> tensorlist;
    tensorlist.reserve(chunk_sizes_vec.size());

    int start_idx = 0;
    for (size_t i = 0; i < chunk_sizes_vec.size(); i++) {
        at::Tensor indices = torch::arange(start_idx, start_idx + chunk_sizes_vec[i], 1).to(torch::kI32);
        auto indices_tensor = tensor_to_const(engine, indices);

        auto gather_layer = engine->network()->addGather(*in, *indices_tensor, raw_dim);
        auto gather_out = gather_layer->getOutput(0);

        tensorlist.emplace_back(gather_out);
        start_idx = start_idx + chunk_sizes_vec[i];
    }
    for (size_t i = 0; i < chunk_sizes_vec.size(); i++) {
        engine->context().set_tensor(node->outputs()[i], tensorlist[i]);
        LOG(INFO) << "Output tensor shape: " << tensorlist[i]->getDimensions();
    }

    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, SelectConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, SliceConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, EmbeddingConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, NarrowConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, SplitConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, MaskedFillConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, GatherConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, IndexConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, IndexPutConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ScatterConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ChunkConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
