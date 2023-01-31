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
* @file lstm_cell.cpp
* @author wangrui39@baidu.com
* @date Mon December 13 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/lstm_cell.h"
#include "poros/converter/gpu/weight.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"
#include "poros/converter/gpu/add.h"

namespace baidu {
namespace mirana {
namespace poros {

nvinfer1::Dims todims_pad(nvinfer1::Dims s_dim, int32_t pad_to) {
    if (s_dim.nbDims > pad_to){
        LOG(WARNING) << "Requested padding of dimensions to " << pad_to << " but found " << 
            s_dim.nbDims << " dimensions, not going to pad";
        return s_dim;
    }

    nvinfer1::Dims dims;
    dims.nbDims = pad_to;
    for (int32_t i = 0; i < pad_to - s_dim.nbDims; ++i) {
        dims.d[i] = 1;
    }
    for (int32_t i = pad_to - s_dim.nbDims; i < pad_to; ++i) {
        dims.d[i] = s_dim.d[i - (pad_to - s_dim.nbDims)];
    }
    return dims;
}

nvinfer1::ITensor* calculate_gate(
    TensorrtEngine* engine, nvinfer1::ITensor *input, nvinfer1::ITensor *w, 
    std::string b_name = "", nvinfer1::ITensor *b = nullptr) {
    
    auto mm = engine->network()->addMatrixMultiply(
        *input, nvinfer1::MatrixOperation::kNONE, *w, nvinfer1::MatrixOperation::kTRANSPOSE);
    nvinfer1::ITensor *mm_out = mm->getOutput(0);
    
    if (b != nullptr) {
        auto mout_dim = mm_out->getDimensions();
        auto b_dim = b->getDimensions();
        
        if (mout_dim.d != b_dim.d) {
            auto shuffle = engine->network()->addShuffle(*b);
            shuffle->setReshapeDimensions(todims_pad(b_dim, mout_dim.nbDims));
            b = shuffle->getOutput(0);
        }

        auto add_layer = engine->network()->addElementWise(*mm_out, *b, nvinfer1::ElementWiseOperation::kSUM);
        return add_layer->getOutput(0);
    }
    else{
        return mm_out;
    }
    
}

bool LstmCellConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "inputs[0] for LstmCellConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::ListType::ofTensors())), 
        "inputs[1] for LstmCellConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->type()->isSubtypeOf(c10::TensorType::get())), 
        "inputs[2] for LstmCellConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[3]->type()->isSubtypeOf(c10::TensorType::get())), 
        "inputs[3] for LstmCellConverter is not Tensor as expected");
    
    //extract Tensors[]
    std::vector<nvinfer1::ITensor*> state;
    bool ret = engine->context().get_tensorlist(inputs[1], state);
    POROS_CHECK_TRUE((state.size() == 2), "Unable to init input List[tensor] for node: " << *node);
    POROS_CHECK_TRUE(ret, "Unable to init input List[tensor] for node: " << *node);
    
    //extract Tensor
    nvinfer1::ITensor *input = engine->context().get_tensor(inputs[0]);
    nvinfer1::ITensor *w_ih = engine->context().get_tensor(inputs[2]);
    nvinfer1::ITensor *w_hh = engine->context().get_tensor(inputs[3]);

    // calculate first half of gates
    nvinfer1::ITensor *out1 = nullptr;
    nvinfer1::ITensor *out2 = nullptr;
    
    if (inputs[4]->type()->isSubtypeOf(c10::TensorType::get())) {
        nvinfer1::ITensor *b_ih = engine->context().get_tensor(inputs[4]);
        out1 = calculate_gate(engine, input, w_ih, "b_ih", b_ih);
    }
    else {
        out1 = calculate_gate(engine, input, w_ih);
    }
    POROS_CHECK_TRUE((out1 != nullptr), "invaid b_ih size for ConcatConverter");

    // calculate second half of gates
    if (inputs[5]->type()->isSubtypeOf(c10::TensorType::get())) {
        nvinfer1::ITensor *b_hh = engine->context().get_tensor(inputs[5]);
        out2 = calculate_gate(engine, state[0], w_hh, "b_hh", b_hh);
    }
    else {
        out2 = calculate_gate(engine, state[0], w_hh);
    }
    POROS_CHECK_TRUE((out2 != nullptr), "invaid b_hh size for ConcatConverter");

    // get all 4 gates
    auto add_layer = engine->network()->addElementWise(*out1, *out2, nvinfer1::ElementWiseOperation::kSUM);
    add_layer->setName((layer_info(node) + "_sum_" + "for_add_out").c_str());
    nvinfer1::ITensor *add_out = add_layer->getOutput(0);

    // chunk Tensor into 4 parts and apply activation functions
    auto dims = add_out->getDimensions().d;
    auto batch = dims[0];
    auto hidden = dims[1] / 4;

    auto size = nvinfer1::Dims2(batch, hidden);
    auto stride = nvinfer1::Dims2(1, 1);
    auto offset0 = nvinfer1::Dims2(0, 0);
    auto offset1 = nvinfer1::Dims2(0, hidden);
    auto offset2 = nvinfer1::Dims2(0, 2 * hidden);
    auto offset3 = nvinfer1::Dims2(0, 3 * hidden);

    auto slice1 = engine->network()->addSlice(*add_out, offset0, size, stride);
    slice1->setName((layer_info(node) + "_ISliceLayer_" + "for_offset0").c_str());
    auto active1 = engine->network()->addActivation(*slice1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    active1->setName((layer_info(node) + "_IActivationLayer_" + "for_offset0").c_str());
    auto ingate = active1->getOutput(0);
    
    auto slice2 = engine->network()->addSlice(*add_out, offset1, size, stride);
    slice2->setName((layer_info(node) + "_ISliceLayer_" + "for_offset1").c_str());
    auto active2 = engine->network()->addActivation(*slice2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    active2->setName((layer_info(node) + "_IActivationLayer_" + "for_offset1").c_str());
    auto forgetgate = active2->getOutput(0);
    
    auto slice3 = engine->network()->addSlice(*add_out, offset2, size, stride);
    slice3->setName((layer_info(node) + "_ISliceLayer_" + "for_offset2").c_str());
    auto active3 = engine->network()->addActivation(*slice3->getOutput(0), nvinfer1::ActivationType::kTANH);
    active3->setName((layer_info(node) + "_IActivationLayer_" + "for_offset2").c_str());
    auto cellgate = active3->getOutput(0);

    auto slice4 = engine->network()->addSlice(*add_out, offset3, size, stride);
    slice4->setName((layer_info(node) + "_ISliceLayer_" + "for_offset3").c_str());
    auto active4 = engine->network()->addActivation(*slice4->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    active4->setName((layer_info(node) + "_IActivationLayer_" + "for_offset3").c_str());
    auto outgate = active4->getOutput(0);

    // compute cy
    auto forget_cx = engine->network()->addElementWise(*forgetgate, *state[1], nvinfer1::ElementWiseOperation::kPROD);
    forget_cx->setName((layer_info(node) + "_prod_" + "for_forget_cx").c_str());
    auto in_cell = engine->network()->addElementWise(*ingate, *cellgate, nvinfer1::ElementWiseOperation::kPROD);
    in_cell->setName((layer_info(node) + "_prod_" + "for_in_cell").c_str());
    auto cy = engine->network()->addElementWise(
        *forget_cx->getOutput(0), *in_cell->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    cy->setName((layer_info(node) + "_prod_" + "for_cy").c_str());
    auto cy_out = cy->getOutput(0);

    // compute hy
    auto cy_tanh = engine->network()->addActivation(*cy_out, nvinfer1::ActivationType::kTANH);
    cy_tanh->setName((layer_info(node) + "_IActivationLayer_" + "for_cy_tanh").c_str());
    auto hy = engine->network()->addElementWise(*outgate, *cy_tanh->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    hy->setName((layer_info(node) + "_prod_" + "for_hy").c_str());
    auto hy_out = hy->getOutput(0);

    engine->context().set_tensor(node->outputs()[0], hy_out);
    engine->context().set_tensor(node->outputs()[1], cy_out);
   
    LOG(INFO) << "Output tensor shape: " << hy_out->getDimensions();
    LOG(INFO) << "Output tensor shape: " << cy_out->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, LstmCellConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
