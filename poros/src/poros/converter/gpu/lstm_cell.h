/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file lstm_cell.h
* @author wangrui39@baidu.com
* @date Mon December 13 11:36:11 CST 2021
* @brief 
**/

#pragma once

#include <string>

//from pytorch
#include "torch/script.h"

#include "poros/converter/gpu/gpu_converter.h"
#include "poros/engine/tensorrt_engine.h"

namespace baidu {
namespace mirana {
namespace poros {

// Correspons to torch.lstm_cell https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.htmls
class LstmCellConverter : public GpuConverter {
public:
    LstmCellConverter() {}
    virtual ~LstmCellConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
    const std::vector<std::string> schema_string() {
        return {"aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)"};
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::lstm_cell};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)", {0, 0}}});
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
