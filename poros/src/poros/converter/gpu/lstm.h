/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file lstm.h
* @author wangrui39@baidu.com
* @date Mon January 17 11:36:11 CST 2022
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

// Correspons to torch.lstm_cell https://pytorch.org/docs/1.9.1/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
class LstmConverter : public GpuConverter {
public:
    LstmConverter() {}
    virtual ~LstmConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)"};
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * aten::lstm.input(Tensor input,
     *                  Tensor[] hx, Tensor[] params,
     *                  bool has_biases, int num_layers, float dropout,
     *                  bool train,
     *                  bool bidirectional,
     *                  bool batch_first) -> (Tensor, Tensor, Tensor))
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::lstm};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
