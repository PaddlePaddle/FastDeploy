/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file partition.h
* @author tianjinjin@baidu.com
* @date Thu Jun  3 14:57:58 CST 2021
* @brief 
**/

#pragma once

#include "torch/script.h"

#include "poros/engine/iengine.h"

namespace baidu {
namespace mirana {
namespace poros {

bool is_node_fusable(const torch::jit::Node* node, IEngine* engine);
bool is_node_fusable(const torch::jit::Node* fusion, 
                    const torch::jit::Node* node, 
                    IEngine* engine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
