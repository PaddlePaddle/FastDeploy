/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file graph_segment.h
* @author tianjinjin@baidu.com
* @date Thu Mar 18 14:33:54 CST 2021
* @brief 
**/

#pragma once

//pytorch
#include <torch/script.h>

#include "poros/engine/iengine.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 * @brief  graph_segment fullfil the segmentation func of given graph
 * @param [in/out] graph : the graph to be segmented
 * @param [in] engine : backend engine
 * @return
 **/
void graph_segment(std::shared_ptr<torch::jit::Graph>& graph, IEngine* engine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
