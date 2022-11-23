/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file aten_trt_util.h
* @author tianjinjin@baidu.com
* @date Fri Aug  6 10:42:39 CST 2021
* @brief
**/

#pragma once

#include <string>

#include "torch/script.h"
#include "NvInfer.h"

namespace baidu {
namespace mirana {
namespace poros {

//将torchscript中的at::tensor转变成tensorrt中的weights结构
bool at_tensor_to_trt_weignts(at::Tensor tensor, nvinfer1::Weights& weight);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu