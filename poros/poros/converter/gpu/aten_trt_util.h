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