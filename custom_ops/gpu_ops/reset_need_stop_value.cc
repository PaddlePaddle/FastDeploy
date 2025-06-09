// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void SetStopValue(const paddle::Tensor& not_need_stop) {
    bool* stop_data = const_cast<bool*>(not_need_stop.data<bool>());
    stop_data[0] = true;
}

PD_BUILD_STATIC_OP(reset_stop_value)
    .Inputs({"not_need_stop"})
    .Outputs({"not_need_stop_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"}})
    .SetKernelFn(PD_KERNEL(SetStopValue));
