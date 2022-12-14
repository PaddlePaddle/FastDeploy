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
* @file: fuse_conv_mul.h
* @author: zhangfan51@baidu.com
* @data: 2022-04-24 18:41:20
* @brief: 
**/ 

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 *  %3 : int = prim::Constant[value=1]()
 *  %4 : float = prim::Constant[value=2.0]()
 *  %1 : Tensor = aten::conv2d(%0, %conv_w, %conv_b, %conv_stride, %conv_padding, %conv_dilation, %3)
 *  %2 : Tensor = aten::mul(%1, %4)
 *
 * 如上面的IR，FuseConvMul是将conv + mul中的mul融到conv中，减少一次mul计算，融后在图上可以匹配到更多的针对conv的优化pass；
 * 限制：aten::mul的输入%4需为constant类型。
 */
class FuseConvMul : public IFuser {
public:
    FuseConvMul();

    bool fuse(std::shared_ptr<torch::jit::Graph> graph);

private:
    bool try_to_fuse_conv_mul(torch::jit::Block *block);

    std::shared_ptr<torch::jit::Graph> graph_;
};

}
}
}
