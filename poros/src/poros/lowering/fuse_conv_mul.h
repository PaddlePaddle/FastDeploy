/***************************************************************************
*
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
*
**************************************************************************/
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
