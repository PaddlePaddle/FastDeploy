/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_conv_bn.h
 @author Lin Xiao Chun (linxiaochun@baidu.com)
 @date 2022-03-31 16:11:19
 @brief
 */

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

class FuseConvBatchNorm : public IFuser {
public:
    FuseConvBatchNorm();

    bool fuse(std::shared_ptr<torch::jit::Graph> graph);

private:
    bool try_to_fuse_conv_batchnorm(torch::jit::Block *block);

    std::shared_ptr<torch::jit::Graph> graph_;
};

}
}
}