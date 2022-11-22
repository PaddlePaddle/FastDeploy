/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_gelu.h
 @author tianshaoqing@baidu.com
 @date 2022-10-20 14:39:32
 @brief
 */

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

class FuseGelu : public IFuser {
public:
    FuseGelu();

    bool fuse(std::shared_ptr<torch::jit::Graph> graph);

private:
    bool try_to_find_gelu(torch::jit::Block *block);
};

}
}
}