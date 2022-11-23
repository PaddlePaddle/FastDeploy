/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_meshgrid.h
 @author Lin Xiao Chun (linxiaochun@baidu.com)
 @date 2022-04-29 14:56:57
 @brief
 */

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

class FuseMeshgrid : public IFuser {
public:
    FuseMeshgrid();

    bool fuse(std::shared_ptr<torch::jit::Graph> graph);

private:
    bool try_to_find_meshgrid(torch::jit::Block *block);

};

}
}
}