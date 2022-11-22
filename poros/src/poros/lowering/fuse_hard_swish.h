/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_hard_swish.h
 @author Lin Xiao Chun (linxiaochun@baidu.com)
 @date 2022-04-07 15:31:26
 @brief

 */

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

class FuseHardSwish : public IFuser {
public:
    FuseHardSwish();

    /**
     * FuseHardSwish
     * @param graph
     * @return true if graph changed, false if not
     */
    bool fuse(std::shared_ptr<torch::jit::Graph> graph);
private:
    /**
     * search for hardswish activation recursively, record all findings
     * @param block
     * @return true if at least one hardswish found, false if none found
     */
    bool try_to_find_hardswish(torch::jit::Block *block);
};

}
}
}