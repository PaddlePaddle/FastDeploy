/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_clip.h
 @author tianshaoqing@baidu.com
 @date 2022-08-01 16:08:26
 @brief
 */

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

class FuseClip : public IFuser {
public:
    FuseClip();
    /**
     * FuseClip
     * @param graph
     * @return true if graph changed, false if not
     */
    bool fuse(std::shared_ptr<torch::jit::Graph> graph);
private:
    /**
     * search for aten::clip recursively, record all findings
     * @param block
     * @return true if at least one clip found, false if none found
     */
    bool try_to_replace_clip(torch::jit::Block *block);
};

}
}
}