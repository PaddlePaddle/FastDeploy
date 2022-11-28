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
* @file fuse_hard_swish.h
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-04-07 15:31:26
* @brief
**/

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