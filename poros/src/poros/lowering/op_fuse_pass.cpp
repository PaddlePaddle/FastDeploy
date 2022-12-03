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
* @file op_fuse_pass.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-03-31 16:11:18
* @brief
**/
#include "poros/lowering/op_fuse_pass.h"

#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <utility>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

std::string IFuser::info() {
    std::string info = "OP Fuser:" + IFuser::name_ + " ";

    for (auto ori: IFuser::fused_ops) {
        info += "[" + ori->info() + "]";
    }
    return info;

}

void IFuser::setName(const std::string name) {
    IFuser::name_ = name;
}

void IFuser::reset() {
    IFuser::fused_ops.clear();
}

std::string FusedOpsRecord::info() {
    std::string info;
    for (auto it = from_ops_.begin(); it != from_ops_.end(); it++) {
        info += std::string(it->toUnqualString());
        if (it != from_ops_.end() - 1) {
            info += ",";
        }
    }
    info += " => ";
    for (auto it = to_ops_.begin(); it != to_ops_.end(); it++) {
        info += std::string(it->toUnqualString());
        if (it != to_ops_.end() - 1) {
            info += ",";
        }
    }
    return info;
}

void FusedOpsRecord::from() {

}

void FusedOpsRecord::to() {

}

void fuse_ops_preprocess(std::shared_ptr<torch::jit::Graph> graph) {
    IFuserManager *manager = IFuserManager::get_instance();
    manager->preprocess_fuse(std::move(graph));

}

void fuse_ops_prewarm(std::shared_ptr<torch::jit::Graph> graph) {
    IFuserManager *manager = IFuserManager::get_instance();
    manager->prewarm_fuse(std::move(graph));

}

IFuserManager *IFuserManager::get_instance() {
    static IFuserManager manager;
    return &manager;
}

std::string IFuserManager::register_fuser(const std::shared_ptr<IFuser> &fuser, const std::string &name) {
    fuser->setName(name);
    preprocess_fusers.push_back(fuser);
    prewarm_fusers.push_back(fuser);
    return name;
}

void IFuserManager::preprocess_fuse(std::shared_ptr<torch::jit::Graph> graph) {
    bool graph_changed = false;
    for (auto &&fuser: preprocess_fusers) {
        fuser->reset();
        if (fuser->fuse(graph)) {
            LOG(INFO) << fuser->info();
            graph_changed = true;
        }
    }
    if (graph_changed) {
        ConstantPropagation(graph);
        EliminateDeadCode(graph);
        EliminateCommonSubexpression(graph);
        ConstantPooling(graph);
    }
}

void IFuserManager::prewarm_fuse(std::shared_ptr<torch::jit::Graph> graph) {
    bool graph_changed = false;
    for (auto &&fuser: prewarm_fusers) {
        fuser->reset();
        if (fuser->fuse(graph)) {
            LOG(INFO) << fuser->info();
            graph_changed = true;
        }
    }
    if (graph_changed) {
        ConstantPropagation(graph);
        EliminateDeadCode(graph);
        EliminateCommonSubexpression(graph);
        ConstantPooling(graph);
    }
}

}
}
}