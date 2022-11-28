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
* @file op_fuse_pass.h
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-03-31 16:11:18
* @brief
**/

#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

#include <vector>

namespace baidu {
namespace mirana {
namespace poros {

/**
 * FusedOpsRecord
 * only used for recording fusion infomation, DO NOT affect actual fusing logic
 */
class FusedOpsRecord {
public:
    void from();

    template<typename... Rest>
    void from(torch::jit::Node *first,  Rest ... rest);

    template<typename... Rest>
    void from(torch::jit::NodeKind first, Rest ... rest);

    void to();

    template<typename... Rest>
    void to(torch::jit::Node *first,  Rest ... rest);

    template<typename... Rest>
    void to(torch::jit::NodeKind first, Rest ... rest);

    std::string info();

private:

    std::vector<torch::jit::NodeKind> from_ops_;
    std::vector<torch::jit::NodeKind> to_ops_;
};

template<typename... Rest>
void FusedOpsRecord::from( torch::jit::Node *first,  Rest ... rest) {
    from_ops_.push_back(first->kind());
    from(rest...); // recursive call using pack expansion syntax
}

template<typename... Rest>
void FusedOpsRecord::from( torch::jit::NodeKind first,  Rest ... rest) {
    from_ops_.push_back(first);
    from(rest...); // recursive call using pack expansion syntax
}

template<typename... Rest>
void FusedOpsRecord::to(torch::jit::Node *first,  Rest ... rest) {
    to_ops_.push_back(first->kind());
    to(rest...);
}

template<typename... Rest>
void FusedOpsRecord::to(torch::jit::NodeKind first, Rest ... rest) {
    to_ops_.push_back(first);
    to(rest...); // recursive call using pack expansion syntax
}

/**
 * IFuser
 * base class of all fusers
 */
class IFuser {
public:
    IFuser() = default;;

    virtual ~IFuser() = default;;

    virtual bool fuse(std::shared_ptr<torch::jit::Graph> graph) = 0;

    std::string info();

    void reset();


    void setName(const std::string name);

    template<typename First, typename...Rest>
    std::shared_ptr<FusedOpsRecord> record_transform(First first, Rest ...rest);

private:
    std::vector<std::shared_ptr<FusedOpsRecord>> fused_ops;

    std::string name_;

};

template<typename First, typename... Rest>
std::shared_ptr<FusedOpsRecord> IFuser::record_transform(First first, Rest ... rest) {
    auto f = std::make_shared<FusedOpsRecord>();
    f->from(first, rest...); // recursive call using pack expansion syntax
    fused_ops.push_back(f);
    return f;
}

/**
 * IFuserManager
 * manage the registration and application of fusers
 */
class IFuserManager {
public:

    static IFuserManager *get_instance();

    std::string register_fuser(const std::shared_ptr<IFuser> &fuser, const std::string &name);

    /**
     * apply all fusers in preprocess_fusers
     * @param graph
     */
    void preprocess_fuse(std::shared_ptr<torch::jit::Graph> graph);

    /**
     * apply all fusers in prewarm_fusers
     * @param graph
     */
    void prewarm_fuse(std::shared_ptr<torch::jit::Graph> graph);

private:
    std::vector<std::shared_ptr<IFuser>> preprocess_fusers;
    std::vector<std::shared_ptr<IFuser>> prewarm_fusers;

};

/**
 *  trying to fuse ops during preprocessing stage
 * @param graph
 */
void fuse_ops_preprocess(std::shared_ptr<torch::jit::Graph> graph);

/**
 * trying to fuse ops during pre-warming stage, now it's same to fuse_ops_preprocess.
 * @param graph
 */
void fuse_ops_prewarm(std::shared_ptr<torch::jit::Graph> graph);

#define REGISTER_OP_FUSER(T)                                     \
    const std::string _G_NAME = []() -> std::string {            \
         return IFuserManager::get_instance()->register_fuser(   \
            std::make_shared<T>(), #T);                          \
    }();

}
}
}
