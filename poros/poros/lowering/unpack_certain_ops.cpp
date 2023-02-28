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

// Part of the following code in this file refs to
// https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/unpack_var.cpp
//
// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD 3-Clause "New" or "Revised" License

/**
* @file unpack_certain_ops.cpp
* @author tianjinjin@baidu.com
* @date Thu Sep 23 20:15:53 CST 2021
* @brief
**/

#include "poros/lowering/lowering_pass.h"

#include "torch/csrc/jit/passes/subgraph_rewrite.h"

namespace baidu {
namespace mirana {
namespace poros {

void unpack_std(std::shared_ptr<torch::jit::Graph>& graph) {
    std::string std_pattern = R"IR(
    graph(%1, %dim, %unbiased, %keepdim):
        %out: Tensor = aten::std(%1, %dim, %unbiased, %keepdim)
        return (%out))IR";

    std::string unpacked_pattern = R"IR(
    graph(%1, %dim, %unbiased, %keepdim):
        %z: Tensor = aten::var(%1, %dim, %unbiased, %keepdim)
        %out: Tensor = aten::sqrt(%z)
        return (%out))IR";

    torch::jit::SubgraphRewriter std_rewriter;
    std_rewriter.RegisterRewritePattern(std_pattern, unpacked_pattern);
    std_rewriter.runOnGraph(graph);
}

void unpack_var(std::shared_ptr<torch::jit::Graph>& graph) {
    std::string var_pattern = R"IR(
    graph(%input, %dim, %unbiased, %keepdim):
        %out: Tensor = aten::var(%input, %dim, %unbiased, %keepdim)
        return (%out))IR";
    std::string unpacked_pattern = R"IR(
    graph(%input, %dims, %unbiased, %keepdim):
        %none: None = prim::Constant()
        %false: bool = prim::Constant[value=0]()
        %0: int = prim::Constant[value=0]()
        %f32_dtype: int = prim::Constant[value=6]()
        %1: int = prim::Constant[value=1]()
        %sqrd: Tensor = aten::mul(%input, %input)
        %sqrdmean: Tensor = aten::mean(%sqrd, %dims, %keepdim, %none)
        %mean: Tensor = aten::mean(%input, %dims, %keepdim, %none)
        %meansqrd: Tensor = aten::mul(%mean, %mean)
        %var: Tensor = aten::sub(%sqrdmean, %meansqrd, %1)
        %varout : Tensor = prim::If(%unbiased)
            block0():
                %shape: int[] = aten::size(%input)
                %shapet: Tensor = aten::tensor(%shape, %f32_dtype, %none, %false)
                %dim: int = prim::ListUnpack(%dims)
                %reduceddims: Tensor = aten::select(%shapet, %0, %dim)
                %numel: Tensor = aten::prod(%reduceddims, %dim, %keepdim, %none)
                %mul: Tensor = aten::mul(%var, %numel)
                %sub: Tensor = aten::sub(%numel, %1, %1)
                %v: Tensor = aten::div(%mul, %sub)
                -> (%v)
            block1():
                -> (%var)
        return(%varout))IR";

    torch::jit::SubgraphRewriter var_rewriter;
    var_rewriter.RegisterRewritePattern(var_pattern, unpacked_pattern);
    var_rewriter.runOnGraph(graph);
}

void replace_log_softmax(std::shared_ptr<torch::jit::Graph> graph) {
    std::string old_pattern = R"IR(
    graph(%1, %dim, %dtype):
        %out: Tensor = aten::log_softmax(%1, %dim, %dtype)
        return (%out))IR";

    std::string new_pattern = R"IR(
    graph(%1, %dim, %dtype):
        %2: Tensor = aten::softmax(%1, %dim, %dtype)
        %out: Tensor = aten::log(%2)
        return (%out))IR";

    torch::jit::SubgraphRewriter std_rewriter;
    std_rewriter.RegisterRewritePattern(old_pattern, new_pattern);
    std_rewriter.runOnGraph(graph);
}

void replace_log_sigmoid(std::shared_ptr<torch::jit::Graph> graph) {
    std::string old_pattern = R"IR(
    graph(%1):
        %out: Tensor = aten::log_sigmoid(%1)
        return (%out))IR";

    std::string new_pattern = R"IR(
    graph(%1):
        %2: Tensor = aten::sigmoid(%1)
        %out: Tensor = aten::log(%2)
        return (%out))IR";

    torch::jit::SubgraphRewriter std_rewriter;
    std_rewriter.RegisterRewritePattern(old_pattern, new_pattern);
    std_rewriter.runOnGraph(graph);
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu