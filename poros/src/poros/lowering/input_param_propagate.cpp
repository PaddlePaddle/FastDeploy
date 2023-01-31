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
* @file input_param_propagate.cpp
* @author huangben@baidu.com
* @date 2021-08-18 14:56:57
* @brief
**/
#include "poros/lowering/lowering_pass.h"

#include <ATen/core/jit_type.h>
#include <ATen/ExpandUtils.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;
struct InputParamPropagate {
    InputParamPropagate(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run(std::vector<Stack>& stack_vec) {
        if (stack_vec.size() == 0) {
            return;
        }

        auto check_input_param_unchanged = [](std::vector<Stack>& stack_vec, size_t offset) {
            if (stack_vec.size() == 1) {
                return true;
            }
            auto ivalue = stack_vec[0][offset];
            for (size_t idx = 1; idx < stack_vec.size(); ++idx) {
                if (stack_vec[idx][offset] != ivalue) {
                    return false;
                }
            }
            return true;
        };

        auto g_inputs = graph_->inputs();
        size_t extra_offset = 0;
        for (size_t offset = 0; offset < stack_vec[0].size(); ++offset) {
            if (stack_vec[0][offset].isBool() || stack_vec[0][offset].isInt()) {
                if (check_input_param_unchanged(stack_vec, offset)) {
                    WithInsertPoint guard(graph_->block()->nodes().front());
                    auto insert_value = graph_->insertConstant(stack_vec[0][offset]);
                    if (g_inputs.size() == stack_vec[0].size()) {
                        g_inputs[offset]->replaceAllUsesWith(insert_value);
                    } else {
                        //TODO: this type check is not comprehensive. It may lead some bug with unexpected input data.
                        while (c10::ClassTypePtr c = g_inputs[offset + extra_offset]->type()->cast<c10::ClassType>()) {
                            if (c->is_module()) {
                                extra_offset++;
                            }
                        }
                        g_inputs[offset + extra_offset]->replaceAllUsesWith(insert_value);
                    }
                }
            }
        }
        return;
    }

private:
    std::shared_ptr<Graph> graph_;
};
} // namespace

void input_param_propagate(std::shared_ptr<torch::jit::Graph> graph,
                        std::vector<std::vector<c10::IValue>>& stack_vec) {
    InputParamPropagate ipp(std::move(graph));
    ipp.run(stack_vec);
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
