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
* @file try_to_freeze_percentformat.cpp
* @author tianjinjin@baidu.com
* @date Wed Nov 24 15:13:00 CST 2021
* @brief
**/

#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {

using namespace torch::jit;

struct FreezePercentFormat {
    FreezePercentFormat(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        bool changed = freeze_percentformats(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
            //PeepholeOptimize(graph_, /*addmm_fusion_enabled*/false);
            //CheckInplace(graph_);
            //runRequiredPasses(graph_);
        }
        return;
    }

private:

    bool is_percent_format_node(Node* node) {
        if (node->kind() != aten::percentFormat) {
            return false;
        }
        //maybe to add more
        return true;
    }

    // IValue tags are intentionally private, so we need additional logic to cast
    // the IValue type to the specified format.
    void add_formatted_arg(char key,
                    const IValue& ival,
                    std::stringstream& ss,
                    int precision = 6) {
        // TODO: Implement precison-based formatting
        std::stringstream tmp;
        switch (key) {
            case 'd':
            case 'i':
                if (ival.isInt()) {
                    ss << ival.toInt();
                } else {
                    ss << static_cast<int>(ival.toDouble());
                }
                break;
            case 'e':
            case 'E':
                tmp << std::setprecision(precision) << std::scientific;
                if (key == 'E') {
                    tmp << std::uppercase;
                }
                if (ival.isInt()) {
                    tmp << static_cast<float>(ival.toInt());
                } else {
                    tmp << static_cast<float>(ival.toDouble());
                }
                ss << tmp.str();
                break;
            case 'f':
            case 'F':
                tmp << std::setprecision(precision) << std::fixed;
                if (ival.isInt()) {
                    tmp << static_cast<float>(ival.toInt());
                } else {
                    tmp << static_cast<float>(ival.toDouble());
                }
                ss << tmp.str();
                break;
            case 'c':
                if (ival.isInt()) {
                    ss << static_cast<char>(ival.toInt());
                } else {
                    ss << ival.toStringRef();
                }
                break;
            case 's':
                if (ival.isString()) {
                    ss << ival.toStringRef();
                } else {
                    ss << ival;
                }
                break;
            default:
                TORCH_CHECK(false, "The specifier %", key, " is not supported in TorchScript format strings");
        }
    }

    std::string interprete_percent_format(std::vector<IValue>& stack, size_t num_inputs) {
        auto format_str = peek(stack, 0, num_inputs).toStringRef();
        auto args = last(stack, num_inputs - 1)[0];
        size_t args_size = 1; // assumed size
        if (args.isTuple()) {
            args_size = args.toTuple()->elements().size();
        }
        std::stringstream ss;
        size_t used_args = 0;
        size_t begin = 0;
        
        while (true) {
            size_t percent_idx = format_str.find('%', begin);
            if (percent_idx == std::string::npos) {
                ss << format_str.substr(begin);
                break;
            }
            size_t format_idx = percent_idx + 1;
            TORCH_CHECK(
                percent_idx < format_str.length() - 1, "Incomplete format specifier");
            ss << format_str.substr(begin, percent_idx - begin);

            if (format_str.at(format_idx) == '%') {
                ss << '%';
                begin = percent_idx + 2; // skip the `%` and the format specifier
                continue;
            }

            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            TORCH_CHECK(used_args < args_size, "Too few arguments for format string");
            char key = format_str.at(format_idx);
            IValue arg;
            if (args.isTuple()) {
                arg = args.toTuple()->elements()[used_args];
            } else {
                arg = args;
            }
            add_formatted_arg(key, arg, ss);
            begin = percent_idx + 2;
            ++used_args;
        }
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        TORCH_CHECK(used_args == args_size, "Too many arguments for format string");
        std::string result = ss.str();
        return result;
    }

    /**
     * the schema of percentformat is :  "aten::percentFormat(str self, ...) -> str"
     * **/
    bool try_to_freeze_percentformat(Node* format_node) {
        //Graph* graph = format_node->owningGraph();
        at::ArrayRef<Value*> inputs = format_node->inputs();
        size_t num_inputs = inputs.size();
        
        //no format input situation.
        if (num_inputs < 2) {
            LOG(INFO) << "should not freeze node: " << node_info(format_node); 
            return false;
        }
        
        //bool all_input_constant = true;
        std::vector<IValue> stack;
        for(size_t index = 0; index < num_inputs; index++) {
            if (inputs[index]->node()->kind() != prim::Constant) {
                LOG(INFO) << "should not freeze node: " << node_info(format_node); 
                return false;
            } else {
                c10::optional<IValue> ivalue = toIValue(inputs[index]->node()->output());
                if (ivalue.has_value()) {
                    stack.push_back(ivalue.value());
                }
            }
        }

        if (stack.size() != num_inputs) {
            LOG(INFO) << "should not freeze node: " << node_info(format_node);     
            return false;
        }

        //if we reach here, that means all inputs are constant. let's calculate the result
        std::string result = interprete_percent_format(stack, num_inputs);
        LOG(INFO) << "try to replace the output of node :" << node_info(format_node)
                << " with constant value: " << result;
        WithInsertPoint guard(graph_->block()->nodes().front());
        Value* string_const = graph_->insertConstant(result);
        format_node->outputs().at(0)->replaceAllUsesWith(string_const);
        format_node->destroy();
        return true;
    }

    bool freeze_percentformats(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // we might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= freeze_percentformats(subblock);
            }
            if (is_percent_format_node(node)) {
                LOG(INFO) << "meet percent format node :" << node_info(node);
                changed |= try_to_freeze_percentformat(node);
            }
        }
        return changed;        
    }

std::shared_ptr<Graph> graph_;
};

} // namespace

void freeze_percentformat(std::shared_ptr<torch::jit::Graph> graph) {
    LOG(INFO) << "Running poros freeze_percentformat passes";
    FreezePercentFormat fpf(std::move(graph));
    fpf.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
