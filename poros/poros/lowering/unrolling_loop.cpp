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
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/loop_unrolling.cpp
//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the 3-Clause BSD License

/**
* @file unrolling_loop.cpp
* @author tianjinjin@baidu.com
* @date Mon Nov 22 16:59:25 CST 2021
* @brief this file is modified from torch/csrc/jit/passes/loop_unrolling.cpp
*        and some parameters are different from the original funciton
**/
#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;

static constexpr int64_t UnrollFactor = 8;
static constexpr int64_t MaxBodySize = 256;
static constexpr int64_t MaxBodyRepeats = 64;
static constexpr int64_t MaxLoopMulResult = 32 * 64;

struct UnrollingLoop {
    UnrollingLoop(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        bool changed = unroll_loops(graph_->block(), true);
        GRAPH_DUMP("afte unroll_loop graph:", graph_);
        changed |= eliminate_useless_loop_count_body(graph_->block());
        GRAPH_DUMP("afte eliminate_useless_loop_count_body graph:", graph_);
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        return;
    }

private:

    bool is_for_loop(Node* node) {
        if (node->kind() != prim::Loop) {
            return false;
        }
        Value* start_cond = node->inputs().at(1);
        c10::optional<bool> maybe_start_value = constant_as<bool>(start_cond);
        Value* continue_cond = node->blocks().at(0)->outputs().at(0);
        c10::optional<bool> maybe_continue_value = constant_as<bool>(continue_cond);
        return maybe_start_value && *maybe_start_value && maybe_continue_value && *maybe_continue_value;
    }

    int64_t limited_block_size(Block* body, int64_t limit) {
        auto it = body->nodes().begin();
        auto end = body->nodes().end();
        for (int64_t i = 0; i < limit; ++it) {
            for (Block* subblock : it->blocks()) {
                i += limited_block_size(subblock, limit - i);
            }
            if (!it->notExecutedOp()) {
                ++i;
            }
            if (it == end) {
                return i;
            }
        }
        return limit;
    }

    int64_t calculate_block_size(Block* body) {
        auto it = body->nodes().begin();
        int64_t count = 0;
        while (it != body->nodes().end()) {
            auto node = *it;
            ++it;  //先++it
            for (auto block : node->blocks()) {
                count += calculate_block_size(block);
            }
            if (!node->notExecutedOp()) {
                ++count;
            }
        }
        return count;
    }

    bool is_small_block(Block* body) {
        return limited_block_size(body, MaxBodySize + 1) <= MaxBodySize;
    }

    // XXX: This function can only be called with a loop that is guaranteed to
    // execute EXACTLY ONCE.
    void inline_body(Node* loop) {
        auto graph = loop->owningGraph();
        auto body = loop->blocks().at(0);
        WithInsertPoint insert_point_guard{loop};

        std::unordered_map<Value*, Value*> value_map;
        auto get_value = [&](Value* v) {
            auto it = value_map.find(v);
            if (it != value_map.end())
            return it->second;
            return v;
        };

        // Loop node has extra (max_iters, initial_cond) inputs,
        // body has an extra (loop_counter) input.
        for (size_t i = 2; i < loop->inputs().size(); ++i) {
            value_map[body->inputs()[i - 1]] = loop->inputs()[i];
        }

        for (Node* orig : body->nodes()) {
            Node* clone = graph->insertNode(graph->createClone(orig, get_value));
            for (size_t i = 0; i < orig->outputs().size(); ++i) {
                value_map[orig->outputs()[i]] = clone->outputs()[i];
            }
        }
        for (size_t i = 0; i < loop->outputs().size(); ++i) {
            loop->outputs().at(i)->replaceAllUsesWith(
                get_value(body->outputs().at(i + 1)));
        }
        // XXX: it is extremely important to destroy the loop in here. DCE might not
        // be able to conclude that it's safe, because the loop might contain side
        // effects.
        loop->destroy();
    }

    // inserts a copy of body, passing inputs to the inputs of the block
    // it returns the a list of the Values for the output of the block
    std::vector<Value*> insert_block_copy(Graph& graph,
                            Block* body,
                            at::ArrayRef<Value*> inputs) {
        TORCH_INTERNAL_ASSERT(inputs.size() == body->inputs().size());
        std::unordered_map<Value*, Value*> value_map;
        auto get_value = [&](Value* v) {
            auto it = value_map.find(v);
            if (it != value_map.end())
            return it->second;
            return v;
        };
        auto inputs_it = inputs.begin();
        for (Value* input : body->inputs()) {
            value_map[input] = *inputs_it++;
        }
        for (Node* node : body->nodes()) {
            Node* new_node = graph.insertNode(graph.createClone(node, get_value));
            auto outputs_it = new_node->outputs().begin();
            for (Value* output : node->outputs()) {
            value_map[output] = *outputs_it++;
            }
        }
        return fmap(body->outputs(), get_value);  //maybe not recognized
    }

    void repeat_body(Block* body, size_t times, Block* dest) {
        auto graph = body->owningGraph();
        WithInsertPoint insert_point_guard(dest);
        for (Value* input : body->inputs()) {
            dest->addInput()->copyMetadata(input);
        }

        std::vector<Value*> io = dest->inputs().vec();
        TORCH_INTERNAL_ASSERT(
            !body->inputs().at(0)->hasUses(), "loop counter should be unused");
        for (size_t i = 0; i < times; ++i) {
            io[0] = body->inputs().at(0);
            io = insert_block_copy(*graph, body, io);
        }
        for (Value* output : io) {
            dest->registerOutput(output);
        }

        // It's likely that we have some dead nodes now - for example the "true"
        // constant that prevents the loop from breaking. We shouldn't wait too long
        // before removing them because they might artificially increase the loop size
        // and prevent outer loop unrolling.
        torch::jit::EliminateDeadCode(dest, false);
    }

    // Replaces the builtin loop counter with a "mutable" variable outside of the
    // loop.
    void replace_loop_counter(Node* loop) {
        Graph* graph = loop->owningGraph();
        Block* body = loop->blocks().at(0);
        WithInsertPoint guard(loop);
        Value* init_counter = graph->insertConstant(0);

        loop->insertInput(2, init_counter);
        loop->insertOutput(0)->setType(IntType::get());

        Value* internal_counter = body->insertInput(1)->setType(init_counter->type());
        body->inputs()[0]->replaceAllUsesWith(internal_counter);

        WithInsertPoint insertPointGuard{body->return_node()};
        Value* result = graph->insert(aten::add, {internal_counter, 1});
        body->insertOutput(1, result);
    }

    bool unroll(Node* loop) {
        Graph* graph = loop->owningGraph();
        Block* body = loop->blocks().at(0);

        int64_t block_size = calculate_block_size(body);
        if (block_size > MaxBodySize) {
            return false;
        }

        // if (!is_small_block(body)) {
        //     return false;
        // }

        // We will be using a "mutable" counter outside of the loop instead of the
        // default one, because this will allow us to share it between the unrolled
        // loop and its epilogue. This is necessary only if the loop counter is
        // actually used in the body.
        if (body->inputs()[0]->uses().size() > 0)
            replace_loop_counter(loop);

        // Some optimization for constant-length loops. If we know they won't run too
        // many times, then we can unroll them entirely.
        Value* trip_count = loop->inputs().at(0);
        c10::optional<int64_t> const_len = constant_as<int64_t>(trip_count);
        //auto loop_mul_result = block_size * const_len;
        if (const_len && *const_len < MaxBodyRepeats && (block_size * (*const_len)) < MaxLoopMulResult) {
            Block* dest = loop->addBlock();
            repeat_body(body, *const_len, dest);
            loop->eraseBlock(0);
            inline_body(loop);
            return true;
        }

        WithInsertPoint insert_point_guard{loop};

        // Clone the loop before we unroll it. The clone will become the epilogue.
        Node* loop_epilogue =
            graph->createClone(loop, [](Value* v) { return v; })->insertAfter(loop);
        for (size_t i = 0; i < loop->outputs().size(); ++i) {
            loop->outputs()[i]->replaceAllUsesWith(loop_epilogue->outputs()[i]);
            loop_epilogue->replaceInput(i + 2, loop->outputs()[i]);
        }

        Block* dest = loop->addBlock();
        repeat_body(body, UnrollFactor, dest);
        loop->eraseBlock(0);

        // Change the iteration counts of both loops
        Value* iter_count = loop->inputs().at(0);
        Value* unrolled_iter_count = graph->insert(
            aten::__round_to_zero_floordiv, {iter_count, UnrollFactor});
        loop->replaceInput(0, unrolled_iter_count);
        loop_epilogue->replaceInput(
            0,
            graph->insert(
                aten::sub,
                {iter_count,
                graph->insert(aten::mul, {unrolled_iter_count, UnrollFactor})}));
        return true;
    }

    bool unroll_loops(Block* block, bool constant_only) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // XXX: unroll might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= unroll_loops(subblock, constant_only);
            }
            if (!is_for_loop(node)) {
                continue;
            }
            //only handle max loop is constant situation.
            if (constant_only && node->inputs().at(0)->node()->kind() != prim::Constant) {
                continue;
            }
            changed |= unroll(node);
        }
        return changed;        
    }
    
    // 去掉像以下形式的prim::loop计数block
    // %943 : int = prim::Loop(%65, %4641, %idx.5)
    //   block0(%944 : int, %945 : int):
    //     %0 : int = prim::Constant[value=1]()
    //     %7285 : int = aten::add(%945, %0)
    //     %947 : bool = aten::lt(%7285, %idx.13)
    //     %948 : bool = aten::__and__(%947, %4641)
    //     -> (%948, %7285)
    // 其中原来的节点已经展开到parent graph中了，只剩下计数模块，没有实质作用，可以删掉。
    bool eliminate_useless_loop_count_body(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // XXX: unroll might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= eliminate_useless_loop_count_body(subblock);
            }
            // pattern
            if (!is_uesless_loop_count_body(node)) {
                continue;
            }
            changed |= destory_useles_loop_count_body(node);
        }
        return changed;        
    }
    // 判断是否是无用的prim::loop计数模块
    bool is_uesless_loop_count_body(Node* node) {
        // 输入node类型必须是prim::loop
        if (node->kind() != prim::Loop) {
            return false;
        }
        // prim::loop的输出必须无user
        if (node->hasUses()) {
            return false;
        }
        auto loop_block = node->blocks().at(0);
        std::vector<Node*> loop_block_nodes;
        for (auto it = loop_block->nodes().begin(); it != loop_block->nodes().end(); it++) {
            loop_block_nodes.push_back(*it);
            if (loop_block_nodes.size() > 3) {
                return false;
            }
        }
        // block必须只有3个nodes且顺序必须是1-->aten::add、2-->aten::lt、3-->aten::__and__
        if (loop_block_nodes.size() == 3 &&
            loop_block_nodes[0]->kind() == aten::add && 
            loop_block_nodes[1]->kind() == aten::lt &&
            loop_block_nodes[2]->kind() == aten::__and__) {
                LOG(INFO) << "Find useless loop counter body on node: [ " << node_info(node) << " ]";
                return true;
        }
        return false;
    }
    // 删掉无用的prim::loop计数节点
    bool destory_useles_loop_count_body(Node* node) {
        if (node->kind() != prim::Loop) {
            return false;
        }
        LOG(INFO) << "Destory useless loop counter node: [ " << node_info(node) << " ]";
        node->destroy();
        return true;
    }

    std::shared_ptr<Graph> graph_;
};

} // namespace

void unrolling_loop(std::shared_ptr<torch::jit::Graph> graph) {
    LOG(INFO) << "Running poros unrolling_loop passes";
    UnrollingLoop ul(std::move(graph));
    ul.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu