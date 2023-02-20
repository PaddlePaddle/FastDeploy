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
* @file graph_prewarm.cpp
* @author tianjinjin@baidu.com
* @date Fri Apr 23 11:41:59 CST 2021
* @brief 
**/
#include "poros/compile/graph_prewarm.h"

//pytorch
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inplace_check.h>
// #include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include "poros/compile/ivalues_analysis.h"
#include "poros/lowering/lowering_pass.h"
#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;
using Stack = std::vector<c10::IValue>;

struct PorosGraphPrewarm {
    PorosGraphPrewarm(std::shared_ptr<Graph> graph):graph_(std::move(graph)) {}

    //graph 预热核心逻辑
    std::shared_ptr<Graph> run(std::vector<Stack>& stack_vec) {

        //step1: back up the prewarm data
        //attention: the data in stack has changed during the interpreterstate procedure.
        //so we should copy twice the stack before interpreter execution
        //one for input_param_propagate, and the other for final interpreterState.
        std::vector<Stack> stack_vec_final = stack_vec;
        std::vector<Stack> stack_vec_copy = stack_vec;

        //step2: the first round ivalue analysis
        torch::jit::getProfilingMode() = true;
        torch::jit::getExecutorMode() = true;
        torch::jit::setGraphExecutorOptimize(false);
        torch::jit::getNumProfiledRuns() = stack_vec.size();
        GRAPH_DEBUG("before first round IvalueAnalysis Graph: ", graph_);
        std::unique_ptr<IvalueAnalysis> ia = IvalueAnalysis::analysis_ivalue_for_graph(graph_);
        ExecutionPlan plan = ExecutionPlan(ia->graph(), "first_round_prewarm");
        for (size_t i = 0; i < stack_vec.size(); i++) {
            InterpreterState(plan.code).run(stack_vec[i]);
        }
        std::shared_ptr<torch::jit::Graph> output_graph = ia->graph();
        GRAPH_DEBUG("after first round IvalueAnalysis Graph: ", output_graph);
        
        //step3: necessary passes to eliminate the profile information in graph
        {
            baidu::mirana::poros::input_param_propagate(output_graph, stack_vec_copy);
            std::vector<Stack>().swap(stack_vec_copy);
            GRAPH_DEBUG("after input_param_propagate Graph: ", output_graph);
            torch::jit::ProfilingRecord::removeProfileCounter(output_graph->block());
            GRAPH_DEBUG("after removeProfileCounter Graph: ", output_graph);
            baidu::mirana::poros::remove_simple_type_profile_nodes(output_graph);
            GRAPH_DEBUG("after remove_simple_type_profile_nodes Graph: ", output_graph);
            torch::jit::RemoveProfileNodesAndSpecializeTypes(output_graph);
            GRAPH_DEBUG("after RemoveProfileNodesAndSpecializeTypes Graph: ", output_graph);
        }

        // step4: some passes can be run based on the prifiled graph and data
        {
            torch::jit::runRequiredPasses(output_graph);
            torch::jit::EliminateDeadCode(output_graph);
            torch::jit::EliminateCommonSubexpression(output_graph);
            /* addmm is only done as an optimization for onnx, so we disable it */
            torch::jit::PeepholeOptimize(output_graph, /*addmm_fusion_enabled*/false);
            torch::jit::ConstantPropagation(output_graph);  //this is very necessary for prone if block!!!!
            torch::jit::ConstantPooling(output_graph);
            // torch::jit::UnrollLoops(output_graph);
            baidu::mirana::poros::unrolling_loop(output_graph);
            baidu::mirana::poros::freeze_percentformat(output_graph);
            baidu::mirana::poros::freeze_aten_size(output_graph);
            baidu::mirana::poros::freeze_aten_len(output_graph);
            baidu::mirana::poros::unrolling_loop(output_graph);

            //some mutation handle pass below
            GRAPH_DEBUG("before remove mutation Graph: ", output_graph);
            //TODO: handle this later, it cores when we using prepare_inplace_ops.
            //prepare_inplace_ops(output_graph);
            torch::jit::RemoveListMutation(output_graph);
            torch::jit::RemoveTensorMutation(output_graph);
            GRAPH_DEBUG("after remove mutation Graph: ", output_graph);

            baidu::mirana::poros::fuse_ops_prewarm(output_graph);

            // run some pass again after unrolled loops
            torch::jit::PeepholeOptimize(output_graph, /*addmm_fusion_enabled*/false);
            torch::jit::ConstantPropagation(output_graph);
            torch::jit::EliminateCommonSubexpression(output_graph);
            torch::jit::CheckInplace(output_graph);
            torch::jit::runRequiredPasses(output_graph);

            baidu::mirana::poros::eliminate_some_dict(output_graph);
            baidu::mirana::poros::eliminate_some_list(output_graph);

            torch::jit::PeepholeOptimize(output_graph, /*addmm_fusion_enabled*/false);
            torch::jit::ConstantPropagation(output_graph);
            torch::jit::ConstantPooling(output_graph);
            baidu::mirana::poros::unrolling_loop(output_graph);
            torch::jit::EliminateCommonSubexpression(output_graph);
            baidu::mirana::poros::link_mutable_list(output_graph);
            torch::jit::CheckInplace(output_graph);
            torch::jit::runRequiredPasses(output_graph);

            torch::jit::LowerSimpleTuples(output_graph);
        }

        //step5: prepare for second round ivalue analysis
        //reset the profile number & clean profile information last round.
        torch::jit::getNumProfiledRuns() = stack_vec_final.size();
        torch::jit::ClearProfilingInformation(output_graph);
        std::unique_ptr<IvalueAnalysis> ia_final = IvalueAnalysis::analysis_ivalue_for_graph(output_graph);
        ExecutionPlan plan_final = ExecutionPlan(ia_final->graph(), "second_round_prewarm");
        for (size_t i = 0; i < stack_vec_final.size(); i++) {
            InterpreterState(plan_final.code).run(stack_vec_final[i]);
        }
        std::shared_ptr<torch::jit::Graph> final_graph = ia_final->graph();

        //step6: store the final dynamic information
        bool is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;
        if (is_dynamic_shape) {
            ia_final->gen_value_dyanamic_shape();
        }
        ia_final->gen_list_size();
        ia_final->gen_int_intlist_value();

        //step7: necessary passes to eliminate the profile record
        {
            torch::jit::ProfilingRecord::removeProfileCounter(final_graph->block());
            baidu::mirana::poros::remove_simple_type_profile_nodes(final_graph);
            torch::jit::RemoveProfileNodesAndSpecializeTypes(final_graph);
            baidu::mirana::poros::freeze_aten_dim(final_graph);
            baidu::mirana::poros::freeze_list_construct(final_graph);
        }
        
        GRAPH_DUMP("final graph_prewarm Graph: ", final_graph);
        return final_graph;
    }

private:
    std::shared_ptr<Graph> graph_;

}; // struct PorosGraphPrewarm

}  // anonymous namespace

std::shared_ptr<Graph> graph_prewarm(std::shared_ptr<Graph>& graph, 
                            const std::vector<std::vector<c10::IValue> >& prewarm_datas) {
    std::vector<std::vector<c10::IValue>> stacks;
    for (size_t i = 0; i < prewarm_datas.size(); ++i) { //TODO: Make it better here
        std::vector<c10::IValue> stack;
        for (c10::IValue input : prewarm_datas[i]) {
            stack.push_back(input);
        }
        stacks.push_back(stack);
    }
    std::shared_ptr<Graph> new_graph = PorosGraphPrewarm(graph).run(stacks);
    return new_graph;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
