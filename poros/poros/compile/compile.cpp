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
* @file compile.h
* @author tianjinjin@baidu.com
* @author huangben@baidu.com
* @date Fri Mar  5 11:39:03 CST 2021
* @brief 
**/
#include "poros/compile/compile.h"

//pytorch
#include <torch/script.h>

//pytorch passes
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
//#includ <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include "poros/compile/graph_prewarm.h"
#include "poros/compile/graph_segment.h"
#include "poros/context/poros_global.h"
#include "poros/lowering/lowering_pass.h"
#include "poros/lowering/op_fuse_pass.h"
#include "poros/lowering/segment_post_processing.h"
#include "poros/util/poros_util.h"
// #include "poros/iplugin/plugin_create.h"

namespace baidu {
namespace mirana {
namespace poros {

Compiler::~Compiler() {
    close();
}

int Compiler::init(const PorosOptions& options) {
    _options = options;
    if (options.debug == true) {
        // when setting this, all the INFO level will be printed
        c10::ShowLogInfoToStderr();
    }
    if (_options.unconst_ops_thres == -1) {
        _options.unconst_ops_thres = 10;
        if (_options.device == Device::XPU) {
            _options.unconst_ops_thres = 1;
        }
    }
    PorosGlobalContext::instance().set_poros_options(_options);
    return 0;
}

int Compiler::preprocess_graph(std::shared_ptr<torch::jit::Graph>& graph) {
    GRAPH_DEBUG("Before preprocess_graph:", graph);
    
    //step1: some passes provided by pytorch that insensitive for profile
    {
        torch::jit::Inline(*graph);
        GRAPH_DUMP("inline graph:", graph);
        /*PropagateInputShapes and  PropagateRequiresGrad maybe no need anymore*/
        torch::jit::PropagateInputShapes(graph);
        
        torch::jit::ClearProfilingInformation(graph);
        torch::jit::LowerGradOf(*graph);  //TODO: maybe no need
        torch::jit::PropagateRequiresGrad(graph);

        torch::jit::runRequiredPasses(graph);
        /* DecomposeOps try to replace addmm & batch_norm & layer_norm. 
        considering poros can handle batch_norm. so we do not unfold this. */
        //torch::jit::DecomposeOps(graph);
        torch::jit::ConstantPropagation(graph);
        torch::jit::EliminateDeadCode(graph);
        torch::jit::EliminateCommonSubexpression(graph);
        torch::jit::ConstantPooling(graph);
        torch::jit::PeepholeOptimize(graph, false);
        torch::jit::EliminateDeadCode(graph);
        torch::jit::LowerSimpleTuples(graph);  // TODO: maybe should not be here
        torch::jit::CheckInplace(graph);
        /*DO NOT set LowerAllTuples. 
        * this may lead method output is not tuple. 
        * and get error message like this:  "Method (but not graphs in general) require a single output." */
        // torch::jit::LowerAllTuples(graph);
    }

    //step2: some passes provided by poros that insensitive for profile
    {   
        baidu::mirana::poros::replace_illegal_constant(graph);
        baidu::mirana::poros::eliminate_exception_pass(graph);
        baidu::mirana::poros::eliminate_maxpool_with_indices(graph);
        baidu::mirana::poros::eliminate_simple_useless_nodes(graph);
        baidu::mirana::poros::unpack_std(graph);
        baidu::mirana::poros::unpack_var(graph);
        baidu::mirana::poros::replace_log_softmax(graph);
        baidu::mirana::poros::replace_log_sigmoid(graph);
        baidu::mirana::poros::replace_pad(graph);
        baidu::mirana::poros::fuse_ops_preprocess(graph);
        torch::jit::runRequiredPasses(graph);
    }

    GRAPH_DEBUG("After preprocess_graph:", graph);
    return 0;
}

int Compiler::compile(const torch::jit::Module& origin_module, 
        const ivalue_vec_t& prewarm_datas, torch::jit::Module* optimized_module) {

    _origin_module = &origin_module;
    _prewarm_datas = prewarm_datas;

    GRAPH_DUMP("origin_module graph:", origin_module.get_method("forward").graph());
    torch::jit::setGraphExecutorOptimize(true);

    std::shared_ptr<torch::jit::Graph> opt_graph = nullptr;
    {
        //step1: clone orign module to unfold module
        torch::jit::Module intermediate_module = torch::jit::freeze_module(origin_module);
        auto method = intermediate_module.get_method("forward");
        auto graph = method.graph();
        int ret = preprocess_graph(graph);
        if (ret < 0) {
            LOG(ERROR) << "preprocess_graph failed!";
            return -1;
        }
        //attention. graph copy happened inside LowerGraph function
        auto graph_and_ivalues = torch::jit::LowerGraph(*graph, intermediate_module._ivalue()); 
        opt_graph = graph_and_ivalues.first;
    }

    std::shared_ptr<torch::jit::Graph> prewarm_graph = graph_prewarm(opt_graph, prewarm_datas);
    GRAPH_DUMP("prewarmed_module graph:", prewarm_graph);

    //cpu的话，预热后就返回
    if (_options.device == Device::CPU) {
        merge_graph_to_module(prewarm_graph, *optimized_module, true);
        return 0;
    }

    //step2: try to find segments in unfold module
    //划分完子图的模型
    int ret = segment_graph(prewarm_graph);
    if (ret < 0) {
        LOG(ERROR) << "segment_graph failed!";
        return -1;
    }
    GRAPH_DUMP("segmented_module graph:", prewarm_graph);

    //step3: try to replace subgraph to engine graph
    merge_graph_to_module(prewarm_graph, *optimized_module, true);
    ret = optimize_subgraph(prewarm_graph, optimized_module); 
    if (ret < 0) {
        LOG(ERROR) << "optimize_subgraph failed!";
        return -1;
    }
    GRAPH_DUMP("optimized_module graph:", optimized_module->get_method("forward").graph()); 
    return 0;
}

int Compiler::segment_graph(std::shared_ptr<torch::jit::Graph>& g) {

    IEngine* engine(nullptr);
    std::string engine_name("");
    if (_options.device == Device::GPU) {
        engine_name = "TensorrtEngine";
    } else if (_options.device == Device::XPU) {
        engine_name = "XtclEngine";
    } else {
        engine = nullptr;
    }

    if (engine_name != "") {
        engine = dynamic_cast<IEngine*>(create_plugin(engine_name, 
                    PorosGlobalContext::instance()._engine_creator_map));
        if (engine->init() < 0) {
            delete engine;
            return -1;
        }
    }
    
    graph_segment(g, engine);
    GRAPH_DEBUG("After segment graph:", g);
    delete engine;
    return 0;
}

IEngine* Compiler::select_engine(const torch::jit::Node* n) {
    if (n == nullptr || n->kind() != torch::jit::prim::CudaFusionGroup) {
        return nullptr;
    }

    IEngine* engine(nullptr);
    std::string engine_name("");
    if (_options.device == Device::GPU) {
        engine_name = "TensorrtEngine";
    } else if (_options.device == Device::XPU) {
        engine_name = "XtclEngine";
    } else {
        engine = nullptr;
    } 

    if (engine_name != "") {
        engine = dynamic_cast<IEngine*>(create_plugin(engine_name, 
                    PorosGlobalContext::instance()._engine_creator_map));
        if (engine->init() < 0) {
            return nullptr;
        }
        _engine_map[n] = engine;
    }

    return engine;
}


int Compiler::optimize_subgraph(const std::shared_ptr<torch::jit::Graph>& opt_graph, 
        torch::jit::Module* optimized_module) {
    auto block = opt_graph->block();
    auto ret = optimize_subblock(block, optimized_module);
    return ret;
}


int Compiler::optimize_subblock(torch::jit::Block* block,
        torch::jit::Module* optimized_module) {

    std::vector<torch::jit::Node*> to_optimized_nodes;
    // 避免有的cudafusiongroup在子block内，需要这样遍历
    find_to_optimized_nodes(block, to_optimized_nodes);
    //保险起见，再sort一遍。
    std::sort(to_optimized_nodes.begin(), to_optimized_nodes.end(), [&](torch::jit::Node* a, torch::jit::Node* b) {
        return a->isBefore(b);
    });

    //size_t i = to_optimized_nodes.size();
    for (auto iter = to_optimized_nodes.rbegin(); iter != to_optimized_nodes.rend(); iter++) {
        auto node = *iter;

        // todo:
        // 1、目前不支持scalar的输出。若强行转engine输出的是tensor，与后面scalar类型不匹配，整个模型跑不起来。
        // 2、subgraph没有输入的情况。
        // 遇到这两种情况先unmerge掉，输出scalar的情况待支持 06.20
        if (node->inputs().size() == 0) {
            LOG(WARNING) << "Subgraph: " << node_info_with_attr(node) << " has no input. unmerge it.";
            baidu::mirana::poros::unmerge_subgraph(node);
            continue;
        }
        bool node_should_be_unmerged = false;
        for (size_t i = 0; i < node->outputs().size(); i++) {
            if (node->outputs()[i]->type()->kind() != c10::TypeKind::TensorType &&
                !node->outputs()[i]->type()->isSubtypeOf(c10::ListType::ofTensors())) {
                LOG(WARNING) << "Subgraph: " << node_info_with_attr(node) << " outputs contain non-tensor or non-tensor[] values. unmerge it.";
                node_should_be_unmerged = true;
                baidu::mirana::poros::unmerge_subgraph(node);
                break;
            }
        }

        if (node_should_be_unmerged) {
            continue;
        }

        IEngine* engine = select_engine(node);
        if (engine == nullptr) {
            LOG(ERROR) << "can't find Engine for node: " << node->kind().toQualString();
            return -1;
        }
        std::shared_ptr<torch::jit::Graph> subgraph = node->g(torch::jit::attr::Subgraph);
        LOG(INFO) << "\n                     \n          ###########\n                     \n"
                << "start to optimize graph: " << node_info_with_attr(node);

        int non_constant_node_num = 0;
        auto subblock = subgraph->block();

        //engine->transform(*node, *optimized_module);
        for (auto it = subblock->nodes().begin(); it != subblock->nodes().end(); ++it) {
            if (it->kind() != torch::jit::prim::Constant) {
                non_constant_node_num ++;
                if (non_constant_node_num > _options.unconst_ops_thres) {
                    break;
                }
            }
        }
        if (non_constant_node_num <= _options.unconst_ops_thres) {
            LOG(INFO) << "subgraph size is too small, unmerge it.";
            baidu::mirana::poros::unmerge_subgraph(node);
        } else {
            if (transform(engine, *node, *optimized_module) < 0) {
                LOG(WARNING) << "transform failed, use origin sub_graph";
                if (_options.debug) {
                    GRAPH_DUMP("transform failed graph: ", subgraph);
                    return -1;
                }
            }
        }
    }

    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
        for (torch::jit::Block* ib : it->blocks()) {
            optimize_subblock(ib, optimized_module);
        }
    }
    return 0;
}

void Compiler::close() {

    for (auto&e : _engine_map) {
        delete e.second;
    }
}

int Compiler::transform(IEngine* engine, torch::jit::Node& subgraph_node,
                            torch::jit::Module& module) {

    AT_ASSERT(subgraph_node.kind() == torch::jit::prim::CudaFusionGroup);
    std::shared_ptr<torch::jit::Graph> sub_graph_copy = subgraph_node.g(torch::jit::attr::Subgraph)->copy();
    
    std::string serialized_engine;

    // 在拷贝的子图上删除无用的节点
    if (!eliminate_subgraph_useless_nodes(sub_graph_copy, subgraph_node, false)) {
        baidu::mirana::poros::unmerge_subgraph(&subgraph_node);
        return -1;
    }
    
    PorosGraph poros_graph = {sub_graph_copy.get(), &subgraph_node};
    
    //++poros_graph.allocated_index;
    //int ret = engine->transform(poros_graph, serialized_engine);
    int ret = engine->transform(poros_graph);
    if (ret < 0) {
        baidu::mirana::poros::unmerge_subgraph(&subgraph_node);
        return ret;
    }
    // 子图转换成功，删除（替换）子图输入无用的节点
    std::shared_ptr<torch::jit::Graph> sub_graph = subgraph_node.g(torch::jit::attr::Subgraph);
    eliminate_subgraph_useless_nodes(sub_graph, subgraph_node, true);

    auto parent_graph = subgraph_node.owningGraph();
    
    // 有的子图输出的tensor原本是long类型，但是trtengine只支持int类型。
    // 那么就需要在engine后添加aten::to(long)的操作将其还原回去。
    // 避免有的op会强制检查long类型（例如：aten::index）
    subgraph_outputs_int2long(parent_graph, subgraph_node, sub_graph_copy);

    
    //std::pair<uint64_t, uint64_t> num_io = engine_ptr->num_io;
    //AT_ASSERT(num_io.first == sub_graph->inputs().size());
    //AT_ASSERT(num_io.second == sub_graph->outputs().size());

    //add engine to attribute 
    std::string name = engine->who_am_i() + "_" + std::to_string(_engine_index++);
    engine->register_module_attribute(name, module);

    //get self input. it's about the module func
    torch::jit::Value* self = nullptr;
    auto first_input_c = parent_graph->inputs()[0]->type()->cast<c10::ClassType>();
    if (first_input_c->is_module()) {
        self = parent_graph->inputs()[0];
    } else {
        self = parent_graph->insertInput(0, "self");  //should set as the first input param
        self->setType(module._ivalue()->type());
    }

    torch::jit::WithInsertPoint guard(&subgraph_node);
    //build new node & remove old graph
    auto engine_node = parent_graph->createGetAttr(self, name);
    engine_node->insertBefore(&subgraph_node);

    std::vector<torch::jit::Value*> engine_inputs;
    for (auto input : subgraph_node.inputs()) {
        //TODO: consider situation that when input is not a tensor
        engine_inputs.push_back(input);
    }
    
    auto input_list_node = parent_graph->createList(c10::TensorType::get(), 
            torch::jit::ArrayRef<torch::jit::Value*>(engine_inputs));
    input_list_node->insertBefore(&subgraph_node);

    std::vector<torch::jit::Value*> execute_node_inputs;
    execute_node_inputs.push_back(input_list_node->outputs()[0]);
    execute_node_inputs.push_back(engine_node->outputs()[0]);

    auto execute_node = parent_graph->create(
        c10::Symbol::fromQualString(engine->who_am_i() + "::execute_engine"),
        torch::jit::ArrayRef<torch::jit::Value*>(execute_node_inputs),
        1);
    execute_node->insertBefore(&subgraph_node);
    execute_node->outputs()[0]->setType(c10::ListType::ofTensors());

    //auto unpack_node = parent_graph->createListUnpack(execute_node->outputs()[0], num_io.second);
    auto unpack_node = parent_graph->createListUnpack(execute_node->outputs()[0], subgraph_node.outputs().size());
    unpack_node->insertBefore(&subgraph_node);

    //AT_ASSERT(subgraph_node.outputs().size() == unpack_node->outputs().size());  
    for (size_t idx = 0; idx < unpack_node->outputs().size(); idx++) {
        subgraph_node.outputs()[idx]->replaceAllUsesWith(unpack_node->outputs()[idx]);
    }

    subgraph_node.removeAllInputs();
    subgraph_node.destroy();  //TODO: 没有清理subgraph，可能会有内存泄漏，确认一下
    return 0;
}

/**
 * @brief  compile graph
 *
 * @param [in] module : 原始module
 * @param [in] input_ivalues : 预热数据
 * @param [in] options : 参数
 * @return optimized_module
 * @retval !nullptr => succeed  nullptr => failed
 **/
std::unique_ptr<torch::jit::Module> CompileGraph(const torch::jit::Module& origin_module, 
                            const std::vector<std::vector<c10::IValue> >& prewarm_datas, 
                            const PorosOptions& options) {
    Compiler compiler;
    if (compiler.init(options) < 0) {
        return nullptr;
    }
    
    try {
        std::unique_ptr<torch::jit::Module> optimized_module(new torch::jit::Module(origin_module._ivalue()->name() + "_poros"));
        if (compiler.compile(origin_module, prewarm_datas, optimized_module.get()) < 0) {
            return nullptr;
        }

        return optimized_module;
    } catch (const c10::Error& e) {
        LOG(ERROR) << e.msg();
        return nullptr;
    }
}

std::unique_ptr<PorosModule> Compile(const torch::jit::Module& module,
        const std::vector<std::vector<c10::IValue> >& prewarm_datas,
        const PorosOptions& options) {

    auto compiled_module = CompileGraph(module, prewarm_datas, options);
    if (compiled_module) {
        std::unique_ptr<PorosModule> poros_module(new PorosModule(*compiled_module));
        poros_module->_options = options;

        if (options.device == Device::GPU) {
            poros_module->to(at::kCUDA);
        }

        if (options.debug == true) {
            // when setting this, all the INFO level will be printed
            c10::ShowLogInfoToStderr();        
        }
        return poros_module;
    } else {
        return nullptr;
    }
}

}//poros
}//mirana
}//baidu
