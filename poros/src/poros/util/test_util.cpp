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
* @file test_util.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include "test_util.h"

#include <cuda_runtime.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include "poros/compile/graph_prewarm.h"
#include "poros/context/poros_global.h"
#include "poros/engine/iengine.h"
#include "poros/iplugin/plugin_create.h"

namespace baidu {
namespace mirana {
namespace poros {
namespace testutil {

static inline void clone_tensor_vector(const std::vector<at::Tensor> &old_vec, std::vector<at::Tensor> &new_vec) {
    for (auto i: old_vec) {
        new_vec.push_back(i.clone());
    }
}

static std::string get_engine_name(const baidu::mirana::poros::PorosOptions &poros_option) {
    std::string engine_name("");
    if (poros_option.device == Device::GPU) {
        engine_name = "TensorrtEngine";
    } else if (poros_option.device == Device::XPU) {
        engine_name = "XtclEngine";
    } else {
        engine_name = "";
    }
    return engine_name;
}

static bool write_to_log(const std::string &log_path, const std::string &inform) {
    if (log_path.empty()) {
        return true;
    } else {
        std::ofstream log_file(log_path, std::ios::app);
        if (log_file) {
            log_file << inform;
            log_file.close();
            return true;
        } else {
            return false;
        }
    }
};

static baidu::mirana::poros::IEngine *
select_engine(const torch::jit::Node *n, const baidu::mirana::poros::PorosOptions &poros_option) {
    baidu::mirana::poros::IEngine *engine(nullptr);
    if (n == nullptr || n->kind() != torch::jit::prim::CudaFusionGroup) {
        return nullptr;
    }
    std::string engine_name = get_engine_name(poros_option);
    if (engine_name.empty()) {
        return nullptr;
    }
    engine = dynamic_cast<baidu::mirana::poros::IEngine *>(create_plugin(engine_name, \
    baidu::mirana::poros::PorosGlobalContext::instance()._engine_creator_map));
    if (engine == nullptr || engine->init() < 0) {
        return nullptr;
    }
    return engine;
};

static bool is_node_supported(const torch::jit::Node *node, const baidu::mirana::poros::PorosOptions &poros_option) {
    std::string engine_name = get_engine_name(poros_option);
    auto converter_map = baidu::mirana::poros::PorosGlobalContext::instance().get_converter_map(engine_name);
    if (converter_map != nullptr && converter_map->node_converterable(node)) {
        LOG(INFO) << "supported node: " << node->kind().toQualString();
        return true;
    } else {
        if (node->kind() != torch::jit::prim::Loop &&
            node->kind() != torch::jit::prim::If &&
            node->kind() != torch::jit::prim::CudaFusionGroup) {
            LOG(WARNING) << "not supported node: " << node->kind().toQualString()
                         << ", detail info: " << *node;
        }
        return false;
    }
}

static std::vector<at::Tensor> run_graph(const std::shared_ptr<torch::jit::Graph> &graph,
                                         const std::vector<at::Tensor> &input_data,
                                         const baidu::mirana::poros::PorosOptions &poros_option,
                                         const std::string &log_path) {
    std::vector<at::Tensor> graph_output;
    // 构建graph exe
    std::string function_name("test tensor");
    torch::jit::GraphExecutor graph_exe(graph, function_name);
    // 将输入导入ivalue vector
    std::vector<c10::IValue> graph_input;
    for (size_t i = 0; i < input_data.size(); i++) {
        graph_input.push_back(input_data[i]);
    }
    // 执行graph
    std::clock_t start, end;
    if (poros_option.device == Device::GPU) {
        cudaDeviceSynchronize();
    }
    start = std::clock();
    graph_exe.run(graph_input);
    if (poros_option.device == Device::GPU) {
        cudaDeviceSynchronize();
    }
    end = std::clock();
    std::string log_inform = "graph time:" + std::to_string(double(end - start) / CLOCKS_PER_SEC * 1000.0) + "ms\t";
    std::cout << log_inform;
    if (!write_to_log(log_path, log_inform)) {
        LOG(WARNING) << "write to log failed";
    }
    // 提取结果
    for (size_t i = 0; i < graph_input.size(); i++) {
        auto tmp_ivalue = graph_input[i];
        graph_output.push_back(tmp_ivalue.toTensor());
    }
    return graph_output;
};

static const std::vector<torch::jit::NodeKind> has_constant_tensor_inputs_node() {
    return {torch::jit::aten::batch_norm,
            torch::jit::aten::_convolution,
            torch::jit::aten::conv1d,
            torch::jit::aten::conv2d,
            torch::jit::aten::conv3d,
            torch::jit::aten::layer_norm,
            torch::jit::aten::lstm,
            torch::jit::aten::group_norm,
            torch::jit::aten::instance_norm};
}

static std::vector<at::Tensor> run_engine(std::shared_ptr<torch::jit::Graph> &graph,
                                          const baidu::mirana::poros::PorosOptions &poros_option,
                                          baidu::mirana::poros::IConverter *converter,
                                          const std::vector<at::Tensor> &input_data,
                                          const std::string &log_path,
                                          const std::vector<std::vector<at::Tensor>> &prewarm_data_of_engine) {
    std::vector<at::Tensor> engine_output;
    // 避免inplace的op,prewarm后会更改原数据,例如:add_,所以先clone
    PorosGlobalContext::instance()._value_dynamic_shape_map = {};

    PorosGlobalContext::instance().set_poros_options(poros_option);
    std::vector<std::vector<at::Tensor>> prewarm_clone;
    for (auto it = prewarm_data_of_engine.begin(); it != prewarm_data_of_engine.end(); ++it) {
        std::vector<at::Tensor> input_clone;
        clone_tensor_vector(*it, input_clone);
        prewarm_clone.push_back(input_clone);
    }
    std::vector<std::vector<c10::IValue>> prewarm_datas;
    for (auto it = prewarm_clone.begin(); it != prewarm_clone.end(); ++it) {
        std::vector<c10::IValue> prewarm_input_data;
        for (size_t i = 0; i < (*it).size(); i++) {
            prewarm_input_data.push_back((*it)[i]);
        }
        prewarm_datas.push_back(prewarm_input_data);
    }

    // 检查graph中是否包含待converter的node
    bool converter_node_exist = false;
    torch::jit::Node *converter_node = nullptr;;
    std::string converter_node_kind_name;
    for (auto node_it: graph->nodes()) {
        for (auto converter_node_kind: converter->node_kind()) { // compare node kind
            if (node_it->kind().toQualString() == converter_node_kind.toQualString()
                && is_node_supported(node_it, poros_option)) {
                converter_node_exist = true;
                converter_node_kind_name = node_it->kind().toQualString();
                converter_node = node_it;
                break;
            }
        }
    }
    if (!converter_node_exist) {
        LOG(WARNING) << "Can't find converter node.";
        return engine_output;
    }

    // 判断是否是batchnormal类型
    bool convter_has_constant_tensor_inputs = false;
    for (auto node_kind_it: has_constant_tensor_inputs_node()) {
        if (node_kind_it.toQualString() == converter_node->kind().toQualString()) {
            convter_has_constant_tensor_inputs = true;
            break;
        }
    }

    // 插入constant tensor inputs
    if (convter_has_constant_tensor_inputs) {
        torch::jit::WithInsertPoint guard(graph->block()->nodes().front());
        if (converter_node->kind().toQualString() == has_constant_tensor_inputs_node()[6].toQualString()) {
            auto lt = c10::List<at::Tensor>({});
            for (size_t i = 3; i < prewarm_clone[0].size(); i++) {
                lt.push_back(prewarm_clone[0][i]);
            }
            auto lt_ivalue = c10::IValue(lt);
            auto len_const = graph->insertConstant(lt_ivalue);
            converter_node->replaceInput(2, len_const);
        } else {
            for (size_t i = 1; i < prewarm_datas[0].size(); i++) {
                auto len_const = graph->insertConstant(prewarm_datas[0][i]);
                if (converter_node->kind().toQualString() == has_constant_tensor_inputs_node()[5].toQualString()
                    || converter_node->kind().toQualString() == has_constant_tensor_inputs_node()[7].toQualString()) {
                    converter_node->replaceInput(i + 1, len_const);
                } else {
                    converter_node->replaceInput(i, len_const);
                }
            }
        }
    }
    // 得到预热图
    auto prewarm_graph = baidu::mirana::poros::graph_prewarm(graph, prewarm_datas);

    // 将graph全部加入subgraph
    torch::jit::Node *subgraph_node = torch::jit::SubgraphUtils::createSingletonSubgraph(
            *(prewarm_graph->nodes().begin()),
            torch::jit::prim::CudaFusionGroup);
    auto node_it = ++prewarm_graph->nodes().begin();
    while (node_it != prewarm_graph->nodes().end()) {
        torch::jit::SubgraphUtils::mergeNodeIntoSubgraph(*node_it, subgraph_node);
        node_it = ++prewarm_graph->nodes().begin();
    }

    // 选取与初始化engine
    baidu::mirana::poros::IEngine *engine = select_engine(subgraph_node, poros_option);
    if (engine == nullptr) {
        LOG(WARNING) << "select engine failed";
        return engine_output;
    }

    // 将graph转到engine(包括op替换)
    std::shared_ptr<torch::jit::Graph> sub_graph = subgraph_node->g(torch::jit::attr::Subgraph);
    baidu::mirana::poros::PorosGraph poros_graph = {sub_graph.get(), subgraph_node};
    if (engine->transform(poros_graph) < 0) {
        LOG(WARNING) << "engine transform failed";
        return engine_output;
    }

    // 测试engine输出
    std::clock_t start, end;
    if (poros_option.device == Device::GPU) {
        cudaDeviceSynchronize();
    }
    start = std::clock();
    if (convter_has_constant_tensor_inputs) {
        std::vector<at::Tensor> input_data_without_constant;
        input_data_without_constant.push_back(input_data[0].clone());
        if (converter_node->kind().toQualString() == has_constant_tensor_inputs_node()[6].toQualString()) {
            input_data_without_constant.push_back(input_data[1].clone());
            input_data_without_constant.push_back(input_data[2].clone());
        }
        engine_output = engine->excute_engine(input_data_without_constant);
    } else {
        engine_output = engine->excute_engine(input_data);
    }
    if (poros_option.device == Device::GPU) {
        cudaDeviceSynchronize();
    }
    end = std::clock();
    std::string log_inform = "engine time:" + std::to_string(double(end - start) / CLOCKS_PER_SEC * 1000.0) + "ms\t" +
                             converter_node_kind_name + "\n";

    std::cout << log_inform;

    if (!write_to_log(log_path, log_inform)) {
        LOG(WARNING) << "write to log failed";
    }
    return engine_output;
};

bool run_graph_and_poros(const std::string &graph_IR,
                         const baidu::mirana::poros::PorosOptions &poros_option,
                         baidu::mirana::poros::IConverter *converter,
                         const std::vector<at::Tensor> &input_data,
                         std::vector<at::Tensor> &graph_output,
                         std::vector<at::Tensor> &poros_output,
                         const std::vector<std::vector<at::Tensor>> *prewarm_data,
                         std::string log_path,
                         const std::vector<size_t> const_input_indices) {
    try {
        // 解析graph
        auto graph = std::make_shared<torch::jit::Graph>();
        torch::jit::parseIR(graph_IR, graph.get());
        std::vector<at::Tensor> real_input;
        clone_tensor_vector(input_data, real_input);

        if (!const_input_indices.empty()) {
            torch::jit::WithInsertPoint guard(graph->block()->nodes().front());
            for (const size_t &index : const_input_indices) {
                graph->inputs()[index]->replaceAllUsesWith(graph->insertConstant(input_data[index]));
            }
            for (auto it = const_input_indices.rbegin(); it != const_input_indices.rend(); it++) {
                graph->eraseInput(*it);
                real_input.erase(real_input.begin() + *it);
            }
        }
        // 运行原始graph
        std::vector<at::Tensor> input_of_graph;
        clone_tensor_vector(real_input, input_of_graph);
        graph_output = run_graph(graph, input_of_graph, poros_option, log_path);

        // convert op并运行engine
        std::vector<at::Tensor> input_of_engine;
        clone_tensor_vector(real_input, input_of_engine);

        // 准备预热数据
        std::vector<std::vector<at::Tensor>> prewarm_data_of_engine;
        if (prewarm_data == nullptr) {
            prewarm_data_of_engine.push_back(input_of_engine);
        } else {
            for (size_t i = 0; i < (*prewarm_data).size(); ++i) {
                std::vector<at::Tensor> tmp_clone_data;
                clone_tensor_vector((*prewarm_data)[i], tmp_clone_data);
                //讲道理，不应该出现这个情况，预防万一...
                if (!const_input_indices.empty() && tmp_clone_data.size() == input_data.size()) {
                    for (auto it = const_input_indices.rbegin(); it != const_input_indices.rend(); it++) {
                        tmp_clone_data.erase(tmp_clone_data.begin() + *it);
                    }
                }
                prewarm_data_of_engine.push_back(tmp_clone_data);
            }
        }
        poros_output = run_engine(graph, poros_option, converter, input_of_engine, log_path, prewarm_data_of_engine);
    } catch (const char* err) {
        LOG(ERROR) <<  " Exception: " << err;
        return false;
    }
    return true;
}


bool almost_equal(const at::Tensor &a, const at::Tensor &b, const float &threshold) {
    auto a_float = a.toType(at::kFloat);
    auto b_float = b.toType(at::kFloat);
    double maxValue = 0.0;
    maxValue = fmax(a.abs().max().item<float>(), maxValue);
    maxValue = fmax(b.abs().max().item<float>(), maxValue);
    at::Tensor diff = a_float - b_float;
    return diff.abs().max().item<float>() <= threshold * maxValue;
}

}// namespace testutil
}// namespace poros
}// namespace mirana
}// namespace baidu
