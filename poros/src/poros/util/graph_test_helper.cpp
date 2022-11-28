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
#include "graph_test_helper.h"

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace baidu {
namespace mirana {
namespace poros {
namespace graphtester {

static inline void clone_tensor_vector(const std::vector<at::Tensor> &old_vec, std::vector<at::Tensor> &new_vec) {
    for (auto i: old_vec) {
        new_vec.push_back(i.clone());
    }
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
    start = std::clock();
    graph_exe.run(graph_input);
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


std::vector<at::Tensor>
replace_input_tensor_to_constant(std::shared_ptr<torch::jit::Graph> graph, const std::vector<at::IValue> &input_data,
                                 const std::vector<InputTypeEnum> &input_data_type_mask) {
    torch::jit::WithInsertPoint guard(graph->block()->nodes().front());
    std::vector<at::Tensor> graph_input_tensor;
    std::vector<size_t> eraseInputIdx;
    for (size_t i = 0; i < input_data_type_mask.size() && i < graph->inputs().size() && i < input_data.size(); i++) {
        switch (input_data_type_mask[i]) {
            case InputTensor: //正常输入Tensor
                graph_input_tensor.push_back(input_data[i].toTensor());
                break;
            case ConstantTensor: //op的weights和bias
                graph->inputs()[i]->replaceAllUsesWith(graph->insertConstant(input_data[i]));
                eraseInputIdx.push_back(i);
                break;
            case ConstantIntVector: // int[] = prim::Constant[value=[1, 1, 1]]()
                graph->inputs()[i]->replaceAllUsesWith(graph->insertConstant(input_data[i].toIntList()));
                eraseInputIdx.push_back(i);
                break;
        }
    }
    // 从后向前删除多余的input
    for (auto it = eraseInputIdx.rbegin(); it != eraseInputIdx.rend(); it++) {
        graph->eraseInput(*it);
    }

    return graph_input_tensor;
}

bool run_graph_and_fused_graph(const std::string &graph_IR,
                               const baidu::mirana::poros::PorosOptions &poros_option,
                               std::shared_ptr<baidu::mirana::poros::IFuser> fuser,
                               const std::vector<at::IValue> &input_data,
                               const std::vector<InputTypeEnum> &input_data_type_mask,
                               std::vector<at::Tensor> &original_graph_output,
                               std::vector<at::Tensor> &fused_graph_output,
                               std::string log_path) {
    try {
        fuser->reset();
        // 解析graph
        auto graph = std::make_shared<torch::jit::Graph>();
        torch::jit::parseIR(graph_IR, graph.get());
        auto input_tensor = replace_input_tensor_to_constant(graph, input_data, input_data_type_mask);
        // 冷启动运行原始graph
        std::vector<at::Tensor> input_of_replaced_graph;
        clone_tensor_vector(input_tensor, input_of_replaced_graph);
        std::cout << "input replaced ";
        original_graph_output = run_graph(graph, input_of_replaced_graph, poros_option, log_path);

        // 运行常量化后的graph
        std::vector<at::Tensor> input_of_ori_graph;
        clone_tensor_vector(input_tensor, input_of_ori_graph);
        std::cout << std::endl << "original ";
        original_graph_output = run_graph(graph, input_of_ori_graph, poros_option, log_path);

        // 运行fuse后的graph
        std::vector<at::Tensor> input_of_fused_graph;
        clone_tensor_vector(input_tensor, input_of_fused_graph);
        fuser->fuse(graph);
        std::cout << std::endl << "op fused ";
        fused_graph_output = run_graph(graph, input_of_fused_graph, poros_option, log_path);
        std::cout << std::endl << fuser->info() << std::endl << std::endl;
    } catch (...) {
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

}// namespace graphtester
}// namespace poros
}// namespace mirana
}// namespace baidu
