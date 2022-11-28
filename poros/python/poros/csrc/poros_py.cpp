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
* @file poros_py.cpp
* @author tianjinjin@baidu.com
* @date Thu Jul  1 10:25:01 CST 2021
* @brief 
**/

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "Python.h"

#include "torch/csrc/jit/python/pybind_utils.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/custom_class.h"
#include "torch/script.h"
#include "torch/torch.h"

#include "poros/compile/compile.h"

namespace py = pybind11;

namespace poros {
namespace pyapi {

torch::jit::Module compile_graph(const torch::jit::Module& mod, 
                                const py::list& input_list,
                                baidu::mirana::poros::PorosOptions& poros_option) {
    auto function_schema = mod.get_method("forward").function().getSchema();
    py::gil_scoped_acquire gil;
    std::vector<torch::jit::Stack> prewarm_datas;
    for (auto& input_tuple : input_list) {
        torch::jit::Stack stack;
        for (auto& input: input_tuple) {
            stack.push_back(torch::jit::toTypeInferredIValue(input));
        }
        prewarm_datas.push_back(stack);
    }
    
    auto poros_mod = baidu::mirana::poros::CompileGraph(mod, prewarm_datas, poros_option);
    if (poros_mod) {
        return *poros_mod;
    } else {
        throw c10::Error("comile failed", "");
    }
}

torch::jit::Module load(const std::string& filename, const baidu::mirana::poros::PorosOptions& options) {
    auto poros_mod = baidu::mirana::poros::Load(filename, options);
    return *poros_mod;
}

PYBIND11_MODULE(_C, m) {
    py::enum_<baidu::mirana::poros::Device>(m, "Device", "Enum to specify device kind to build poros Module")
        .value("GPU", baidu::mirana::poros::Device::GPU, "Spiecify using GPU as the backend of poros Module")
        .value("CPU", baidu::mirana::poros::Device::CPU, "Spiecify using CPU as the backend of poros Module")
        .value("XPU", baidu::mirana::poros::Device::XPU, "Spiecify using XPU as the backend of poros Module")
        .export_values();

    py::class_<baidu::mirana::poros::PorosOptions>(m, "PorosOptions")
        .def(py::init<>())
        .def_readwrite("device", &baidu::mirana::poros::PorosOptions::device)
        .def_readwrite("debug", &baidu::mirana::poros::PorosOptions::debug)
        .def_readwrite("use_fp16", &baidu::mirana::poros::PorosOptions::use_fp16)
        .def_readwrite("is_dynamic", &baidu::mirana::poros::PorosOptions::is_dynamic)
        .def_readwrite("long_to_int", &baidu::mirana::poros::PorosOptions::long_to_int)
        .def_readwrite("device_id", &baidu::mirana::poros::PorosOptions::device_id)
        .def_readwrite("max_workspace_size", &baidu::mirana::poros::PorosOptions::max_workspace_size)
        .def_readwrite("unconst_ops_thres", &baidu::mirana::poros::PorosOptions::unconst_ops_thres)
        .def_readwrite("use_nvidia_tf32", &baidu::mirana::poros::PorosOptions::use_nvidia_tf32)
        .def_readwrite("preprocess_mode", &baidu::mirana::poros::PorosOptions::preprocess_mode)
        .def_readwrite("unsupport_op_list", &baidu::mirana::poros::PorosOptions::unsupport_op_list);
        

    m.doc() = "Poros C Bindings";
    m.def(
        "compile_graph",
        &poros::pyapi::compile_graph,
        "compile a PyTorch module and return a Poros module \
        that can significantly lower the inference latency");
    m.def(
        "load",
        &poros::pyapi::load,
        "load poros model");
}

} // namespace pyapi
} // namespace poros
