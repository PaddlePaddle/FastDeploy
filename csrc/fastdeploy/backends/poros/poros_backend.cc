// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/backends/poros/poros_backend.h"
#include <sys/time.h>

namespace fastdeploy {

TensorInfo PorosBackend::GetInputInfo(int index) {
    // eager mode cann't obtain input information before infer
    TensorInfo info_input;
    return info_input;
}

TensorInfo PorosBackend::GetOutputInfo(int index) {
    // eager mode cann't obtain output information before infer
    TensorInfo info_output;
    return info_output;
}

void PorosBackend::BuildOption(const PorosBackendOption& option) {
    _options.device = option.use_gpu ? baidu::mirana::poros::Device::GPU : baidu::mirana::poros::Device::CPU;
    _options.long_to_int = option.long_to_int;
    _options.use_nvidia_tf32 = option.use_nvidia_tf32;
    _options.device_id = option.gpu_id;
    _options.unconst_ops_thres = option.unconst_ops_thres;
    std::cout << "test_wjj_option: " << option.is_dynamic << std::endl;
    // _options.is_dynamic = option.max_shape.empty() ? false : true;
    _options.is_dynamic = option.is_dynamic;
    std::cout << "test_wjj_option: " << _options.is_dynamic << std::endl;
    _options.is_dynamic = true;
    _options.max_workspace_size = option.max_workspace_size;
    _options.use_fp16 = option.enable_fp16;
    // int input_index = 0;
    // if (_options.is_dynamic) {
    //     std::vector<torch::jit::IValue> inputs_min;
    //     std::vector<torch::jit::IValue> inputs_opt;
    //     std::vector<torch::jit::IValue> inputs_max;
    //     for (auto iter = option.min_shape.begin(); iter != option.min_shape.end(); ++iter) {
    //         std::vector<int64_t> min_shape;
    //         std::vector<int64_t> opt_shape;
    //         std::vector<int64_t> max_shape;
    //         auto prewarm_dtype = GetPorosDtype(option.prewarm_datatypes[input_index]);
    //         auto max_iter = option.max_shape.find(iter->first);
    //         auto opt_iter = option.opt_shape.find(iter->first);
    //         FDASSERT(max_iter != option.max_shape.end(), "Cannot find " + iter->first + " in TrtBackendOption::max_shape.");
    //         FDASSERT(opt_iter != option.opt_shape.end(), "Cannot find " + iter->first + " in TrtBackendOption::opt_shape.");
    //         min_shape.assign(iter->second.begin(), iter->second.end());
    //         opt_shape.assign(opt_iter->second.begin(), opt_iter->second.end());
    //         max_shape.assign(max_iter->second.begin(), max_iter->second.end());
    //         //min
    //         if (option.use_gpu) {
    //             std::cout << "test_wjj0000000: " << min_shape[0] << " " << min_shape[1] << std::endl;
    //             auto min_tensor = at::ones(min_shape, {at::kCUDA}).to(prewarm_dtype);
    //             std::cout << min_tensor << std::endl;
    //             inputs_min.push_back(min_tensor);
    //             // inputs_min.push_back(at::randint(1, 10, min_shape, {at::kCUDA}));
    //         } else{
    //             std::cout << "test_wjj0000000: " << min_shape[0] << " " << min_shape[1] << std::endl;
    //             inputs_min.push_back(at::randn(min_shape, {at::kCPU}).to(prewarm_dtype));
    //         }
    //         //opt
    //         if (option.use_gpu) {
    //             std::cout << "test_wjj1111111: " << opt_shape[0] << " " << opt_shape[1] << std::endl;
    //             auto opt_tensor = at::ones(opt_shape, {at::kCUDA}).to(prewarm_dtype);
    //             inputs_opt.push_back(opt_tensor);
    //             // inputs_opt.push_back(at::randint(1, 10, opt_shape, {at::kCUDA}));
    //         } else {
    //             std::cout << "test_wjj1111111: " << opt_shape[0] << " " << opt_shape[1] << std::endl;
    //             inputs_opt.push_back(at::randn(opt_shape, {at::kCPU}).to(prewarm_dtype));
    //         }
    //         //max
    //         if (option.use_gpu) {
    //             std::cout << "test_wjj2222222: " << max_shape[0] << " " << max_shape[1] << std::endl;
    //             auto max_tensor = at::ones(max_shape, {at::kCUDA}).to(prewarm_dtype);
    //             inputs_max.push_back(max_tensor);
    //             // inputs_max.push_back(at::randint(1, 10, max_shape, {at::kCUDA}));
    //         } else {
    //             std::cout << "test_wjj2222222: " << max_shape[0] << " " << max_shape[1] << std::endl;
    //             inputs_max.push_back(at::randn(max_shape, {at::kCPU}).to(prewarm_dtype));
    //         }
    //         input_index += 1;
    //     }
    //     std::cout << "test_wjj4444444: " << inputs_max.size() << std::endl;
    //     std::cout << "test_wjj5555555: " << inputs_min.size() << std::endl;
    //     std::cout << "test_wjj6666666: " << inputs_opt.size() << std::endl;
    //     _prewarm_datas.push_back(inputs_max);
    //     _prewarm_datas.push_back(inputs_min);
    //     _prewarm_datas.push_back(inputs_opt);
    // }
    // else {
    //     std::vector<torch::jit::IValue> inputs_min;    
    //     for (auto iter:option.min_shape) {
    //         auto prewarm_dtype = GetPorosDtype(option.prewarm_datatypes[input_index]);
    //         std::vector<int64_t> min_shape;
    //         min_shape.assign(iter.second.begin(), iter.second.end());
    //         //min
    //         if (option.use_gpu) {
    //             inputs_min.push_back(at::randn(min_shape, {at::kCUDA}).to(prewarm_dtype));
    //         } else{
    //             inputs_min.push_back(at::randn(min_shape, {at::kCPU}).to(prewarm_dtype));
    //         }
    //         input_index += 1;
    //     }
    //     _prewarm_datas.push_back(inputs_min);
    // }
    return;
}

bool PorosBackend::Compile(const std::string& model_file, std::vector<std::vector<FDTensor>>& prewarm_tensors, const PorosBackendOption& option) {
    if (initialized_) {
        FDERROR << "PorosBackend is already initlized, cannot initialize again."
                << std::endl;
        return false;
    }
    BuildOption(option);
    torch::jit::Module mod;
    mod = torch::jit::load(model_file);
    mod.eval();
    if (option.use_gpu) {
        mod.to(at::kCUDA);
    } else {
        mod.to(at::kCPU);
    }
    // get inputs_nums and outputs_nums
    auto graph = mod.get_method("forward").graph();
    auto inputs = graph->inputs();
    // remove self node
    _numinputs = inputs.size() - 1;
    // TODO:tuple to solve
    auto outputs = graph->outputs();
    _numoutputs = outputs.size();
    std::cout << "test_wjj7777777777: " << _numinputs << std::endl;
    std::cout << "test_wjj8888888888: " << _numoutputs << std::endl;
    // FDTensor to at::Tensor
    std::vector<std::vector<c10::IValue>> prewarm_datas;
    bool is_backend_cuda = option.use_gpu ? true : false;
    std::cout << "test_wjj999999999: " << is_backend_cuda << std::endl;
    std::cout << "test_wjj_prewarm0: " << prewarm_tensors.size() << prewarm_tensors[0].size() << std::endl;
    for (size_t i = 0; i < prewarm_tensors.size(); ++i) {
        std::vector<c10::IValue> prewarm_data;
        for (size_t j = 0; j < prewarm_tensors[i].size(); ++j) {
            auto tensor = CreatePorosValue(prewarm_tensors[i][j], is_backend_cuda);
            prewarm_data.push_back(tensor);
        }
        prewarm_datas.push_back(prewarm_data);
    }
    std::cout << "test_wjj_prewarm1: " << prewarm_datas.size() << std::endl;
    std::cout << "test_wjj_prewarm2: " << prewarm_datas[0].size() << prewarm_datas[1].size() << prewarm_datas[2].size() << std::endl;
    std::cout << "test_wjj=============: " << _options.is_dynamic << std::endl;
    _poros_module = baidu::mirana::poros::Compile(mod, prewarm_datas, _options);
    std::cout << "test_wjjfinished: " << std::endl;
    if (_poros_module == nullptr) {
        FDERROR << "PorosBackend initlize Failed, try initialize again."
                << std::endl;
        return false;
    }
    initialized_ = true;
    return true;
}

bool PorosBackend::InitFromTorchscript(const std::string& model_file, const PorosBackendOption& option) {
    if (initialized_) {
        FDERROR << "PorosBackend is already initlized, cannot initialize again."
                << std::endl;
        return false;
    }
    if (option.poros_file != "") {
        std::ifstream fin(option.poros_file, std::ios::binary | std::ios::in);
        if (fin) {
            FDINFO << "Detect compiled Poros file in "
                    << option.poros_file << ", will load it directly."
                    << std::endl;
            fin.close();
            return InitFromPoros(option.poros_file, option);
        }
    }
    BuildOption(option);
    torch::jit::Module mod;
    mod = torch::jit::load(model_file);
    mod.eval();
    if (option.use_gpu) {
        mod.to(at::kCUDA);
    } else {
        mod.to(at::kCPU);
    }
    // get inputs_nums and outputs_nums
    auto graph = mod.get_method("forward").graph();
    auto inputs = graph->inputs();
    // remove self node
    _numinputs = inputs.size() - 1;
    auto outputs = graph->outputs();
    _numoutputs = outputs.size();
    std::cout << "test_wjj7777777777: " << _numinputs << std::endl;
    std::cout << "test_wjj8888888888: " << _numoutputs << std::endl;
    _poros_module = baidu::mirana::poros::Compile(mod, _prewarm_datas, _options);
    std::cout << "test_wjj9999999999: " << std::endl;
    if (_poros_module == nullptr) {
        FDERROR << "PorosBackend initlize Failed, try initialize again."
                << std::endl;
        return false;
    }
    initialized_ = true;
    return true;
}

bool PorosBackend::InitFromPoros(const std::string& model_file, const PorosBackendOption& option) {
    if (initialized_) {
        FDERROR << "PorosBackend is already initlized, cannot initialize again."
                << std::endl;
        return false;
    }
    BuildOption(option);
    _poros_module = baidu::mirana::poros::Load(model_file, _options);
    if (_poros_module == nullptr) {
        FDERROR << "PorosBackend initlize Failed, try initialize again."
                << std::endl;
        return false;
    }
    // get inputs_nums and outputs_nums
    auto graph = _poros_module->get_method("forward").graph();
    auto inputs = graph->inputs();
    // remove self node
    _numinputs = inputs.size() - 1;
    auto outputs = graph->outputs();
    _numoutputs = outputs.size();
    initialized_ = true;
    return true;
}

bool PorosBackend::Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs) {
    // Convert FD Tensor to PyTorch Tensor
    std::vector<torch::jit::IValue> poros_inputs;
    bool is_backend_cuda = _options.device == baidu::mirana::poros::Device::GPU ? true : false; 
    for (size_t i = 0; i < inputs.size(); ++i) {
        poros_inputs.push_back(CreatePorosValue(inputs[i], is_backend_cuda));
    }
    std::cout << "test_wjj1111111111: " << poros_inputs.size() << std::endl;
    // Infer
    auto poros_outputs = _poros_module->forward(poros_inputs);
    std::cout << "test_wjj222222222: " << std::endl;
    std::cout << "test_wjj : " << poros_outputs.isTensor() << std::endl;
    std::cout << "test_wjj : " << poros_outputs.isList() << std::endl;
    std::cout << "test_wjj : " << poros_outputs.isTuple() << std::endl;
    // Convert PyTorch Tensor to FD Tensor
    if (poros_outputs.isTensor()) {
        std::cout << "test_wjj++++++: " << std::endl;
        CopyTensorToCpu(poros_outputs.toTensor().to(at::kCPU), &((*outputs)[0]));
        std::cout << "test_wjj------: " << std::endl;
    } else if (poros_outputs.isTuple()) {
        // deal with multi outputs
        auto poros_outputs_list = poros_outputs.toTuple();
        size_t index = 0;
        for (size_t i = 0; i < poros_outputs_list->elements().size(); ++i) {
            std::cout << "test_wjj33333: " << poros_outputs_list->elements()[i].isTensor() << std::endl;
            std::cout << "test_wjj33333: " << poros_outputs_list->elements()[i].isList() << std::endl;
            std::cout << "test_wjj33333: " << poros_outputs_list->elements()[i].isTuple() << std::endl;
            auto poros_tensor = poros_outputs_list->elements()[i];
            if (poros_tensor.isTensor()) {
                CopyTensorToCpu(poros_tensor.toTensor().to(at::kCPU), &((*outputs)[index]));
                index += 1;
            } else if (poros_tensor.isList()) {
                auto poros_tensor_list = poros_tensor.toList();
                // size_t tensor_size = poros_tensor_list.size();
                // for (size_t j = 0; j < tensor_size; ++j) {
                //     CopyTensorToCpu(poros_tensor_list.get(j).to(at::kCPU), &((*outputs)[index]));
                //     index += 1;
                // }
                for (const auto list_idx : c10::irange(0, poros_tensor_list.size())) {
                    const auto& elt = poros_tensor_list.get(list_idx);
                    CopyTensorToCpu(elt.toTensor().to(at::kCPU), &((*outputs)[index]));
                    index += 1;
                }
            } else if (poros_tensor.isTuple()) {
                auto poros_tensor_tuple = poros_tensor.toTuple();
                for (size_t j = 0; j < poros_tensor_tuple->elements().size(); ++j) {
                    CopyTensorToCpu(poros_tensor_tuple->elements()[j].toTensor().to(at::kCPU), &((*outputs)[index]));
                    index += 1;
                }
            } else {
                continue;
            }
            std::cout << "test_wjj finished!!! " << std::endl;
        }
    } else {
        FDERROR << "Convert to FDTensor Failed!!!!!" << std::endl;
    }
    return true;
}

}  // namespace fastdeploy