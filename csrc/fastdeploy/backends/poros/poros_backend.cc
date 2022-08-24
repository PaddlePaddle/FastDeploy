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
#ifdef ENABLE_TRT_BACKEND
    _options.is_dynamic = option.trt_option.max_shape.empty() ? false : true;
    _options.max_workspace_size = option.trt_option.max_workspace_size;
    _options.use_fp16 = option.trt_option.enable_fp16;
    if (_options.is_dynamic) {
        std::vector<int64_t> min_shape;
        std::vector<int64_t> opt_shape;
        std::vector<int64_t> max_shape;
        for (auto iter:option.trt_option.min_shape) {
            auto max_iter = option.trt_option.max_shape.find(iter.first);
            auto opt_iter = option.trt_option.opt_shape.find(iter.first);
            FDASSERT(max_iter != option.trt_option.max_shape.end(), "Cannot find " + iter.first + " in TrtBackendOption::max_shape.");
            FDASSERT(opt_iter != option.trt_option.opt_shape.end(), "Cannot find " + iter.first + " in TrtBackendOption::opt_shape.");
            min_shape.assign(iter.second.begin(), iter.second.end());
            opt_shape.assign(opt_iter.second.begin(), opt_iter.second.end());
            max_shape.assign(max_iter.second.begin(), max_iter.second.end());
        }
        //min
        std::vector<torch::jit::IValue> inputs_min;
        if (option.use_gpu) {
            inputs_min.push_back(at::randn(min_shape, {at::kCUDA}));
        } else{
            inputs_min.push_back(at::randn(min_shape, {at::kCPU}));
        }
        _prewarm_datas.push_back(inputs_min);
        //opt
        std::vector<torch::jit::IValue> inputs_opt;
        if (option.use_gpu) {
            inputs_opt.push_back(at::randn(opt_shape, {at::kCUDA}));
        } else {
            inputs_opt.push_back(at::randn(opt_shape, {at::kCPU}));
        }
        _prewarm_datas.push_back(inputs_opt);
        //max
        std::vector<torch::jit::IValue> inputs_max;
        if (option.use_gpu) {
            inputs_max.push_back(at::randn(max_shape, {at::kCUDA}));
        } else {
            inputs_max.push_back(at::randn(max_shape, {at::kCPU}));
        }
        _prewarm_datas.push_back(inputs_max);
    }
    else {
        std::vector<int64_t> min_shape;
        for (auto iter:option.trt_option.min_shape) {
            min_shape.assign(iter.second.begin(), iter.second.end());
        }
        //min
        std::vector<torch::jit::IValue> inputs_min;    
        if (option.use_gpu) {
            inputs_min.push_back(at::randn(min_shape, {at::kCUDA}));
        } else{
            inputs_min.push_back(at::randn(min_shape, {at::kCPU}));
        }
        _prewarm_datas.push_back(inputs_min);
    }
#endif
    return;
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
    _poros_module = baidu::mirana::poros::Compile(mod, _prewarm_datas, _options);
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
    // Infer
    auto poros_outputs = _poros_module->forward(poros_inputs).toTensorList();
    // Convert PyTorch Tensor to FD Tensor
    for (size_t i = 0; i < poros_outputs.size(); ++i) {
        CopyTensorToCpu(poros_outputs[i], &((*outputs)[i]));
    }
    return true;
}

}  // namespace fastdeploy