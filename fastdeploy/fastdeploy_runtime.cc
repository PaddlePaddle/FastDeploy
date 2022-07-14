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

#include "fastdeploy/fastdeploy_runtime.h"
#include "fastdeploy/utils/utils.h"
#ifdef ENABLE_ORT_BACKEND
#include "fastdeploy/backends/ort/ort_backend.h"
#endif

#ifdef ENABLE_TRT_BACKEND
#include "fastdeploy/backends/tensorrt/trt_backend.h"
#endif

namespace fastdeploy {

std::vector<Backend> GetAvailableBackends() {
  std::vector<Backend> backends;
#ifdef ENABLE_ORT_BACKEND
  backends.push_back(Backend::ORT);
#endif
#ifdef ENABLE_TRT_BACKEND
  backends.push_back(Backend::TRT);
#endif
  return backends;
}

bool IsBackendAvailable(const Backend& backend) {
  std::vector<Backend> backends = GetAvailableBackends();
  for (size_t i = 0; i < backends.size(); ++i) {
    if (backend == backends[i]) {
      return true;
    }
  }
  return false;
}

bool ModelFormatCheck(const std::string& model_file,
                      const Frontend& model_format) {
  if (model_format == Frontend::PADDLE) {
    if (model_file.size() < 8 ||
        model_file.substr(model_file.size() - 8, 8) != ".pdmodel") {
      FDLogger() << "With model format of Frontend::PADDLE, the model file "
                    "should ends with `.pdmodel`, but now it's "
                 << model_file << std::endl;
      return false;
    }
  } else if (model_format == Frontend::ONNX) {
    if (model_file.size() < 5 ||
        model_file.substr(model_file.size() - 5, 5) != ".onnx") {
      FDLogger() << "With model format of Frontend::ONNX, the model file "
                    "should ends with `.onnx`, but now it's "
                 << model_file << std::endl;
      return false;
    }
  } else {
    FDLogger() << "Only support model format with frontend Frontend::PADDLE / "
                  "Frontend::ONNX."
               << std::endl;
    return false;
  }
  return true;
}

bool Runtime::Init(const RuntimeOption& _option) {
  option = _option;
  if (option.backend == Backend::ORT) {
    CreateOrtBackend();
  } else if (option.backend == Backend::TRT) {
    CreateTrtBackend();
  } else {
    FDERROR << "Runtime only support Backend::ORT/Backend::TRT as backend now."
            << std::endl;
    return false;
  }
  return true;
}

TensorInfo Runtime::GetInputInfo(int index) {
  return backend_->GetInputInfo(index);
}

TensorInfo Runtime::GetOutputInfo(int index) {
  return backend_->GetOutputInfo(index);
}

bool Runtime::Infer(std::vector<FDTensor>& input_tensors,
                    std::vector<FDTensor>* output_tensors) {
  return backend_->Infer(input_tensors, output_tensors);
}

void Runtime::CreateOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  auto ort_option = OrtBackendOption();
  ort_option.graph_optimization_level = option.ort_graph_opt_level;
  ort_option.intra_op_num_threads = option.cpu_thread_num;
  ort_option.inter_op_num_threads = option.ort_inter_op_num_threads;
  ort_option.execution_mode = option.ort_execution_mode;
  ort_option.use_gpu = (option.device == Device::GPU) ? true : false;
  ort_option.gpu_id = option.device_id;
  FDASSERT(option.model_format == Frontend::PADDLE ||
               option.model_format == Frontend::ONNX,
           "OrtBackend only support model format of Frontend::PADDLE / "
           "Frontend::ONNX.");
  backend_ = new OrtBackend();
  auto casted_backend = dynamic_cast<OrtBackend*>(backend_);
  if (option.model_format == Frontend::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, ort_option),
             "Load model from ONNX failed while initliazing OrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, ort_option),
             "Load model from Paddle failed while initliazing OrtBackend.");
  }
#else
  FDASSERT(false, "OrtBackend is not available, please compiled with "
                  "ENABLE_ORT_BACKEND=ON.");
#endif
}

void Runtime::CreateTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  auto trt_option = TrtBackendOption();
  trt_option.gpu_id = option.device_id;
  trt_option.enable_fp16 = option.trt_enable_fp16;
  trt_option.enable_int8 = option.trt_enable_int8;
  trt_option.max_batch_size = option.trt_max_batch_size;
  trt_option.max_workspace_size = option.trt_max_workspace_size;
  trt_option.fixed_shape = option.trt_fixed_shape;
  trt_option.max_shape = option.trt_max_shape;
  trt_option.min_shape = option.trt_min_shape;
  trt_option.opt_shape = option.trt_opt_shape;
  trt_option.serialize_file = option.trt_serialize_file;
  FDASSERT(option.model_format == Frontend::PADDLE ||
               option.model_format == Frontend::ONNX,
           "TrtBackend only support model format of Frontend::PADDLE / "
           "Frontend::ONNX.");
  backend_ = new TrtBackend();
  auto casted_backend = dynamic_cast<TrtBackend*>(backend_);
  if (option.model_format == Frontend::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, trt_option),
             "Load model from ONNX failed while initliazing TrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, trt_option),
             "Load model from Paddle failed while initliazing TrtBackend.");
  }
#else
  FDASSERT(false, "TrtBackend is not available, please compiled with "
                  "ENABLE_TRT_BACKEND=ON.");
#endif
}
} // namespace fastdeploy
