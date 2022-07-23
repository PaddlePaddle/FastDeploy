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
#pragma once

#include <map>
#include <vector>
#include "fastdeploy/backends/backend.h"
#include "fastdeploy/utils/perf.h"

namespace fastdeploy {

enum FASTDEPLOY_DECL Backend { UNKNOWN, ORT, TRT, PDINFER };
// AUTOREC will according to the name of model file
// to decide which Frontend is
enum FASTDEPLOY_DECL Frontend { AUTOREC, PADDLE, ONNX };

FASTDEPLOY_DECL std::string Str(const Backend& b);
FASTDEPLOY_DECL std::string Str(const Frontend& f);
FASTDEPLOY_DECL std::vector<Backend> GetAvailableBackends();

FASTDEPLOY_DECL bool IsBackendAvailable(const Backend& backend);

bool CheckModelFormat(const std::string& model_file,
                      const Frontend& model_format);
Frontend GuessModelFormat(const std::string& model_file);

struct FASTDEPLOY_DECL RuntimeOption {
  Backend backend = Backend::UNKNOWN;

  // for cpu inference and preprocess
  int cpu_thread_num = 8;
  int device_id = 0;

#ifdef WITH_GPU
  Device device = Device::GPU;
#else
  Device device = Device::CPU;
#endif

  // ======Only for ORT Backend========
  // -1 means use default value by ort
  // 0: ORT_DISABLE_ALL 1: ORT_ENABLE_BASIC 2: ORT_ENABLE_EXTENDED 3:
  // ORT_ENABLE_ALL
  int ort_graph_opt_level = -1;
  int ort_inter_op_num_threads = -1;
  // 0: ORT_SEQUENTIAL 1: ORT_PARALLEL
  int ort_execution_mode = -1;

  // ======Only for Paddle Backend=====
  bool pd_enable_mkldnn = true;
  int pd_mkldnn_cache_size = 1;

  // ======Only for Trt Backend=======
  std::map<std::string, std::vector<int32_t>> trt_fixed_shape;
  std::map<std::string, std::vector<int32_t>> trt_max_shape;
  std::map<std::string, std::vector<int32_t>> trt_min_shape;
  std::map<std::string, std::vector<int32_t>> trt_opt_shape;
  std::string trt_serialize_file = "";
  bool trt_enable_fp16 = false;
  bool trt_enable_int8 = false;
  size_t trt_max_batch_size = 32;
  size_t trt_max_workspace_size = 1 << 30;

  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty
  Frontend model_format = Frontend::AUTOREC;  // format of input model
};

struct FASTDEPLOY_DECL Runtime {
 public:
  //  explicit Runtime(const RuntimeOption& _option = RuntimeOption());

  bool Init(const RuntimeOption& _option);

  bool Infer(std::vector<FDTensor>& input_tensors,
             std::vector<FDTensor>* output_tensors);

  void CreateOrtBackend();

  void CreatePaddleBackend();

  void CreateTrtBackend();

  int NumInputs() { return backend_->NumInputs(); }
  int NumOutputs() { return backend_->NumOutputs(); }
  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);

  RuntimeOption option;

 private:
  BaseBackend* backend_;
};
}  // namespace fastdeploy
