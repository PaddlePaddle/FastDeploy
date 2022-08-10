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
  // set path of model file and params file
  // for onnx, only need to define model_file, but also need to
  // define model_format
  // model_format support 'paddle' / 'onnx' now.
  void SetModelPath(const std::string& model_path,
                    const std::string& params_path = "",
                    const std::string& _model_format = "paddle");

  // set model inference in GPU
  void UseCpu();

  // set model inference in CPU
  void UseGpu(int gpu_id = 0);

  // set number of thread while inference in CPU
  void SetCpuThreadNum(int thread_num);

  // use paddle inference backend
  void UsePaddleBackend();

  // use onnxruntime backend
  void UseOrtBackend();

  // use tensorrt backend
  void UseTrtBackend();

  // enable mkldnn while use paddle inference in CPU
  void EnablePaddleMKLDNN();
  // disable mkldnn while use paddle inference in CPU
  void DisablePaddleMKLDNN();

  // enable debug information of paddle backend
  void EnablePaddleLogInfo();
  // disable debug information of paddle backend
  void DisablePaddleLogInfo();

  // set size of cached shape while enable mkldnn with paddle inference backend
  void SetPaddleMKLDNNCacheSize(int size);

  // set tensorrt shape while the inputs of model contain dynamic shape
  // min_shape: the minimum shape
  // opt_shape: the most common shape while inference, default be empty
  // max_shape: the maximum shape, default be empty

  // if opt_shape, max_shape are empty, they will keep same with the min_shape
  // which means the shape will be fixed as min_shape while inference
  void SetTrtInputShape(
      const std::string& input_name, const std::vector<int32_t>& min_shape,
      const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
      const std::vector<int32_t>& max_shape = std::vector<int32_t>());

  // enable half precision while use tensorrt backend
  void EnableTrtFP16();
  // disable half precision, change to full precision(float32)
  void DisableTrtFP16();

  void SetTrtCacheFile(const std::string& cache_file_path);

  Backend backend = Backend::UNKNOWN;
  // for cpu inference and preprocess
  int cpu_thread_num = 8;
  int device_id = 0;

  Device device = Device::CPU;

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
  bool pd_enable_log_info = false;
  int pd_mkldnn_cache_size = 1;

  // ======Only for Trt Backend=======
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

  // inside parameters, only for inside usage
  // remove multiclass_nms in Paddle2ONNX
  bool remove_multiclass_nms_ = false;
  // for Paddle2ONNX to export custom operators
  std::map<std::string, std::string> custom_op_info_;
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
  std::unique_ptr<BaseBackend> backend_;
};
}  // namespace fastdeploy
