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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fastdeploy/backends/backend.h"
#include "paddle_api.h"  // NOLINT

namespace fastdeploy {

struct LiteBackendOption {
  // cpu num threads
  int threads = 1;
  // lite power mode
  // 0: LITE_POWER_HIGH
  // 1: LITE_POWER_LOW
  // 2: LITE_POWER_FULL
  // 3: LITE_POWER_NO_BIND
  // 4: LITE_POWER_RAND_HIGH
  // 5: LITE_POWER_RAND_LOW
  int power_mode = 3;
  // enable fp16
  bool enable_fp16 = false;
  // enable int8
  bool enable_int8 = false;
  // optimized model dir for CxxConfig
  std::string optimized_model_dir = "";
  // TODO(qiuyanjun): support more options for lite backend.
  // Such as fp16, different device target (kARM/kXPU/kNPU/...)
  std::string nnadapter_subgraph_partition_config_path = "";
  std::string nnadapter_subgraph_partition_config_buffer = "";
  std::string nnadapter_context_properties = "";
  std::string nnadapter_model_cache_dir = "";
  std::string nnadapter_mixed_precision_quantization_config_path = "";
  std::map<std::string, std::vector<std::vector<int64_t>>>
    nnadapter_dynamic_shape_info = {{"", {{0}}}};
  std::vector<std::string> nnadapter_device_names = {};
  bool enable_timvx = false;
  bool enable_ascend = false;
  bool enable_kunlunxin = false;
  int device_id = 0;
  int kunlunxin_l3_workspace_size = 0xfffc00;
  bool kunlunxin_locked = false;
  bool kunlunxin_autotune = true;
  std::string kunlunxin_autotune_file = "";
  std::string kunlunxin_precision = "int16";
  bool kunlunxin_adaptive_seqlen = false;
  bool kunlunxin_enable_multi_stream = false;
};

// Convert data type from paddle lite to fastdeploy
FDDataType LiteDataTypeToFD(const paddle::lite_api::PrecisionType& dtype);

class LiteBackend : public BaseBackend {
 public:
  LiteBackend() {}
  virtual ~LiteBackend() = default;
  void BuildOption(const LiteBackendOption& option);

  bool InitFromPaddle(const std::string& model_file,
                      const std::string& params_file,
                      const LiteBackendOption& option = LiteBackendOption());

  bool Infer(std::vector<FDTensor>& inputs,
            std::vector<FDTensor>* outputs,
            bool copy_to_fd = true) override; // NOLINT

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

 private:
  paddle::lite_api::CxxConfig config_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::map<std::string, int> inputs_order_;
  LiteBackendOption option_;
  bool supported_fp16_ = false;
  bool ReadFile(const std::string& filename,
               std::vector<char>* contents,
               const bool binary = true);
};
}  // namespace fastdeploy
