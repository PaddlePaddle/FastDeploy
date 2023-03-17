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

#include "fastdeploy/runtime.h"
#include <cassert>

namespace fd = fastdeploy;

int main(int argc, char* argv[]) {
  // Download from https://bj.bcebos.com/paddle2onnx/model_zoo/pplcnet.tar.gz
  std::string model_file = "pplcnet/inference.pdmodel";
  std::string params_file = "pplcnet/inference.pdiparams";

  // configure runtime
  // How to configure by RuntimeOption, refer its api doc for more information
  // https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1RuntimeOption.html
  fd::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, params_file);
  runtime_option.UseCpu();
 
  // If need to configure Paddle Inference backend for more option, we can configure runtime_option.paddle_infer_option
  // refer https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1PaddleBackendOption.html
  runtime_option.paddle_infer_option.enable_mkldnn = true;

  fd::Runtime runtime;
  assert(runtime.Init(runtime_option));

  // Get model's inputs information
  // API doc refer https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1Runtime.html
  std::vector<fd::TensorInfo> inputs_info = runtime.GetInputInfos();

  // Create dummy data fill with 0.5
  std::vector<float> dummy_data(1 * 3 * 224 * 224, 0.5);

  // Create inputs/outputs tensors
  std::vector<fd::FDTensor> inputs(inputs_info.size());
  std::vector<fd::FDTensor> outputs;

  // Initialize input tensors
  // API doc refer https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1FDTensor.html
  inputs[0].SetData({1, 3, 224, 224}, fd::FDDataType::FP32, dummy_data.data());
  inputs[0].name = inputs_info[0].name;

  // Inference
  assert(runtime.Infer(inputs, &outputs));
 
  // Print debug information of outputs 
  outputs[0].PrintInfo();

  // Get data pointer and print it's elements
  const float* data_ptr = reinterpret_cast<const float*>(outputs[0].GetData());
  for (size_t i = 0; i < 10 && i < outputs[0].Numel(); ++i) {
    std::cout << data_ptr[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
