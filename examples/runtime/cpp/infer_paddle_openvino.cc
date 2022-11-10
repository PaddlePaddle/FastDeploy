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

namespace fd = fastdeploy;

int main(int argc, char* argv[]) {
  std::string model_file = "mobilenetv2/inference.pdmodel";
  std::string params_file = "mobilenetv2/inference.pdiparams";

  // setup option
  fd::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, params_file, fd::ModelFormat::PADDLE);
  runtime_option.UseOpenVINOBackend();
  runtime_option.SetCpuThreadNum(12);
  // init runtime
  std::unique_ptr<fd::Runtime> runtime =
      std::unique_ptr<fd::Runtime>(new fd::Runtime());
  if (!runtime->Init(runtime_option)) {
    std::cerr << "--- Init FastDeploy Runitme Failed! "
              << "\n--- Model:  " << model_file << std::endl;
    return -1;
  } else {
    std::cout << "--- Init FastDeploy Runitme Done! "
              << "\n--- Model:  " << model_file << std::endl;
  }
  // init input tensor shape
  fd::TensorInfo info = runtime->GetInputInfo(0);
  info.shape = {1, 3, 224, 224};

  std::vector<fd::FDTensor> input_tensors(1);
  std::vector<fd::FDTensor> output_tensors(1);

  std::vector<float> inputs_data;
  inputs_data.resize(1 * 3 * 224 * 224);
  for (size_t i = 0; i < inputs_data.size(); ++i) {
    inputs_data[i] = std::rand() % 1000 / 1000.0f;
  }
  input_tensors[0].SetExternalData({1, 3, 224, 224}, fd::FDDataType::FP32, inputs_data.data());
  
  //get input name
  input_tensors[0].name = info.name;

  runtime->Infer(input_tensors, &output_tensors);

  output_tensors[0].PrintInfo();
  return 0;
}