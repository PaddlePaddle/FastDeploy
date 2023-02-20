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

void build_test_data(std::vector<std::vector<fd::FDTensor>> &prewarm_datas, bool is_dynamic) {
    if (is_dynamic == false) {
        std::vector<float> inputs_data;
        inputs_data.resize(1 * 3 * 224 * 224);
        for (size_t i = 0; i < inputs_data.size(); ++i) {
            inputs_data[i] = std::rand() % 1000 / 1000.0f;
        }
        prewarm_datas[0][0].Resize({1, 3, 224, 224}, fd::FDDataType::FP32);
        fd::FDTensor::CopyBuffer(prewarm_datas[0][0].Data(),
               inputs_data.data(),
               prewarm_datas[0][0].Nbytes());
        return;
    }
    //max
    std::vector<float> inputs_data_max;
    inputs_data_max.resize(1 * 3 * 224 * 224);
    for (size_t i = 0; i < inputs_data_max.size(); ++i) {
        inputs_data_max[i] = std::rand() % 1000 / 1000.0f;
    }
    prewarm_datas[0][0].Resize({1, 3, 224, 224}, fd::FDDataType::FP32);
    fd::FDTensor::CopyBuffer(prewarm_datas[0][0].Data(),
           inputs_data_max.data(),
           prewarm_datas[0][0].Nbytes());
    //min
    std::vector<float> inputs_data_min;
    inputs_data_min.resize(1 * 3 * 224 * 224);
    for (size_t i = 0; i < inputs_data_min.size(); ++i) {
        inputs_data_min[i] = std::rand() % 1000 / 1000.0f;
    }
    prewarm_datas[1][0].Resize({1, 3, 224, 224}, fd::FDDataType::FP32);
    fd::FDTensor::CopyBuffer(prewarm_datas[1][0].Data(),
           inputs_data_min.data(),
           prewarm_datas[1][0].Nbytes());

    //opt
    std::vector<float> inputs_data_opt;
    inputs_data_opt.resize(1 * 3 * 224 * 224);
    for (size_t i = 0; i < inputs_data_opt.size(); ++i) {
        inputs_data_opt[i] = std::rand() % 1000 / 1000.0f;
    }
    prewarm_datas[2][0].Resize({1, 3, 224, 224}, fd::FDDataType::FP32);
    fd::FDTensor::CopyBuffer(prewarm_datas[2][0].Data(),
           inputs_data_opt.data(),
           prewarm_datas[2][0].Nbytes());

}

int main(int argc, char* argv[]) {
  // prewarm_datas
  bool is_dynamic = true;
  std::vector<std::vector<fd::FDTensor>> prewarm_datas;
  if (is_dynamic) {
    prewarm_datas.resize(3);
    prewarm_datas[0].resize(1);
    prewarm_datas[1].resize(1);
    prewarm_datas[2].resize(1);
  } else {
    prewarm_datas.resize(1);
    prewarm_datas[0].resize(1);
  }
  build_test_data(prewarm_datas, is_dynamic);
  std::string model_file = "std_resnet50_script.pt";

  // setup option
  fd::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, "", fd::ModelFormat::TORCHSCRIPT);
  runtime_option.UsePorosBackend();
  runtime_option.UseGpu(0);

  // Compile runtime
  std::unique_ptr<fd::Runtime> runtime =
      std::unique_ptr<fd::Runtime>(new fd::Runtime());

  runtime->Init(runtime_option);

  if (!runtime->Compile(prewarm_datas, runtime_option)) {
    std::cerr << "--- Init FastDeploy Runitme Failed! "
              << "\n--- Model:  " << model_file << std::endl;
    return -1;
  } else {
    std::cout << "--- Init FastDeploy Runitme Done! "
              << "\n--- Model:  " << model_file << std::endl;
  }

  std::vector<fd::FDTensor> input_tensors;
  input_tensors.resize(1);
  std::vector<fd::FDTensor> output_tensors;
  output_tensors.resize(1);

  std::vector<float> inputs_data;
  inputs_data.resize(1 * 3 * 224 * 224);
  for (size_t i = 0; i < inputs_data.size(); ++i) {
    inputs_data[i] = std::rand() % 1000 / 1000.0f;
  }
  input_tensors[0].SetExternalData({1, 3, 224, 224}, fd::FDDataType::FP32, inputs_data.data());
  
  runtime->Infer(input_tensors, &output_tensors);

  output_tensors[0].PrintInfo();
  return 0;
}
