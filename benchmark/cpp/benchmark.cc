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
#include "fastdeploy/function/functions.h"
#include "flags.h"
#include "macros.h"
#include "option.h"

namespace vision = fastdeploy::vision;
namespace benchmark = fastdeploy::benchmark;

// TOOD: Support TRT shape
DEFINE_string(trt_shape, "1,3,224,224:1,3,224,224:1,3,224,224",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,224,224:1,3,224,224:1,3,224,224");

DEFINE_string(shapes, "1,3,224,224",
              "Set input shape for model."
              "eg:--shapes 1,3,224,224");

DEFINE_string(names, "DEFAULT",
              "Set input names for model."
              "eg:--names x");

DEFINE_string(dtypes, "FP32",
              "Set input dtypes for model."
              "eg:--dtypes FP32");

static std::vector<int64_t> GetInt64Shape(const std::vector<int>& shape) {
  std::vector<int64_t> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int64_t>(shape[i]);
  }
  return new_shape;
}

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  std::string model_name, params_name, config_name;
  auto model_format = fastdeploy::ModelFormat::PADDLE;
  if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                               &model_format, config_info, false)) {
    return -1;
  }
  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;

  option.SetModelPath(model_file, params_file, model_format);

  fastdeploy::Runtime runtime;
  runtime.Init(option);

  std::vector<std::vector<int32_t>> input_shapes =
      benchmark::ResultManager::GetInputShapes(FLAGS_shapes);
  assert(runtime.NumInputs() == input_shapes.size());

  std::vector<std::string> input_names =
      benchmark::ResultManager::GetInputNames(FLAGS_names);
  std::vector<fastdeploy::FDDataType> input_dtypes =
      benchmark::ResultManager::GetInputDtypes(FLAGS_dtypes);
  if (input_names[0] == "DEFAULT") {
    input_names.clear();
    for (int i = 0; i < runtime.NumInputs(); ++i) {
      input_names.push_back(runtime.GetInputInfo(i).name);
    }
  }
  assert(runtime.NumInputs() == input_names.size());
  assert(runtime.NumInputs() == input_dtypes.size());

  std::vector<fastdeploy::FDTensor> inputs(runtime.NumInputs());

  for (int i = 0; i < inputs.size(); ++i) {
    fastdeploy::function::Full(1, GetInt64Shape(input_shapes[i]), &inputs[i],
                               input_dtypes[i]);
    inputs[i].name = input_names[i];
  }

  std::vector<fastdeploy::FDTensor> outputs;
  runtime.Infer(inputs, &outputs);

  auto profile_time = runtime.GetProfileTime() * 1000.0;

  std::cout << "Runtime: " << profile_time << " ms" << std::endl;
#endif
  return 0;
}