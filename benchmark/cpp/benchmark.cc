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

DEFINE_string(shapes, "1,3,224,224", "Set input shape for model.");
DEFINE_string(names, "DEFAULT", "Set input names for model.");
DEFINE_string(dtypes, "FP32", "Set input dtypes for model.");
DEFINE_string(trt_shapes, "1,3,224,224:1,3,224,224:1,3,224,224",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,224,224:1,3,224,224:1,3,224,224");
DEFINE_bool(dump, false, "whether to dump output tensors.");
DEFINE_bool(diff, false, "check the diff between two tensors.");
DEFINE_string(tensors, "a.txt", "a.txt:b.txt");

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
  // Only check tensor diff
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_diff) {
    std::cout << "Check tensor diff ..." << std::endl;
    std::vector<std::string> tensor_paths =
        benchmark::ResultManager::SplitStr(FLAGS_tensors);
    assert(tensor_paths.size() == 2);
    fastdeploy::FDTensor tensor_a, tensor_b;
    benchmark::ResultManager::LoadFDTensor(&tensor_a, tensor_paths[0]);
    benchmark::ResultManager::LoadFDTensor(&tensor_b, tensor_paths[1]);
    auto tensor_diff =
        benchmark::ResultManager::CalculateDiffStatis(tensor_a, tensor_b);
    std::cout << "Tensor diff: mean=" << tensor_diff.data.mean
              << ", max=" << tensor_diff.data.max
              << ", min=" << tensor_diff.data.min << std::endl;

    return 0;
  }

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

  // Init flags infos
  std::vector<std::vector<int32_t>> input_shapes =
      benchmark::ResultManager::GetInputShapes(FLAGS_shapes);
  assert(runtime.NumInputs() == input_shapes.size());

  std::vector<std::string> input_names =
      benchmark::ResultManager::GetInputNames(FLAGS_names);
  std::vector<fastdeploy::FDDataType> input_dtypes =
      benchmark::ResultManager::GetInputDtypes(FLAGS_dtypes);

  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shapes);
    if (input_names[0] == "DEFAULT") {
      std::cout << "Please set the input names for TRT/Paddle-TRT backend!"
                << std::endl;
      return -1;
    }
    assert(input_names.size() == (trt_shapes.size() / 3));
    for (int i = 0; i < input_shapes.size(); ++i) {
      option.trt_option.SetShape(input_names[i], trt_shapes[i * 3],
                                 trt_shapes[i * 3 + 1], trt_shapes[i * 3 + 2]);
    }
  }

  // Init runtime
  fastdeploy::Runtime runtime;
  if (!runtime.Init(option)) {
    std::cout << "Initial Runtime failed!" << std::endl;
  }

  if (input_names[0] == "DEFAULT") {
    input_names.clear();
    for (int i = 0; i < runtime.NumInputs(); ++i) {
      input_names.push_back(runtime.GetInputInfo(i).name);
    }
  }
  assert(runtime.NumInputs() == input_names.size());
  assert(runtime.NumInputs() == input_dtypes.size());

  std::vector<fastdeploy::FDTensor> inputs(runtime.NumInputs());

  // Feed inputs
  for (int i = 0; i < inputs.size(); ++i) {
    fastdeploy::function::Full(1, GetInt64Shape(input_shapes[i]), &inputs[i],
                               input_dtypes[i]);
    inputs[i].name = input_names[i];
  }

  std::vector<fastdeploy::FDTensor> outputs;
  runtime.Infer(inputs, &outputs);

  auto profile_time = runtime.GetProfileTime() * 1000.0;

  // Dump outputs
  if (FLAGS_dump) {
    for (int i = 0; i < outputs.size(); ++i) {
      auto name_tokens =
          benchmark::ResultManager::SplitStr(outputs[i].name, '/');
      std::string out_name = name_tokens[0];
      for (int j = 1; j < name_tokens.size(); ++j) {
        out_name += "_";
        out_name += name_tokens[j];
      }
      std::string out_file = config_info["backend"] + "_" + out_name + ".txt";
      benchmark::ResultManager::SaveFDTensor(outputs[i], out_file);
      outputs[i].PrintInfo();
      std::cout << "Saved: " << out_file << std::endl;
    }
  }

  std::stringstream ss;
  ss.precision(6);
  std::cout << "Runtime(ms): " << profile_time << "ms." << std::endl;
  ss << "Runtime(ms): " << profile_time << "ms." << std::endl;
  benchmark::ResultManager::SaveBenchmarkResult(ss.str(),
                                                config_info["result_path"]);
#endif
  return 0;
}