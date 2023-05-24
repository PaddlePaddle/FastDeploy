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

DEFINE_string(shapes, "1,3,224,224",
              "Required, set input shape for model."
              "default 1,3,224,224");
DEFINE_string(names, "DEFAULT", "Required, set input names for model.");
DEFINE_string(dtypes, "FP32",
              "Required, set input dtypes for model."
              "default FP32.");
DEFINE_string(trt_shapes, "1,3,224,224:1,3,224,224:1,3,224,224",
              "Optional, set min/opt/max shape for trt/paddle_trt."
              "default 1,3,224,224:1,3,224,224:1,3,224,224");
DEFINE_int32(batch, 1,
             "Optional, set trt max batch size, "
             "default 1");
DEFINE_bool(dump, false,
            "Optional, whether to dump output tensors, "
            "default false.");
DEFINE_bool(info, false,
            "Optional, only check the input infos of model."
            "default false.");
DEFINE_bool(diff, false,
            "Optional, check the diff between two tensors."
            "default false.");
DEFINE_string(tensors, "tensor_a.txt:tensor_b.txt",
              "Optional, the paths to dumped tensors, "
              "default tensor_a.txt:tensor_b.txt");
DEFINE_bool(mem, false,
            "Optional, whether to force to collect memory info, "
            "default false.");
DEFINE_int32(interval, -1,
             "Optional, sampling interval for collect memory info, "
             "default false.");
DEFINE_string(model_format, "PADDLE",
              "Optional, set specific model format,"
              "eg, PADDLE/ONNX/RKNN/TORCHSCRIPT/SOPHGO"
              "default PADDLE.");
DEFINE_bool(disable_mkldnn, false,
            "Optional, disable mkldnn for paddle backend. "
            "default false.");
DEFINE_string(optimized_model_dir, "",
              "Optional, set optimized model dir for lite."
              "eg: model.opt.nb, "
              "default ''");

#if defined(ENABLE_BENCHMARK)
static std::vector<int64_t> GetInt64Shape(const std::vector<int>& shape) {
  std::vector<int64_t> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int64_t>(shape[i]);
  }
  return new_shape;
}

static fastdeploy::ModelFormat GetModelFormat(const std::string& model_format) {
  if (model_format == "PADDLE") {
    return fastdeploy::ModelFormat::PADDLE;
  } else if (model_format == "ONNX") {
    return fastdeploy::ModelFormat::ONNX;
  } else if (model_format == "RKNN") {
    return fastdeploy::ModelFormat::RKNN;
  } else if (model_format == "TORCHSCRIPT") {
    return fastdeploy::ModelFormat::TORCHSCRIPT;
  } else if (model_format == "SOPHGO") {
    return fastdeploy::ModelFormat::SOPHGO;
  } else {
    return fastdeploy::ModelFormat::PADDLE;
  }
}

static void CheckTensorDiff(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
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
}

static void RuntimeProfiling(int argc, char* argv[]) {
  // Init runtime option
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return;
  }
  if (FLAGS_disable_mkldnn) {
    option.paddle_infer_option.enable_mkldnn = false;
  }
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  UpdateBaseCustomFlags(config_info);  // see flags.h
  // Init log recorder
  std::stringstream ss;
  ss.precision(6);

  // Memory resource moniter
  int sampling_interval = FLAGS_interval >= 1
                              ? FLAGS_interval
                              : std::stoi(config_info["sampling_interval"]);

  benchmark::ResourceUsageMonitor resource_moniter(
      sampling_interval, std::stoi(config_info["device_id"]));

  // Check model path and model format
  std::string model_name, params_name, config_name;
  std::string model_file, params_file;
  auto model_format = fastdeploy::ModelFormat::PADDLE;
  if (FLAGS_model_file != "UNKNOWN") {
    // Set model file/param/format via command line
    if (FLAGS_model != "") {
      model_file = FLAGS_model + sep + FLAGS_model_file;
      params_file = FLAGS_model + sep + FLAGS_params_file;
    } else {
      model_file = FLAGS_model_file;
      params_file = FLAGS_params_file;
    }
    model_format = GetModelFormat(FLAGS_model_format);
    if (model_format == fastdeploy::ModelFormat::PADDLE &&
        FLAGS_params_file == "") {
      if (config_info["backend"] != "lite") {
        std::cout << "[ERROR] params_file can not be empty for PADDLE"
                  << " format, Please, set your custom params_file manually."
                  << std::endl;
        return;
      } else {
        std::cout << "[INFO] Will using the lite light api for: " << model_file
                  << std::endl;
      }
    }
  } else {
    // Set model file/param/format via model dir (only support
    // for Paddle model format now)
    if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                                 &model_format, config_info, false)) {
      return;
    }
    model_file = FLAGS_model + sep + model_name;
    params_file = FLAGS_model + sep + params_name;
  }

  option.SetModelPath(model_file, params_file, model_format);

  // Set opt model dir
  if (config_info["backend"] == "lite") {
    if (FLAGS_optimized_model_dir != "") {
      option.paddle_lite_option.optimized_model_dir = FLAGS_optimized_model_dir;
    } else {
      option.paddle_lite_option.optimized_model_dir = FLAGS_model;
    }
  }

  // Get input shapes/names/dtypes
  std::vector<std::vector<int32_t>> input_shapes =
      benchmark::ResultManager::GetInputShapes(FLAGS_shapes);
  std::vector<std::string> input_names =
      benchmark::ResultManager::GetInputNames(FLAGS_names);
  std::vector<fastdeploy::FDDataType> input_dtypes =
      benchmark::ResultManager::GetInputDtypes(FLAGS_dtypes);

  // Set tensorrt shapes
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    option.trt_option.max_batch_size = FLAGS_batch;
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shapes);
    if (input_names[0] == "DEFAULT") {
      std::cout << "Please set the input names for TRT/Paddle-TRT backend!"
                << std::endl;
      return;
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

  // Check default input names
  if (input_names[0] == "DEFAULT") {
    input_names.clear();
    for (int i = 0; i < runtime.NumInputs(); ++i) {
      input_names.push_back(runtime.GetInputInfo(i).name);
    }
  }

  assert(runtime.NumInputs() == input_shapes.size());
  assert(runtime.NumInputs() == input_names.size());
  assert(runtime.NumInputs() == input_dtypes.size());

  // Feed inputs, all values set as 1.
  std::vector<fastdeploy::FDTensor> inputs(runtime.NumInputs());
  for (int i = 0; i < inputs.size(); ++i) {
    fastdeploy::function::Full(1, GetInt64Shape(input_shapes[i]), &inputs[i],
                               input_dtypes[i]);
    inputs[i].name = input_names[i];
  }

  // Start memory resource moniter
  if (config_info["collect_memory_info"] == "true" || FLAGS_mem) {
    resource_moniter.Start();
  }

  // Run runtime profiling
  std::vector<fastdeploy::FDTensor> outputs;
  if (!runtime.Infer(inputs, &outputs)) {
    std::cerr << "Failed to predict." << std::endl;
    ss << "Runtime(ms): Failed" << std::endl;
    if (config_info["collect_memory_info"] == "true") {
      ss << "cpu_rss_mb: Failed" << std::endl;
      ss << "gpu_rss_mb: Failed" << std::endl;
      ss << "gpu_util: Failed" << std::endl;
      resource_moniter.Stop();
    }
    benchmark::ResultManager::SaveBenchmarkResult(ss.str(),
                                                  config_info["result_path"]);
    return;
  }

  double profile_time = runtime.GetProfileTime() * 1000.0;
  std::cout << "Runtime(ms): " << profile_time << "ms." << std::endl;
  ss << "Runtime(ms): " << profile_time << "ms." << std::endl;

  // Collect memory info
  if (config_info["collect_memory_info"] == "true" || FLAGS_mem) {
    float cpu_mem = resource_moniter.GetMaxCpuMem();
    float gpu_mem = resource_moniter.GetMaxGpuMem();
    float gpu_util = resource_moniter.GetMaxGpuUtil();
    std::cout << "cpu_rss_mb: " << cpu_mem << "MB." << std::endl;
    ss << "cpu_rss_mb: " << cpu_mem << "MB." << std::endl;
    std::cout << "gpu_rss_mb: " << gpu_mem << "MB." << std::endl;
    ss << "gpu_rss_mb: " << gpu_mem << "MB." << std::endl;
    std::cout << "gpu_util: " << gpu_util << std::endl;
    ss << "gpu_util: " << gpu_util << "MB." << std::endl;
    resource_moniter.Stop();
  }
  benchmark::ResultManager::SaveBenchmarkResult(ss.str(),
                                                config_info["result_path"]);

  // Dump output tensors
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
}

static void showInputInfos(int argc, char* argv[]) {
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return;
  }
  if (FLAGS_disable_mkldnn) {
    option.paddle_infer_option.enable_mkldnn = false;
  }
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  std::string model_name, params_name, config_name;
  auto model_format = fastdeploy::ModelFormat::PADDLE;
  if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                               &model_format, config_info, false)) {
    return;
  }
  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;

  option.SetModelPath(model_file, params_file, model_format);

  // Init runtime
  fastdeploy::Runtime runtime;
  if (!runtime.Init(option)) {
    std::cout << "Initial Runtime failed!" << std::endl;
  }
  // Show input tensor infos
  auto input_infos = runtime.GetInputInfos();
  for (int i = 0; i < input_infos.size(); ++i) {
    std::cout << input_infos[i] << std::endl;
  }
}
#endif

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK)
  google::SetVersionString("0.0.0");
  google::SetUsageMessage(
      "./benchmark -[info|diff|check|dump|mem] -model xxx -config_path xxx "
      "-[shapes|dtypes|names|tensors] -[model_file|params_file|model_format]");
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_diff) {
    CheckTensorDiff(argc, argv);
    return 0;
  } else if (FLAGS_info) {
    showInputInfos(argc, argv);
    return 0;
  } else {
    RuntimeProfiling(argc, argv);
    return 0;
  }
#endif
  return 0;
}
