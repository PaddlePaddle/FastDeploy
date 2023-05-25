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

#include "fastdeploy/runtime/backends/paddle/paddle_backend.h"

#include <sstream>

#include "fastdeploy/utils/path.h"

namespace fastdeploy {

void PaddleBackend::BuildOption(const PaddleBackendOption& option) {
  option_ = option;
  if (option.device == Device::GPU) {
    config_.EnableUseGpu(option.gpu_mem_init_size, option.device_id);
    if (option_.switch_ir_debug) {
      FDINFO << "Will Enable ir_debug for Paddle Backend." << std::endl;
      config_.SwitchIrDebug();
    }
    if (option_.external_stream_) {
      FDINFO << "Will use external stream for Paddle Backend." << std::endl;
      config_.SetExecStream(option_.external_stream_);
    }
    if (option.enable_trt) {
      if (!option.trt_option.enable_fp16) {
        FDINFO << "Will try to use tensorrt inference with Paddle Backend."
               << std::endl;
      }
      config_.Exp_DisableTensorRtOPs(option.trt_disabled_ops_);
      auto precision = paddle_infer::PrecisionType::kFloat32;
      if (option.trt_option.enable_fp16) {
        FDINFO << "Will try to use tensorrt fp16 inference with Paddle Backend."
               << std::endl;
        precision = paddle_infer::PrecisionType::kHalf;
      }
      bool use_static = false;
      if (option.trt_option.serialize_file != "") {
        FDWARNING
            << "Detect that tensorrt cache file has been set to "
            << option.trt_option.serialize_file
            << ", but while enable paddle2trt, please notice that the cache "
               "file will save to the directory where paddle model saved."
            << std::endl;
        use_static = true;
        std::string opt_cache_dir =
            GetDirFromPath(option.trt_option.serialize_file);

        config_.SetOptimCacheDir(opt_cache_dir);
      }
      config_.EnableTensorRtEngine(option.trt_option.max_workspace_size,
                                   option.trt_option.max_batch_size, 3,
                                   precision, use_static);
      SetTRTDynamicShapeToConfig(option);
      if (option_.enable_fixed_size_opt) {
        paddle_infer::experimental::InternalUtils::SetTransformerMaskid(
            &config_, "opt");
      }
    }
  } else if (option.device == Device::IPU) {
#ifdef WITH_IPU
    config_.EnableIpu(option.ipu_option.ipu_device_num,
                      option.ipu_option.ipu_micro_batch_size,
                      option.ipu_option.ipu_enable_pipelining,
                      option.ipu_option.ipu_batches_per_step);
    config_.SetIpuConfig(option.ipu_option.ipu_enable_fp16,
                         option.ipu_option.ipu_replica_num,
                         option.ipu_option.ipu_available_memory_proportion,
                         option.ipu_option.ipu_enable_half_partial);
#else
    FDWARNING << "The FastDeploy is not compiled with IPU device, so will "
                 "fallback to CPU with Paddle Inference Backend."
              << std::endl;
#endif
  } else if (option.device == Device::KUNLUNXIN) {
#ifdef WITH_KUNLUNXIN
    config_.EnableXpu(option.xpu_option.kunlunxin_l3_workspace_size,
                      option.xpu_option.kunlunxin_locked,
                      option.xpu_option.kunlunxin_autotune,
                      option.xpu_option.kunlunxin_autotune_file,
                      option.xpu_option.kunlunxin_precision,
                      option.xpu_option.kunlunxin_adaptive_seqlen,
                      option.xpu_option.kunlunxin_enable_multi_stream);
    config_.SetXpuConfig(
        option.xpu_option.kunlunxin_quant_post_dynamic_weight_bits,
        option.xpu_option.kunlunxin_quant_post_dynamic_op_types);
    config_.SetXpuDeviceId(option.xpu_option.kunlunxin_device_id);
#else
    FDWARNING
        << "The FastDeploy is not compiled with KUNLUNXIN device, so will "
           "fallback to CPU with Paddle Inference Backend."
        << std::endl;
#endif
  } else {
    config_.DisableGpu();
    if (option.enable_mkldnn) {
      config_.EnableMKLDNN();
      config_.SetMkldnnCacheCapacity(option.mkldnn_cache_size);
    }
  }

  if (!option.enable_log_info) {
    config_.DisableGlogInfo();
  }
  if (option.cpu_thread_num <= 0) {
    config_.SetCpuMathLibraryNumThreads(8);
  } else {
    config_.SetCpuMathLibraryNumThreads(option.cpu_thread_num);
  }
}

bool PaddleBackend::Init(const RuntimeOption& runtime_option) {
  if (!(Supported(runtime_option.model_format, Backend::PDINFER) &&
        Supported(runtime_option.device, Backend::PDINFER))) {
    return false;
  }

  auto option = runtime_option;
  // Collect basic paddle inference option and trt option.
  option.paddle_infer_option.model_file = runtime_option.model_file;
  option.paddle_infer_option.params_file = runtime_option.params_file;
  option.paddle_infer_option.model_from_memory_ =
      runtime_option.model_from_memory_;
  option.paddle_infer_option.device = runtime_option.device;
  option.paddle_infer_option.device_id = runtime_option.device_id;
  option.paddle_infer_option.enable_pinned_memory =
      runtime_option.enable_pinned_memory;
  option.paddle_infer_option.external_stream_ = runtime_option.external_stream_;
  option.paddle_infer_option.trt_option = runtime_option.trt_option;
  option.paddle_infer_option.trt_option.gpu_id = runtime_option.device_id;
  // Note(qiuyanjun): For Ipu option and XPU option, please check the
  // details of RuntimeOption::UseIpu() and RuntimeOption::UseKunlunXin().
  // Futhermore, please check paddle_infer_option.SetIpuConfig() and
  // paddle_infer_option.SetXpuConfig() for more details of extra configs.
  return InitFromPaddle(option.model_file, option.params_file,
                        option.model_from_memory_, option.paddle_infer_option);
}

bool PaddleBackend::InitFromPaddle(const std::string& model,
                                   const std::string& params,
                                   bool model_from_memory,
                                   const PaddleBackendOption& option) {
  if (initialized_) {
    FDERROR << "PaddleBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  if (model_from_memory) {
    config_.SetModelBuffer(model.c_str(), model.size(), params.c_str(),
                           params.size());
  } else {
    config_.SetModel(model, params);
  }
  if (option.enable_memory_optimize) {
    config_.EnableMemoryOptim();
  }
  BuildOption(option);

  // The input/output information get from predictor is not right, use
  // PaddleReader instead now
  std::string model_content = model;
  if (!model_from_memory) {
    FDASSERT(ReadBinaryFromFile(model, &model_content),
             "Failed to read file %s.", model.c_str());
  }
  auto reader =
      paddle2onnx::PaddleReader(model_content.c_str(), model_content.size());
  // If it's a quantized model, and use cpu with mkldnn, automaticaly switch to
  // int8 mode
  if (reader.is_quantize_model) {
    if (option.device == Device::GPU) {
      FDWARNING << "The loaded model is a quantized model, while inference on "
                   "GPU, please use TensorRT backend to get better performance."
                << std::endl;
      if (option.enable_trt) {
        bool use_static = false;
        if (option.trt_option.serialize_file != "") {
          FDWARNING
              << "Detect that tensorrt cache file has been set to "
              << option.trt_option.serialize_file
              << ", but while enable paddle2trt, please notice that the cache "
                 "file will save to the directory where paddle model saved."
              << std::endl;
          use_static = true;
        }
        config_.EnableTensorRtEngine(option.trt_option.max_workspace_size,
                                     option.trt_option.max_batch_size, 3,
                                     paddle_infer::PrecisionType::kInt8,
                                     use_static, false);
        SetTRTDynamicShapeToConfig(option);
      }
    }
    if (option.enable_mkldnn) {
      config_.EnableMkldnnInt8();
    } else {
      FDWARNING << "The loaded model is a quantized model, while inference on "
                   "CPU, please enable MKLDNN to get better performance."
                << std::endl;
    }
  }

  inputs_desc_.resize(reader.num_inputs);
  for (int i = 0; i < reader.num_inputs; ++i) {
    std::string name(reader.inputs[i].name);
    std::vector<int64_t> shape(reader.inputs[i].shape,
                               reader.inputs[i].shape + reader.inputs[i].rank);
    inputs_desc_[i].name = name;
    inputs_desc_[i].shape.assign(shape.begin(), shape.end());
    inputs_desc_[i].dtype = ReaderDataTypeToFD(reader.inputs[i].dtype);
  }
  outputs_desc_.resize(reader.num_outputs);
  for (int i = 0; i < reader.num_outputs; ++i) {
    std::string name(reader.outputs[i].name);
    std::vector<int64_t> shape(
        reader.outputs[i].shape,
        reader.outputs[i].shape + reader.outputs[i].rank);
    outputs_desc_[i].name = name;
    outputs_desc_[i].shape.assign(shape.begin(), shape.end());
    outputs_desc_[i].dtype = ReaderDataTypeToFD(reader.outputs[i].dtype);
  }
  if (option.collect_trt_shape) {
    // Set the shape info file.
    std::string curr_model_dir = "./";
    if (!option.model_from_memory_) {
      curr_model_dir = GetDirFromPath(option.model_file);
    }
    std::string shape_range_info =
        PathJoin(curr_model_dir, "shape_range_info.pbtxt");
    if (!CheckFileExists(shape_range_info)) {
      FDINFO << "Start generating shape range info file." << std::endl;
      paddle_infer::Config analysis_config;
      if (model_from_memory) {
        analysis_config.SetModelBuffer(model.c_str(), model.size(),
                                       params.c_str(), params.size());
      } else {
        analysis_config.SetModel(model, params);
      }
      analysis_config.CollectShapeRangeInfo(shape_range_info);
      auto predictor_tmp = paddle_infer::CreatePredictor(analysis_config);
      std::map<std::string, std::vector<int>> max_shape;
      std::map<std::string, std::vector<int>> min_shape;
      std::map<std::string, std::vector<int>> opt_shape;
      GetDynamicShapeFromOption(option, &max_shape, &min_shape, &opt_shape);
      std::map<std::string, std::vector<float>> max_input_data;
      std::map<std::string, std::vector<float>> min_input_data;
      std::map<std::string, std::vector<float>> opt_input_data;
      if (!option.trt_option.min_input_data.empty()) {
        GetInputDataFromOption(option, &max_input_data, &min_input_data,
                               &opt_input_data);
      }
      // Need to run once to get the shape range info file.
      CollectShapeRun(predictor_tmp.get(), max_shape, max_input_data);
      CollectShapeRun(predictor_tmp.get(), min_shape, min_input_data);
      CollectShapeRun(predictor_tmp.get(), opt_shape, min_input_data);
      FDINFO << "Finish generating shape range info file." << std::endl;
    }
    FDINFO << "Start loading shape range info file " << shape_range_info
           << " to set TensorRT dynamic shape." << std::endl;
    config_.EnableTunedTensorRtDynamicShape(shape_range_info, false);
  }
  // Note(zhoushunjie): The pass deletion should be executed just before
  // creating predictor.
  if (!option.delete_pass_names.empty()) {
    auto pass_builder = config_.pass_builder();
    for (int i = 0; i < option.delete_pass_names.size(); i++) {
      FDINFO << "Delete pass : " << option.delete_pass_names[i] << std::endl;
      pass_builder->DeletePass(option.delete_pass_names[i]);
    }
  }
  predictor_ = paddle_infer::CreatePredictor(config_);
  initialized_ = true;
  return true;
}

TensorInfo PaddleBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo PaddleBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetOutputInfos() {
  return outputs_desc_;
}

bool PaddleBackend::Infer(std::vector<FDTensor>& inputs,
                          std::vector<FDTensor>* outputs, bool copy_to_fd) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[PaddleBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  // output share backend memory only support CPU or GPU
  if (option_.device == Device::IPU) {
    copy_to_fd = true;
  }

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto handle = predictor_->GetInputHandle(inputs[i].name);
    ShareTensorFromFDTensor(handle.get(), inputs[i]);
  }
  // prebinded output only support for GPU
  // if (!copy_to_fd) {
  //   for (size_t i = 0; i < (*outputs).size(); ++i) {
  //     auto output_name = (*outputs)[i].name;
  //     // if a output is not prebinded,
  //     // the name of output is expected to be empty.
  //     // We skip here
  //     if (output_name.empty()) {
  //       continue;
  //     }
  //     // Record the prebinded output_name.
  //     // Those outputs do not need PaddleTensorToFDTensor
  //     // after predictor_.Run()
  //     auto handle = predictor_->GetOutputHandle(output_name);
  //     ShareOutTensorFromFDTensor(handle.get(), (*outputs)[i]);
  //   }
  // }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  predictor_->Run();
  RUNTIME_PROFILE_LOOP_END

  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto handle = predictor_->GetOutputHandle(outputs_desc_[i].name);
    if (copy_to_fd) {
      (*outputs)[i].is_pinned_memory = option_.enable_pinned_memory;
    }
    PaddleTensorToFDTensor(handle, &((*outputs)[i]), copy_to_fd);
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

std::unique_ptr<BaseBackend> PaddleBackend::Clone(RuntimeOption& runtime_option,
                                                  void* stream, int device_id) {
  std::unique_ptr<BaseBackend> new_backend =
      utils::make_unique<PaddleBackend>();
  auto casted_backend = dynamic_cast<PaddleBackend*>(new_backend.get());
  if (device_id > 0 && (option_.device == Device::GPU) &&
      device_id != option_.device_id) {
    auto clone_option = option_;
    clone_option.device_id = device_id;
    clone_option.external_stream_ = stream;
    FDASSERT(casted_backend->InitFromPaddle(
                 runtime_option.model_file, runtime_option.params_file,
                 runtime_option.model_from_memory_, clone_option),
             "Clone model from Paddle failed while initialize PaddleBackend.");
    FDWARNING << "The target device id:" << device_id
              << " is different from current device id:" << option_.device_id
              << ", cannot share memory with current engine." << std::endl;
    return new_backend;
  }
  casted_backend->inputs_desc_.assign(inputs_desc_.begin(), inputs_desc_.end());
  casted_backend->outputs_desc_.assign(outputs_desc_.begin(),
                                       outputs_desc_.end());
  casted_backend->predictor_ = std::move(predictor_->Clone(stream));
  return new_backend;
}

void PaddleBackend::SetTRTDynamicShapeToConfig(
    const PaddleBackendOption& option) {
  std::map<std::string, std::vector<int>> max_shape;
  std::map<std::string, std::vector<int>> min_shape;
  std::map<std::string, std::vector<int>> opt_shape;
  GetDynamicShapeFromOption(option, &max_shape, &min_shape, &opt_shape);
  if (min_shape.size() > 0) {
    FDINFO << "Start setting trt dynamic shape." << std::endl;
    config_.SetTRTDynamicShapeInfo(min_shape, max_shape, opt_shape);
    FDINFO << "Finish setting trt dynamic shape." << std::endl;
  }
}

void PaddleBackend::GetDynamicShapeFromOption(
    const PaddleBackendOption& option,
    std::map<std::string, std::vector<int>>* max_shape,
    std::map<std::string, std::vector<int>>* min_shape,
    std::map<std::string, std::vector<int>>* opt_shape) const {
  auto print_shape = [](const std::vector<int>& shape) -> std::string {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i < shape.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    return oss.str();
  };
  for (const auto& item : option.trt_option.min_shape) {
    auto max_iter = option.trt_option.max_shape.find(item.first);
    auto opt_iter = option.trt_option.opt_shape.find(item.first);
    FDASSERT(max_iter != option.trt_option.max_shape.end(),
             "Cannot find %s in TrtBackendOption::min_shape.",
             item.first.c_str());
    FDASSERT(opt_iter != option.trt_option.opt_shape.end(),
             "Cannot find %s in TrtBackendOption::opt_shape.",
             item.first.c_str());
    (*max_shape)[item.first].assign(max_iter->second.begin(),
                                    max_iter->second.end());
    (*opt_shape)[item.first].assign(opt_iter->second.begin(),
                                    opt_iter->second.end());
    (*min_shape)[item.first].assign(item.second.begin(), item.second.end());
    FDINFO << item.first
           << ": the max shape = " << print_shape(max_iter->second)
           << ", the min shape = " << print_shape(item.second)
           << ", the opt shape = " << print_shape(opt_iter->second)
           << std::endl;
  }
}

void PaddleBackend::GetInputDataFromOption(
    const PaddleBackendOption& option,
    std::map<std::string, std::vector<float>>* max_input_data,
    std::map<std::string, std::vector<float>>* min_input_data,
    std::map<std::string, std::vector<float>>* opt_input_data) const {
  for (const auto& item : option.trt_option.min_input_data) {
    auto max_iter = option.trt_option.max_input_data.find(item.first);
    auto opt_iter = option.trt_option.opt_input_data.find(item.first);
    FDASSERT(max_iter != option.trt_option.max_input_data.end(),
             "Cannot find %s in TrtBackendOption::min_input_data.",
             item.first.c_str());
    FDASSERT(opt_iter != option.trt_option.opt_input_data.end(),
             "Cannot find %s in TrtBackendOption::opt_input_data.",
             item.first.c_str());
    (*max_input_data)[item.first].assign(max_iter->second.begin(),
                                         max_iter->second.end());
    (*opt_input_data)[item.first].assign(opt_iter->second.begin(),
                                         opt_iter->second.end());
    (*min_input_data)[item.first].assign(item.second.begin(),
                                         item.second.end());
  }
}

void PaddleBackend::CollectShapeRun(
    paddle_infer::Predictor* predictor,
    const std::map<std::string, std::vector<int>>& shape,
    const std::map<std::string, std::vector<float>>& data) const {
  auto input_names = predictor->GetInputNames();
  auto input_type = predictor->GetInputTypes();
  for (const auto& name : input_names) {
    FDASSERT(shape.find(name) != shape.end() &&
                 input_type.find(name) != input_type.end(),
             "When collect_trt_shape is true, please define max/opt/min shape "
             "for model's input:[\"%s\"] by "
             "(C++)RuntimeOption.trt_option.SetShape/"
             "(Python)RuntimeOption.trt_option.set_shape.",
             name.c_str());
    auto tensor = predictor->GetInputHandle(name);
    auto shape_value = shape.at(name);
    int shape_num = std::accumulate(shape_value.begin(), shape_value.end(), 1,
                                    std::multiplies<int>());
    tensor->Reshape(shape_value);

    if (data.find(name) != data.end()) {
      FDASSERT(data.at(name).size() == shape_num,
               "The data num and accumulate of shape must be equal for input: "
               "[\"%s\"], "
               " When Use the (C++)RuntimeOption.trt_option.SetInputData/ "
               " (Python)RuntimeOption.trt_option.set_input_data/",
               name.c_str());
    }

    auto dtype = input_type[name];
    switch (dtype) {
      case paddle_infer::DataType::FLOAT32: {
        if (data.find(name) != data.end()) {
          tensor->CopyFromCpu(data.at(name).data());
        } else {
          std::vector<float> input_data(shape_num, 1.0);
          tensor->CopyFromCpu(input_data.data());
        }
        break;
      }
      case paddle_infer::DataType::INT32: {
        if (data.find(name) != data.end()) {
          std::vector<int> input_data(data.at(name).begin(),
                                      data.at(name).end());
          tensor->CopyFromCpu(input_data.data());
        } else {
          std::vector<int> input_data(shape_num, 1);
          tensor->CopyFromCpu(input_data.data());
        }
        break;
      }
      case paddle_infer::DataType::INT64: {
        if (data.find(name) != data.end()) {
          std::vector<int64_t> input_data(data.at(name).begin(),
                                          data.at(name).end());
          tensor->CopyFromCpu(input_data.data());
        } else {
          std::vector<int64_t> input_data(shape_num, 1);
          tensor->CopyFromCpu(input_data.data());
        }
        break;
      }
      default: {
        FDASSERT(false,
                 "Input data Paddle backend only supports "
                 "FP32/INT32/INT64 currently.");
        break;
      }
    }
  }
  predictor->Run();
}

}  // namespace fastdeploy
