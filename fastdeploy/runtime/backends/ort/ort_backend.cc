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

#include "fastdeploy/runtime/backends/ort/ort_backend.h"

#include <memory>

#include "fastdeploy/core/float16.h"
#include "fastdeploy/runtime/backends/ort/ops/adaptive_pool2d.h"
#include "fastdeploy/runtime/backends/ort/ops/multiclass_nms.h"
#include "fastdeploy/runtime/backends/ort/utils.h"
#include "fastdeploy/utils/utils.h"
#ifdef ENABLE_PADDLE2ONNX
#include "paddle2onnx/converter.h"
#endif

namespace fastdeploy {

std::vector<OrtCustomOp*> OrtBackend::custom_operators_ =
    std::vector<OrtCustomOp*>();

void OrtBackend::BuildOption(const OrtBackendOption& option) {
  option_ = option;
  if (option.graph_optimization_level >= 0) {
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel(option.graph_optimization_level));
  }
  if (option.intra_op_num_threads > 0) {
    session_options_.SetIntraOpNumThreads(option.intra_op_num_threads);
  }
  if (option.inter_op_num_threads > 0) {
    session_options_.SetInterOpNumThreads(option.inter_op_num_threads);
  }
  if (option.execution_mode >= 0) {
    session_options_.SetExecutionMode(ExecutionMode(option.execution_mode));
  }
  if (option.device == Device::GPU) {
    auto all_providers = Ort::GetAvailableProviders();
    bool support_cuda = false;
    std::string providers_msg = "";
    for (size_t i = 0; i < all_providers.size(); ++i) {
      providers_msg = providers_msg + all_providers[i] + ", ";
      if (all_providers[i] == "CUDAExecutionProvider") {
        support_cuda = true;
      }
    }
    if (!support_cuda) {
      FDWARNING << "Compiled fastdeploy with onnxruntime doesn't "
                   "support GPU, the available providers are "
                << providers_msg << "will fallback to CPUExecutionProvider."
                << std::endl;
      option_.device = Device::CPU;
    } else {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = option.device_id;
      if (option.external_stream_) {
        cuda_options.has_user_compute_stream = 1;
        cuda_options.user_compute_stream = option.external_stream_;
      }
      session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }
  }
}

bool OrtBackend::Init(const RuntimeOption& option) {
  if (option.device != Device::CPU && option.device != Device::GPU) {
    FDERROR
        << "Backend::ORT only supports Device::CPU/Device::GPU, but now its "
        << option.device << "." << std::endl;
    return false;
  }
  OrtBackendOption ort_option = option.ort_option;
  ort_option.device = option.device;
  ort_option.device_id = option.device_id;
  ort_option.external_stream_ = option.external_stream_;

  if (option.model_format == ModelFormat::PADDLE) {
    if (option.model_from_memory_) {
      return InitFromPaddle(option.model_file, option.params_file, ort_option);
    }
    std::string model_buffer, params_buffer;
    FDASSERT(ReadBinaryFromFile(option.model_file, &model_buffer),
             "Failed to read model file.");
    FDASSERT(ReadBinaryFromFile(option.params_file, &params_buffer),
             "Failed to read parameters file.");
    return InitFromPaddle(model_buffer, params_buffer, ort_option);
  } else if (option.model_format == ModelFormat::ONNX) {
    if (option.model_from_memory_) {
      return InitFromOnnx(option.model_file, ort_option);
    }
    std::string model_buffer;
    FDASSERT(ReadBinaryFromFile(option.model_file, &model_buffer),
             "Failed to read model file.");
    return InitFromOnnx(model_buffer, ort_option);
  } else {
    FDERROR << "Only support Paddle/ONNX model format for OrtBackend."
            << std::endl;
    return false;
  }
  return false;
}

bool OrtBackend::InitFromPaddle(const std::string& model_buffer,
                                const std::string& params_buffer,
                                const OrtBackendOption& option, bool verbose) {
  if (initialized_) {
    FDERROR << "OrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  char* model_content_ptr;
  int model_content_size = 0;
  bool save_external = false;
#ifdef ENABLE_PADDLE2ONNX
  std::vector<paddle2onnx::CustomOp> ops;
  ops.resize(2);
  strcpy(ops[0].op_name, "multiclass_nms3");
  strcpy(ops[0].export_op_name, "MultiClassNMS");
  strcpy(ops[1].op_name, "pool2d");
  strcpy(ops[1].export_op_name, "AdaptivePool2d");

  if (!paddle2onnx::Export(model_buffer.c_str(), model_buffer.size(),
                           params_buffer.c_str(), params_buffer.size(),
                           &model_content_ptr, &model_content_size, 11, true,
                           verbose, true, true, true, ops.data(), 2,
                           "onnxruntime", nullptr, 0, "", &save_external)) {
    FDERROR << "Error occured while export PaddlePaddle to ONNX format."
            << std::endl;
    return false;
  }

  std::string onnx_model_proto(model_content_ptr,
                               model_content_ptr + model_content_size);
  delete[] model_content_ptr;
  model_content_ptr = nullptr;
  if (save_external) {
    std::string model_file_name = "model.onnx";
    std::fstream f(model_file_name, std::ios::out);
    FDASSERT(f.is_open(), "Can not open file: %s to save model.",
             model_file_name.c_str());
    f << onnx_model_proto;
    f.close();
  }
  return InitFromOnnx(onnx_model_proto, option);
#else
  FDERROR << "Didn't compile with PaddlePaddle Frontend, you can try to "
             "call `InitFromOnnx` instead."
          << std::endl;
#endif
  return false;
}

bool OrtBackend::InitFromOnnx(const std::string& model_file,
                              const OrtBackendOption& option) {
  if (initialized_) {
    FDERROR << "OrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }

  BuildOption(option);
  InitCustomOperators();
  session_ = {env_, model_file.data(), model_file.size(), session_options_};
  binding_ = std::make_shared<Ort::IoBinding>(session_);

  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);
  size_t n_inputs = session_.GetInputCount();
  for (size_t i = 0; i < n_inputs; ++i) {
    auto input_name = session_.GetInputName(i, allocator);
    auto type_info = session_.GetInputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    inputs_desc_.emplace_back(OrtValueInfo{input_name, shape, data_type});
    allocator.Free(input_name);
  }

  size_t n_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < n_outputs; ++i) {
    auto output_name = session_.GetOutputName(i, allocator);
    auto type_info = session_.GetOutputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    outputs_desc_.emplace_back(OrtValueInfo{output_name, shape, data_type});

    Ort::MemoryInfo out_memory_info("Cpu", OrtDeviceAllocator, 0,
                                    OrtMemTypeDefault);
    binding_->BindOutput(output_name, out_memory_info);

    allocator.Free(output_name);
  }
  initialized_ = true;
  return true;
}

void OrtBackend::OrtValueToFDTensor(const Ort::Value& value, FDTensor* tensor,
                                    const std::string& name, bool copy_to_fd) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto data_type = info.GetElementType();
  size_t numel = info.GetElementCount();
  auto shape = info.GetShape();
  FDDataType dtype;

  if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    dtype = FDDataType::FP32;
    numel *= sizeof(float);
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    dtype = FDDataType::INT32;
    numel *= sizeof(int32_t);
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    dtype = FDDataType::INT64;
    numel *= sizeof(int64_t);
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    dtype = FDDataType::FP64;
    numel *= sizeof(double);
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    dtype = FDDataType::FP16;
    numel *= sizeof(float16);
  } else {
    FDASSERT(
        false,
        "Unrecognized data type of %d while calling OrtBackend::CopyToCpu().",
        data_type);
  }
  const void* value_ptr = value.GetTensorData<void*>();
  if (copy_to_fd) {
    tensor->Resize(shape, dtype, name);
    memcpy(tensor->MutableData(), value_ptr, numel);
  } else {
    tensor->name = name;
    tensor->SetExternalData(shape, dtype, const_cast<void*>(value_ptr),
                            Device::CPU);
  }
}

bool OrtBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs, bool copy_to_fd) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[OrtBackend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  // from FDTensor to Ort Inputs
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto ort_value = CreateOrtValue(inputs[i], option_.device == Device::GPU);
    binding_->BindInput(inputs[i].name.c_str(), ort_value);
  }

  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0,
                                OrtMemTypeDefault);
    binding_->BindOutput(outputs_desc_[i].name.c_str(), memory_info);
  }

  // Inference with inputs
  try {
    session_.Run({}, *(binding_.get()));
  } catch (const std::exception& e) {
    FDERROR << "Failed to Infer: " << e.what() << std::endl;
    return false;
  }

  // Convert result after inference
  std::vector<Ort::Value> ort_outputs = binding_->GetOutputValues();
  outputs->resize(ort_outputs.size());
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    OrtValueToFDTensor(ort_outputs[i], &((*outputs)[i]), outputs_desc_[i].name,
                       copy_to_fd);
  }

  return true;
}

TensorInfo OrtBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  TensorInfo info;
  info.name = inputs_desc_[index].name;
  info.shape.assign(inputs_desc_[index].shape.begin(),
                    inputs_desc_[index].shape.end());
  info.dtype = GetFdDtype(inputs_desc_[index].dtype);
  return info;
}

std::vector<TensorInfo> OrtBackend::GetInputInfos() {
  auto size = inputs_desc_.size();
  std::vector<TensorInfo> infos;
  infos.reserve(size);
  for (auto i = 0; i < size; i++) {
    infos.emplace_back(GetInputInfo(i));
  }
  return infos;
}

TensorInfo OrtBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  TensorInfo info;
  info.name = outputs_desc_[index].name;
  info.shape.assign(outputs_desc_[index].shape.begin(),
                    outputs_desc_[index].shape.end());
  info.dtype = GetFdDtype(outputs_desc_[index].dtype);
  return info;
}

std::vector<TensorInfo> OrtBackend::GetOutputInfos() {
  std::vector<TensorInfo> infos;
  for (auto i = 0; i < outputs_desc_.size(); i++) {
    infos.emplace_back(GetOutputInfo(i));
  }
  return infos;
}

void OrtBackend::InitCustomOperators() {
#ifndef NON_64_PLATFORM
  if (custom_operators_.size() == 0) {
    MultiClassNmsOp* multiclass_nms = new MultiClassNmsOp{};
    custom_operators_.push_back(multiclass_nms);
    if (option_.device == Device::GPU) {
      AdaptivePool2dOp* adaptive_pool2d =
          new AdaptivePool2dOp{"CUDAExecutionProvider"};
      custom_operators_.push_back(adaptive_pool2d);
    } else {
      AdaptivePool2dOp* adaptive_pool2d =
          new AdaptivePool2dOp{"CPUExecutionProvider"};
      custom_operators_.push_back(adaptive_pool2d);
    }
  }
  for (size_t i = 0; i < custom_operators_.size(); ++i) {
    custom_op_domain_.Add(custom_operators_[i]);
  }
  session_options_.Add(custom_op_domain_);
#endif
}

}  // namespace fastdeploy
