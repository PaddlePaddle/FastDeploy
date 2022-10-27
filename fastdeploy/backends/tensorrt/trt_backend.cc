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

#include "fastdeploy/backends/tensorrt/trt_backend.h"
#include "fastdeploy/function/cuda_cast.h"

#include <cstring>
#include <unordered_map>

#include "NvInferRuntime.h"
#include "fastdeploy/utils/utils.h"
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

namespace fastdeploy {

FDTrtLogger* FDTrtLogger::logger = nullptr;

// Check if the model can build tensorrt engine now
// If the model has dynamic input shape, it will require defined shape
// information We can set the shape range information by function
// SetTrtInputShape() But if the shape range is not defined, then the engine
// cannot build, in this case, The engine will build once there's data feeded,
// and the shape range will be updated
bool CanBuildEngine(
    const std::map<std::string, ShapeRangeInfo>& shape_range_info) {
  for (auto iter = shape_range_info.begin(); iter != shape_range_info.end();
       ++iter) {
    bool is_full_static = true;
    for (size_t i = 0; i < iter->second.shape.size(); ++i) {
      if (iter->second.shape[i] < 0) {
        is_full_static = false;
        break;
      }
    }

    if (is_full_static) {
      continue;
    }
    for (size_t i = 0; i < iter->second.shape.size(); ++i) {
      if (iter->second.min[i] < 0 || iter->second.max[i] < 0) {
        return false;
      }
    }
  }
  return true;
}

bool TrtBackend::LoadTrtCache(const std::string& trt_engine_file) {
  cudaSetDevice(option_.gpu_id);

  std::string engine_buffer;
  if (!ReadBinaryFromFile(trt_engine_file, &engine_buffer)) {
    FDERROR << "Failed to load TensorRT Engine from " << trt_engine_file << "."
            << std::endl;
    return false;
  }

  FDUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*FDTrtLogger::Get())};
  if (!runtime) {
    FDERROR << "Failed to call createInferRuntime()." << std::endl;
    return false;
  }
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_buffer.data(),
                                     engine_buffer.size()),
      FDInferDeleter());
  if (!engine_) {
    FDERROR << "Failed to call deserializeCudaEngine()." << std::endl;
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  GetInputOutputInfo();

  for (int32_t i = 0; i < engine_->getNbBindings(); ++i) {
    if (!engine_->bindingIsInput(i)) {
      continue;
    }
    auto min = ToVec(engine_->getProfileDimensions(
        i, 0, nvinfer1::OptProfileSelector::kMAX));
    auto max = ToVec(engine_->getProfileDimensions(
        i, 0, nvinfer1::OptProfileSelector::kMIN));
    auto name = std::string(engine_->getBindingName(i));
    auto iter = shape_range_info_.find(name);
    if (iter == shape_range_info_.end()) {
      FDERROR << "There's no input named '" << name << "' in loaded model."
              << std::endl;
      return false;
    }
    iter->second.Update(min);
    iter->second.Update(max);
  }
  FDINFO << "Build TensorRT Engine from cache file: " << trt_engine_file
         << " with shape range information as below," << std::endl;
  for (const auto& item : shape_range_info_) {
    FDINFO << item.second << std::endl;
  }
  return true;
}

bool TrtBackend::InitFromPaddle(const std::string& model_file,
                                const std::string& params_file,
                                const TrtBackendOption& option, bool verbose) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;

#ifdef ENABLE_PADDLE_FRONTEND
  std::vector<paddle2onnx::CustomOp> custom_ops;
  for (auto& item : option_.custom_op_info_) {
    paddle2onnx::CustomOp op;
    std::strcpy(op.op_name, item.first.c_str());
    std::strcpy(op.export_op_name, item.second.c_str());
    custom_ops.emplace_back(op);
  }
  char* model_content_ptr;
  int model_content_size = 0;
  char* calibration_cache_ptr;
  int calibration_cache_size = 0;
  if (!paddle2onnx::Export(model_file.c_str(), params_file.c_str(),
                           &model_content_ptr, &model_content_size, 11, true,
                           verbose, true, true, true, custom_ops.data(),
                           custom_ops.size(), "tensorrt",
                           &calibration_cache_ptr, &calibration_cache_size)) {
    FDERROR << "Error occured while export PaddlePaddle to ONNX format."
            << std::endl;
    return false;
  }

  if (option_.remove_multiclass_nms_) {
    char* new_model = nullptr;
    int new_model_size = 0;
    if (!paddle2onnx::RemoveMultiClassNMS(model_content_ptr, model_content_size,
                                          &new_model, &new_model_size)) {
      FDERROR << "Try to remove MultiClassNMS failed." << std::endl;
      return false;
    }
    delete[] model_content_ptr;
    std::string onnx_model_proto(new_model, new_model + new_model_size);
    delete[] new_model;
    if (calibration_cache_size) {
      std::string calibration_str(
          calibration_cache_ptr,
          calibration_cache_ptr + calibration_cache_size);
      calibration_str_ = calibration_str;
      delete[] calibration_cache_ptr;
    }
    return InitFromOnnx(onnx_model_proto, option, true);
  }

  std::string onnx_model_proto(model_content_ptr,
                               model_content_ptr + model_content_size);
  delete[] model_content_ptr;
  model_content_ptr = nullptr;
  if (calibration_cache_size) {
    std::string calibration_str(calibration_cache_ptr,
                                calibration_cache_ptr + calibration_cache_size);
    calibration_str_ = calibration_str;
    delete[] calibration_cache_ptr;
  }
  return InitFromOnnx(onnx_model_proto, option, true);
#else
  FDERROR << "Didn't compile with PaddlePaddle frontend, you can try to "
             "call `InitFromOnnx` instead."
          << std::endl;
  return false;
#endif
}

bool TrtBackend::InitFromOnnx(const std::string& model_file,
                              const TrtBackendOption& option,
                              bool from_memory_buffer) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;
  cudaSetDevice(option_.gpu_id);

  std::string onnx_content = "";
  if (!from_memory_buffer) {
    std::ifstream fin(model_file.c_str(), std::ios::binary | std::ios::in);
    if (!fin) {
      FDERROR << "[ERROR] Failed to open ONNX model file: " << model_file
              << std::endl;
      return false;
    }
    fin.seekg(0, std::ios::end);
    onnx_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(onnx_content.at(0)), onnx_content.size());
    fin.close();
  } else {
    onnx_content = model_file;
  }

  // This part of code will record the original outputs order
  // because the converted tensorrt network may exist wrong order of outputs
  outputs_order_.clear();
  auto onnx_reader =
      paddle2onnx::OnnxReader(onnx_content.c_str(), onnx_content.size());
  for (int i = 0; i < onnx_reader.num_outputs; ++i) {
    std::string name(onnx_reader.outputs[i].name);
    outputs_order_[name] = i;
  }

  shape_range_info_.clear();
  inputs_desc_.clear();
  outputs_desc_.clear();
  inputs_desc_.resize(onnx_reader.num_inputs);
  outputs_desc_.resize(onnx_reader.num_outputs);
  for (int i = 0; i < onnx_reader.num_inputs; ++i) {
    std::string name(onnx_reader.inputs[i].name);
    std::vector<int64_t> shape(
        onnx_reader.inputs[i].shape,
        onnx_reader.inputs[i].shape + onnx_reader.inputs[i].rank);
    inputs_desc_[i].name = name;
    inputs_desc_[i].shape.assign(shape.begin(), shape.end());
    inputs_desc_[i].dtype = ReaderDtypeToTrtDtype(onnx_reader.inputs[i].dtype);
    inputs_desc_[i].original_dtype = ReaderDtypeToFDDtype(onnx_reader.inputs[i].dtype);
    auto info = ShapeRangeInfo(shape);
    info.name = name;
    auto iter_min = option.min_shape.find(name);
    auto iter_max = option.max_shape.find(name);
    auto iter_opt = option.opt_shape.find(name);
    if (iter_min != option.min_shape.end()) {
      info.min.assign(iter_min->second.begin(), iter_min->second.end());
      info.max.assign(iter_max->second.begin(), iter_max->second.end());
      info.opt.assign(iter_opt->second.begin(), iter_opt->second.end());
    }
    shape_range_info_.insert(std::make_pair(name, info));
  }

  for (int i = 0; i < onnx_reader.num_outputs; ++i) {
    std::string name(onnx_reader.outputs[i].name);
    std::vector<int64_t> shape(
        onnx_reader.outputs[i].shape,
        onnx_reader.outputs[i].shape + onnx_reader.outputs[i].rank);
    outputs_desc_[i].name = name;
    outputs_desc_[i].shape.assign(shape.begin(), shape.end());
    outputs_desc_[i].dtype =
        ReaderDtypeToTrtDtype(onnx_reader.outputs[i].dtype);
    outputs_desc_[i].original_dtype =
        ReaderDtypeToFDDtype(onnx_reader.outputs[i].dtype);
  }

  if (option_.external_stream_) {
    stream_ = reinterpret_cast<cudaStream_t>(option_.external_stream_);
  } else {
    FDASSERT(cudaStreamCreate(&stream_) == 0,
           "[ERROR] Error occurs while calling cudaStreamCreate().");
  }

  if (!CreateTrtEngineFromOnnx(onnx_content)) {
    FDERROR << "Failed to create tensorrt engine." << std::endl;
    return false;
  }
  initialized_ = true;
  return true;
}

int TrtBackend::ShapeRangeInfoUpdated(const std::vector<FDTensor>& inputs) {
  bool need_update_engine = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = shape_range_info_.find(inputs[i].name);
    if (iter == shape_range_info_.end()) {
      FDERROR << "There's no input named '" << inputs[i].name
              << "' in loaded model." << std::endl;
    }
    if (iter->second.Update(inputs[i].shape) == 1) {
      need_update_engine = true;
    }
  }
  return need_update_engine;
}

bool TrtBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs) {
  if (inputs.size() != NumInputs()) {
    FDERROR << "Require " << NumInputs() << "inputs, but get " << inputs.size()
            << "." << std::endl;
    return false;
  }
  if (ShapeRangeInfoUpdated(inputs)) {
    // meet new shape output of predefined max/min shape
    // rebuild the tensorrt engine
    FDWARNING
        << "TensorRT engine will be rebuilt once shape range information "
           "changed, this may take lots of time, you can set a proper shape "
           "range before loading model to avoid rebuilding process. refer "
           "https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/"
           "tensorrt_tricks.md for more details."
        << std::endl;
    BuildTrtEngine();
  }

  SetInputs(inputs);
  AllocateOutputsBuffer(outputs);

  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    FDERROR << "Failed to Infer with TensorRT." << std::endl;
    return false;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    // if the final output tensor's dtype is different from the model output tensor's dtype,
    // then we need cast the data to the final output's dtype
    auto model_output_dtype = GetFDDataType(outputs_device_buffer_[(*outputs)[i].name].dtype());
    if ((*outputs)[i].dtype != model_output_dtype) {
      FDTensor output_tensor;
      output_tensor.SetExternalData((*outputs)[i].shape, model_output_dtype,
                                    outputs_device_buffer_[(*outputs)[i].name].data(),
                                    Device::GPU);
  
      casted_output_tensors_[(*outputs)[i].name].Resize((*outputs)[i].shape, (*outputs)[i].dtype,
                                                        (*outputs)[i].name, Device::GPU);
      CudaCast(output_tensor, &casted_output_tensors_[(*outputs)[i].name], stream_);
    } else {
      casted_output_tensors_[(*outputs)[i].name].SetExternalData(
          (*outputs)[i].shape, model_output_dtype,
          outputs_device_buffer_[(*outputs)[i].name].data(),
          Device::GPU);
    }
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    FDASSERT(cudaMemcpyAsync((*outputs)[i].Data(),
                             casted_output_tensors_[(*outputs)[i].name].Data(),
                             (*outputs)[i].Nbytes(), cudaMemcpyDeviceToHost,
                             stream_) == 0,
             "[ERROR] Error occurs while copy memory from GPU to CPU.");
  }
  FDASSERT(cudaStreamSynchronize(stream_) == cudaSuccess,
           "[ERROR] Error occurs while sync cuda stream.");

  return true;
}

void TrtBackend::GetInputOutputInfo() {
  // Read the original dtypes from inputs_desc_ and outputs_desc_
  std::unordered_map<std::string, FDDataType> inputs_original_dtype_map;
  std::unordered_map<std::string, FDDataType> outputs_original_dtype_map;
  for (size_t i = 0; i < inputs_desc_.size(); ++i) {
    inputs_original_dtype_map[inputs_desc_[i].name] = inputs_desc_[i].original_dtype;
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    outputs_original_dtype_map[outputs_desc_[i].name] = outputs_desc_[i].original_dtype;
  }

  // Re-read the tensor infos from TRT model and write into inputs_desc_ and outputs_desc_
  std::vector<TrtValueInfo>().swap(inputs_desc_);
  std::vector<TrtValueInfo>().swap(outputs_desc_);
  inputs_desc_.clear();
  outputs_desc_.clear();
  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    auto shape = ToVec(engine_->getBindingDimensions(i));
    auto dtype = engine_->getBindingDataType(i);
    if (engine_->bindingIsInput(i)) {
      auto original_dtype = inputs_original_dtype_map.count(name) ? inputs_original_dtype_map[name] : GetFDDataType(dtype);
      inputs_desc_.emplace_back(TrtValueInfo{name, shape, dtype, original_dtype});
      inputs_device_buffer_[name] = FDDeviceBuffer(dtype);
    } else {
      auto original_dtype = outputs_original_dtype_map.count(name) ? outputs_original_dtype_map[name] : GetFDDataType(dtype);
      outputs_desc_.emplace_back(TrtValueInfo{name, shape, dtype, original_dtype});
      outputs_device_buffer_[name] = FDDeviceBuffer(dtype);
      casted_output_tensors_[name] = FDTensor();
    }
  }
  bindings_.resize(num_binds);
}

void TrtBackend::SetInputs(const std::vector<FDTensor>& inputs) {
  for (const auto& item : inputs) {
    auto idx = engine_->getBindingIndex(item.name.c_str());
    std::vector<int> shape(item.shape.begin(), item.shape.end());
    auto dims = ToDims(shape);
    context_->setBindingDimensions(idx, dims);

    if (item.device == Device::GPU) {
      if (item.dtype == FDDataType::INT64) {
        FDWARNING << "TRT doesn't support INT64 input, the input tensor will be "
                     "casted to INT32" << std::endl;

        inputs_device_buffer_[item.name].resize(dims);
        FDTensor input_tensor;
        input_tensor.SetExternalData(item.shape, FDDataType::INT32,
                                     inputs_device_buffer_[item.name].data(),
                                     Device::GPU);
        CudaCast(item, &input_tensor, stream_);
      } else {
        // no copy
        inputs_device_buffer_[item.name].SetExternalData(dims, item.Data());
      }
    } else {
      // Allocate input buffer memory
      inputs_device_buffer_[item.name].resize(dims);

      // copy from cpu to gpu
      if (item.dtype == FDDataType::INT64) {
        int64_t* data = static_cast<int64_t*>(const_cast<void*>(item.Data()));
        std::vector<int32_t> casted_data(data, data + item.Numel());
        FDASSERT(cudaMemcpyAsync(inputs_device_buffer_[item.name].data(),
                                 static_cast<void*>(casted_data.data()),
                                 item.Nbytes() / 2, cudaMemcpyHostToDevice,
                                 stream_) == 0,
                 "Error occurs while copy memory from CPU to GPU.");
      } else {
        FDASSERT(cudaMemcpyAsync(inputs_device_buffer_[item.name].data(),
                                 item.Data(),
                                 item.Nbytes(), cudaMemcpyHostToDevice,
                                 stream_) == 0,
                 "Error occurs while copy memory from CPU to GPU.");
      }
    }
    // binding input buffer
    bindings_[idx] = inputs_device_buffer_[item.name].data();
  }
}

void TrtBackend::AllocateOutputsBuffer(std::vector<FDTensor>* outputs) {
  if (outputs->size() != outputs_desc_.size()) {
    outputs->resize(outputs_desc_.size());
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto idx = engine_->getBindingIndex(outputs_desc_[i].name.c_str());
    auto output_dims = context_->getBindingDimensions(idx);

    // find the original index of output
    auto iter = outputs_order_.find(outputs_desc_[i].name);
    FDASSERT(
        iter != outputs_order_.end(),
        "Cannot find output: %s of tensorrt network from the original model.",
        outputs_desc_[i].name.c_str());
    auto ori_idx = iter->second;

    // set user's outputs info
    std::vector<int64_t> shape(output_dims.d,
                               output_dims.d + output_dims.nbDims);
    (*outputs)[ori_idx].is_pinned_memory = option_.enable_pinned_memory;
    (*outputs)[ori_idx].Resize(shape, outputs_desc_[i].original_dtype,
                               outputs_desc_[i].name);

    // Allocate output buffer memory
    outputs_device_buffer_[outputs_desc_[i].name].resize(output_dims);
  
    // binding output buffer
    bindings_[idx] = outputs_device_buffer_[outputs_desc_[i].name].data();
  }
}

bool TrtBackend::BuildTrtEngine() {
  auto config =
      FDUniquePtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
  if (!config) {
    FDERROR << "Failed to call createBuilderConfig()." << std::endl;
    return false;
  }

  if (option_.enable_fp16) {
    if (!builder_->platformHasFastFp16()) {
      FDWARNING << "Detected FP16 is not supported in the current GPU, "
                   "will use FP32 instead."
                << std::endl;
    } else {
      FDINFO << "[TrtBackend] Use FP16 to inference." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  FDINFO << "Start to building TensorRT Engine..." << std::endl;

  if (context_) {
    context_.reset();
    engine_.reset();
  }

  builder_->setMaxBatchSize(option_.max_batch_size);
  config->setMaxWorkspaceSize(option_.max_workspace_size);
  auto profile = builder_->createOptimizationProfile();
  for (const auto& item : shape_range_info_) {
    FDASSERT(
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kMIN,
                               ToDims(item.second.min)),
        "[TrtBackend] Failed to set min_shape for input: %s in TrtBackend.",
        item.first.c_str());
    FDASSERT(
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kMAX,
                               ToDims(item.second.max)),
        "[TrtBackend] Failed to set max_shape for input: %s in TrtBackend.",
        item.first.c_str());
    if (item.second.opt.size() == 0) {
      FDASSERT(
          profile->setDimensions(item.first.c_str(),
                                 nvinfer1::OptProfileSelector::kOPT,
                                 ToDims(item.second.max)),
          "[TrtBackend] Failed to set opt_shape for input: %s in TrtBackend.",
          item.first.c_str());
    } else {
      FDASSERT(
          item.second.opt.size() == item.second.shape.size(),
          "Require the dimension of opt in shape range information equal to "
          "dimension of input: %s in this model, but now it's %zu != %zu.",
          item.first.c_str(), item.second.opt.size(), item.second.shape.size());
      FDASSERT(
          profile->setDimensions(item.first.c_str(),
                                 nvinfer1::OptProfileSelector::kOPT,
                                 ToDims(item.second.opt)),
          "[TrtBackend] Failed to set opt_shape for input: %s in TrtBackend.",
          item.first.c_str());
    }
  }
  config->addOptimizationProfile(profile);

  if (calibration_str_.size()) {
    if (!builder_->platformHasFastInt8()) {
      FDWARNING << "Detected INT8 is not supported in the current GPU, "
                   "will use FP32 instead."
                << std::endl;
    } else {
      FDINFO << "[TrtBackend] Use INT8 to inference." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
      Int8EntropyCalibrator2* calibrator =
          new Int8EntropyCalibrator2(calibration_str_);
      config->setInt8Calibrator(calibrator);
    }
  }

  FDUniquePtr<nvinfer1::IHostMemory> plan{
      builder_->buildSerializedNetwork(*network_, *config)};
  if (!plan) {
    FDERROR << "Failed to call buildSerializedNetwork()." << std::endl;
    return false;
  }

  FDUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*FDTrtLogger::Get())};
  if (!runtime) {
    FDERROR << "Failed to call createInferRuntime()." << std::endl;
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      FDInferDeleter());
  if (!engine_) {
    FDERROR << "Failed to call deserializeCudaEngine()." << std::endl;
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  GetInputOutputInfo();

  FDINFO << "TensorRT Engine is built successfully." << std::endl;
  if (option_.serialize_file != "") {
    FDINFO << "Serialize TensorRTEngine to local file "
           << option_.serialize_file << "." << std::endl;
    std::ofstream engine_file(option_.serialize_file.c_str(), std::ios::binary | std::ios::out);  
    if (!engine_file) {
      FDERROR << "Failed to open " << option_.serialize_file << " to write."
              << std::endl;
      return false;
    }
    engine_file.write(static_cast<char*>(plan->data()), plan->size());
    engine_file.close();
    FDINFO << "TensorRTEngine is serialized to local file "
           << option_.serialize_file
           << ", we can load this model from the seralized engine "
              "directly next time."
           << std::endl;
  }
  return true;
}

bool TrtBackend::CreateTrtEngineFromOnnx(const std::string& onnx_model_buffer) {
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  builder_ = FDUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(*FDTrtLogger::Get()));
  if (!builder_) {
    FDERROR << "Failed to call createInferBuilder()." << std::endl;
    return false;
  }
  network_ = FDUniquePtr<nvinfer1::INetworkDefinition>(
      builder_->createNetworkV2(explicitBatch));
  if (!network_) {
    FDERROR << "Failed to call createNetworkV2()." << std::endl;
    return false;
  }
  parser_ = FDUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_, *FDTrtLogger::Get()));
  if (!parser_) {
    FDERROR << "Failed to call createParser()." << std::endl;
    return false;
  }
  if (!parser_->parse(onnx_model_buffer.data(), onnx_model_buffer.size())) {
    FDERROR << "Failed to parse ONNX model by TensorRT." << std::endl;
    return false;
  }

  if (option_.serialize_file != "") {
    std::ifstream fin(option_.serialize_file, std::ios::binary | std::ios::in);
    if (fin) {
      FDINFO << "Detect serialized TensorRT Engine file in "
             << option_.serialize_file << ", will load it directly."
             << std::endl;
      fin.close();
      // clear memory buffer of the temporary member
      std::string().swap(onnx_model_buffer_);
      return LoadTrtCache(option_.serialize_file);
    }
  }

  if (!CanBuildEngine(shape_range_info_)) {
    onnx_model_buffer_ = onnx_model_buffer;
    FDWARNING << "Cannot build engine right now, because there's dynamic input "
                 "shape exists, list as below,"
              << std::endl;
    for (int i = 0; i < NumInputs(); ++i) {
      FDWARNING << "Input " << i << ": " << GetInputInfo(i) << std::endl;
    }
    FDWARNING
        << "FastDeploy will build the engine while inference with input data, "
           "and will also collect the input shape range information. You "
           "should be noticed that FastDeploy will rebuild the engine while "
           "new input shape is out of the collected shape range, this may "
           "bring some time consuming problem, refer "
           "https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/"
           "tensorrt_tricks.md for more details."
        << std::endl;
    initialized_ = true;
    return true;
  }

  if (!BuildTrtEngine()) {
    FDERROR << "Failed to build tensorrt engine." << std::endl;
  }

  // clear memory buffer of the temporary member
  std::string().swap(onnx_model_buffer_);
  return true;
}

TensorInfo TrtBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  TensorInfo info;
  info.name = inputs_desc_[index].name;
  info.shape.assign(inputs_desc_[index].shape.begin(),
                    inputs_desc_[index].shape.end());
  info.dtype = GetFDDataType(inputs_desc_[index].dtype);
  return info;
}

std::vector<TensorInfo> TrtBackend::GetInputInfos() {
  std::vector<TensorInfo> infos;
  for (auto i = 0; i < inputs_desc_.size(); i++) {
    infos.emplace_back(GetInputInfo(i));
  }
  return infos;
}

TensorInfo TrtBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  TensorInfo info;
  info.name = outputs_desc_[index].name;
  info.shape.assign(outputs_desc_[index].shape.begin(),
                    outputs_desc_[index].shape.end());
  info.dtype = GetFDDataType(outputs_desc_[index].dtype);
  return info;
}

std::vector<TensorInfo> TrtBackend::GetOutputInfos() {
  std::vector<TensorInfo> infos;
  for (auto i = 0; i < outputs_desc_.size(); i++) {
    infos.emplace_back(GetOutputInfo(i));
  }
  return infos;
}

}  // namespace fastdeploy
