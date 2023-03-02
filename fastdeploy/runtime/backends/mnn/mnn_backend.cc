// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/runtime/backends/mnn/mnn_backend.h"

#include <memory>
#include <string>

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
// Convert data type from MNN to fastdeploy
FDDataType MNNDataTypeToFD(const halide_type_t& dtype) {
  halide_type_code_t code = dtype.code;
  uint8_t bits = dtype.bits;
  // halide_type_int = 0, signed integers
  if (code == halide_type_code_t::halide_type_int) {
    if (bits == 8) {
      return FDDataType::INT8;
    } else if (bits == 16) {
      return FDDataType::INT16;
    } else if (bits == 32) {
      return FDDataType::INT32;
    } else {
      return FDDataType::INT64;
    }
  } else if (code == halide_type_code_t::halide_type_uint) {
    if (bits == 8) {
      return FDDataType::UINT8;
    } else {
      FDASSERT(false,
               "FastDeploy only support UINT8 for unsigned integers now"
               "but got %d",
               bits);
    }
  } else if (code == halide_type_code_t::halide_type_float) {
    if (bits == 16) {
      return FDDataType::FP16;
    } else if (bits == 32) {
      return FDDataType::FP32;
    } else if (bits == 64) {
      return FDDataType::FP64;
    } else {
      FDASSERT(false,
               "FastDeploy only support FP16/FP32/FP64 for float now"
               "but got %d",
               bits);
    }
  }
  FDASSERT(false, "Unexpected data type of halide_type_handle.");
  return FDDataType::FP32;
}

MNNBackend::~MNNBackend() {
  if (interpreter_.get()) {
    interpreter_->releaseModel();
    if (session_) {
      interpreter_->releaseSession(session_);
    }
  }
}

void MNNBackend::BuildOption(const MNNBackendOption& option) {
  option_ = option;
  // cpu num threads
  if (option_.cpu_threads > 0) {
    schedule_config_.numThread = option_.cpu_threads;
  }
  // device
  if (option_.device == Device::CPU) {
    schedule_config_.type = MNN_FORWARD_CPU;
  } else {
    FDERROR << "Backend::MNN only support CPU now, but got" << option_.device
            << ", fallback to CPU now." << std::endl;
  }
  // power mode
  if (option_.power_mode != MNNPowerMode::MNN_POWER_NORMAL) {
    backend_config_.power =
        static_cast<MNN::BackendConfig::PowerMode>(option_.power_mode);
  }
  // fp32/fp16/int8
  if (option_.enable_fp16) {
    backend_config_.precision = MNN::BackendConfig::Precision_Low;
    FDINFO << "FP16 precision is enabled for Backend::MNN" << std::endl;
  }
  schedule_config_.backendConfig = &backend_config_;
  // custom tensors to keep to avoid memory reuse
  if (!option_.save_tensors.empty()) {
    schedule_config_.saveTensors = option_.save_tensors;
  }
}

bool MNNBackend::SetTensorInfoByCustomOrder(
    const std::map<std::string, int>& custom_orders,
    const std::map<std::string, MNN::Tensor*>& tensors_info,
    std::vector<TensorInfo>* desc, std::map<std::string, int>* order) {
  desc->clear();
  order->clear();
  for (int i = 0; i < custom_orders.size(); ++i) {
    bool find = false;
    for (const auto& it : custom_orders) {
      if (it.second == i) {
        auto iter = tensors_info.find(it.first);
        if (iter == tensors_info.end()) {
          FDERROR << "Cannot find name:[" << it.first << "] from MNN model."
                  << std::endl;
          return false;
        }
        // Check data layout NCHW
        FDASSERT(iter->second->getDimensionType() == MNN::Tensor::CAFFE,
                 "Backend::MNN only support NCHW data layout now!")
        // Fill i-th info
        TensorInfo info;
        info.name = iter->first;
        info.shape = iter->second->shape();
        info.dtype = MNNDataTypeToFD(iter->second->getType());
        desc->push_back(info);

        (*order)[info.name] = i;
        find = true;
      }
    }
    FDASSERT(find, "Can not match ordered info %d from MNN Model", i)
  }
  return true;
}

bool MNNBackend::SetTensorInfo(
    const std::map<std::string, MNN::Tensor*>& tensors_info,
    std::vector<TensorInfo>* desc, std::map<std::string, int>* order) {
  desc->clear();
  order->clear();
  int i = 0;
  for (const auto& iter : tensors_info) {
    // Check data layout NCHW
    FDASSERT(iter.second->getDimensionType() == MNN::Tensor::CAFFE,
             "Backend::MNN only support NCHW data layout now!")
    // Fill i-th info
    TensorInfo info;
    info.name = iter.first;
    info.shape = iter.second->shape();
    info.dtype = MNNDataTypeToFD(iter.second->getType());
    desc->push_back(info);

    (*order)[info.name] = i;
    i += 1;
  }
  return true;
}

bool MNNBackend::Init(const RuntimeOption& runtime_option) {
  if (initialized_) {
    FDERROR << "Backend::MNN is already initialized, cannot initialize again."
            << std::endl;
    return false;
  }
  if (runtime_option.model_format != ModelFormat::MNN_MODEL) {
    FDERROR << "Backend::MNN only supports model format MNN, but now it's "
            << runtime_option.model_format << "." << std::endl;
    return false;
  }
  if (runtime_option.model_from_memory_) {
    FDERROR << "Backend::MNN doesn't support load model from memory, "
               "please load model from disk."
            << std::endl;
    return false;
  }

  // Build schedule, interpreter and session
  BuildOption(runtime_option.mnn_option);
  interpreter_ = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(runtime_option.model_file.c_str()));
  session_ = interpreter_->createSession(schedule_config_);

  // Get input/output infos
  inputs_desc_.clear();
  outputs_desc_.clear();
  // MNN model may not keep the same order with original model
  // So here will reorder it's inputs and outputs according to
  // custom input/output orders set by the user.
  const std::map<std::string, MNN::Tensor*>& inputs_info =
      interpreter_->getSessionInputAll(session_);
  const std::map<std::string, MNN::Tensor*>& outputs_info =
      interpreter_->getSessionOutputAll(session_);
  // Check the size of custom inputs/outputs orders
  if (!option_.in_orders.empty()) {
    FDASSERT(option_.in_orders.size() == inputs_info.size(),
             "The size of custom input orders must be equal the size of"
             " inputs info, but got %d != %d now!",
             static_cast<int>(option_.in_orders.size()),
             static_cast<int>(inputs_info.size()))
  }
  if (!option_.out_orders.empty()) {
    int saved_size = option_.save_tensors.size();
    FDASSERT(option_.out_orders.size() == outputs_info.size() - saved_size,
             "The size of custom output orders must be equal the size of"
             " outputs info, but got %d != %d now!",
             static_cast<int>(option_.out_orders.size()),
             static_cast<int>(outputs_info.size()) - saved_size)
  }
  // Check the name in custom inputs/outputs orders and reorder
  if (!option_.save_tensors.empty()) {
    FDASSERT(!option_.out_orders.empty(),
             "Out orders can not be"
             "empty if mnn_option.save_tensors has been set.")
  }
  if (!option_.in_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.in_orders, inputs_info,
                                        &inputs_desc_, &inputs_order_),
             "Backend::MNN set input info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(inputs_info, &inputs_desc_, &inputs_order_),
             "Backend::MNN set input info falied!")
  }
  if (!option_.out_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.out_orders, outputs_info,
                                        &outputs_desc_, &outputs_order_),
             "Backend::MNN set output info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(outputs_info, &outputs_desc_, &outputs_order_),
             "Backend::MNN set output info falied!")
  }
  return true;
}

std::vector<int> MNNBackend::GetMNNShape(const std::vector<int64_t>& shape) {
  std::vector<int> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int>(shape[i]);
  }
  return new_shape;
}

std::vector<int64_t> MNNBackend::GetFDShape(const std::vector<int>& shape) {
  std::vector<int64_t> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int64_t>(shape[i]);
  }
  return new_shape;
}

bool MNNBackend::IsTensorShapeDirty(const std::vector<int>& old_tensor_shape,
                                    const std::vector<int>& new_data_shape) {
  if (old_tensor_shape.size() != new_data_shape.size()) {
    return true;
  }
  for (int i = 0; i < old_tensor_shape.size(); ++i) {
    if (old_tensor_shape[i] != new_data_shape[i]) {
      return true;
    }
  }
  return false;
}

std::string MNNBackend::ShapeStr(const std::vector<int>& shape) {
  std::string str = "[";
  for (int j = 0; j < shape.size(); ++j) {
    str += std::to_string(shape[j]);
    if (j == shape.size() - 1) {
      str += "]";
    } else {
      str += ",";
    }
  }
  return str;
}

bool MNNBackend::UpdateInputShapeAndDesc(const std::vector<FDTensor>& inputs) {
  const std::map<std::string, MNN::Tensor*>& old_inputs_info =
      interpreter_->getSessionInputAll(session_);
  bool tensor_shape_is_dirty = false;
  for (int i = 0; i < inputs.size(); ++i) {
    auto iter = old_inputs_info.find(inputs[i].name);
    if (iter == old_inputs_info.end()) {
      FDERROR << "Cannot find input name:" << inputs[i].name
              << " from MNN model." << std::endl;
      return false;
    }
    auto new_data_shape = GetMNNShape(inputs[i].shape);
    auto old_tensor_shape = iter->second->shape();
    // Resize tensor -> fixed shape, this method will resize
    // the tensor when the shape is already dirty (the shape
    // of tensor is not match current input data).
    if (IsTensorShapeDirty(old_tensor_shape, new_data_shape)) {
      interpreter_->resizeTensor(iter->second, new_data_shape);
      FDWARNING << "Tensor is dirty, Backend::MNN choose to "
                << "resize tensor [" << iter->first << "] "
                << ShapeStr(old_tensor_shape) << " -> "
                << ShapeStr(new_data_shape) << std::endl;
      tensor_shape_is_dirty = true;
    }
  }
  // Resize session if old tensor shape is dirty.
  if (tensor_shape_is_dirty) {
    interpreter_->resizeSession(session_);
  };
  // Update inputs_desc for new input shapes.
  const std::map<std::string, MNN::Tensor*>& new_inputs_info =
      interpreter_->getSessionInputAll(session_);
  if (!option_.in_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.in_orders, new_inputs_info,
                                        &inputs_desc_, &inputs_order_),
             "Backend::MNN update input info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(new_inputs_info, &inputs_desc_, &inputs_order_),
             "Backend::MNN update input info falied!")
  }
  return true;
}

bool MNNBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs, bool copy_to_fd) {
  if (benchmark_option_.enable_profile) {
    FDWARNING << "Backend::MNN change the input tensor's values "
              << "according to it's memory reuse policy. So, the " << std::endl;
    FDWARNING << "output tensors will tend to be randomly after "
              << "the first inference of the profile loop." << std::endl;
  }
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[MNNBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  FDASSERT(UpdateInputShapeAndDesc(inputs),
           "Backend::MNN update input tensor shape failed!");

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = inputs_order_.find(inputs[i].name);
    if (iter == inputs_order_.end()) {
      FDERROR << "Cannot find input with name:" << inputs[i].name
              << " in loaded model." << std::endl;
      return false;
    }
    // Get input tensor by name in MNN, only support NCHW now.
    auto input_tensor =
        interpreter_->getSessionInput(session_, inputs[i].name.c_str());
#define MNN_COPY_NCHW_DATA_FROM_HOST(__type__)                                \
  {                                                                           \
    auto host_nchw_tensor = MNN::Tensor::create<__type__>(                    \
        GetMNNShape(inputs[i].shape), const_cast<void*>(inputs[i].CpuData()), \
        MNN::Tensor::CAFFE);                                                  \
    input_tensor->copyFromHostTensor(host_nchw_tensor);                       \
    delete host_nchw_tensor;                                                  \
  }

    switch (inputs[i].dtype) {
      case FDDataType::FP32:
        MNN_COPY_NCHW_DATA_FROM_HOST(float)
        break;
      case FDDataType::INT32:
        MNN_COPY_NCHW_DATA_FROM_HOST(int)
        break;
      case FDDataType::INT8:
        MNN_COPY_NCHW_DATA_FROM_HOST(int8_t)
        break;
      case FDDataType::INT64:
        MNN_COPY_NCHW_DATA_FROM_HOST(int64_t)
        break;
      default:
        FDASSERT(false, "Unexpected data type of %s.",
                 Str(inputs[i].dtype).c_str())
        break;
    }
#undef MNN_COPY_NCHW_DATA_FROM_HOST
  }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  FDASSERT(interpreter_->runSession(session_) == MNN::NO_ERROR,
           "Backend::MNN runSession falied!")
  RUNTIME_PROFILE_LOOP_END

  // Get output tensors, outputs_desc_ is already
  // reordered by custom output orders.
  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    // Fetch device output tensor by ordered name
    auto device_tensor =
        interpreter_->getSessionOutput(session_, outputs_desc_[i].name.c_str());
    FDASSERT(device_tensor, "Backend::MNN failed to get output tensors: %s",
             outputs_desc_[i].name.c_str())
    auto tensor_dtype = MNNDataTypeToFD(device_tensor->getType());
    // Update the shape in desc
    outputs_desc_[i].dtype = tensor_dtype;
    outputs_desc_[i].shape = device_tensor->shape();
    // From device -> host, NCHW -> FDTensor
    MNN::Tensor host_tensor(device_tensor, device_tensor->getDimensionType());
    device_tensor->copyToHostTensor(&host_tensor);
    (*outputs)[i].Resize(GetFDShape(host_tensor.shape()),
                         outputs_desc_[i].dtype, outputs_desc_[i].name);
    std::memcpy((*outputs)[i].MutableData(), host_tensor.host<void>(),
                (*outputs)[i].Nbytes());
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

TensorInfo MNNBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> MNNBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo MNNBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> MNNBackend::GetOutputInfos() { return outputs_desc_; }

}  // namespace fastdeploy