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

#include "fastdeploy/runtime/backends/tnn/tnn_backend.h"

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
// Convert data type from TNN to fastdeploy
FDDataType TNNDataTypeToFD(const tnn::DataType& dtype) {
  if (dtype == tnn::DataType::DATA_TYPE_AUTO) {
    FDASSERT(false, "Not support DATA_TYPE_AUTO with Backend::TNN")
  } else if (dtype == tnn::DataType::DATA_TYPE_FLOAT) {
    return FDDataType::FP32;
  } else if (dtype == tnn::DataType::DATA_TYPE_HALF) {
    return FDDataType::FP16;
  } else if (dtype == tnn::DataType::DATA_TYPE_INT8) {
    return FDDataType::INT8;
  } else if (dtype == tnn::DataType::DATA_TYPE_INT32) {
    return FDDataType::INT32;
  } else if (dtype == tnn::DataType::DATA_TYPE_INT64) {
    return FDDataType::INT64;
  } else {
    FDASSERT(false,
             "Not support tnn::DataType %d for"
             " Backend::TNN now!",
             dtype)
  }
  return FDDataType::FP32;
}

// Convert mat type from TNN to fastdeploy
FDDataType TNNMatTypeToFD(const tnn::MatType& mtype) {
  if (mtype == tnn::MatType::NCHW_FLOAT) {
    return FDDataType::FP32;
  } else if (mtype == tnn::MatType::NC_INT32) {
    return FDDataType::INT32;
  } else if (mtype == tnn::MatType::N8UC3) {
    return FDDataType::UINT8;
  } else {
    FDASSERT(false,
             "Not support tnn::MatType %d for"
             " Backend::TNN now!",
             mtype)
  }
  return FDDataType::FP32;
}

// reference:
// https://github.com/Tencent/TNN/blob/master/examples/base/utils/utils.cc
std::string TNNBackend::ContentBufferFromFile(const char* proto_or_model_path) {
  std::ifstream file(proto_or_model_path, std::ios::binary);
  if (file.is_open()) {
    file.seekg(0, file.end);
    int size = file.tellg();
    char* content = new char[size];
    file.seekg(0, file.beg);
    file.read(content, size);
    std::string file_content;
    file_content.assign(content, size);
    delete[] content;
    file.close();
    return file_content;
  }
  return "";
}

void TNNBackend::BuildOption(const TNNBackendOption& option) {
  option_ = option;
  // NetworkConfig: device
  if (option_.device == Device::CPU) {
#if defined(__ANDROID__)
    network_config_.device_type = tnn::DeviceType::DEVICE_ARM;
#else
    network_config_.device_type = tnn::DeviceType::DEVICE_X86;
#endif
  } else {
    FDERROR << "Backend::TNN only support CPU now, but got" << option_.device
            << ", fallback to CPU now." << std::endl;
  }
  // NetworkConfig: fp32/fp16/int8(not support)
  network_config_.precision = tnn::Precision::PRECISION_HIGH;  // default fp32
  if (option_.enable_fp16) {
    network_config_.precision = tnn::Precision::PRECISION_NORMAL;  // fp16
    FDINFO << "FP16 precision is enabled for Backend::TNN" << std::endl;
  } else if (option_.enable_bf16) {
    network_config_.precision = tnn::Precision::PRECISION_LOW;  // bf16
    FDINFO << "BF16 precision is enabled for Backend::TNN" << std::endl;
  } else if (option_.enable_int8) {
    network_config_.precision = tnn::Precision::PRECISION_AUTO;
    FDINFO << "INT8 precision is enabled for Backend::TNN" << std::endl;
  }
  // NetworkConfig: data format, only support NCHW now
  network_config_.data_format = tnn::DataFormat::DATA_FORMAT_NCHW;
  // Cpu power mode
  tnn::CpuUtils::SetCpuPowersave(option_.cpu_powersave);
}

bool TNNBackend::SetTensorInfoByCustomOrder(
    const std::map<std::string, int>& custom_orders, const tnn::BlobMap& blobs,
    std::vector<TensorInfo>* desc, std::map<std::string, int>* order) {
  desc->clear();
  order->clear();
  for (int i = 0; i < custom_orders.size(); ++i) {
    bool find = false;
    for (const auto& it : custom_orders) {
      if (it.second == i) {
        auto iter = blobs.find(it.first);
        if (iter == blobs.end()) {
          FDERROR << "Cannot find name:[" << it.first << "] from TNN model."
                  << std::endl;
          return false;
        }
        // Check data layout NCHW
        auto data_format = iter->second->GetBlobDesc().data_format;
        FDASSERT((data_format == tnn::DataFormat::DATA_FORMAT_NCHW) ||
                     (data_format == tnn::DataFormat::DATA_FORMAT_NC4HW4),
                 "Backend::TNN only support NCHW/_NC4HW4 data layout now!"
                 " but got %d !",
                 data_format)
        // Fill i-th info
        TensorInfo info;
        info.name = iter->first;
        info.shape = iter->second->GetBlobDesc().dims;
        info.dtype = TNNDataTypeToFD(iter->second->GetBlobDesc().data_type);
        desc->push_back(info);

        (*order)[info.name] = i;
        find = true;
      }
    }
    FDASSERT(find, "Can not match ordered from TNN Model", i)
  }
  return true;
}

bool TNNBackend::SetTensorInfo(const tnn::BlobMap& blobs,
                               std::vector<TensorInfo>* desc,
                               std::map<std::string, int>* order) {
  desc->clear();
  order->clear();
  int i = 0;
  for (const auto& iter : blobs) {
    // Check data layout NCHW
    auto data_format = iter.second->GetBlobDesc().data_format;
    FDASSERT((data_format == tnn::DataFormat::DATA_FORMAT_NCHW) ||
                 (data_format == tnn::DataFormat::DATA_FORMAT_NC4HW4),
             "Backend::TNN only support NCHW/NC4HW4 data layout now!"
             " but got %d !",
             data_format)
    // Fill i-th info
    TensorInfo info;
    info.name = iter.first;
    info.shape = iter.second->GetBlobDesc().dims;
    info.dtype = TNNDataTypeToFD(iter.second->GetBlobDesc().data_type);
    desc->push_back(info);

    (*order)[info.name] = i;
    i += 1;
  }
  return true;
}

std::vector<int> TNNBackend::GetTNNShape(const std::vector<int64_t>& shape) {
  std::vector<int> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int>(shape[i]);
  }
  return new_shape;
}

std::vector<int64_t> TNNBackend::GetFDShape(const std::vector<int>& shape) {
  std::vector<int64_t> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int64_t>(shape[i]);
  }
  return new_shape;
}

std::string TNNBackend::ShapeStr(const std::vector<int>& shape) {
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

bool TNNBackend::Init(const RuntimeOption& runtime_option) {
  if (initialized_) {
    FDERROR << "Backend::TNN is already initialized, cannot initialize again."
            << std::endl;
    return false;
  }
  if (runtime_option.model_format != ModelFormat::TNN_MODEL) {
    FDERROR << "Backend::TNN only supports model format TNN, but now it's "
            << runtime_option.model_format << "." << std::endl;
    return false;
  }
  if (runtime_option.model_from_memory_) {
    FDERROR << "Backend::TNN doesn't support load model from memory, "
               "please load model from disk."
            << std::endl;
    return false;
  }

  // Init network config
  BuildOption(runtime_option.tnn_option);
  // Init model config
  model_config_.model_type = tnn::ModelType::MODEL_TYPE_TNN;
  std::string proto_content_buffer, model_content_buffer;
  model_content_buffer =
      ContentBufferFromFile(runtime_option.model_file.c_str());
  proto_content_buffer =
      ContentBufferFromFile(runtime_option.params_file.c_str());
  model_config_.params = {proto_content_buffer, model_content_buffer};
  // Create TNN Net and Instance.
  tnn::Status status;
  net_ = std::make_shared<tnn::TNN>();
  FDASSERT(net_->Init(model_config_) == tnn::TNN_OK,
           "Backend::TNN init failed!")
  instance_ = net_->CreateInst(network_config_, status);
  FDASSERT(status == tnn::TNN_OK, "Backend::TNN CreateInst failed!")
  // Cpu num threads
  instance_->SetCpuNumThreads(option_.cpu_threads);

  // Get input/output infos
  inputs_desc_.clear();
  outputs_desc_.clear();
  FDASSERT(instance_->GetAllInputBlobs(inputs_blob_map_) == tnn::TNN_OK,
           "Backend::TNN get all input blobs failed!");
  FDASSERT(instance_->GetAllOutputBlobs(outputs_blob_map_) == tnn::TNN_OK,
           "Backend::TNN get all output blobs failed!");
  // Check the size of custom inputs/outputs orders
  if (!option_.in_orders.empty()) {
    FDASSERT(option_.in_orders.size() == inputs_blob_map_.size(),
             "The size of custom input orders must be equal the size of"
             " inputs info, but got %d != %d now!",
             static_cast<int>(option_.in_orders.size()),
             static_cast<int>(inputs_blob_map_.size()))
  }
  if (!option_.out_orders.empty()) {
    FDASSERT(option_.out_orders.size() == outputs_blob_map_.size(),
             "The size of custom output orders must be equal the size of"
             " outputs info, but got %d != %d now!",
             static_cast<int>(option_.out_orders.size()),
             static_cast<int>(outputs_blob_map_.size()))
  }
  if (!option_.in_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.in_orders, inputs_blob_map_,
                                        &inputs_desc_, &inputs_order_),
             "Backend::TNN set input info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(inputs_blob_map_, &inputs_desc_, &inputs_order_),
             "Backend::TNN set input info falied!")
  }
  if (!option_.out_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.out_orders, outputs_blob_map_,
                                        &outputs_desc_, &outputs_order_),
             "Backend::TNN set output info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(outputs_blob_map_, &outputs_desc_, &outputs_order_),
             "Backend::TNN set output info falied!")
  }
  return true;
}

bool TNNBackend::UpdateInputShapeAndDesc(const std::vector<FDTensor>& inputs) {
  for (int i = 0; i < inputs.size(); ++i) {
    bool find = false;
    for (int j = 0; j < inputs_desc_.size(); ++j) {
      if (inputs[i].name == inputs_desc_[j].name) {
        inputs_desc_[j].shape = GetTNNShape(inputs[i].Shape());
        inputs_desc_[j].dtype = inputs[j].Dtype();
        find = true;
        break;
      }
    }
    if (!find) {
      return false;
    }
  }
  return true;
}

bool TNNBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs, bool copy_to_fd) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[TNNBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  FDASSERT(UpdateInputShapeAndDesc(inputs),
           "Backend::TNN update input tensor shape failed!")

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = inputs_order_.find(inputs[i].name);
    if (iter == inputs_order_.end()) {
      FDERROR << "Cannot find input with name:" << inputs[i].name
              << " in loaded model." << std::endl;
      return false;
    }

#define TNN_COPY_NCHW_DATA_FROM_HOST(__mat_type__)                          \
  {                                                                         \
    auto in_mat = std::make_shared<tnn::Mat>(                               \
        network_config_.device_type, tnn::MatType::__mat_type__,            \
        GetTNNShape(inputs[i].shape), inputs[i].MutableData());             \
    FDASSERT(in_mat->GetData(), "Init input data for Backend::TNN failed!") \
    FDASSERT(                                                               \
        instance_->SetInputMat(in_mat, {}, inputs[i].name) == tnn::TNN_OK,  \
        "Set input data for Backend::TNN instance_ failed!");               \
  }

    switch (inputs[i].dtype) {
      case FDDataType::FP32:
        TNN_COPY_NCHW_DATA_FROM_HOST(NCHW_FLOAT)
        break;
      case FDDataType::INT32:
        TNN_COPY_NCHW_DATA_FROM_HOST(NC_INT32)
        break;
      case FDDataType::UINT8:
        TNN_COPY_NCHW_DATA_FROM_HOST(N8UC3)
        break;
      default:
        FDASSERT(false, "Unsupport data type of %s for Backend::TNN",
                 Str(inputs[i].dtype).c_str())
        break;
    }
#undef TNN_COPY_NCHW_DATA_FROM_HOST
  }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  FDASSERT(instance_->Forward() == tnn::TNN_OK,
           "Backend::TNN instance_ Forward failed!");  // Forward sync
  RUNTIME_PROFILE_LOOP_END

  // Get output tensors, outputs_desc_ is already
  // reordered by custom output orders.
  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    std::shared_ptr<tnn::Mat> out_mat;
    FDASSERT(
        instance_->GetOutputMat(out_mat, {}, outputs_desc_[i].name,
                                network_config_.device_type) == tnn::TNN_OK,
        "Backend::TNN instance_ GetOutputMat [%s] failed!",
        outputs_desc_[i].name.c_str())
    outputs_desc_[i].shape = out_mat->GetDims();
    // Copy -> FD Tensor
    (*outputs)[i].Resize(GetFDShape(outputs_desc_[i].shape),
                         outputs_desc_[i].dtype, outputs_desc_[i].name);
    std::memcpy((*outputs)[i].MutableData(), out_mat->GetData(),
                (*outputs)[i].Nbytes());
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

TensorInfo TNNBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> TNNBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo TNNBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> TNNBackend::GetOutputInfos() { return outputs_desc_; }

}  // namespace fastdeploy