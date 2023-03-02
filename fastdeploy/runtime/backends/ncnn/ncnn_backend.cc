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

#include "fastdeploy/runtime/backends/ncnn/ncnn_backend.h"

#include <memory>
#include <string>

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
/// Convert data type from NCNN to FastDeploy
FDDataType NCNNDataTypeToFD(size_t elemsize, bool integer) {
  FDASSERT(elemsize == 4,
           "Only support float32/int32 NCNN dtype(elemsize=4)"
           " in FastDeploy now, but got elemsize %d",
           static_cast<int>(elemsize))
  // The default dtype of NCNN is fp32.
  return integer ? FDDataType::INT32 : FDDataType::FP32;
}

void NCNNBackend::BuildOption(const NCNNBackendOption& option) {
  option_ = option;
  // cpu num threads
  if (option_.cpu_threads > 0) {
    opt_.num_threads = option_.cpu_threads;
  }
  // device
  if (option_.device != Device::CPU) {
    FDERROR << "Backend::NCNN only support CPU now, but got" << option_.device
            << ", fallback to CPU now." << std::endl;
  }
  // lightmode
  opt_.lightmode = option_.light_mode;
  // fp32/fp16/int8
  if (option_.enable_fp16) {
#if defined(__aarch64__) || defined(_M_ARM64)
    opt_.use_fp16_packed = true;
    opt_.use_fp16_storage = true;
    opt_.use_fp16_arithmetic = true;
    FDINFO << "FP16 precision is enabled for Backend::NCNN" << std::endl;
#endif
  }
  if (option_.enable_bf16) {
#if defined(__aarch64__) || defined(_M_ARM64)
    opt_.use_bf16_storage = true;  // only storage
    opt_.use_packing_layout = true;
#endif
  }
  if (option_.enable_int8) {
    opt_.use_int8_packed = true;
    opt_.use_int8_storage = true;
    opt_.use_int8_arithmetic = true;
    opt_.use_int8_inference = true;
    FDINFO << "INT8 precision is enabled for Backend::NCNN" << std::endl;
  }
  // cpu power mode
  ncnn::set_cpu_powersave(option_.cpu_powersave);
}

std::vector<int> NCNNBackend::GetMatShapeByBlob(int id, size_t* elemsize) {
  auto blob_info = net_->blobs().at(id);
  return GetMatShape(blob_info.shape, elemsize);
}

std::vector<int> NCNNBackend::GetMatShape(const ncnn::Mat& mat,
                                          size_t* elemsize) {
  std::vector<int> shape;
  int dims = mat.dims;
  // Only support batch=1
  shape.resize(dims + 1);
  shape[0] = 1;  // batch always 1
  dims += 1;     // + batch dim
  if (dims == 4) {
    shape[1] = mat.c;  // channel
    shape[2] = mat.h;  // h
    shape[3] = mat.w;  // w
  } else if (dims == 3) {
    shape[1] = mat.h;  // h
    shape[2] = mat.w;  // w
  } else if (dims == 2) {
    shape[1] = mat.w;  // w
  } else {
    FDASSERT(false, "Not support %d dims", dims)
  }
  *elemsize = mat.elemsize;
  return shape;
}

bool NCNNBackend::SetTensorInfoByCustomOrder(
    const std::map<std::string, int>& custom_orders,
    const std::vector<const char*>& tensor_names,
    const std::vector<int>& tensor_indexes, std::vector<TensorInfo>* desc,
    std::map<std::string, int>* order) {
  desc->clear();
  order->clear();
  for (int i = 0; i < custom_orders.size(); ++i) {
    bool find = false;
    for (const auto& it : custom_orders) {
      if (it.second == i) {
        int id = -1;
        // match name
        for (int j = 0; j < tensor_names.size(); ++j) {
          if (it.first == tensor_names[j]) {
            id = j;
            break;
          }
        }
        if (id < 0) {
          FDERROR << "Cannot find name:[" << it.first << "] from NCNN model."
                  << std::endl;
          return false;
        }
        // Fill i-th info
        TensorInfo info;
        info.name = tensor_names[id];
        size_t elemsize = -1;
        info.shape = GetMatShapeByBlob(tensor_indexes[id], &elemsize);
        // Only support FP32 input/output
        info.dtype = NCNNDataTypeToFD(elemsize);
        desc->push_back(info);

        (*order)[info.name] = i;
        find = true;
      }
    }
    FDASSERT(find, "Can not match ordered info %d from NCNN Model", i)
  }
  return false;
}

bool NCNNBackend::SetTensorInfo(const std::vector<const char*>& tensor_names,
                                const std::vector<int>& tensor_indexes,
                                std::vector<TensorInfo>* desc,
                                std::map<std::string, int>* order) {
  desc->clear();
  order->clear();
  for (int i = 0; i < tensor_names.size(); ++i) {
    // Fill i-th info
    TensorInfo info;
    info.name = tensor_names[i];
    size_t elemsize = -1;
    info.shape = GetMatShapeByBlob(tensor_indexes[i], &elemsize);
    // Only support FP32 input/output
    info.dtype = NCNNDataTypeToFD(elemsize);
    desc->push_back(info);

    (*order)[info.name] = i;
  }
  return true;
}

bool NCNNBackend::Init(const RuntimeOption& runtime_option) {
  if (initialized_) {
    FDERROR << "Backend::NCNN is already initialized, cannot initialize again."
            << std::endl;
    return false;
  }
  if (runtime_option.model_format != ModelFormat::NCNN_MODEL) {
    FDERROR << "Backend::NCNN only supports model format NCNN, but now it's "
            << runtime_option.model_format << "." << std::endl;
    return false;
  }
  if (runtime_option.model_from_memory_) {
    FDERROR << "Backend::NCNN doesn't support load model from memory, "
               "please load model from disk."
            << std::endl;
    return false;
  }

  // Build ncnn::Net option, net and extractor.
  BuildOption(runtime_option.ncnn_option);
  net_ = std::shared_ptr<ncnn::Net>(new ncnn::Net());
  net_->opt = opt_;
  // Load model files
  FDASSERT(!(net_->load_model(runtime_option.model_file.c_str())),
           "Can not load model file for Backend::NCNN: %s",
           runtime_option.model_file.c_str())  // *.bin
  FDASSERT(!(net_->load_param(runtime_option.params_file.c_str())),
           "Can not load model file for Backend::NCNN: %s",
           runtime_option.params_file.c_str())  // *.param

  // Get input/output infos
  inputs_desc_.clear();
  outputs_desc_.clear();
  // NCNN will return 0 value for input/output ncnn::Mat before
  // the true inference process happended.
  input_indexes_ = net_->input_indexes();
  output_indexes_ = net_->output_indexes();
#ifndef NCNN_STRING
  FDASSERT(false,
           "NCNN_STRING is not defined, please re-compile "
           "NCNN with NCNN_STRING=ON")
#endif
  input_names_ = net_->input_names();
  output_names_ = net_->output_names();
  // Check the size of custom inputs/outputs orders
  if (!option_.in_orders.empty()) {
    FDASSERT(option_.in_orders.size() == input_indexes_.size(),
             "The size of custom input orders must be equal the size of"
             " inputs info, but got %d != %d now!",
             static_cast<int>(option_.in_orders.size()),
             static_cast<int>(input_indexes_.size()))
  }
  if (!option_.out_orders.empty()) {
    FDASSERT(option_.out_orders.size() == output_indexes_.size(),
             "The size of custom output orders must be equal the size of"
             " inputs info, but got %d != %d now!",
             static_cast<int>(option_.out_orders.size()),
             static_cast<int>(input_indexes_.size()))
  }
  if (!option_.in_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.in_orders, input_names_,
                                        input_indexes_, &inputs_desc_,
                                        &inputs_order_),
             "Backend::NCNN set input info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(input_names_, input_indexes_, &inputs_desc_,
                           &inputs_order_),
             "Backend::NCNN set input info falied!")
  }
  if (!option_.out_orders.empty()) {
    FDASSERT(SetTensorInfoByCustomOrder(option_.out_orders, output_names_,
                                        output_indexes_, &outputs_desc_,
                                        &outputs_order_),
             "Backend::NCNN set input info by custom order falied!")
  } else {
    FDASSERT(SetTensorInfo(output_names_, output_indexes_, &outputs_desc_,
                           &outputs_order_),
             "Backend::NCNN set input info falied!")
  }
  return true;
}

std::vector<int> NCNNBackend::GetNCNNShape(const std::vector<int64_t>& shape) {
  std::vector<int> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int>(shape[i]);
  }
  return new_shape;
}

std::vector<int64_t> NCNNBackend::GetFDShape(const std::vector<int>& shape) {
  std::vector<int64_t> new_shape;
  new_shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = static_cast<int64_t>(shape[i]);
  }
  return new_shape;
}

bool NCNNBackend::UpdateInputShapeAndDesc(const std::vector<FDTensor>& inputs) {
  for (int i = 0; i < inputs.size(); ++i) {
    bool find = false;
    for (int j = 0; j < inputs_desc_.size(); ++j) {
      if (inputs[i].name == inputs_desc_[j].name) {
        inputs_desc_[j].shape = GetNCNNShape(inputs[i].Shape());
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

bool NCNNBackend::Infer(std::vector<FDTensor>& inputs,
                        std::vector<FDTensor>* outputs, bool copy_to_fd) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[NCNNBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  FDASSERT(UpdateInputShapeAndDesc(inputs),
           "Backend::NCNN update input tensor shape failed!");

  ncnn::Extractor extractor = net_->create_extractor();

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = inputs_order_.find(inputs[i].name);
    if (iter == inputs_order_.end()) {
      FDERROR << "Cannot find input with name:" << inputs[i].name
              << " in loaded model." << std::endl;
      return false;
    }

    auto shape = GetNCNNShape(inputs[i].shape);
    FDASSERT(shape[0] == 1, "Only support batch=1, but got %d", shape[0])
    // Default dtype of ncnn::Mat is FP32. We only support
    // FP32 input/output now.
    if (shape.size() == 4) {  // dims = 4
      int c = shape[1], h = shape[2], w = shape[3];
      ncnn::Mat in = ncnn::Mat(inputs[i].Numel(), inputs[i].MutableData())
                         .reshape(w, h, c)
                         .clone();
      extractor.input(inputs[i].name.c_str(), in);
    } else if (shape.size() == 3) {
      int h = shape[1], w = shape[2];
      ncnn::Mat in = ncnn::Mat(inputs[i].Numel(), inputs[i].MutableData())
                         .reshape(w, h)
                         .clone();
      extractor.input(inputs[i].name.c_str(), in);
    } else if (shape.size() == 2) {
      int w = shape[1];
      ncnn::Mat in = ncnn::Mat(inputs[i].Numel(), inputs[i].MutableData())
                         .reshape(w)
                         .clone();
      extractor.input(inputs[i].name.c_str(), in);
    } else {
      FDASSERT(false, "%d dims for Backend::NCNN is not support!",
               static_cast<int>(shape.size()))
    }
  }

  // extract outputs
  std::vector<ncnn::Mat> outs(outputs_desc_.size());

  // Get output tensors, outputs_desc_ is already
  // reordered by custom output orders.
  RUNTIME_PROFILE_LOOP_BEGIN(1)
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    // type = 0, default
    // type = 1, do not convert fp16/bf16 or / and packing
    int extract_type = 0;  // may use type 1
    FDASSERT(!extractor.extract(outputs_desc_[i].name.c_str(), outs[i],
                                extract_type),
             "Cannot extract output: %s", outputs_desc_[i].name.c_str())
  }
  RUNTIME_PROFILE_LOOP_END

  // Copy -> output tensors.
  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    size_t elemsize = -1;
    outputs_desc_[i].shape = GetMatShape(outs[i], &elemsize);
    // Only support FP32 input/output now.
    outputs_desc_[i].dtype = NCNNDataTypeToFD(elemsize);
    (*outputs)[i].Resize(GetFDShape(outputs_desc_[i].shape),
                         outputs_desc_[i].dtype, outputs_desc_[i].name);
    // Handle the data via 16 bytes aligned, use
    // ncnn::Mat::channel(i) with CHW order instead of
    // copy from raw data directly. c >= 1 (1,2,3,...)
    // Refence: ncnn/blob/master/docs/faq.md
    const size_t cbytes = (*outputs)[i].Nbytes() / outs[i].c;
    for (int j = 0; j < outs[i].c; ++j) {
      uint8_t* raw_mutable_data =
          static_cast<uint8_t*>((*outputs)[i].MutableData()) + cbytes * i;
      std::memcpy(static_cast<void*>(raw_mutable_data),
                  static_cast<void*>(static_cast<float*>(outs[i].channel(j))),
                  cbytes);
    }
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

std::string NCNNBackend::ShapeStr(const std::vector<int>& shape) {
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

TensorInfo NCNNBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> NCNNBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo NCNNBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> NCNNBackend::GetOutputInfos() { return outputs_desc_; }

}  // namespace fastdeploy