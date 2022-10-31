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
#include "fastdeploy/backends/rknpu/rknpu2/rknpu2_backend.h"

namespace fastdeploy {
RKNPU2Backend::~RKNPU2Backend() {
  if(input_attrs != nullptr){
    free(input_attrs);
  }
  if(output_attrs != nullptr){
    free(output_attrs);
  }
}
/***************************************************************
 *  @name       GetSDKAndDeviceVersion
 *  @brief      get RKNN sdk and device version
 *  @param      None
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::GetSDKAndDeviceVersion() {
  int ret;
  // 获取sdk 和 驱动的版本
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return false;
  }
  FDINFO << "rknn_api/rknnrt version: " << sdk_ver.api_version
         << ", driver version: " << sdk_ver.drv_version << std::endl;
  return true;
}

/***************************************************************
 *  @name      BuildOption
 *  @brief     save option
 *  @param     RKNPU2BackendOption
 *  @note      None
 ***************************************************************/
void RKNPU2Backend::BuildOption(const RKNPU2BackendOption& option) {
  this->option_ = option;
  // 保存cpu参数，部分函数只使用于指定cpu
  // cpu默认设置为RK3588
  this->option_.cpu_name = option.cpu_name;

  // 保存context参数
  this->option_.core_mask = option.core_mask;
}

/***************************************************************
 *  @name       InitFromRKNN
 *  @brief      Initialize RKNN model
 *  @param      model_file: Binary data for the RKNN model or the path of RKNN model.
 *              params_file: None
 *              option: config
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::InitFromRKNN(const std::string& model_file,
                                 const RKNPU2BackendOption& option) {
  // LoadModel
  if (!this->LoadModel((char*)model_file.data())) {
    FDERROR << "load model failed" << std::endl;
    return false;
  }

  // GetSDKAndDeviceVersion
  if (!this->GetSDKAndDeviceVersion()) {
    FDERROR << "get SDK and device version failed" << std::endl;
    return false;
  }

  // BuildOption
  this->BuildOption(option);

  // SetCoreMask if RK3588
  if (this->option_.cpu_name == rknpu2::CpuName::RK3588) {
    if (!this->SetCoreMask(option_.core_mask)) {
      FDERROR << "set core mask failed" << std::endl;
      return false;
    }
  }

  // GetModelInputOutputInfos
  if (!this->GetModelInputOutputInfos()) {
    FDERROR << "get model input output infos failed" << std::endl;
    return false;
  }

  return true;
}

/***************************************************************
 *  @name       SetCoreMask
 *  @brief      设置运行的 NPU 核心
 *  @param      core_mask: The specification of NPU core setting.
 *  @return     bool
 *  @note       Only support RK3588
 ***************************************************************/
bool RKNPU2Backend::SetCoreMask(rknpu2::CoreMask& core_mask) const {
  int ret = rknn_set_core_mask(ctx, static_cast<rknn_core_mask>(core_mask));
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn_set_core_mask fail! ret=" << ret << std::endl;
    return false;
  }
  return true;
}

/***************************************************************
 *  @name       LoadModel
 *  @brief      read rknn model
 *  @param      model: Binary data for the RKNN model or the path of RKNN model.
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::LoadModel(void* model) {
  int ret = RKNN_SUCC;
  ret = rknn_init(&ctx, model, 0, 0, nullptr);
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn_init fail! ret=" << ret << std::endl;
    return false;
  }
  return true;
}

/***************************************************************
 *  @name       GetModelInputOutputInfos
 *  @brief      Get the detailed input and output infos of Model
 *  @param      None
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::GetModelInputOutputInfos() {
  int ret = RKNN_SUCC;

  // Get the number of model inputs and outputs
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    return false;
  }

  // Get detailed input parameters
  input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  inputs_desc_.resize(io_num.n_input);
  // FDINFO << "========== RKNNInputTensorInfo ==========" << std::endl;
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_init error! ret=%d\n", ret);
      return false;
    }
    std::string temp_name = input_attrs[i].name;
    std::vector<int> temp_shape{};
    temp_shape.resize(input_attrs[i].n_dims);
    for (int j = 0; j < input_attrs[i].n_dims; j++) {
      temp_shape[j] = (int)input_attrs[i].dims[j];
    }

    FDDataType temp_dtype =
        fastdeploy::RKNPU2Backend::RknnTensorTypeToFDDataType(
            input_attrs[i].type);
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[i] = temp_input_info;
    // DumpTensorAttr(input_attrs[i]);
  }

  // Get detailed output parameters
  output_attrs =
      (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  outputs_desc_.resize(io_num.n_output);
  // FDINFO << "========== RKNNOutputTensorInfo ==========" << std::endl;
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      FDERROR << "rknn_query fail! ret = " << ret << std::endl;
      return false;
    }
    std::string temp_name = output_attrs[i].name;
    std::vector<int> temp_shape{};
    temp_shape.resize(output_attrs[i].n_dims);
    for (int j = 0; j < output_attrs[i].n_dims; j++) {
      temp_shape[j] = (int)output_attrs[i].dims[j];
    }
    FDDataType temp_dtype =
        fastdeploy::RKNPU2Backend::RknnTensorTypeToFDDataType(
            output_attrs[i].type);
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_[i] = temp_input_info;
    // DumpTensorAttr(output_attrs[i]);
  }
  return true;
}

/***************************************************************
 *  @name       DumpTensorAttr
 *  @brief      Get the model's detailed inputs and outputs
 *  @param      rknn_tensor_attr
 *  @return     None
 *  @note       None
 ***************************************************************/
void RKNPU2Backend::DumpTensorAttr(rknn_tensor_attr& attr) {
  printf("index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], "
         "n_elems=%d, size=%d, fmt=%s, type=%s, "
         "qnt_type=%s, zp=%d, scale=%f\n",
         attr.index, attr.name, attr.n_dims, attr.dims[0], attr.dims[1],
         attr.dims[2], attr.dims[3], attr.n_elems, attr.size,
         get_format_string(attr.fmt), get_type_string(attr.type),
         get_qnt_type_string(attr.qnt_type), attr.zp, attr.scale);
}

TensorInfo RKNPU2Backend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs())
  return inputs_desc_[index];
}

std::vector<TensorInfo> RKNPU2Backend::GetInputInfos() { return inputs_desc_; }

TensorInfo RKNPU2Backend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs())
  return outputs_desc_[index];
}

std::vector<TensorInfo> RKNPU2Backend::GetOutputInfos() {
  return outputs_desc_;
}

bool RKNPU2Backend::Infer(std::vector<FDTensor>& inputs,
                          std::vector<FDTensor>* outputs) {
  int ret = RKNN_SUCC;
  // Judge whether the input and output size are the same
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[RKNPU2Backend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  // the input size only can be one
  if (inputs.size() > 1) {
    FDERROR << "[RKNPU2Backend] Size of the inputs only support 1."
            << std::endl;
    return false;
  }

  // Judge whether the input and output types are the same
  rknn_tensor_type input_type =
      fastdeploy::RKNPU2Backend::FDDataTypeToRknnTensorType(inputs[0].dtype);
  if (input_type != input_attrs[0].type) {
    FDWARNING << "The input tensor type != model's inputs type."
              << "The input_type need " << get_type_string(input_attrs[0].type)
              << ",but inputs[0].type is " << get_type_string(input_type)
              << std::endl;
  }

  rknn_tensor_format input_layout =
      RKNN_TENSOR_NHWC; // RK3588 only support NHWC
  input_attrs[0].type = input_type;
  input_attrs[0].fmt = input_layout;
  input_attrs[0].size = inputs[0].Nbytes();
  input_attrs[0].size_with_stride = inputs[0].Nbytes();
  input_attrs[0].pass_through = 0;

  // create input tensor memory
  rknn_tensor_mem* input_mems[1];
  input_mems[0] = rknn_create_mem(ctx, inputs[0].Nbytes());
  if (input_mems[0] == nullptr) {
    FDERROR << "rknn_create_mem input_mems error." << std::endl;
    return false;
  }

  // Copy input data to input tensor memory
  uint32_t width = input_attrs[0].dims[2];
  uint32_t stride = input_attrs[0].w_stride;
  if (width == stride) {
    if (inputs[0].Data() == nullptr) {
      FDERROR << "inputs[0].Data is NULL." << std::endl;
      return false;
    }
    memcpy(input_mems[0]->virt_addr, inputs[0].Data(), inputs[0].Nbytes());
  } else {
    FDERROR << "[RKNPU2Backend] only support width == stride." << std::endl;
    return false;
  }

  // Create output tensor memory
  rknn_tensor_mem* output_mems[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // Most post-processing does not support the fp16 format.
    // The unified output here is float32
    uint32_t output_size = output_attrs[i].n_elems * sizeof(float);
    output_mems[i] = rknn_create_mem(ctx, output_size);
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
  if (ret != RKNN_SUCC) {
    FDERROR << "input tensor memory rknn_set_io_mem fail! ret=" << ret
            << std::endl;
    return false;
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this requires float32 to compute top5
    output_attrs[i].type = RKNN_TENSOR_FLOAT32;
    ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    // set output memory and attribute
    if (ret != RKNN_SUCC) {
      FDERROR << "output tensor memory rknn_set_io_mem fail! ret=" << ret
              << std::endl;
      return false;
    }
  }

  // run rknn
  ret = rknn_run(ctx, nullptr);
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn run error! ret=" << ret << std::endl;
    return false;
  }
  rknn_destroy_mem(ctx, input_mems[0]);

  // get result
  outputs->resize(outputs_desc_.size());
  std::vector<int64_t> temp_shape(4);
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    temp_shape.resize(outputs_desc_[i].shape.size());
    for (int j = 0; j < outputs_desc_[i].shape.size(); ++j) {
      temp_shape[j] = outputs_desc_[i].shape[j];
    }
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    memcpy((*outputs)[i].MutableData(), (float*)output_mems[i]->virt_addr, (*outputs)[i].Nbytes());
    rknn_destroy_mem(ctx, output_mems[i]);
  }

  return true;
}

/***************************************************************
 *  @name       RknnTensorTypeToFDDataType
 *  @brief      Change RknnTensorType To FDDataType
 *  @param      rknn_tensor_type
 *  @return     None
 *  @note       Most post-processing does not support the fp16 format. 
 *              Therefore, if the input is FP16, the output will be FP32.
 ***************************************************************/
FDDataType RKNPU2Backend::RknnTensorTypeToFDDataType(rknn_tensor_type type) {
  if (type == rknn_tensor_type::RKNN_TENSOR_FLOAT16) {
    return FDDataType::FP32;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_FLOAT32) {
    return FDDataType::FP32;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_INT8) {
    return FDDataType::INT8;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_INT16) {
    return FDDataType::INT16;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_INT32) {
    return FDDataType::INT32;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_UINT8) {
    return FDDataType::UINT8;
  }
  if (type == rknn_tensor_type::RKNN_TENSOR_BOOL) {
    return FDDataType::BOOL;
  }
  FDERROR << "FDDataType don't support this type" << std::endl;
  return FDDataType::UNKNOWN1;
}

/***************************************************************
 *  @name       FDDataTypeToRknnTensorType
 *  @brief      Change FDDataType To RknnTensorType
 *  @param      FDDataType
 *  @return     None
 *  @note       None
 ***************************************************************/
rknn_tensor_type
RKNPU2Backend::FDDataTypeToRknnTensorType(fastdeploy::FDDataType type) {
  if (type == FDDataType::FP16) {
    return rknn_tensor_type::RKNN_TENSOR_FLOAT16;
  }
  if (type == FDDataType::FP32) {
    return rknn_tensor_type::RKNN_TENSOR_FLOAT32;
  }
  if (type == FDDataType::INT8) {
    return rknn_tensor_type::RKNN_TENSOR_INT8;
  }
  if (type == FDDataType::INT16) {
    return rknn_tensor_type::RKNN_TENSOR_INT16;
  }
  if (type == FDDataType::INT32) {
    return rknn_tensor_type::RKNN_TENSOR_INT32;
  }
  if (type == FDDataType::UINT8) {
    return rknn_tensor_type::RKNN_TENSOR_UINT8;
  }
  if (type == FDDataType::BOOL) {
    return rknn_tensor_type::RKNN_TENSOR_BOOL;
  }
  FDERROR << "rknn_tensor_type don't support this type" << std::endl;
  return RKNN_TENSOR_TYPE_MAX;
}
} // namespace fastdeploy