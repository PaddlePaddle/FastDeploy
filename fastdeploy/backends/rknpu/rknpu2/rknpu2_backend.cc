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
  // Release memory uniformly here
  if (input_attrs_ != nullptr) {
    free(input_attrs_);
  }

  if (output_attrs_ != nullptr) {
    free(output_attrs_);
  }

  for (uint32_t i = 0; i < io_num.n_input; i++) {
    rknn_destroy_mem(ctx, input_mems_[i]);
  }
  if(input_mems_ != nullptr){
    free(input_mems_);
  }

  for (uint32_t i = 0; i < io_num.n_output; i++) {
    rknn_destroy_mem(ctx, output_mems_[i]);
  }
  if(output_mems_ != nullptr){
    free(output_mems_);
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
  // get sdk and device version
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
  // save cpu_name
  this->option_.cpu_name = option.cpu_name;

  // save context
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
 *  @brief      set NPU core for model
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
  input_attrs_ =
      (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
  memset(input_attrs_, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  inputs_desc_.resize(io_num.n_input);

  // create input tensor memory
  // rknn_tensor_mem* input_mems[io_num.n_input];
  input_mems_ = (rknn_tensor_mem**)malloc(sizeof(rknn_tensor_mem*) * io_num.n_input);

  // get input info and copy to input tensor info
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs_[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_init error! ret=%d\n", ret);
      return false;
    }
    if((input_attrs_[i].fmt != RKNN_TENSOR_NHWC) &&
        (input_attrs_[i].fmt != RKNN_TENSOR_UNDEFINED)){
      FDERROR << "rknpu2_backend only support input format is NHWC or UNDEFINED" << std::endl;
    }
    DumpTensorAttr(input_attrs_[i]);

    // copy input_attrs_ to input tensor info
    std::string temp_name = input_attrs_[i].name;
    std::vector<int> temp_shape{};
    temp_shape.resize(input_attrs_[i].n_dims);
    for (int j = 0; j < input_attrs_[i].n_dims; j++) {
      temp_shape[j] = (int)input_attrs_[i].dims[j];
    }
    FDDataType temp_dtype =
        fastdeploy::RKNPU2Backend::RknnTensorTypeToFDDataType(
            input_attrs_[i].type);
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[i] = temp_input_info;
  }

  // Get detailed output parameters
  output_attrs_ =
      (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
  memset(output_attrs_, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  outputs_desc_.resize(io_num.n_output);

  // Create output tensor memory
  output_mems_ = (rknn_tensor_mem**)malloc(sizeof(rknn_tensor_mem*) * io_num.n_output);;

  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs_[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      FDERROR << "rknn_query fail! ret = " << ret << std::endl;
      return false;
    }

    // If the output dimension is 3, the runtime will automatically change it to 4. 
    // Obviously, this is wrong, and manual correction is required here.
    int n_dims = output_attrs_[i].n_dims;
    if((n_dims == 4) && (output_attrs_[i].dims[3] == 1)){
      n_dims--;
      FDWARNING << "The output[" 
                << i
                << "].shape[3] is 1, remove this dim." 
                << std::endl;
    }

    DumpTensorAttr(output_attrs_[i]);
    // copy output_attrs_ to output tensor
    std::string temp_name = output_attrs_[i].name;
    std::vector<int> temp_shape{};
    temp_shape.resize(n_dims);
    for (int j = 0; j < n_dims; j++) {
      temp_shape[j] = (int)output_attrs_[i].dims[j];
    }

    FDDataType temp_dtype =
        fastdeploy::RKNPU2Backend::RknnTensorTypeToFDDataType(
            output_attrs_[i].type);
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_[i] = temp_input_info;
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

  if(!this->infer_init){
    for (uint32_t i = 0; i < io_num.n_input; i++) {
      // Judge whether the input and output types are the same
      rknn_tensor_type input_type =
          fastdeploy::RKNPU2Backend::FDDataTypeToRknnTensorType(inputs[i].dtype);
      if (input_type != input_attrs_[i].type) {
        FDWARNING << "The input tensor type != model's inputs type."
                  << "The input_type need " << get_type_string(input_attrs_[i].type)
                  << ",but inputs["<< i << "].type is " << get_type_string(input_type)
                  << std::endl;
      }

      // Create input tensor memory
      input_attrs_[i].type = input_type;
      input_attrs_[i].size = inputs[0].Nbytes();
      input_attrs_[i].size_with_stride = inputs[0].Nbytes();
      input_attrs_[i].pass_through = 0;
      input_mems_[i] = rknn_create_mem(ctx, inputs[i].Nbytes());
      if (input_mems_[i] == nullptr) {
        FDERROR << "rknn_create_mem input_mems_ error." << std::endl;
        return false;
      }

      // Set input tensor memory
      ret = rknn_set_io_mem(ctx, input_mems_[i], &input_attrs_[i]);
      if (ret != RKNN_SUCC) {
        FDERROR << "input tensor memory rknn_set_io_mem fail! ret=" << ret
                << std::endl;
        return false;
      }
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
      // Most post-processing does not support the fp16 format.
      // The unified output here is float32
      uint32_t output_size = output_attrs_[i].n_elems * sizeof(float);
      output_mems_[i] = rknn_create_mem(ctx, output_size);
      if (output_mems_[i] == nullptr) {
        FDERROR << "rknn_create_mem output_mems_ error." << std::endl;
        return false;
      }
      // default output type is depend on model, this requires float32 to compute top5
      output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
      ret = rknn_set_io_mem(ctx, output_mems_[i], &output_attrs_[i]);
      // set output memory and attribute
      if (ret != RKNN_SUCC) {
        FDERROR << "output tensor memory rknn_set_io_mem fail! ret=" << ret
                << std::endl;
        return false;
      }
    }

    this->infer_init = true;
  }
  
  // Copy input data to input tensor memory
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    uint32_t width = input_attrs_[i].dims[2];
    uint32_t stride = input_attrs_[i].w_stride;
    if (width == stride) {
      if (inputs[i].Data() == nullptr) {
        FDERROR << "inputs[0].Data is NULL." << std::endl;
        return false;
      }
      memcpy(input_mems_[i]->virt_addr, inputs[i].Data(), inputs[i].Nbytes());
    } else {
      FDERROR << "[RKNPU2Backend] only support width == stride." << std::endl;
      return false;
    }
  }
  

  // run rknn
  ret = rknn_run(ctx, nullptr);
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn run error! ret=" << ret << std::endl;
    return false;
  }

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
    memcpy((*outputs)[i].MutableData(), (float*)output_mems_[i]->virt_addr,
           (*outputs)[i].Nbytes());
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