#include "fastdeploy/backends/rknpu/rknpu2/rknpu2_backend.h"

namespace fastdeploy {

/***************************************************************
 *  @name       GetSDKAndDeviceVersion
 *  @brief      获取SDK和驱动的版本号
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
  std::cout << "rknn_api/rknnrt version: " << sdk_ver.api_version
            << ", driver version: " << sdk_ver.drv_version << std::endl;
  return true;
}

/***************************************************************
 *  @name      BuildOption
 *  @brief     把输入的配置参数保存到私有成员option_中
 *  @param     RKNPU2BackendOption: 见头文件定义
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
 *  @brief      读取RKNN模型，保存配置参数
 *  @param      model_file: 初始化RKNN模型
 *              params_file: 统一样式，无实际作用
 *              option: 配置参数，见头文件定义
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::InitFromRKNN(const std::string& model_file,
                                 const std::string& params_file,
                                 const RKNPU2BackendOption& option) {
  // 读取模型
  if (!this->LoadModel((char*)model_file.data())) {
    std::cout << "load model failed" << std::endl;
    return false;
  }

  // 获取sdk和驱动版本号
  if (!this->GetSDKAndDeviceVersion()) {
    std::cout << "get SDK and device version failed" << std::endl;
    return false;
  }

  // 保存配置参数
  this->BuildOption(option);

  // 设置使用的核心，只在RK3588生效
  if (this->option_.cpu_name == rknpu2_cpu_name::RK3588) {
    if (!this->SetCoreMask(option_.core_mask)) {
      std::cout << "set core mask failed" << std::endl;
      return false;
    }
  }

  // 获取模型输入输出参数
  if (!this->GetModelInputOutputInfos()) {
    std::cout << "get model input output infos failed" << std::endl;
    return false;
  }

  return true;
}

/***************************************************************
 *  @name       SetCoreMask
 *  @brief      设置运行的 NPU 核心
 *  @param      core_mask: NPU 核心的枚举类型，详情见头文件
 *                         RKNPU2BackendOption结构体
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::SetCoreMask(rknpu2_core_mask& core_mask) const {
  int ret = rknn_set_core_mask(ctx, core_mask);
  if (ret != RKNN_SUCC) {
    std::cout << "rknn_set_core_mask fail! ret=" << ret << std::endl;
    return false;
  }
  return true;
}

/***************************************************************
 *  @name       LoadModel
 *  @brief      读取模型
 *  @param      model: RKNN 模型的二进制数据或者 RKNN 模型路径。
 *  @return     bool
 *  @note       None
 ***************************************************************/
bool RKNPU2Backend::LoadModel(void* model) {
  int ret = RKNN_SUCC;
  ret = rknn_init(&ctx, model, 0, 0, nullptr);
  if (ret != RKNN_SUCC) {
    std::cout << "rknn_init fail! ret=" << ret << std::endl;
    return false;
  }
  return true;
}

/***************************************************************
 *  @name       GetModelInputOutputInfos
 *  @brief      获取模型输入输出参数
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
  input_attrs =
      (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  inputs_desc_.resize(io_num.n_input);
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error! ret=%d\n", ret);
      return false;
    }
    std::string temp_name = input_attrs[i].name;
    std::vector<int> temp_shape = {
        (int)input_attrs[i].dims[0], (int)input_attrs[i].dims[1],
        (int)input_attrs[i].dims[2], (int)input_attrs[i].dims[3]};

    FDDataType temp_dtype = FDDataType::FP32;
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_[i] = temp_input_info;
    // DumpTensorAttr(input_attrs[i]);
  }

  // Get detailed output parameters
  output_attrs =
      (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  outputs_desc_.resize(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return false;
    }
    std::string temp_name = output_attrs[i].name;
    // std::vector<int> temp_shape = {(int)output_attrs[i].dims[0], (int)output_attrs[i].dims[1],
    //                           (int)output_attrs[i].dims[2], (int)output_attrs[i].dims[3]};
    std::vector<int> temp_shape = {(int)output_attrs[i].dims[0],
                                   (int)output_attrs[i].dims[1],
                                   (int)output_attrs[i].dims[2]};
    FDDataType temp_dtype = FDDataType::FP32;
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_[i] = temp_input_info;
    DumpTensorAttr(output_attrs[i]);
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
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> RKNPU2Backend::GetInputInfos() { return inputs_desc_; }

TensorInfo RKNPU2Backend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> RKNPU2Backend::GetOutputInfos() {
  return outputs_desc_;
}

bool RKNPU2Backend::Infer(std::vector<FDTensor>& inputs,
                          std::vector<FDTensor>* outputs) {
  int ret = 0;
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[RKNPU2Backend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  // get input_data
  std::cout << "get input_data" << std::endl;
  // 需要新增判断输入输出类型是否相同

  rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
  rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;
  input_attrs[0].type = input_type;
  input_attrs[0].fmt = input_layout;

  // create input tensor memory
  rknn_tensor_mem* input_mems[1];
  input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

  // Copy input data to input tensor memory
  int height = 0;
  int width = 0;
  int channel = 0;
  width = input_attrs[0].dims[2];
  int stride = input_attrs[0].w_stride;
  if (width == stride) {
    memcpy(input_mems[0]->virt_addr, inputs[0].Data(),
           width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
  } else {
    int height = input_attrs[0].dims[1];
    int channel = input_attrs[0].dims[3];
    // copy from src to dst with stride
    uint8_t* src_ptr = (uint8_t*)inputs[0].Data();
    uint8_t* dst_ptr = (uint8_t*)input_mems[0]->virt_addr;
    // width-channel elements
    int src_wc_elems = width * channel;
    int dst_wc_elems = stride * channel;
    for (int h = 0; h < height; ++h) {
      memcpy(dst_ptr, src_ptr, src_wc_elems);
      src_ptr += src_wc_elems;
      dst_ptr += dst_wc_elems;
    }
  }

  // Create output tensor memory
  rknn_tensor_mem* output_mems[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this requires float32 to compute top5
    // allocate float32 output tensor
    int output_size = output_attrs[i].n_elems * sizeof(float);
    output_mems[i] = rknn_create_mem(ctx, output_size);
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn_set_io_mem fail! ret=" << ret << std::endl;
    return false;
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this requires float32 to compute top5
    output_attrs[i].type = RKNN_TENSOR_FLOAT32;
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    if (ret < 0) {
      FDERROR << "rknn_set_io_mem fail! ret=" << ret << std::endl;
      return false;
    }
  }

  // run rknn
  ret = rknn_run(ctx, NULL);
  if (ret != RKNN_SUCC) {
    FDERROR << "rknn run error! ret=" << ret << std::endl;
    return false;
  }

  // get result
  outputs->resize(outputs_desc_.size());
  std::vector<int64_t> temp_shape(3);
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    for (int j = 0; j < outputs_desc_[i].shape.size(); ++j) {
      temp_shape[j] = outputs_desc_[i].shape[j];
    }
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    memcpy((*outputs)[i].MutableData(), output_mems[i]->virt_addr,
           (*outputs)[i].Nbytes());
  }

  return true;
}
} // namespace fastdeploy