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

#include "fastdeploy/runtime/backends/horizon/horizon_backend.h"
namespace fastdeploy{

bool HorizonBackend::LoadModel(void *model){
    int ret = hbDNNInitializeFromFiles(&packed_dnn_handle, &model , 1);
    if(ret != 0){
        FDERROR << "horizon fail! ret=" << ret << std::endl;
        return false;
    }
    return true;
}

bool HorizonBackend::GetModelInputOutputInfos(){
    const char **model_name_list;
    int model_count = 0;
    int ret;
    // get model name
    ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    if(ret != 0){
        FDERROR << "get model name fail! ret=" << ret << std::endl;
        return false;
    }
    // get dnn handle
    ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    if(ret != 0){
        FDERROR << "get dnn handle fail! ret=" << ret << std::endl;
        return false;
    }
    // get input infos
    // Get detailed input parameters
    int input_count = 0;
    ret = hbDNNGetInputCount(&input_count, dnn_handle);
    if(ret != 0){
        FDERROR << "get input count fail! ret=" << ret << std::endl;
        return false;
    }
    input_properties_ = (hbDNNTensorProperties*)malloc(sizeof(hbDNNTensorProperties) * input_count);
    memset(input_properties_, 0, input_count * sizeof(hbDNNTensorProperties));

    inputs_desc_.resize(input_count);

    // create input tensor memory
    input_mems_ = (hbDNNTensor**)malloc(sizeof(hbDNNTensor*) * input_count);

    // get input info and copy to input tensor info
    for (uint32_t i = 0; i < input_count; i++) {
        ret = hbDNNGetInputTensorProperties(&input_properties_[i], dnn_handle, i);

        if(ret != 0){
            FDERROR << "get input tensor properties fail! ret=" << ret << std::endl;
            return false;
        }

        if ((input_properties_[i].tensorLayout != HB_DNN_LAYOUT_NCHW)) {
            FDERROR << "horizon_backend only support input layout is NCHW"
                    << std::endl;
        }
        if(input_properties_[i].tensorType!= HB_DNN_IMG_TYPE_RGB){
            FDERROR << "horizon_backend only support input format is RGB"
                    << std::endl;
        }

        char *name = "";

        ret = hbDNNGetInputName(&name, dnn_handle, i);
        if(ret != 0){
            FDERROR << "get input tensor name fail! ret=" << ret << std::endl;
            return false;
        }
        // copy input proper to input tensor info
        std::string temp_name = name;
        std::vector<int> temp_shape{};
        int n_dims = input_properties_[i].validShape.numDimensions;

        temp_shape.resize(n_dims);
        for (int j = 0; j < n_dims; j++) {
            temp_shape[j] = (int)input_properties_[i].validShape.dimensionSize[j];
        }
        // TODO: how to get input dtype ?
        FDDataType temp_dtype = FDDataType::UINT8;
        TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
        inputs_desc_[i] = temp_input_info;
    }
    
    // get output infos
    // Get detailed output parameters
    int output_count = 0;
    ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    if(ret != 0){
        FDERROR << "get output count fail! ret=" << ret << std::endl;
        return false;
    }
    output_properties_ = (hbDNNTensorProperties*)malloc(sizeof(hbDNNTensorProperties) * output_count);
    memset(output_properties_, 0, output_count * sizeof(hbDNNTensorProperties));

    outputs_desc_.resize(output_count);
    output_mems_ = (hbDNNTensor**)malloc(sizeof(hbDNNTensor*) * output_count);

    for (uint32_t i = 0; i < output_count; i++){
        // get model output size
        ret = hbDNNGetOutputTensorProperties(&output_properties_[i], dnn_handle, i);
        
        char *name = "";
        ret = hbDNNGetOutputName(&name, dnn_handle, i);
        if(ret != 0){
            FDERROR << "get output tensor name fail! ret=" << ret << std::endl;
            return false;
        }

        // copy output proper to output tensor info
        std::string temp_name = name;
        std::vector<int> temp_shape{};
        int n_dims = output_properties_[i].validShape.numDimensions;
        
        if ((n_dims == 4) && (output_properties_[i].validShape.dimensionSize[3] == 1)) {
            n_dims--;
        }
        temp_shape.resize(n_dims);
        for (int j = 0; j < n_dims; j++) {
            temp_shape[j] = (int)output_properties_[i].validShape.dimensionSize[j];
        }

        FDDataType temp_dtype = HorizonTensorTypeToFDDataType(output_properties_[i].tensorType);

        TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
        outputs_desc_[i] = temp_input_info;
    }

    return true;
}

bool HorizonBackend::InitFromHorizon(const std::string& model_file){

}

FDDataType HorizonBackend::HorizonTensorTypeToFDDataType(hbDNNDataType type){
    if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_F16) {
        return FDDataType::FP16;
    }
    if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_F32) {
        return FDDataType::FP32;
    }
    if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_S8) {
        return FDDataType::INT8;
    }
    if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_S16) {
        return FDDataType::INT16;
    }
    if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_S32) {
        return FDDataType::INT32;
    }
    if (type == hbDNNDataType::HB_DNN_TENSOR_TYPE_U8) {
        return FDDataType::UINT8;
    }

    FDERROR << "FDDataType don't support this type" << std::endl;
    return FDDataType::UNKNOWN1;
}

} //namespace fastdeploy