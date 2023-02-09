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

#include "fastdeploy_capi/runtime_option.h"

#include "fastdeploy/utils/utils.h"
#include "fastdeploy_capi/types_internal.h"

extern "C" {

FD_C_RuntimeOptionWrapper* FD_C_CreateRuntimeOptionWrapper() {
  FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper =
      new FD_C_RuntimeOptionWrapper();
  fd_c_runtime_option_wrapper->runtime_option =
      std::unique_ptr<fastdeploy::RuntimeOption>(
          new fastdeploy::RuntimeOption());
  return fd_c_runtime_option_wrapper;
}

void FD_C_DestroyRuntimeOption(
    __fd_take FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  delete fd_c_runtime_option_wrapper;
}

void FD_C_RuntimeOptionWrapperSetModelPath(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* model_path, const char* params_path,
    const FD_C_ModelFormat format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetModelPath(std::string(model_path),
                               std::string(params_path),
                               static_cast<fastdeploy::ModelFormat>(format));
}

void FD_C_RuntimeOptionWrapperSetModelBuffer(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* model_buffer, const char* params_buffer,
    const FD_C_ModelFormat format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetModelBuffer(model_buffer, params_buffer,
                                 static_cast<fastdeploy::ModelFormat>(format));
}

void FD_C_RuntimeOptionWrapperUseCpu(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseCpu();
}

void FD_C_RuntimeOptionWrapperUseGpu(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int gpu_id) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseGpu(gpu_id);
}

void FD_C_RuntimeOptionWrapperUseRKNPU2(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    FD_C_rknpu2_CpuName rknpu2_name, FD_C_rknpu2_CoreMask rknpu2_core) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseRKNPU2(
      static_cast<fastdeploy::rknpu2::CpuName>(rknpu2_name),
      static_cast<fastdeploy::rknpu2::CoreMask>(rknpu2_core));
}

void FD_C_RuntimeOptionWrapperUseTimVX(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseTimVX();
}

void FD_C_RuntimeOptionWrapperUseAscend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseAscend();
}

void FD_C_RuntimeOptionWrapperUseKunlunXin(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int kunlunxin_id, int l3_workspace_size, FD_C_Bool locked,
    FD_C_Bool autotune, const char* autotune_file, const char* precision,
    FD_C_Bool adaptive_seqlen, FD_C_Bool enable_multi_stream) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseKunlunXin(kunlunxin_id, l3_workspace_size, bool(locked),
                               bool(autotune), std::string(autotune_file),
                               std::string(precision), bool(adaptive_seqlen),
                               bool(enable_multi_stream));
}

void FD_C_RuntimeOptionWrapperUseSophgo(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseSophgo();
}

void FD_C_RuntimeOptionWrapperSetExternalStream(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    void* external_stream) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetExternalStream(external_stream);
}

void FD_C_RuntimeOptionWrapperSetCpuThreadNum(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int thread_num) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetCpuThreadNum(thread_num);
}

void FD_C_RuntimeOptionWrapperSetOrtGraphOptLevel(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int level) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetOrtGraphOptLevel(level);
}

void FD_C_RuntimeOptionWrapperUsePaddleBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UsePaddleBackend();
}

void FD_C_RuntimeOptionWrapperUsePaddleInferBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  return FD_C_RuntimeOptionWrapperUsePaddleBackend(fd_c_runtime_option_wrapper);
}

void FD_C_RuntimeOptionWrapperUseOrtBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseOrtBackend();
}

void FD_C_RuntimeOptionWrapperUseSophgoBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseSophgoBackend();
}

void FD_C_RuntimeOptionWrapperUseTrtBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseTrtBackend();
}

void FD_C_RuntimeOptionWrapperUsePorosBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UsePorosBackend();
}

void FD_C_RuntimeOptionWrapperUseOpenVINOBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseOpenVINOBackend();
}

void FD_C_RuntimeOptionWrapperUseLiteBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseLiteBackend();
}

void FD_C_RuntimeOptionWrapperUsePaddleLiteBackend(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  return FD_C_RuntimeOptionWrapperUseLiteBackend(fd_c_runtime_option_wrapper);
}

void FD_C_RuntimeOptionWrapperSetPaddleMKLDNN(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    FD_C_Bool pd_mkldnn) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetPaddleMKLDNN(pd_mkldnn);
}

void FD_C_RuntimeOptionWrapperEnablePaddleToTrt(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnablePaddleToTrt();
}

void FD_C_RuntimeOptionWrapperDeletePaddleBackendPass(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* delete_pass_name) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DeletePaddleBackendPass(std::string(delete_pass_name));
}

void FD_C_RuntimeOptionWrapperEnablePaddleLogInfo(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnablePaddleLogInfo();
}

void FD_C_RuntimeOptionWrapperDisablePaddleLogInfo(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DisablePaddleLogInfo();
}

void FD_C_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int size) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetPaddleMKLDNNCacheSize(size);
}

void FD_C_RuntimeOptionWrapperSetOpenVINODevice(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* name) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetOpenVINODevice(std::string(name));
}

void FD_C_RuntimeOptionWrapperSetLiteOptimizedModelDir(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* optimized_model_dir) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetLiteOptimizedModelDir(std::string(optimized_model_dir));
}

void FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionPath(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* nnadapter_subgraph_partition_config_path) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetLiteSubgraphPartitionPath(
      std::string(nnadapter_subgraph_partition_config_path));
}

void FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* nnadapter_subgraph_partition_config_buffer) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetLiteSubgraphPartitionConfigBuffer(
      std::string(nnadapter_subgraph_partition_config_buffer));
}

void FD_C_RuntimeOptionWrapperSetLiteContextProperties(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* nnadapter_context_properties) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetLiteContextProperties(
      std::string(nnadapter_context_properties));
}

void FD_C_RuntimeOptionWrapperSetLiteModelCacheDir(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* nnadapter_model_cache_dir) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetLiteModelCacheDir(std::string(nnadapter_model_cache_dir));
}

void FD_C_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* nnadapter_mixed_precision_quantization_config_path) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
}

void FD_C_RuntimeOptionWrapperEnableLiteFP16(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnableLiteFP16();
}

void FD_C_RuntimeOptionWrapperDisableLiteFP16(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DisableLiteFP16();
}

void FD_C_RuntimeOptionWrapperEnableLiteInt8(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnableLiteInt8();
}

void FD_C_RuntimeOptionWrapperDisableLiteInt8(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DisableLiteInt8();
}

void FD_C_RuntimeOptionWrapperSetLitePowerMode(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    FD_C_LitePowerMode mode) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetLitePowerMode(
      static_cast<fastdeploy::LitePowerMode>(mode));
}

void FD_C_RuntimeOptionWrapperEnableTrtFP16(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnableTrtFP16();
}

void FD_C_RuntimeOptionWrapperDisableTrtFP16(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DisableTrtFP16();
}

void FD_C_RuntimeOptionWrapperSetTrtCacheFile(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const char* cache_file_path) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetTrtCacheFile(std::string(cache_file_path));
}

void FD_C_RuntimeOptionWrapperEnablePinnedMemory(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnablePinnedMemory();
}

void FD_C_RuntimeOptionWrapperDisablePinnedMemory(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DisablePinnedMemory();
}

void FD_C_RuntimeOptionWrapperEnablePaddleTrtCollectShape(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->EnablePaddleTrtCollectShape();
}

void FD_C_RuntimeOptionWrapperDisablePaddleTrtCollectShape(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->DisablePaddleTrtCollectShape();
}

void FD_C_RuntimeOptionWrapperSetOpenVINOStreams(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int num_streams) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetOpenVINOStreams(num_streams);
}

void FD_C_RuntimeOptionWrapperUseIpu(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int device_num, int micro_batch_size, FD_C_Bool enable_pipelining,
    int batches_per_step) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->UseIpu(device_num, micro_batch_size, enable_pipelining,
                         batches_per_step);
}

void FD_C_RuntimeOptionWrapperSetIpuConfig(
    __fd_keep FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    FD_C_Bool enable_fp16, int replica_num, float available_memory_proportion,
    FD_C_Bool enable_half_partial) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  runtime_option->SetIpuConfig(enable_fp16, replica_num,
                               available_memory_proportion,
                               enable_half_partial);
}

}  // extern "C"
