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

namespace fastdeploy {
std::unique_ptr<fastdeploy::RuntimeOption> &CheckAndConvertFD_RuntimeOption(
    FD_RuntimeOption *fd_runtime_option) {
  FDASSERT(fd_runtime_option != nullptr,
           "The pointer of fd_runtime_option shouldn't be nullptr.");
  return fd_runtime_option->runtime_option;
}
}  // namespace fastdeploy

extern "C" {

FD_RuntimeOption *FD_CreateRuntimeOption() {
  FD_RuntimeOption *fd_runtime_option = new FD_RuntimeOption();
  fd_runtime_option->runtime_option =
      std::unique_ptr<fastdeploy::RuntimeOption>(
          new fastdeploy::RuntimeOption());
  return fd_runtime_option;
}

void FD_DestroyRuntimeOption(__fd_take FD_RuntimeOption *fd_runtime_option) {
  delete fd_runtime_option;
}

void FD_RuntimeOptionSetModelPath(__fd_keep FD_RuntimeOption *fd_runtime_option,
                                  const char *model_path,
                                  const char *params_path,
                                  const FD_ModelFormat format) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetModelPath(std::string(model_path),
                               std::string(params_path),
                               static_cast<fastdeploy::ModelFormat>(format));
}

void FD_RuntimeOptionSetModelBuffer(
    __fd_keep FD_RuntimeOption *fd_runtime_option, const char *model_buffer,
    const char *params_buffer,
    const FD_ModelFormat format) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetModelBuffer(model_buffer, params_buffer,
                                 static_cast<fastdeploy::ModelFormat>(format));
}

void FD_RuntimeOptionUseCpu(__fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseCpu();
}

void FD_RuntimeOptionUseGpu(__fd_keep FD_RuntimeOption *fd_runtime_option,
                            int gpu_id) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseGpu(gpu_id);
}

void FD_RuntimeOptionUseRKNPU2(__fd_keep FD_RuntimeOption *fd_runtime_option,
                               FD_rknpu2_CpuName rknpu2_name,
                               FD_rknpu2_CoreMask rknpu2_core) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseRKNPU2(
      static_cast<fastdeploy::rknpu2::CpuName>(rknpu2_name),
      static_cast<fastdeploy::rknpu2::CoreMask>(rknpu2_core));
}

void FD_RuntimeOptionUseTimVX(__fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseTimVX();
}

void FD_RuntimeOptionUseAscend(__fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseAscend();
}

void FD_RuntimeOptionUseKunlunXin(__fd_keep FD_RuntimeOption *fd_runtime_option,
                                  int kunlunxin_id, int l3_workspace_size,
                                  FD_Bool locked, FD_Bool autotune,
                                  const char *autotune_file,
                                  const char *precision,
                                  FD_Bool adaptive_seqlen,
                                  FD_Bool enable_multi_stream) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseKunlunXin(kunlunxin_id, l3_workspace_size, bool(locked),
                               bool(autotune), std::string(autotune_file),
                               std::string(precision), bool(adaptive_seqlen),
                               bool(enable_multi_stream));
}

void FD_RuntimeOptionUseSophgo(__fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseSophgo();
}

void FD_RuntimeOptionSetExternalStream(
    __fd_keep FD_RuntimeOption *fd_runtime_option, void *external_stream) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetExternalStream(external_stream);
}

void FD_RuntimeOptionSetCpuThreadNum(
    __fd_keep FD_RuntimeOption *fd_runtime_option, int thread_num) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetCpuThreadNum(thread_num);
}

void FD_RuntimeOptionSetOrtGraphOptLevel(
    __fd_keep FD_RuntimeOption *fd_runtime_option, int level) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetOrtGraphOptLevel(level);
}

void FD_RuntimeOptionUsePaddleBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UsePaddleBackend();
}

void FD_RuntimeOptionUsePaddleInferBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  return FD_RuntimeOptionUsePaddleBackend(fd_runtime_option);
}

void FD_RuntimeOptionUseOrtBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseOrtBackend();
}

void FD_RuntimeOptionUseSophgoBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseSophgoBackend();
}

void FD_RuntimeOptionUseTrtBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseTrtBackend();
}

void FD_RuntimeOptionUsePorosBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UsePorosBackend();
}

void FD_RuntimeOptionUseOpenVINOBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseOpenVINOBackend();
}

void FD_RuntimeOptionUseLiteBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseLiteBackend();
}

void FD_RuntimeOptionUsePaddleLiteBackend(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  return FD_RuntimeOptionUseLiteBackend(fd_runtime_option);
}

void FD_RuntimeOptionSetPaddleMKLDNN(
    __fd_keep FD_RuntimeOption *fd_runtime_option, FD_Bool pd_mkldnn) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetPaddleMKLDNN(pd_mkldnn);
}

void FD_RuntimeOptionEnablePaddleToTrt(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnablePaddleToTrt();
}

void FD_RuntimeOptionDeletePaddleBackendPass(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *delete_pass_name) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DeletePaddleBackendPass(std::string(delete_pass_name));
}

void FD_RuntimeOptionEnablePaddleLogInfo(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnablePaddleLogInfo();
}

void FD_RuntimeOptionDisablePaddleLogInfo(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DisablePaddleLogInfo();
}

void FD_RuntimeOptionSetPaddleMKLDNNCacheSize(
    __fd_keep FD_RuntimeOption *fd_runtime_option, int size) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetPaddleMKLDNNCacheSize(size);
}

void FD_RuntimeOptionSetOpenVINODevice(
    __fd_keep FD_RuntimeOption *fd_runtime_option, const char *name) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetOpenVINODevice(std::string(name));
}

void FD_RuntimeOptionSetLiteOptimizedModelDir(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *optimized_model_dir) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetLiteOptimizedModelDir(std::string(optimized_model_dir));
}

void FD_RuntimeOptionSetLiteSubgraphPartitionPath(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *nnadapter_subgraph_partition_config_path) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetLiteSubgraphPartitionPath(
      std::string(nnadapter_subgraph_partition_config_path));
}

void FD_RuntimeOptionSetLiteSubgraphPartitionConfigBuffer(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *nnadapter_subgraph_partition_config_buffer) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetLiteSubgraphPartitionConfigBuffer(
      std::string(nnadapter_subgraph_partition_config_buffer));
}

void FD_RuntimeOptionSetLiteContextProperties(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *nnadapter_context_properties) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetLiteContextProperties(
      std::string(nnadapter_context_properties));
}

void FD_RuntimeOptionSetLiteModelCacheDir(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *nnadapter_model_cache_dir) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetLiteModelCacheDir(std::string(nnadapter_model_cache_dir));
}

void FD_RuntimeOptionSetLiteMixedPrecisionQuantizationConfigPath(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *nnadapter_mixed_precision_quantization_config_path) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
}

void FD_RuntimeOptionEnableLiteFP16(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnableLiteFP16();
}

void FD_RuntimeOptionDisableLiteFP16(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DisableLiteFP16();
}

void FD_RuntimeOptionEnableLiteInt8(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnableLiteInt8();
}

void FD_RuntimeOptionDisableLiteInt8(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DisableLiteInt8();
}

void FD_RuntimeOptionSetLitePowerMode(
    __fd_keep FD_RuntimeOption *fd_runtime_option, FD_LitePowerMode mode) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetLitePowerMode(
      static_cast<fastdeploy::LitePowerMode>(mode));
}

void FD_RuntimeOptionEnableTrtFP16(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnableTrtFP16();
}

void FD_RuntimeOptionDisableTrtFP16(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DisableTrtFP16();
}

void FD_RuntimeOptionSetTrtCacheFile(
    __fd_keep FD_RuntimeOption *fd_runtime_option,
    const char *cache_file_path) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetTrtCacheFile(std::string(cache_file_path));
}

void FD_RuntimeOptionEnablePinnedMemory(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnablePinnedMemory();
}

void FD_RuntimeOptionDisablePinnedMemory(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DisablePinnedMemory();
}

void FD_RuntimeOptionEnablePaddleTrtCollectShape(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->EnablePaddleTrtCollectShape();
}

void FD_RuntimeOptionDisablePaddleTrtCollectShape(
    __fd_keep FD_RuntimeOption *fd_runtime_option) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->DisablePaddleTrtCollectShape();
}

void FD_RuntimeOptionSetOpenVINOStreams(
    __fd_keep FD_RuntimeOption *fd_runtime_option, int num_streams) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetOpenVINOStreams(num_streams);
}

void FD_RuntimeOptionUseIpu(__fd_keep FD_RuntimeOption *fd_runtime_option,
                            int device_num, int micro_batch_size,
                            FD_Bool enable_pipelining, int batches_per_step) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->UseIpu(device_num, micro_batch_size, enable_pipelining,
                         batches_per_step);
}

void FD_RuntimeOptionSetIpuConfig(__fd_keep FD_RuntimeOption *fd_runtime_option,
                                  FD_Bool enable_fp16, int replica_num,
                                  float available_memory_proportion,
                                  FD_Bool enable_half_partial) {
  CHECK_AND_CONVERT_FD_RuntimeOption;
  runtime_option->SetIpuConfig(enable_fp16, replica_num,
                               available_memory_proportion,
                               enable_half_partial);
}

}  // extern "C"
