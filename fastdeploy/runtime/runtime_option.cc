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

#include "fastdeploy/runtime/runtime.h"
#include "fastdeploy/utils/unique_ptr.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

void RuntimeOption::SetModelPath(const std::string& model_path,
                                 const std::string& params_path,
                                 const ModelFormat& format) {
  model_file = model_path;
  params_file = params_path;
  model_format = format;
  model_from_memory_ = false;
}

void RuntimeOption::SetModelBuffer(const std::string& model_buffer,
                                   const std::string& params_buffer,
                                   const ModelFormat& format) {
  model_file = model_buffer;
  params_file = params_buffer;
  model_format = format;
  model_from_memory_ = true;
}

void RuntimeOption::UseGpu(int gpu_id) {
#ifdef WITH_GPU
  device = Device::GPU;
  device_id = gpu_id;
#else
  FDWARNING << "The FastDeploy didn't compile with GPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseCpu() { device = Device::CPU; }

void RuntimeOption::UseRKNPU2(fastdeploy::rknpu2::CpuName rknpu2_name,
                              fastdeploy::rknpu2::CoreMask rknpu2_core) {
  rknpu2_cpu_name_ = rknpu2_name;
  rknpu2_core_mask_ = rknpu2_core;
  device = Device::RKNPU;
}

void RuntimeOption::UseTimVX() {
  device = Device::TIMVX;
  paddle_lite_option.device = device;
}

void RuntimeOption::UseKunlunXin(int kunlunxin_id, int l3_workspace_size,
                                 bool locked, bool autotune,
                                 const std::string& autotune_file,
                                 const std::string& precision,
                                 bool adaptive_seqlen,
                                 bool enable_multi_stream) {
  device = Device::KUNLUNXIN;
  paddle_lite_option.device = device;
  paddle_lite_option.device_id = kunlunxin_id;
  paddle_lite_option.kunlunxin_l3_workspace_size = l3_workspace_size;
  paddle_lite_option.kunlunxin_locked = locked;
  paddle_lite_option.kunlunxin_autotune = autotune;
  paddle_lite_option.kunlunxin_autotune_file = autotune_file;
  paddle_lite_option.kunlunxin_precision = precision;
  paddle_lite_option.kunlunxin_adaptive_seqlen = adaptive_seqlen;
  paddle_lite_option.kunlunxin_enable_multi_stream = enable_multi_stream;
}

void RuntimeOption::UseAscend() {
  device = Device::ASCEND;
  paddle_lite_option.device = device;
}

void RuntimeOption::UseSophgo() {
  device = Device::SOPHGOTPUD;
  UseSophgoBackend();
}

void RuntimeOption::SetExternalStream(void* external_stream) {
  external_stream_ = external_stream;
}

void RuntimeOption::SetCpuThreadNum(int thread_num) {
  FDASSERT(thread_num > 0, "The thread_num must be greater than 0.");
  cpu_thread_num = thread_num;
  paddle_lite_option.cpu_threads = thread_num;
  ort_option.intra_op_num_threads = thread_num;
}

void RuntimeOption::SetOrtGraphOptLevel(int level) {
  std::vector<int> supported_level{-1, 0, 1, 2};
  auto valid_level = std::find(supported_level.begin(), supported_level.end(),
                               level) != supported_level.end();
  FDASSERT(valid_level, "The level must be -1, 0, 1, 2.");
  ort_option.graph_optimization_level = level;
}

// use paddle inference backend
void RuntimeOption::UsePaddleBackend() {
#ifdef ENABLE_PADDLE_BACKEND
  backend = Backend::PDINFER;
#else
  FDASSERT(false, "The FastDeploy didn't compile with Paddle Inference.");
#endif
}

// use onnxruntime backend
void RuntimeOption::UseOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend = Backend::ORT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OrtBackend.");
#endif
}

// use sophgoruntime backend
void RuntimeOption::UseSophgoBackend() {
#ifdef ENABLE_SOPHGO_BACKEND
  backend = Backend::SOPHGOTPU;
#else
  FDASSERT(false, "The FastDeploy didn't compile with SophgoBackend.");
#endif
}

// use poros backend
void RuntimeOption::UsePorosBackend() {
#ifdef ENABLE_POROS_BACKEND
  backend = Backend::POROS;
#else
  FDASSERT(false, "The FastDeploy didn't compile with PorosBackend.");
#endif
}

void RuntimeOption::UseTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  backend = Backend::TRT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with TrtBackend.");
#endif
}

void RuntimeOption::UseOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  backend = Backend::OPENVINO;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OpenVINO.");
#endif
}

void RuntimeOption::UseLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  backend = Backend::LITE;
#else
  FDASSERT(false, "The FastDeploy didn't compile with Paddle Lite.");
#endif
}

void RuntimeOption::SetPaddleMKLDNN(bool pd_mkldnn) {
  pd_enable_mkldnn = pd_mkldnn;
}

void RuntimeOption::DeletePaddleBackendPass(const std::string& pass_name) {
  pd_delete_pass_names.push_back(pass_name);
}
void RuntimeOption::EnablePaddleLogInfo() { pd_enable_log_info = true; }

void RuntimeOption::DisablePaddleLogInfo() { pd_enable_log_info = false; }

void RuntimeOption::EnablePaddleToTrt() {
  FDASSERT(backend == Backend::TRT,
           "Should call UseTrtBackend() before call EnablePaddleToTrt().");
#ifdef ENABLE_PADDLE_BACKEND
  FDINFO << "While using TrtBackend with EnablePaddleToTrt, FastDeploy will "
            "change to use Paddle Inference Backend."
         << std::endl;
  backend = Backend::PDINFER;
  pd_enable_trt = true;
#else
  FDASSERT(false,
           "While using TrtBackend with EnablePaddleToTrt, require the "
           "FastDeploy is compiled with Paddle Inference Backend, "
           "please rebuild your FastDeploy.");
#endif
}

void RuntimeOption::SetPaddleMKLDNNCacheSize(int size) {
  FDASSERT(size > 0, "Parameter size must greater than 0.");
  pd_mkldnn_cache_size = size;
}

void RuntimeOption::SetOpenVINODevice(const std::string& name) {
  openvino_device = name;
}

void RuntimeOption::EnableLiteFP16() { paddle_lite_option.enable_fp16 = true; }

void RuntimeOption::DisableLiteFP16() {
  paddle_lite_option.enable_fp16 = false;
}

void RuntimeOption::EnableLiteInt8() { paddle_lite_option.enable_int8 = true; }

void RuntimeOption::DisableLiteInt8() {
  paddle_lite_option.enable_int8 = false;
}

void RuntimeOption::SetLitePowerMode(LitePowerMode mode) {
  paddle_lite_option.power_mode = mode;
}

void RuntimeOption::SetLiteOptimizedModelDir(
    const std::string& optimized_model_dir) {
  paddle_lite_option.optimized_model_dir = optimized_model_dir;
}

void RuntimeOption::SetLiteSubgraphPartitionPath(
    const std::string& nnadapter_subgraph_partition_config_path) {
  paddle_lite_option.nnadapter_subgraph_partition_config_path =
      nnadapter_subgraph_partition_config_path;
}

void RuntimeOption::SetLiteSubgraphPartitionConfigBuffer(
    const std::string& nnadapter_subgraph_partition_config_buffer) {
  paddle_lite_option.nnadapter_subgraph_partition_config_buffer =
      nnadapter_subgraph_partition_config_buffer;
}

void RuntimeOption::SetLiteDeviceNames(
    const std::vector<std::string>& nnadapter_device_names) {
  paddle_lite_option.nnadapter_device_names = nnadapter_device_names;
}

void RuntimeOption::SetLiteContextProperties(
    const std::string& nnadapter_context_properties) {
  paddle_lite_option.nnadapter_context_properties =
      nnadapter_context_properties;
}

void RuntimeOption::SetLiteModelCacheDir(
    const std::string& nnadapter_model_cache_dir) {
  paddle_lite_option.nnadapter_model_cache_dir = nnadapter_model_cache_dir;
}

void RuntimeOption::SetLiteDynamicShapeInfo(
    const std::map<std::string, std::vector<std::vector<int64_t>>>&
        nnadapter_dynamic_shape_info) {
  paddle_lite_option.nnadapter_dynamic_shape_info =
      nnadapter_dynamic_shape_info;
}

void RuntimeOption::SetLiteMixedPrecisionQuantizationConfigPath(
    const std::string& nnadapter_mixed_precision_quantization_config_path) {
  paddle_lite_option.nnadapter_mixed_precision_quantization_config_path =
      nnadapter_mixed_precision_quantization_config_path;
}

void RuntimeOption::SetTrtInputShape(const std::string& input_name,
                                     const std::vector<int32_t>& min_shape,
                                     const std::vector<int32_t>& opt_shape,
                                     const std::vector<int32_t>& max_shape) {
  trt_min_shape[input_name].clear();
  trt_max_shape[input_name].clear();
  trt_opt_shape[input_name].clear();
  trt_min_shape[input_name].assign(min_shape.begin(), min_shape.end());
  if (opt_shape.size() == 0) {
    trt_opt_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_opt_shape[input_name].assign(opt_shape.begin(), opt_shape.end());
  }
  if (max_shape.size() == 0) {
    trt_max_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_max_shape[input_name].assign(max_shape.begin(), max_shape.end());
  }
}

void RuntimeOption::SetTrtMaxWorkspaceSize(size_t max_workspace_size) {
  trt_max_workspace_size = max_workspace_size;
}
void RuntimeOption::SetTrtMaxBatchSize(size_t max_batch_size) {
  trt_max_batch_size = max_batch_size;
}

void RuntimeOption::EnableTrtFP16() { trt_enable_fp16 = true; }

void RuntimeOption::DisableTrtFP16() { trt_enable_fp16 = false; }

void RuntimeOption::EnablePinnedMemory() { enable_pinned_memory = true; }

void RuntimeOption::DisablePinnedMemory() { enable_pinned_memory = false; }

void RuntimeOption::SetTrtCacheFile(const std::string& cache_file_path) {
  trt_serialize_file = cache_file_path;
}

void RuntimeOption::SetOpenVINOStreams(int num_streams) {
  ov_num_streams = num_streams;
}

void RuntimeOption::EnablePaddleTrtCollectShape() { pd_collect_shape = true; }

void RuntimeOption::DisablePaddleTrtCollectShape() { pd_collect_shape = false; }

void RuntimeOption::DisablePaddleTrtOPs(const std::vector<std::string>& ops) {
  trt_disabled_ops_.insert(trt_disabled_ops_.end(), ops.begin(), ops.end());
}

void RuntimeOption::UseIpu(int device_num, int micro_batch_size,
                           bool enable_pipelining, int batches_per_step) {
#ifdef WITH_IPU
  device = Device::IPU;
  ipu_device_num = device_num;
  ipu_micro_batch_size = micro_batch_size;
  ipu_enable_pipelining = enable_pipelining;
  ipu_batches_per_step = batches_per_step;
#else
  FDWARNING << "The FastDeploy didn't compile with IPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::SetIpuConfig(bool enable_fp16, int replica_num,
                                 float available_memory_proportion,
                                 bool enable_half_partial) {
  ipu_enable_fp16 = enable_fp16;
  ipu_replica_num = replica_num;
  ipu_available_memory_proportion = available_memory_proportion;
  ipu_enable_half_partial = enable_half_partial;
}

}  // namespace fastdeploy
