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

#ifdef ENABLE_ORT_BACKEND
#include "fastdeploy/runtime/backends/ort/ort_backend.h"
#endif

#ifdef ENABLE_TRT_BACKEND
#include "fastdeploy/runtime/backends/tensorrt/trt_backend.h"
#endif

#ifdef ENABLE_PADDLE_BACKEND
#include "fastdeploy/runtime/backends/paddle/paddle_backend.h"
#endif

#ifdef ENABLE_POROS_BACKEND
#include "fastdeploy/runtime/backends/poros/poros_backend.h"
#endif

#ifdef ENABLE_OPENVINO_BACKEND
#include "fastdeploy/runtime/backends/openvino/ov_backend.h"
#endif

#ifdef ENABLE_LITE_BACKEND
#include "fastdeploy/runtime/backends/lite/lite_backend.h"
#endif

#ifdef ENABLE_RKNPU2_BACKEND
#include "fastdeploy/runtime/backends/rknpu2/rknpu2_backend.h"
#endif

#ifdef ENABLE_SOPHGO_BACKEND
#include "fastdeploy/runtime/backends/sophgo/sophgo_backend.h"
#endif

namespace fastdeploy {

bool Runtime::Init(const RuntimeOption& _option) {
  option = _option;
  // Choose default backend by model format
  if (option.backend == Backend::UNKNOWN) {
    auto iter = s_default_backends_cfg.find(option.model_format);
    if (iter == s_default_backends_cfg.end()) {
      FDERROR << "Cannot found a default backend for model format: "
              << option.model_format
              << ", please define the inference backend in RuntimeOption."
              << std::endl;
      return false;
    }
    for (const auto& b : iter->second) {
      if (IsBackendAvailable(b)) {
        option.backend = b;
        FDINFO << "FastDeploy will choose " << b << " to inference this model."
               << std::endl;
      }
    }
    if (option.backend == Backend::UNKNOWN) {
      FDERROR << "Cannot found available backends for model format: "
              << option.model_format << "." << std::endl;
      return false;
    }
  }

  if (option.backend == Backend::ORT) {
    FDASSERT(option.device == Device::CPU || option.device == Device::GPU,
             "Backend::ORT only supports Device::CPU/Device::GPU.");
    CreateOrtBackend();
    FDINFO << "Runtime initialized with Backend::ORT in " << option.device
           << "." << std::endl;
  } else if (option.backend == Backend::TRT) {
    FDASSERT(option.device == Device::GPU,
             "Backend::TRT only supports Device::GPU.");
    CreateTrtBackend();
    FDINFO << "Runtime initialized with Backend::TRT in " << option.device
           << "." << std::endl;
  } else if (option.backend == Backend::PDINFER) {
    FDASSERT(
        option.device == Device::CPU || option.device == Device::GPU ||
            option.device == Device::IPU,
        "Backend::PDINFER only supports Device::CPU/Device::GPU/Device::IPU.");
    FDASSERT(
        option.model_format == ModelFormat::PADDLE,
        "Backend::PDINFER only supports model format of ModelFormat::PADDLE.");
    CreatePaddleBackend();
    FDINFO << "Runtime initialized with Backend::PDINFER in " << option.device
           << "." << std::endl;
  } else if (option.backend == Backend::POROS) {
    FDASSERT(option.device == Device::CPU || option.device == Device::GPU,
             "Backend::POROS only supports Device::CPU/Device::GPU.");
    FDASSERT(option.model_format == ModelFormat::TORCHSCRIPT,
             "Backend::POROS only supports model format of "
             "ModelFormat::TORCHSCRIPT.");
    FDINFO << "Runtime initialized with Backend::POROS in " << option.device
           << "." << std::endl;
    return true;
  } else if (option.backend == Backend::OPENVINO) {
    FDASSERT(option.device == Device::CPU,
             "Backend::OPENVINO only supports Device::CPU");
    CreateOpenVINOBackend();
    FDINFO << "Runtime initialized with Backend::OPENVINO in " << option.device
           << "." << std::endl;
  } else if (option.backend == Backend::LITE) {
    FDASSERT(option.device == Device::CPU || option.device == Device::TIMVX ||
                 option.device == Device::KUNLUNXIN ||
                 option.device == Device::ASCEND,
             "Backend::LITE only supports "
             "Device::CPU/Device::TIMVX/Device::KUNLUNXIN.");
    CreateLiteBackend();
    FDINFO << "Runtime initialized with Backend::LITE in " << option.device
           << "." << std::endl;
  } else if (option.backend == Backend::RKNPU2) {
    FDASSERT(option.device == Device::RKNPU,
             "Backend::RKNPU2 only supports Device::RKNPU2");
    CreateRKNPU2Backend();

    FDINFO << "Runtime initialized with Backend::RKNPU2 in " << option.device
           << "." << std::endl;
  } else if (option.backend == Backend::SOPHGOTPU) {
    FDASSERT(option.device == Device::SOPHGOTPUD,
             "Backend::SOPHGO only supports Device::SOPHGO");
    CreateSophgoNPUBackend();

    FDINFO << "Runtime initialized with Backend::SOPHGO in " << option.device
           << "." << std::endl;
  } else {
    FDERROR << "Runtime only support "
               "Backend::ORT/Backend::TRT/Backend::PDINFER/Backend::POROS as "
               "backend now."
            << std::endl;
    return false;
  }
  return true;
}

TensorInfo Runtime::GetInputInfo(int index) {
  return backend_->GetInputInfo(index);
}

TensorInfo Runtime::GetOutputInfo(int index) {
  return backend_->GetOutputInfo(index);
}

std::vector<TensorInfo> Runtime::GetInputInfos() {
  return backend_->GetInputInfos();
}

std::vector<TensorInfo> Runtime::GetOutputInfos() {
  return backend_->GetOutputInfos();
}

bool Runtime::Infer(std::vector<FDTensor>& input_tensors,
                    std::vector<FDTensor>* output_tensors) {
  for (auto& tensor : input_tensors) {
    FDASSERT(tensor.device_id < 0 || tensor.device_id == option.device_id,
             "Device id of input tensor(%d) and runtime(%d) are not same.",
             tensor.device_id, option.device_id);
  }
  return backend_->Infer(input_tensors, output_tensors);
}

bool Runtime::Infer() {
  bool result = backend_->Infer(input_tensors_, &output_tensors_, false);
  for (auto& tensor : output_tensors_) {
    tensor.device_id = option.device_id;
  }
  return result;
}

void Runtime::BindInputTensor(const std::string& name, FDTensor& input) {
  bool is_exist = false;
  for (auto& t : input_tensors_) {
    if (t.name == name) {
      is_exist = true;
      t.SetExternalData(input.shape, input.dtype, input.MutableData(),
                        input.device, input.device_id);
      break;
    }
  }
  if (!is_exist) {
    FDTensor new_tensor(name);
    new_tensor.SetExternalData(input.shape, input.dtype, input.MutableData(),
                               input.device, input.device_id);
    input_tensors_.emplace_back(std::move(new_tensor));
  }
}

FDTensor* Runtime::GetOutputTensor(const std::string& name) {
  for (auto& t : output_tensors_) {
    if (t.name == name) {
      return &t;
    }
  }
  FDWARNING << "The output name [" << name << "] don't exist." << std::endl;
  return nullptr;
}

void Runtime::CreatePaddleBackend() {
#ifdef ENABLE_PADDLE_BACKEND
  auto pd_option = PaddleBackendOption();
  pd_option.model_file = option.model_file;
  pd_option.params_file = option.params_file;
  pd_option.enable_mkldnn = option.pd_enable_mkldnn;
  pd_option.enable_log_info = option.pd_enable_log_info;
  pd_option.mkldnn_cache_size = option.pd_mkldnn_cache_size;
  pd_option.use_gpu = (option.device == Device::GPU) ? true : false;
  pd_option.use_ipu = (option.device == Device::IPU) ? true : false;
  pd_option.gpu_id = option.device_id;
  pd_option.delete_pass_names = option.pd_delete_pass_names;
  pd_option.cpu_thread_num = option.cpu_thread_num;
  pd_option.enable_pinned_memory = option.enable_pinned_memory;
  pd_option.external_stream_ = option.external_stream_;
  pd_option.model_from_memory_ = option.model_from_memory_;
  if (pd_option.model_from_memory_) {
    pd_option.model_buffer_ = option.model_buffer_;
    pd_option.params_buffer_ = option.params_buffer_;
    pd_option.model_buffer_size_ = option.model_buffer_size_;
    pd_option.params_buffer_size_ = option.params_buffer_size_;
  }
#ifdef ENABLE_TRT_BACKEND
  if (pd_option.use_gpu && option.pd_enable_trt) {
    pd_option.enable_trt = true;
    pd_option.collect_shape = option.pd_collect_shape;
    auto trt_option = TrtBackendOption();
    trt_option.gpu_id = option.device_id;
    trt_option.enable_fp16 = option.trt_enable_fp16;
    trt_option.max_batch_size = option.trt_max_batch_size;
    trt_option.max_workspace_size = option.trt_max_workspace_size;
    trt_option.max_shape = option.trt_max_shape;
    trt_option.min_shape = option.trt_min_shape;
    trt_option.opt_shape = option.trt_opt_shape;
    trt_option.serialize_file = option.trt_serialize_file;
    trt_option.enable_pinned_memory = option.enable_pinned_memory;
    pd_option.trt_option = trt_option;
    pd_option.trt_disabled_ops_ = option.trt_disabled_ops_;
  }
#endif
#ifdef WITH_IPU
  if (pd_option.use_ipu) {
    auto ipu_option = IpuOption();
    ipu_option.ipu_device_num = option.ipu_device_num;
    ipu_option.ipu_micro_batch_size = option.ipu_micro_batch_size;
    ipu_option.ipu_enable_pipelining = option.ipu_enable_pipelining;
    ipu_option.ipu_batches_per_step = option.ipu_batches_per_step;
    ipu_option.ipu_enable_fp16 = option.ipu_enable_fp16;
    ipu_option.ipu_replica_num = option.ipu_replica_num;
    ipu_option.ipu_available_memory_proportion =
        option.ipu_available_memory_proportion;
    ipu_option.ipu_enable_half_partial = option.ipu_enable_half_partial;
    pd_option.ipu_option = ipu_option;
  }
#endif
  FDASSERT(option.model_format == ModelFormat::PADDLE,
           "PaddleBackend only support model format of ModelFormat::PADDLE.");
  backend_ = utils::make_unique<PaddleBackend>();
  auto casted_backend = dynamic_cast<PaddleBackend*>(backend_.get());
  if (pd_option.model_from_memory_) {
    FDASSERT(casted_backend->InitFromPaddle(option.model_buffer_,
                                            option.params_buffer_, pd_option),
             "Load model from Paddle failed while initliazing PaddleBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, pd_option),
             "Load model from Paddle failed while initliazing PaddleBackend.");
  }
#else
  FDASSERT(false,
           "PaddleBackend is not available, please compiled with "
           "ENABLE_PADDLE_BACKEND=ON.");
#endif
}

void Runtime::CreateOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  auto ov_option = OpenVINOBackendOption();
  ov_option.cpu_thread_num = option.cpu_thread_num;
  ov_option.device = option.openvino_device;
  ov_option.shape_infos = option.ov_shape_infos;
  ov_option.num_streams = option.ov_num_streams;
  for (const auto& op : option.ov_cpu_operators) {
    ov_option.cpu_operators.insert(op);
  }
  FDASSERT(option.model_format == ModelFormat::PADDLE ||
               option.model_format == ModelFormat::ONNX,
           "OpenVINOBackend only support model format of ModelFormat::PADDLE / "
           "ModelFormat::ONNX.");
  backend_ = utils::make_unique<OpenVINOBackend>();
  auto casted_backend = dynamic_cast<OpenVINOBackend*>(backend_.get());

  if (option.model_format == ModelFormat::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, ov_option),
             "Load model from ONNX failed while initliazing OrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, ov_option),
             "Load model from Paddle failed while initliazing OrtBackend.");
  }
#else
  FDASSERT(false,
           "OpenVINOBackend is not available, please compiled with "
           "ENABLE_OPENVINO_BACKEND=ON.");
#endif
}

void Runtime::CreateOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  auto ort_option = OrtBackendOption();
  ort_option.graph_optimization_level = option.ort_graph_opt_level;
  ort_option.intra_op_num_threads = option.cpu_thread_num;
  ort_option.inter_op_num_threads = option.ort_inter_op_num_threads;
  ort_option.execution_mode = option.ort_execution_mode;
  ort_option.use_gpu = (option.device == Device::GPU) ? true : false;
  ort_option.gpu_id = option.device_id;
  ort_option.external_stream_ = option.external_stream_;

  FDASSERT(option.model_format == ModelFormat::PADDLE ||
               option.model_format == ModelFormat::ONNX,
           "OrtBackend only support model format of ModelFormat::PADDLE / "
           "ModelFormat::ONNX.");
  backend_ = utils::make_unique<OrtBackend>();
  auto casted_backend = dynamic_cast<OrtBackend*>(backend_.get());
  if (option.model_format == ModelFormat::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, ort_option),
             "Load model from ONNX failed while initliazing OrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, ort_option),
             "Load model from Paddle failed while initliazing OrtBackend.");
  }
#else
  FDASSERT(false,
           "OrtBackend is not available, please compiled with "
           "ENABLE_ORT_BACKEND=ON.");
#endif
}

void Runtime::CreateTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  auto trt_option = TrtBackendOption();
  trt_option.model_file = option.model_file;
  trt_option.params_file = option.params_file;
  trt_option.model_format = option.model_format;
  trt_option.gpu_id = option.device_id;
  trt_option.enable_fp16 = option.trt_enable_fp16;
  trt_option.enable_int8 = option.trt_enable_int8;
  trt_option.max_batch_size = option.trt_max_batch_size;
  trt_option.max_workspace_size = option.trt_max_workspace_size;
  trt_option.max_shape = option.trt_max_shape;
  trt_option.min_shape = option.trt_min_shape;
  trt_option.opt_shape = option.trt_opt_shape;
  trt_option.serialize_file = option.trt_serialize_file;
  trt_option.enable_pinned_memory = option.enable_pinned_memory;
  trt_option.external_stream_ = option.external_stream_;

  FDASSERT(option.model_format == ModelFormat::PADDLE ||
               option.model_format == ModelFormat::ONNX,
           "TrtBackend only support model format of ModelFormat::PADDLE / "
           "ModelFormat::ONNX.");
  backend_ = utils::make_unique<TrtBackend>();
  auto casted_backend = dynamic_cast<TrtBackend*>(backend_.get());
  if (option.model_format == ModelFormat::ONNX) {
    FDASSERT(casted_backend->InitFromOnnx(option.model_file, trt_option),
             "Load model from ONNX failed while initliazing TrtBackend.");
  } else {
    FDASSERT(casted_backend->InitFromPaddle(option.model_file,
                                            option.params_file, trt_option),
             "Load model from Paddle failed while initliazing TrtBackend.");
  }
#else
  FDASSERT(false,
           "TrtBackend is not available, please compiled with "
           "ENABLE_TRT_BACKEND=ON.");
#endif
}

void Runtime::CreateLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  auto lite_option = LiteBackendOption();
  lite_option.threads = option.cpu_thread_num;
  lite_option.enable_int8 = option.lite_enable_int8;
  lite_option.enable_fp16 = option.lite_enable_fp16;
  lite_option.power_mode = static_cast<int>(option.lite_power_mode);
  lite_option.optimized_model_dir = option.lite_optimized_model_dir;
  lite_option.nnadapter_subgraph_partition_config_path =
      option.lite_nnadapter_subgraph_partition_config_path;
  lite_option.nnadapter_subgraph_partition_config_buffer =
      option.lite_nnadapter_subgraph_partition_config_buffer;
  lite_option.nnadapter_device_names = option.lite_nnadapter_device_names;
  lite_option.nnadapter_context_properties =
      option.lite_nnadapter_context_properties;
  lite_option.nnadapter_model_cache_dir = option.lite_nnadapter_model_cache_dir;
  lite_option.nnadapter_dynamic_shape_info =
      option.lite_nnadapter_dynamic_shape_info;
  lite_option.nnadapter_mixed_precision_quantization_config_path =
      option.lite_nnadapter_mixed_precision_quantization_config_path;
  lite_option.enable_timvx = option.enable_timvx;
  lite_option.enable_ascend = option.enable_ascend;
  lite_option.enable_kunlunxin = option.enable_kunlunxin;
  lite_option.device_id = option.device_id;
  lite_option.kunlunxin_l3_workspace_size = option.kunlunxin_l3_workspace_size;
  lite_option.kunlunxin_locked = option.kunlunxin_locked;
  lite_option.kunlunxin_autotune = option.kunlunxin_autotune;
  lite_option.kunlunxin_autotune_file = option.kunlunxin_autotune_file;
  lite_option.kunlunxin_precision = option.kunlunxin_precision;
  lite_option.kunlunxin_adaptive_seqlen = option.kunlunxin_adaptive_seqlen;
  lite_option.kunlunxin_enable_multi_stream =
      option.kunlunxin_enable_multi_stream;

  FDASSERT(option.model_format == ModelFormat::PADDLE,
           "LiteBackend only support model format of ModelFormat::PADDLE");
  backend_ = utils::make_unique<LiteBackend>();
  auto casted_backend = dynamic_cast<LiteBackend*>(backend_.get());
  FDASSERT(casted_backend->InitFromPaddle(option.model_file, option.params_file,
                                          lite_option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false,
           "LiteBackend is not available, please compiled with "
           "ENABLE_LITE_BACKEND=ON.");
#endif
}

void Runtime::CreateRKNPU2Backend() {
#ifdef ENABLE_RKNPU2_BACKEND
  auto rknpu2_option = RKNPU2BackendOption();
  rknpu2_option.cpu_name = option.rknpu2_cpu_name_;
  rknpu2_option.core_mask = option.rknpu2_core_mask_;
  FDASSERT(option.model_format == ModelFormat::RKNN,
           "RKNPU2Backend only support model format of ModelFormat::RKNN");
  backend_ = utils::make_unique<RKNPU2Backend>();
  auto casted_backend = dynamic_cast<RKNPU2Backend*>(backend_.get());
  FDASSERT(casted_backend->InitFromRKNN(option.model_file, rknpu2_option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false,
           "RKNPU2Backend is not available, please compiled with "
           "ENABLE_RKNPU2_BACKEND=ON.");
#endif
}

void Runtime::CreateSophgoNPUBackend() {
#ifdef ENABLE_SOPHGO_BACKEND
  auto sophgo_option = SophgoBackendOption();
  FDASSERT(option.model_format == ModelFormat::SOPHGO,
           "SophgoBackend only support model format of ModelFormat::SOPHGO");
  backend_ = utils::make_unique<SophgoBackend>();
  auto casted_backend = dynamic_cast<SophgoBackend*>(backend_.get());
  FDASSERT(casted_backend->InitFromSophgo(option.model_file, sophgo_option),
           "Load model from nb file failed while initializing LiteBackend.");
#else
  FDASSERT(false,
           "SophgoBackend is not available, please compiled with "
           "ENABLE_SOPHGO_BACKEND=ON.");
#endif
}

Runtime* Runtime::Clone(void* stream, int device_id) {
  Runtime* runtime = new Runtime();
  if (option.backend != Backend::OPENVINO &&
      option.backend != Backend::PDINFER) {
    runtime->Init(option);
    FDWARNING << "Only OpenVINO/Paddle Inference support \
                  clone engine to  reduce CPU/GPU memory usage now. For "
              << option.backend
              << ", FastDeploy will create a new engine which \
                  will not share memory  with the current runtime."
              << std::endl;
    return runtime;
  }
  FDINFO << "Runtime Clone with Backend:: " << option.backend << " in "
         << option.device << "." << std::endl;
  runtime->option = option;
  runtime->backend_ = backend_->Clone(stream, device_id);
  return runtime;
}

}  // namespace fastdeploy
