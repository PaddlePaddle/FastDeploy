// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/core/fd_type.h"
#include "fastdeploy/runtime.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy_backend_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

//
// FastDeploy Backend that implements the TRITONBACKEND API.
//
namespace triton {
namespace backend {
namespace fastdeploy_runtime {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);
  virtual ~ModelState() = default;

  // Load an model. If 'instance_group_kind' is not
  // TRITONSERVER_INSTANCEGROUPKIND_AUTO then use it and
  // 'instance_group_device_id' to initialize the appropriate
  // execution providers. Return in 'model_path' the full path to the
  // onnx or paddle file.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id, std::string* model_path,
      std::string* params_path, fastdeploy::Runtime** runtime,
      cudaStream_t stream);

  const std::map<std::string, std::pair<int64_t, int64_t>>& ModelOutputs() {
    return model_outputs_;
  }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();

  TRITONSERVER_Error* AutoCompleteIO(
      const char* key, const std::vector<fastdeploy::TensorInfo>& io_infos);

  // Runtime options used when creating a FastDeploy Runtime.
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_options_;
  bool model_load_;
  fastdeploy::Runtime* main_runtime_;
  bool is_clone_ = true;

  // model_outputs is a map that contains unique outputs that the model must
  // provide. In the model configuration, the output in the state configuration
  // can have intersection with the outputs section of the model. If an output
  // is specified both in the output section and state section, it indicates
  // that the backend must return the output state to the client too.
  std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(triton_model,
                                                        &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());
    // RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  auto& model_outputs = (*state)->model_outputs_;

  // Parse the output states in the model configuration
  triton::common::TritonJson::Value sequence_batching;
  if ((*state)->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string output_state_name;
        RETURN_IF_ERROR(
            state.MemberAsString("output_name", &output_state_name));
        auto it = model_outputs.find(output_state_name);
        if (it == model_outputs.end()) {
          model_outputs.insert({output_state_name, std::make_pair(-1, i)});
        } else {
          it->second.second = i;
        }
      }
    }
  }

  // Parse the output names in the model configuration
  triton::common::TritonJson::Value outputs;
  RETURN_IF_ERROR((*state)->ModelConfig().MemberAsArray("output", &outputs));
  for (size_t i = 0; i < outputs.ArraySize(); i++) {
    triton::common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

    std::string output_name_str;

    RETURN_IF_ERROR(output.MemberAsString("name", &output_name_str));
    auto it = model_outputs.find(output_name_str);
    if (it == model_outputs.end()) {
      model_outputs.insert({output_name_str, {i, -1}});
    } else {
      it->second.first = i;
    }
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model),
      model_load_(false),
      main_runtime_(nullptr),
      is_clone_(true) {
  // Create runtime options that will be cloned and used for each
  // instance when creating that instance's runtime.
  runtime_options_.reset(new fastdeploy::RuntimeOption());

  triton::common::TritonJson::Value optimization;
  if (not ModelConfig().Find("optimization", &optimization)) {
    return;
  }

  triton::common::TritonJson::Value eas;
  if (not optimization.Find("execution_accelerators", &eas)) {
    return;
  }

  // CPU execution providers
  {
    triton::common::TritonJson::Value cpu_eas;
    if (eas.Find("cpu_execution_accelerator", &cpu_eas)) {
      for (size_t idx = 0; idx < cpu_eas.ArraySize(); idx++) {
        triton::common::TritonJson::Value ea;
        THROW_IF_BACKEND_MODEL_ERROR(cpu_eas.IndexAsObject(idx, &ea));
        std::string name;
        THROW_IF_BACKEND_MODEL_ERROR(ea.MemberAsString("name", &name));
        if (name == "onnxruntime") {
          runtime_options_->UseOrtBackend();
        } else if (name == "paddle") {
          runtime_options_->UsePaddleBackend();
        } else if (name == "openvino") {
          runtime_options_->UseOpenVINOBackend();
        } else if (name != "") {
          TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string("unknown cpu_execution_accelerator name '" + name +
                          "' is provided. Available choices are [onnxruntime, "
                          "paddle, openvino]")
                  .c_str());
          THROW_IF_BACKEND_MODEL_ERROR(error);
        }

        triton::common::TritonJson::Value params;
        if (ea.Find("parameters", &params)) {
          std::vector<std::string> param_keys;
          THROW_IF_BACKEND_MODEL_ERROR(params.Members(&param_keys));
          for (const auto& param_key : param_keys) {
            std::string value_string;
            THROW_IF_BACKEND_MODEL_ERROR(
                params.MemberAsString(param_key.c_str(), &value_string));
            if (param_key == "cpu_threads") {
              int cpu_thread_num;
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseIntValue(value_string, &cpu_thread_num));
              runtime_options_->SetCpuThreadNum(cpu_thread_num);
            } else if (param_key == "use_mkldnn") {
              bool pd_enable_mkldnn;
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseBoolValue(value_string, &pd_enable_mkldnn));
              runtime_options_->SetPaddleMKLDNN(pd_enable_mkldnn);
            } else if (param_key == "use_paddle_log") {
              runtime_options_->EnablePaddleLogInfo();
            } else if (param_key == "num_streams") {
              int num_streams;
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseIntValue(value_string, &num_streams));
              runtime_options_->openvino_option.num_streams = num_streams;
            } else if (param_key == "is_clone") {
              THROW_IF_BACKEND_MODEL_ERROR(
                  ParseBoolValue(value_string, &is_clone_));
            } else if (param_key == "use_ipu") {
              // runtime_options_->UseIpu();
            }
          }
        }
      }
    }
  }

  // GPU execution providers
  {
    triton::common::TritonJson::Value gpu_eas;
    if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
      for (size_t idx = 0; idx < gpu_eas.ArraySize(); idx++) {
        triton::common::TritonJson::Value ea;
        THROW_IF_BACKEND_MODEL_ERROR(gpu_eas.IndexAsObject(idx, &ea));
        std::string name;
        THROW_IF_BACKEND_MODEL_ERROR(ea.MemberAsString("name", &name));

        if (name == "onnxruntime") {
          runtime_options_->UseOrtBackend();
        } else if (name == "paddle") {
          runtime_options_->UsePaddleBackend();
        } else if (name == "tensorrt") {
          runtime_options_->UseTrtBackend();
        }
        if (name == "min_shape" or name == "max_shape" or name == "opt_shape") {
          triton::common::TritonJson::Value params;
          if (ea.Find("parameters", &params)) {
            std::vector<std::string> input_names;
            THROW_IF_BACKEND_MODEL_ERROR(params.Members(&input_names));
            for (const auto& input_name : input_names) {
              std::vector<int32_t> shape;
              FDParseShape(params, input_name, &shape);
              if (name == "min_shape") {
                runtime_options_->trt_option.min_shape[input_name] = shape;
              } else if (name == "max_shape") {
                runtime_options_->trt_option.max_shape[input_name] = shape;
              } else {
                runtime_options_->trt_option.opt_shape[input_name] = shape;
              }
            }
          }
        } else {
          triton::common::TritonJson::Value params;
          if (ea.Find("parameters", &params)) {
            std::vector<std::string> param_keys;
            THROW_IF_BACKEND_MODEL_ERROR(params.Members(&param_keys));
            for (const auto& param_key : param_keys) {
              std::string value_string;
              THROW_IF_BACKEND_MODEL_ERROR(
                  params.MemberAsString(param_key.c_str(), &value_string));
              if (param_key == "precision") {
                std::transform(value_string.begin(), value_string.end(),
                               value_string.begin(), ::tolower);
                if (value_string == "trt_fp16") {
                  runtime_options_->trt_option.enable_fp16 = true;
                } else if (value_string == "pd_fp16") {
                  // TODO(liqi): paddle inference don't currently have interface
                  // for fp16.
                }
                // } else if( param_key == "max_batch_size") {
                //   THROW_IF_BACKEND_MODEL_ERROR(ParseUnsignedLongLongValue(
                //       value_string, &runtime_options_->trt_max_batch_size));
                // } else if( param_key == "workspace_size") {
                //   THROW_IF_BACKEND_MODEL_ERROR(ParseUnsignedLongLongValue(
                //       value_string,
                //       &runtime_options_->trt_max_workspace_size));
              } else if (param_key == "cache_file") {
                runtime_options_->trt_option.serialize_file = value_string;
              } else if (param_key == "use_paddle") {
                runtime_options_->EnablePaddleToTrt();
              } else if (param_key == "use_paddle_log") {
                runtime_options_->EnablePaddleLogInfo();
              } else if (param_key == "is_clone") {
                THROW_IF_BACKEND_MODEL_ERROR(
                    ParseBoolValue(value_string, &is_clone_));
              }
            }
          }
        }
      }
    }
  }
}

TRITONSERVER_Error* ModelState::LoadModel(
    const std::string& artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id, std::string* model_path,
    std::string* params_path, fastdeploy::Runtime** runtime,
    cudaStream_t stream) {
  // FastDeploy Runtime creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  // The Clone interface can be invoked only when the main_runtime_ is created.
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  if (model_load_ && is_clone_) {
    if (main_runtime_ == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          std::string("main_runtime is nullptr").c_str());
    }
    *runtime = main_runtime_->Clone((void*)stream, instance_group_device_id);
  } else {
    auto dir_path = JoinPath({RepositoryPath(), std::to_string(Version())});
    {
      // ONNX Format
      bool exists;
      *model_path = JoinPath({dir_path, "model.onnx"});
      RETURN_IF_ERROR(FileExists(*model_path, &exists));

      // Paddle Formax
      if (not exists) {
        *model_path = JoinPath({dir_path, "model.pdmodel"});
        RETURN_IF_ERROR(FileExists(*model_path, &exists));
        if (not exists) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_NOT_FOUND,
              std::string(
                  "Model should be named as 'model.onnx' or 'model.pdmodel'")
                  .c_str());
        }
        *params_path = JoinPath({dir_path, "model.pdiparams"});
        RETURN_IF_ERROR(FileExists(*params_path, &exists));
        if (not exists) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_NOT_FOUND,
              std::string(
                  "Paddle params should be named as 'model.pdiparams' or "
                  "not provided.'")
                  .c_str());
        }
        runtime_options_->SetModelPath(*model_path, *params_path,
                                       fastdeploy::ModelFormat::PADDLE);
      } else {
        runtime_options_->SetModelPath(*model_path, "",
                                       fastdeploy::ModelFormat::ONNX);
      }
    }

    // GPU
#ifdef TRITON_ENABLE_GPU
    if ((instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
        (instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_AUTO)) {
      runtime_options_->UseGpu(instance_group_device_id);
      runtime_options_->SetExternalStream((void*)stream);
    } else if (runtime_options_->device != fastdeploy::Device::IPU) {
      runtime_options_->UseCpu();
    }
#else
    if (runtime_options_->device != fastdeploy::Device::IPU) {
      // If Device is set to IPU, just skip CPU setting.
      runtime_options_->UseCpu();
    }
#endif  // TRITON_ENABLE_GPU

    *runtime = main_runtime_ = new fastdeploy::Runtime();
    if (!(*runtime)->Init(*runtime_options_)) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                                   std::string("Runtime init error").c_str());
    }
    model_load_ = true;
  }
  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::AutoCompleteConfig() {
  // If the model configuration already specifies inputs and outputs
  // then don't perform any auto-completion.
  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  if ((input_cnt > 0) && (output_cnt > 0)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping model configuration auto-complete for '") +
         Name() + "': inputs and outputs already specified")
            .c_str());
    return nullptr;  // success
  }

  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));

  std::string model_path;
  std::string params_path;

  TRITONSERVER_InstanceGroupKind kind = TRITONSERVER_INSTANCEGROUPKIND_CPU;

#ifdef TRITON_ENABLE_GPU
  triton::common::TritonJson::Value instance_group;
  ModelConfig().Find("instance_group", &instance_group);

  // Earlier in the model lifecycle, device checks for the instance group
  // have already occurred. If at least one instance group with
  // "kind" = "KIND_GPU" then allow model to use GPU else autocomplete to
  // "KIND_CPU"
  for (size_t i = 0; i < instance_group.ArraySize(); ++i) {
    triton::common::TritonJson::Value instance_obj;
    instance_group.IndexAsObject(i, &instance_obj);

    triton::common::TritonJson::Value instance_group_kind;
    instance_obj.Find("kind", &instance_group_kind);
    std::string kind_str;
    RETURN_IF_ERROR(instance_group_kind.AsString(&kind_str));

    if (kind_str == "KIND_GPU") {
      kind = TRITONSERVER_INSTANCEGROUPKIND_GPU;
      break;
    }
  }
#endif  // TRITON_ENABLE_GPU

  fastdeploy::Runtime* runtime = nullptr;
  RETURN_IF_ERROR(LoadModel(artifact_name, kind, 0, &model_path, &params_path,
                            &runtime, nullptr));

  // TODO(liqi): need to infer max_batch_size
  int max_batch_size = -1;
  triton::common::TritonJson::Value mbs_value;
  ModelConfig().Find("max_batch_size", &mbs_value);
  mbs_value.SetInt(max_batch_size);
  SetMaxBatchSize(max_batch_size);

  auto input_infos = runtime->GetInputInfos();
  auto output_infos = runtime->GetOutputInfos();
  if (input_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("input", input_infos));
  }
  if (output_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("output", output_infos));
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::AutoCompleteIO(
    const char* key, const std::vector<fastdeploy::TensorInfo>& io_infos) {
  triton::common::TritonJson::Value existing_ios;
  bool found_ios = ModelConfig().Find(key, &existing_ios);

  triton::common::TritonJson::Value ios(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io_info : io_infos) {
    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(io.AddString("name", io_info.name));
    RETURN_IF_ERROR(
        io.AddString("data_type", FDTypeToModelConfigDataType(io_info.dtype)));

    // The model signature supports batching then the first dimension
    // is -1 and should not appear in the model configuration 'dims'
    // that we are creating.
    const auto& io_info_shape = io_info.shape;
    triton::common::TritonJson::Value dims(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_info_shape.size();
         ++i) {
      RETURN_IF_ERROR(dims.AppendInt(io_info_shape[i]));
    }

    // If dims are empty then must use a reshape...
    if (dims.ArraySize() == 0) {
      RETURN_IF_ERROR(dims.AppendInt(1));
      triton::common::TritonJson::Value reshape(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      triton::common::TritonJson::Value reshape_dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      RETURN_IF_ERROR(io.Add("reshape", std::move(reshape)));
    }
    RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(ios.Append(std::move(io)));
  }

  if (found_ios) {
    existing_ios.Swap(ios);
  } else {
    ModelConfig().Add(key, std::move(ios));
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  void ReleaseRunResources();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(TRITONBACKEND_Request** requests,
                       const uint32_t request_count);

 private:
  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance);
  void ReleaseOrtRunResources();
  int GetInfoIndex(const std::string& name,
                   const std::vector<fastdeploy::TensorInfo>& infos);
  void GetInfoNames(const std::vector<fastdeploy::TensorInfo>& infos,
                    std::vector<std::string>& names);
  TRITONSERVER_Error* ValidateInputs();
  TRITONSERVER_Error* ValidateOutputs();
  TRITONSERVER_Error* Run(std::vector<TRITONBACKEND_Response*>* responses,
                          const uint32_t response_count);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, bool* cuda_copy);

  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the model file.
  std::string model_path_;
  std::string params_path_;

  std::shared_ptr<fastdeploy::Runtime> runtime_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<fastdeploy::TensorInfo> input_tensor_infos_;
  std::vector<fastdeploy::TensorInfo> output_tensor_infos_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state),
      runtime_(nullptr) {
  fastdeploy::Runtime* runtime = nullptr;
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), Kind(), DeviceId(), &model_path_, &params_path_,
      &runtime, CudaStream()));
  runtime_.reset(runtime);
  runtime = nullptr;

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs());
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState() { ReleaseRunResources(); }

void ModelInstanceState::ReleaseRunResources() {
  input_names_.clear();
  output_names_.clear();
  input_tensor_infos_.clear();
  output_tensor_infos_.clear();
}

int ModelInstanceState::GetInfoIndex(
    const std::string& name, const std::vector<fastdeploy::TensorInfo>& infos) {
  for (size_t i = 0; i < infos.size(); ++i) {
    if (name == infos[i].name) return int(i);
  }
  return -1;
}

void ModelInstanceState::GetInfoNames(
    const std::vector<fastdeploy::TensorInfo>& infos,
    std::vector<std::string>& names) {
  for (const auto& info : infos) names.emplace_back(info.name);
}

TRITONSERVER_Error* ModelInstanceState::ValidateInputs() {
  input_tensor_infos_ = runtime_->GetInputInfos();
  std::vector<std::string> names;
  GetInfoNames(input_tensor_infos_, names);
  input_names_.clear();

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
  if (input_tensor_infos_.size() != ios.ArraySize()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', configuration expects " + std::to_string(ios.ArraySize()) +
         " inputs, model provides " +
         std::to_string(input_tensor_infos_.size()))
            .c_str());
  }
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    input_names_.emplace_back(io_name);
    int index = GetInfoIndex(io_name, input_tensor_infos_);
    if (index < 0) {
      std::set<std::string> inames(names.begin(), names.end());
      RETURN_IF_ERROR(CheckAllowedModelInput(io, inames));
    }

    auto fd_data_type = ModelConfigDataTypeToFDType(io_dtype);
    if (fd_data_type == fastdeploy::FDDataType::UNKNOWN1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (fd_data_type != input_tensor_infos_[index].dtype) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for input '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFDType(input_tensor_infos_[index].dtype)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    triton::common::TritonJson::Value allow_ragged_batch_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &allow_ragged_batch_json)) {
      RETURN_IF_ERROR(allow_ragged_batch_json.AsBool(&allow_ragged_batch));
    }
    if (allow_ragged_batch) {
      const std::vector<int64_t> model_shape(
          input_tensor_infos_[index].shape.begin(),
          input_tensor_infos_[index].shape.end());
      // Make sure the input has shpae [-1]
      if ((model_shape.size() != 1) || (model_shape[0] != WILDCARD_DIM)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unable to load model '") + model_state_->Name() +
             "', configuration expects model provides input with shape [-1]  "
             "for ragged input '" +
             io_name + "', model provides " + ShapeToString(model_shape))
                .c_str());
      }
    } else {
      // TODO: Implement shape checking
      // RETURN_IF_ERROR(CompareDimsSupported();
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error* ModelInstanceState::ValidateOutputs() {
  output_tensor_infos_ = runtime_->GetOutputInfos();
  std::set<std::string> out_names;
  for (const auto& info : output_tensor_infos_) {
    out_names.insert(info.name);
  }
  output_names_.clear();

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  // It is possible not to return all output!
  // if (output_tensor_infos_.size() != ios.ArraySize()) {
  //   return TRITONSERVER_ErrorNew(
  //       TRITONSERVER_ERROR_INVALID_ARG,
  //       (std::string("unable to load model '") + model_state_->Name() +
  //        "', configuration expects " + std::to_string(ios.ArraySize()) +
  //        " outputs, model provides " +
  //        std::to_string(output_tensor_infos_.size()))
  //           .c_str());
  // }
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    output_names_.emplace_back(io_name);
    int index = GetInfoIndex(io_name, output_tensor_infos_);
    if (index < 0) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, out_names));
    }

    auto fd_data_type = ModelConfigDataTypeToFDType(io_dtype);
    if (fd_data_type == fastdeploy::FDDataType::UNKNOWN1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for output '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (fd_data_type != output_tensor_infos_[index].dtype) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for output '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFDType(output_tensor_infos_[index].dtype)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    // The batch output shape doesn't necessarily match the model
    if (model_state_->FindBatchOutput(io_name) == nullptr) {
      // TODO: Implement shape checking
      // RETURN_IF_ERROR(CompareDimsSupported());
    }
  }
  return nullptr;  // success
}

void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests,
                                         const uint32_t request_count) {
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() +
               " with " + std::to_string(request_count) + " requests")
                  .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();
  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to FastDeploy Runtime backend for '" +
                  Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape,
                                            nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string("batch size " + std::to_string(total_batch_size) +
                        " for '" + Name() + "', max allowed is " +
                        std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());
  FD_RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      SetInputTensors(total_batch_size, requests, request_count, &responses,
                      &collector, &cuda_copy));

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  if (!all_response_failed) {
    FD_RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count,
                                         all_response_failed,
                                         Run(&responses, request_count));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  if (!all_response_failed) {
    FD_RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ReadOutputTensors(total_batch_size, requests, request_count,
                          &responses));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                       response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                   "failed to send fastdeploy backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                     TritonModelInstance(), request,
                     (responses[r] != nullptr) /* success */, exec_start_ns,
                     compute_start_ns, compute_end_ns, exec_end_ns),
                 "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                     TritonModelInstance(), total_batch_size, exec_start_ns,
                     compute_start_ns, compute_end_ns, exec_end_ns),
                 "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error* ModelInstanceState::Run(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count) {
  runtime_->Infer();
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaStreamSynchronize(CudaStream());
  }
#endif
  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, bool* cuda_copy) {
  const int max_batch_size = model_state_->MaxBatchSize();
  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    std::string in_name = std::string(input_name);
    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_InputProperties(input, nullptr, nullptr, &input_shape,
                                          &input_dims_count, nullptr, nullptr));

        batchn_shape[0] += GetElementCount(input_shape, input_dims_count);
      }
    } else {
      // The shape for the entire input batch, [total_batch_size, ...]
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
        allowed_input_types;
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      allowed_input_types = {{TRITONSERVER_MEMORY_GPU, DeviceId()},
                             {TRITONSERVER_MEMORY_CPU_PINNED, 0},
                             {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                             {TRITONSERVER_MEMORY_CPU, 0}};
    }

    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, allowed_input_types, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    int32_t device_id = -1;
    fastdeploy::Device device;
    if (memory_type == TRITONSERVER_MEMORY_GPU) {
      device_id = DeviceId();
      device = fastdeploy::Device::GPU;
    } else {
      device = fastdeploy::Device::CPU;
    }

    fastdeploy::FDTensor fdtensor(in_name);
    fdtensor.SetExternalData(batchn_shape, ConvertDataTypeToFD(input_datatype),
                             const_cast<char*>(input_buffer), device,
                             device_id);
    runtime_->BindInputTensor(in_name, fdtensor);
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses) {
  // r22.03
  // BackendOutputResponder responder(
  //     requests, request_count, responses,
  //     model_state_->TritonMemoryManager(), model_state_->MaxBatchSize() > 0,
  //     model_state_->EnablePinnedOutput(), CudaStream());
  // r21.10
  BackendOutputResponder responder(
      requests, request_count, responses, StateForModel()->MaxBatchSize(),
      StateForModel()->TritonMemoryManager(),
      StateForModel()->EnablePinnedOutput(), CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;

  // It is possible not to return all output!
  // auto& model_outputs = StateForModel()->ModelOutputs();
  // size_t output_count = output_tensors_.size();
  // if (output_count != model_outputs.size()) {
  //   RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
  //       TRITONSERVER_ERROR_INTERNAL,
  //       ("Retrieved output count is not equal to expected count.")));
  // }

  for (auto& output_name : output_names_) {
    auto* output_tensor = runtime_->GetOutputTensor(output_name);
    if (output_tensor == nullptr) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("output tensor '") + output_name + "' is not found")
              .c_str()));
    }
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    if (output_tensor->device == fastdeploy::Device::GPU) {
      memory_type = TRITONSERVER_MEMORY_GPU;
      memory_type_id = DeviceId();
    }
    responder.ProcessTensor(
        output_tensor->name, ConvertFDType(output_tensor->dtype),
        output_tensor->shape,
        reinterpret_cast<char*>(output_tensor->MutableData()), memory_type,
        memory_type_id);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return nullptr;
}

/////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_Initialize(
    TRITONBACKEND_Backend* backend) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton TRITONBACKEND API version: ") +
               std::to_string(api_version_major) + "." +
               std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("'") + name + "' TRITONBACKEND API version: " +
               std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
               std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                  .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // The backend configuration may contain information needed by the
  // ort backend, such as command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message,
                                                      &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  TRITONSERVER_Error* err = nullptr;
  if (byte_size != 0) {
    err = backend_config.Parse(buffer, byte_size);
  }
  RETURN_IF_ERROR(err);

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_Finalize(
    TRITONBACKEND_Backend* backend) {
  void* state = nullptr;
  LOG_IF_ERROR(TRITONBACKEND_BackendState(backend, &state),
               "failed to get backend state");
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(
    TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(
    TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
               " (" + TRITONSERVER_InstanceGroupKindString(kind) + " device " +
               std::to_string(device_id) + ")")
                  .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("model ") + model_state->Name() + ", instance " +
               instance_state->Name() + ", executing " +
               std::to_string(request_count) + " requests")
                  .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}  // namespace fastdeploy_runtime
}  // namespace backend
}  // namespace triton
