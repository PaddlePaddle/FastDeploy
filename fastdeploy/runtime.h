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

/*! \file runtime.h
    \brief A brief file description.

    More details
 */

#pragma once

#include <map>
#include <vector>

#include "fastdeploy/backends/backend.h"
#include "fastdeploy/utils/perf.h"

/** \brief All C++ FastDeploy APIs are defined inside this namespace
 *
 */
namespace fastdeploy {

/*! Inference backend supported in FastDeploy */
enum Backend {
  UNKNOWN,  ///< Unknown inference backend
  ORT,  ///< ONNX Runtime, support Paddle/ONNX format model, CPU / Nvidia GPU
  TRT,  ///< TensorRT, support Paddle/ONNX format model, Nvidia GPU only
  PDINFER,  ///< Paddle Inference, support Paddle format model, CPU / Nvidia GPU
  OPENVINO,  ///< Intel OpenVINO, support Paddle/ONNX format, CPU only
  LITE,      ///< Paddle Lite, support Paddle format model, ARM CPU only
};

/*! Deep learning model format */
enum ModelFormat {
  AUTOREC,  ///< Auto recognize the model format by model file name
  PADDLE,   ///< Model with paddlepaddle format
  ONNX,     ///< Model with ONNX format
};

FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out,
                                         const Backend& backend);
FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out,
                                         const ModelFormat& format);

FASTDEPLOY_DECL std::string Str(const Backend& b);
FASTDEPLOY_DECL std::string Str(const ModelFormat& f);

/**
 * @brief Get all the available inference backend in FastDeploy
 */
FASTDEPLOY_DECL std::vector<Backend> GetAvailableBackends();

/**
 * @brief Check if the inference backend available
 */
FASTDEPLOY_DECL bool IsBackendAvailable(const Backend& backend);

bool CheckModelFormat(const std::string& model_file,
                      const ModelFormat& model_format);
ModelFormat GuessModelFormat(const std::string& model_file);

/*! @brief Option object used when create a new Runtime object
 */
struct FASTDEPLOY_DECL RuntimeOption {
  /** \brief Set path of model file and parameter file
   *
   * \param[in] model_path Path of model file, e.g ResNet50/model.pdmodel for
   * Paddle format model / ResNet50/model.onnx for ONNX format model \param[in]
   * params_path Path of parameter file, this only used when the model format is
   * Paddle, e.g Resnet50/model.pdiparams \param[in] format Format of the loaded
   * model
   */
  void SetModelPath(
      const std::string& model_path, const std::string& params_path = "",
      const ModelFormat& format =
          ModelFormat::PADDLE);  ///< Set path of model file and parameter file,
                                 ///< if the format is ModelFormat::ONNX, the
                                 ///< `params_path` can be set as empty string

  void UseCpu();  ///< Use cpu to inference, the runtime will inference on CPU
                  ///< by default

  void UseGpu(int gpu_id = 0);  ///< Use Nvidia GPU to inference

  void SetCpuThreadNum(
      int thread_num);  ///< Set number of cpu threads while inference on CPU,
                        ///< by default it will decided by the different
                        ///< backends

  void UsePaddleBackend();  ///< Set Paddle Inference as inference backend,
                            ///< support CPU/GPU

  void
  UseOrtBackend();  ///< Set ONNX Runtime as inference backend, support CPU/GPU

  void
  UseTrtBackend();  ///< Set TensorRT as inference backend, only support GPU

  void UseOpenVINOBackend();  ///< Set OpenVINO as inference backend, only
                              ///< support CPU

  void UseLiteBackend();  ///< Set Paddle Lite as inference backend, only
                          ///< support arm cpu

  void EnablePaddleMKLDNN();  ///< Enable mkldnn while using Paddle Inference as
                              ///< inference backend

  void DisablePaddleMKLDNN();  ///< Disable mkldnn while using Paddle Inference
                               ///< as inference backend

  void DeletePaddleBackendPass(
      const std::string&
          delete_pass_name);  ///< Delete pass by name while using Paddle
                              ///< Inference as inference backend, this can be
                              ///< called multiple times to delete mulitple
                              ///< passes

  void
  EnablePaddleLogInfo();  ///< Enable print debug information while using Paddle
                          ///< Inference as inference backend, the backend
                          ///< disable the debug information by default

  void DisablePaddleLogInfo();  ///< Disable print debug information while using
                                ///< Paddle Inference as inference backend

  void SetPaddleMKLDNNCacheSize(
      int size);  ///< Set shape cache size while using Paddle Inference with
                  ///< mkldnn, by default it will cache all the difference shape

  void SetLitePowerMode(
      int mode);  ///< Set power mode while using Paddle Lite as inference
                  ///< backend, mode(0: LITE_POWER_HIGH; 1: LITE_POWER_LOW; 2:
                  ///< LITE_POWER_FULL; 3: LITE_POWER_NO_BIND, 4:
                  ///< LITE_POWER_RAND_HIGH; 5: LITE_POWER_RAND_LOW, refer
                  ///< [paddle
                  ///< lite](https://paddle-lite.readthedocs.io/zh/latest/api_reference/cxx_api_doc.html#set-power-mode)
                  ///< for more details)

  /** \brief Set shape range of input tensor for the model that contain dynamic
   * input shape while using TensorRT backend
   *
   * \param[in] input_name The name of input for the model which is dynamic
   * shape \param[in] min_shape The minimal shape for the input tensor
   * \param[in] opt_shape The optimized shape for the input tensor, just set the
   * most common shape, if set as default value, it will keep same with
   * min_shape \param[in] max_shape The maximum shape for the input tensor, if
   * set as default value, it will keep same with min_shape
   */
  void SetTrtInputShape(
      const std::string& input_name, const std::vector<int32_t>& min_shape,
      const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
      const std::vector<int32_t>& max_shape = std::vector<
          int32_t>());  ///< For model that contain dynamic input shape, this
                        ///< interface allow user to set its shape range while
                        ///< using TensorRT backend

  void SetTrtMaxWorkspaceSize(
      size_t trt_max_workspace_size);  ///< Set max_workspace_size for TensorRT,
                                       ///< default 1<<30

  void EnableTrtFP16();  ///< Enable FP16 inference while using TensorRT
                         ///< backend. Notice: not all the GPU device support
                         ///< FP16, on those device doesn't support FP16,
                         ///< FastDeploy will fallback to FP32 automaticly

  void
  DisableTrtFP16();  ///< Disable FP16 inference while using TensorRT backend

  void SetTrtCacheFile(
      const std::string&
          cache_file_path);  ///< Set cache file path while use TensorRT
                             ///< backend. Loadding a Paddle/ONNX model and
                             ///< initialize TensorRT will take a long time, by
                             ///< this interface it will save the tensorrt
                             ///< engine to `cache_file_path`, and load it
                             ///< directly while execute the code again.

  Backend backend = Backend::UNKNOWN;
  // for cpu inference and preprocess
  // default will let the backend choose their own default value
  int cpu_thread_num = -1;
  int device_id = 0;

  Device device = Device::CPU;

  // ======Only for ORT Backend========
  // -1 means use default value by ort
  // 0: ORT_DISABLE_ALL 1: ORT_ENABLE_BASIC 2: ORT_ENABLE_EXTENDED 3:
  // ORT_ENABLE_ALL
  int ort_graph_opt_level = -1;
  int ort_inter_op_num_threads = -1;
  // 0: ORT_SEQUENTIAL 1: ORT_PARALLEL
  int ort_execution_mode = -1;

  // ======Only for Paddle Backend=====
  bool pd_enable_mkldnn = true;
  bool pd_enable_log_info = false;
  int pd_mkldnn_cache_size = 1;
  std::vector<std::string> pd_delete_pass_names;

  // ======Only for Paddle-Lite Backend=====
  // 0: LITE_POWER_HIGH 1: LITE_POWER_LOW 2: LITE_POWER_FULL
  // 3: LITE_POWER_NO_BIND 4: LITE_POWER_RAND_HIGH
  // 5: LITE_POWER_RAND_LOW
  int lite_power_mode = 0;

  // ======Only for Trt Backend=======
  std::map<std::string, std::vector<int32_t>> trt_max_shape;
  std::map<std::string, std::vector<int32_t>> trt_min_shape;
  std::map<std::string, std::vector<int32_t>> trt_opt_shape;
  std::string trt_serialize_file = "";
  bool trt_enable_fp16 = false;
  bool trt_enable_int8 = false;
  size_t trt_max_batch_size = 32;
  size_t trt_max_workspace_size = 1 << 30;

  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty
  ModelFormat model_format = ModelFormat::AUTOREC;  // format of input model

  // inside parameters, only for inside usage
  // remove multiclass_nms in Paddle2ONNX
  bool remove_multiclass_nms_ = false;
  // for Paddle2ONNX to export custom operators
  std::map<std::string, std::string> custom_op_info_;
};

/*! @brief Runtime object used to inference the loaded model on different
 * devices
 */
struct FASTDEPLOY_DECL Runtime {
 public:
  bool Init(const RuntimeOption&
                _option);  ///< Intialize a Runtime object with RuntimeOption

  /** \brief Inference the model by the input data, and write to the output
   *
   * \param[in] input_tensors Notice the FDTensor::name should keep same with
   * the model's input \param[in] output_tensors Inference results \return true
   * if the inference successed, otherwise false
   */
  bool Infer(std::vector<FDTensor>& input_tensors,
             std::vector<FDTensor>* output_tensors);

  /** \brief Get number of inputs
   */
  int NumInputs() { return backend_->NumInputs(); }
  /** \brief Get number of outputs
   */
  int NumOutputs() { return backend_->NumOutputs(); }
  /** \brief Get input information by index
   */
  TensorInfo GetInputInfo(int index);
  /** \brief Get output information by index
   */
  TensorInfo GetOutputInfo(int index);
  /** \brief Get all the input information
   */
  std::vector<TensorInfo> GetInputInfos();
  /** \brief Get all the output information
   */
  std::vector<TensorInfo> GetOutputInfos();

  RuntimeOption option;

 private:
  void CreateOrtBackend();
  void CreatePaddleBackend();
  void CreateTrtBackend();
  void CreateOpenVINOBackend();
  void CreateLiteBackend();
  std::unique_ptr<BaseBackend> backend_;
};
}  // namespace fastdeploy
