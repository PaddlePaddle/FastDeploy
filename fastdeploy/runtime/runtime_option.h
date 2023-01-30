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

/*! \file runtime_option.h
    \brief A brief file description.

    More details
 */

#pragma once

#include <algorithm>
#include <map>
#include <vector>
#include "fastdeploy/runtime/enum_variables.h"
#include "fastdeploy/runtime/backends/lite/option.h"
#include "fastdeploy/runtime/backends/openvino/option.h"
#include "fastdeploy/runtime/backends/ort/option.h"
#include "fastdeploy/runtime/backends/paddle/option.h"
#include "fastdeploy/runtime/backends/poros/option.h"
#include "fastdeploy/runtime/backends/rknpu2/option.h"
#include "fastdeploy/runtime/backends/sophgo/option.h"
#include "fastdeploy/runtime/backends/tensorrt/option.h"

namespace fastdeploy {

/*! @brief Option object used when create a new Runtime object
 */
struct FASTDEPLOY_DECL RuntimeOption {
  /** \brief Set path of model file and parameter file
   *
   * \param[in] model_path Path of model file, e.g ResNet50/model.pdmodel for Paddle format model / ResNet50/model.onnx for ONNX format model
   * \param[in] params_path Path of parameter file, this only used when the model format is Paddle, e.g Resnet50/model.pdiparams
   * \param[in] format Format of the loaded model
   */
  void SetModelPath(const std::string& model_path,
                    const std::string& params_path = "",
                    const ModelFormat& format = ModelFormat::PADDLE);

  /** \brief Specify the memory buffer of model and parameter. Used when model and params are loaded directly from memory
   *
   * \param[in] model_buffer The memory buffer of model
   * \param[in] model_buffer_size The size of the model data
   * \param[in] params_buffer The memory buffer of the combined parameters file
   * \param[in] params_buffer_size The size of the combined parameters data
   * \param[in] format Format of the loaded model
   */
  void SetModelBuffer(const char* model_buffer, size_t model_buffer_size,
                      const char* params_buffer, size_t params_buffer_size,
                      const ModelFormat& format = ModelFormat::PADDLE);

  /// Use cpu to inference, the runtime will inference on CPU by default
  void UseCpu();

  /// Use Nvidia GPU to inference
  void UseGpu(int gpu_id = 0);

  void UseRKNPU2(fastdeploy::rknpu2::CpuName rknpu2_name =
                     fastdeploy::rknpu2::CpuName::RK3588,
                 fastdeploy::rknpu2::CoreMask rknpu2_core =
                     fastdeploy::rknpu2::CoreMask::RKNN_NPU_CORE_0);

  /// Use TimVX to inference
  void UseTimVX();

  /// Use Huawei Ascend to inference
  void UseAscend();

  ///
  /// \brief Turn on KunlunXin XPU.
  ///
  /// \param kunlunxin_id the KunlunXin XPU card to use (default is 0).
  /// \param l3_workspace_size The size of the video memory allocated by the l3
  ///         cache, the maximum is 16M.
  /// \param locked Whether the allocated L3 cache can be locked. If false,
  ///       it means that the L3 cache is not locked, and the allocated L3
  ///       cache can be shared by multiple models, and multiple models
  ///       sharing the L3 cache will be executed sequentially on the card.
  /// \param autotune Whether to autotune the conv operator in the model. If
  ///       true, when the conv operator of a certain dimension is executed
  ///       for the first time, it will automatically search for a better
  ///       algorithm to improve the performance of subsequent conv operators
  ///       of the same dimension.
  /// \param autotune_file Specify the path of the autotune file. If
  ///       autotune_file is specified, the algorithm specified in the
  ///       file will be used and autotune will not be performed again.
  /// \param precision Calculation accuracy of multi_encoder
  /// \param adaptive_seqlen Is the input of multi_encoder variable length
  /// \param enable_multi_stream Whether to enable the multi stream of
  ///        KunlunXin XPU.
  ///
  void UseKunlunXin(int kunlunxin_id = 0, int l3_workspace_size = 0xfffc00,
                    bool locked = false, bool autotune = true,
                    const std::string& autotune_file = "",
                    const std::string& precision = "int16",
                    bool adaptive_seqlen = false,
                    bool enable_multi_stream = false);

  /// Use Sophgo to inference
  void UseSophgo();

  void SetExternalStream(void* external_stream);

  /*
   * @brief Set number of cpu threads while inference on CPU, by default it will decided by the different backends
   */
  void SetCpuThreadNum(int thread_num);

  /// Set ORT graph opt level, default is decide by ONNX Runtime itself
  void SetOrtGraphOptLevel(int level = -1);

  /// Set Paddle Inference as inference backend, support CPU/GPU
  void UsePaddleBackend();

  /// Wrapper function of UsePaddleBackend()
  void UsePaddleInferBackend() { return UsePaddleBackend(); }

  /// Set ONNX Runtime as inference backend, support CPU/GPU
  void UseOrtBackend();

  /// Set SOPHGO Runtime as inference backend, support CPU/GPU
  void UseSophgoBackend();

  /// Set TensorRT as inference backend, only support GPU
  void UseTrtBackend();

  /// Set Poros backend as inference backend, support CPU/GPU
  void UsePorosBackend();

  /// Set OpenVINO as inference backend, only support CPU
  void UseOpenVINOBackend();

  /// Set Paddle Lite as inference backend, only support arm cpu
  void UseLiteBackend();

  /// Wrapper function of UseLiteBackend()
  void UsePaddleLiteBackend() { return UseLiteBackend(); }

  /// Set mkldnn switch while using Paddle Inference as inference backend
  void SetPaddleMKLDNN(bool pd_mkldnn = true);

  /*
   * @brief If TensorRT backend is used, EnablePaddleToTrt will change to use Paddle Inference backend, and use its integrated TensorRT instead.
   */
  void EnablePaddleToTrt();

  /**
   * @brief Delete pass by name while using Paddle Inference as inference backend, this can be called multiple times to delete a set of passes
   */
  void DeletePaddleBackendPass(const std::string& delete_pass_name);

  /**
   * @brief Enable print debug information while using Paddle Inference as inference backend, the backend disable the debug information by default
   */
  void EnablePaddleLogInfo();

  /**
   * @brief Disable print debug information while using Paddle Inference as inference backend
   */
  void DisablePaddleLogInfo();

  /**
   * @brief Set shape cache size while using Paddle Inference with mkldnn, by default it will cache all the difference shape
   */
  void SetPaddleMKLDNNCacheSize(int size);

  /**
   * @brief Set device name for OpenVINO, default 'CPU', can also be 'AUTO', 'GPU', 'GPU.1'....
   */
  void SetOpenVINODevice(const std::string& name = "CPU");

  /**
   * @brief Set shape info for OpenVINO
   */
  void SetOpenVINOShapeInfo(
      const std::map<std::string, std::vector<int64_t>>& shape_info) {
    ov_shape_infos = shape_info;
  }

  /**
   * @brief While use OpenVINO backend with intel GPU, use this interface to specify operators run on CPU
   */
  void SetOpenVINOCpuOperators(const std::vector<std::string>& operators) {
    ov_cpu_operators = operators;
  }

  /**
   * @brief Set optimzed model dir for Paddle Lite backend.
   */
  void SetLiteOptimizedModelDir(const std::string& optimized_model_dir);

  /**
   * @brief Set subgraph partition path for Paddle Lite backend.
   */
  void SetLiteSubgraphPartitionPath(
      const std::string& nnadapter_subgraph_partition_config_path);

  /**
   * @brief Set subgraph partition path for Paddle Lite backend.
   */
  void SetLiteSubgraphPartitionConfigBuffer(
      const std::string& nnadapter_subgraph_partition_config_buffer);

  /**
   * @brief Set device name for Paddle Lite backend.
   */
  void
  SetLiteDeviceNames(const std::vector<std::string>& nnadapter_device_names);

  /**
   * @brief Set context properties for Paddle Lite backend.
   */
  void
  SetLiteContextProperties(const std::string& nnadapter_context_properties);

  /**
   * @brief Set model cache dir for Paddle Lite backend.
   */
  void SetLiteModelCacheDir(const std::string& nnadapter_model_cache_dir);

  /**
   * @brief Set dynamic shape info for Paddle Lite backend.
   */
  void SetLiteDynamicShapeInfo(
      const std::map<std::string, std::vector<std::vector<int64_t>>>&
          nnadapter_dynamic_shape_info);

  /**
   * @brief Set mixed precision quantization config path for Paddle Lite backend.
   */
  void SetLiteMixedPrecisionQuantizationConfigPath(
      const std::string& nnadapter_mixed_precision_quantization_config_path);

  /**
   * @brief enable half precision while use paddle lite backend
   */
  void EnableLiteFP16();

  /**
   * @brief disable half precision, change to full precision(float32)
   */
  void DisableLiteFP16();

  /**
    * @brief enable int8 precision while use paddle lite backend
    */
  void EnableLiteInt8();

  /**
    * @brief disable int8 precision, change to full precision(float32)
    */
  void DisableLiteInt8();

  /**
   * @brief Set power mode while using Paddle Lite as inference backend, mode(0: LITE_POWER_HIGH; 1: LITE_POWER_LOW; 2: LITE_POWER_FULL; 3: LITE_POWER_NO_BIND, 4: LITE_POWER_RAND_HIGH; 5: LITE_POWER_RAND_LOW, refer [paddle lite](https://paddle-lite.readthedocs.io/zh/latest/api_reference/cxx_api_doc.html#set-power-mode) for more details)
   */
  void SetLitePowerMode(LitePowerMode mode);

  /** \brief Set shape range of input tensor for the model that contain dynamic input shape while using TensorRT backend
   *
   * \param[in] input_name The name of input for the model which is dynamic shape
   * \param[in] min_shape The minimal shape for the input tensor
   * \param[in] opt_shape The optimized shape for the input tensor, just set the most common shape, if set as default value, it will keep same with min_shape
   * \param[in] max_shape The maximum shape for the input tensor, if set as default value, it will keep same with min_shape
   */
  void SetTrtInputShape(
      const std::string& input_name, const std::vector<int32_t>& min_shape,
      const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
      const std::vector<int32_t>& max_shape = std::vector<int32_t>());

  /// Set max_workspace_size for TensorRT, default 1<<30
  void SetTrtMaxWorkspaceSize(size_t trt_max_workspace_size);

  /// Set max_batch_size for TensorRT, default 32
  void SetTrtMaxBatchSize(size_t max_batch_size);

  /**
   * @brief Enable FP16 inference while using TensorRT backend. Notice: not all the GPU device support FP16, on those device doesn't support FP16, FastDeploy will fallback to FP32 automaticly
   */
  void EnableTrtFP16();

  /// Disable FP16 inference while using TensorRT backend
  void DisableTrtFP16();

  /**
   * @brief Set cache file path while use TensorRT backend. Loadding a Paddle/ONNX model and initialize TensorRT will take a long time, by this interface it will save the tensorrt engine to `cache_file_path`, and load it directly while execute the code again
   */
  void SetTrtCacheFile(const std::string& cache_file_path);

  /**
   * @brief Enable pinned memory. Pinned memory can be utilized to speedup the data transfer between CPU and GPU. Currently it's only suppurted in TRT backend and Paddle Inference backend.
   */
  void EnablePinnedMemory();

  /**
   * @brief Disable pinned memory
   */
  void DisablePinnedMemory();

  /**
   * @brief Enable to collect shape in paddle trt backend
   */
  void EnablePaddleTrtCollectShape();

  /**
   * @brief Disable to collect shape in paddle trt backend
   */
  void DisablePaddleTrtCollectShape();

  /**
   * @brief Prevent ops running in paddle trt backend
   */
  void DisablePaddleTrtOPs(const std::vector<std::string>& ops);

  /*
   * @brief Set number of streams by the OpenVINO backends
   */
  void SetOpenVINOStreams(int num_streams);

  /** \Use Graphcore IPU to inference.
   *
   * \param[in] device_num the number of IPUs.
   * \param[in] micro_batch_size the batch size in the graph, only work when graph has no batch shape info.
   * \param[in] enable_pipelining enable pipelining.
   * \param[in] batches_per_step the number of batches per run in pipelining.
   */
  void UseIpu(int device_num = 1, int micro_batch_size = 1,
              bool enable_pipelining = false, int batches_per_step = 1);

  /** \brief Set IPU config.
   *
   * \param[in] enable_fp16 enable fp16.
   * \param[in] replica_num the number of graph replication.
   * \param[in] available_memory_proportion the available memory proportion for matmul/conv.
   * \param[in] enable_half_partial enable fp16 partial for matmul, only work with fp16.
   */
  void SetIpuConfig(bool enable_fp16 = false, int replica_num = 1,
                    float available_memory_proportion = 1.0,
                    bool enable_half_partial = false);

  Backend backend = Backend::UNKNOWN;

  // for cpu inference
  // default will let the backend choose their own default value
  int cpu_thread_num = -1;
  int device_id = 0;

  Device device = Device::CPU;

  void* external_stream_ = nullptr;

  bool enable_pinned_memory = false;

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
  bool pd_enable_trt = false;
  bool pd_collect_shape = false;
  int pd_mkldnn_cache_size = 1;
  std::vector<std::string> pd_delete_pass_names;

  // ======Only for Paddle IPU Backend =======
  int ipu_device_num = 1;
  int ipu_micro_batch_size = 1;
  bool ipu_enable_pipelining = false;
  int ipu_batches_per_step = 1;
  bool ipu_enable_fp16 = false;
  int ipu_replica_num = 1;
  float ipu_available_memory_proportion = 1.0;
  bool ipu_enable_half_partial = false;

  // ======Only for Trt Backend=======
  std::map<std::string, std::vector<int32_t>> trt_max_shape;
  std::map<std::string, std::vector<int32_t>> trt_min_shape;
  std::map<std::string, std::vector<int32_t>> trt_opt_shape;
  std::string trt_serialize_file = "";
  bool trt_enable_fp16 = false;
  bool trt_enable_int8 = false;
  size_t trt_max_batch_size = 1;
  size_t trt_max_workspace_size = 1 << 30;
  // ======Only for PaddleTrt Backend=======
  std::vector<std::string> trt_disabled_ops_{};

  // ======Only for Poros Backend=======
  bool is_dynamic = false;
  bool long_to_int = true;
  bool use_nvidia_tf32 = false;
  int unconst_ops_thres = -1;
  std::string poros_file = "";

  // ======Only for OpenVINO Backend=======
  int ov_num_streams = 0;
  std::string openvino_device = "CPU";
  std::map<std::string, std::vector<int64_t>> ov_shape_infos;
  std::vector<std::string> ov_cpu_operators;

  // ======Only for RKNPU2 Backend=======
  fastdeploy::rknpu2::CpuName rknpu2_cpu_name_ =
      fastdeploy::rknpu2::CpuName::RK3588;
  fastdeploy::rknpu2::CoreMask rknpu2_core_mask_ =
      fastdeploy::rknpu2::CoreMask::RKNN_NPU_CORE_AUTO;


  /// Option to configure Paddle Lite backend
  LiteBackendOption paddle_lite_option;

  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty
  // format of input model
  ModelFormat model_format = ModelFormat::PADDLE;

  std::string model_buffer_ = "";
  std::string params_buffer_ = "";
  size_t model_buffer_size_ = 0;
  size_t params_buffer_size_ = 0;
  bool model_from_memory_ = false;
};

}  // namespace fastdeploy
