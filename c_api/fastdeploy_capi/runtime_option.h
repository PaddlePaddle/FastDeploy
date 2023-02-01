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

#pragma once

#include <memory>

#include "fastdeploy_capi/fd_common.h"

typedef struct FD_RuntimeOption FD_RuntimeOption;


#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a new FD_RuntimeOption
 *
 * \return Return a pointer to FD_RuntimeOption object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_RuntimeOption* FD_CreateRuntimeOption();

/** \brief Destroy a FD_RuntimeOption object
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */

FASTDEPLOY_CAPI_EXPORT extern  void FD_DestroyRuntimeOption(__fd_take FD_RuntimeOption* fd_runtime_option);

/** \brief Set path of model file and parameter file
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] model_path Path of model file, e.g ResNet50/model.pdmodel for Paddle format model / ResNet50/model.onnx for ONNX format model
 * \param[in] params_path Path of parameter file, this only used when the model format is Paddle, e.g Resnet50/model.pdiparams
 * \param[in] format Format of the loaded model
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetModelPath(
                  __fd_keep FD_RuntimeOption *fd_runtime_option,
                  const char* model_path,
                  const char* params_path,
                  const FD_ModelFormat format);

/** \brief Specify the memory buffer of model and parameter. Used when model and params are loaded directly from memory
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] model_buffer The memory buffer of model
 * \param[in] params_buffer The memory buffer of the combined parameters file
 * \param[in] format Format of the loaded model
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetModelBuffer(__fd_keep FD_RuntimeOption *fd_runtime_option,
                      const char* model_buffer,
                      const char* params_buffer,
                      const FD_ModelFormat);


/** \brief Use cpu to inference, the runtime will inference on CPU by default
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseCpu(__fd_keep FD_RuntimeOption *fd_runtime_option);

/** \brief Use Nvidia GPU to inference
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseGpu(__fd_keep FD_RuntimeOption *fd_runtime_option, int gpu_id);


/** \brief Use RKNPU2 to inference
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] rknpu2_name  CpuName enum value
 * \param[in] rknpu2_core CoreMask enum value
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseRKNPU2(
                 __fd_keep FD_RuntimeOption *fd_runtime_option,
                 FD_rknpu2_CpuName rknpu2_name,
                 FD_rknpu2_CoreMask rknpu2_core);

/** \brief Use TimVX to inference
 *
 *  \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseTimVX(__fd_keep FD_RuntimeOption *fd_runtime_option);

/** \brief Use Huawei Ascend to inference
 *
 *  \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseAscend(__fd_keep FD_RuntimeOption *fd_runtime_option);

///
/// \brief Turn on KunlunXin XPU.
///
/// \param[in] fd_runtime_option pointer to FD_RuntimeOption object
/// \param[in] kunlunxin_id the KunlunXin XPU card to use (default is 0).
/// \param[in] l3_workspace_size The size of the video memory allocated by the l3
///         cache, the maximum is 16M.
/// \param[in] locked Whether the allocated L3 cache can be locked. If false,
///       it means that the L3 cache is not locked, and the allocated L3
///       cache can be shared by multiple models, and multiple models
///       sharing the L3 cache will be executed sequentially on the card.
/// \param[in] autotune Whether to autotune the conv operator in the model. If
///       true, when the conv operator of a certain dimension is executed
///       for the first time, it will automatically search for a better
///       algorithm to improve the performance of subsequent conv operators
///       of the same dimension.
/// \param[in] autotune_file Specify the path of the autotune file. If
///       autotune_file is specified, the algorithm specified in the
///       file will be used and autotune will not be performed again.
/// \param[in] precision Calculation accuracy of multi_encoder
/// \param[in] adaptive_seqlen Is the input of multi_encoder variable length
/// \param[in] enable_multi_stream Whether to enable the multi stream of
///        KunlunXin XPU.
///
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseKunlunXin(
                    __fd_keep FD_RuntimeOption *fd_runtime_option,
                    int kunlunxin_id, int l3_workspace_size,
                    FD_Bool locked, FD_Bool autotune,
                    const char* autotune_file,
                    const char* precision,
                    FD_Bool adaptive_seqlen,
                    FD_Bool enable_multi_stream);

/** Use Sophgo to inference
 *
 *  \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseSophgo(__fd_keep FD_RuntimeOption *fd_runtime_option);


FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetExternalStream(__fd_keep FD_RuntimeOption *fd_runtime_option, void* external_stream);

/**
  * @brief Set number of cpu threads while inference on CPU, by default it will decided by the different backends
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] thread_num number of threads
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetCpuThreadNum(__fd_keep FD_RuntimeOption *fd_runtime_option, int thread_num);

/**
  * @brief Set ORT graph opt level, default is decide by ONNX Runtime itself
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] level optimization level
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetOrtGraphOptLevel(__fd_keep FD_RuntimeOption *fd_runtime_option, int level);

/**
  * @brief Set Paddle Inference as inference backend, support CPU/GPU
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */

FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUsePaddleBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Wrapper function of UsePaddleBackend()
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUsePaddleInferBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set ONNX Runtime as inference backend, support CPU/GPU
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseOrtBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set SOPHGO Runtime as inference backend, support CPU/GPU
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseSophgoBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set TensorRT as inference backend, only support GPU
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseTrtBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set Poros backend as inference backend, support CPU/GPU
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUsePorosBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set OpenVINO as inference backend, only support CPU
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseOpenVINOBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set Paddle Lite as inference backend, only support arm cpu
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseLiteBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Wrapper function of UseLiteBackend()
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUsePaddleLiteBackend(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set mkldnn switch while using Paddle Inference as inference backend
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] pd_mkldnn whether to use mkldnn
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetPaddleMKLDNN(__fd_keep FD_RuntimeOption *fd_runtime_option, FD_Bool pd_mkldnn);

/**
  * @brief If TensorRT backend is used, EnablePaddleToTrt will change to use Paddle Inference backend, and use its integrated TensorRT instead.
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnablePaddleToTrt(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Delete pass by name while using Paddle Inference as inference backend, this can be called multiple times to delete a set of passes
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] delete_pass_name pass name
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDeletePaddleBackendPass(__fd_keep FD_RuntimeOption *fd_runtime_option, const char* delete_pass_name);

/**
  * @brief Enable print debug information while using Paddle Inference as inference backend, the backend disable the debug information by default
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnablePaddleLogInfo(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Disable print debug information while using Paddle Inference as inference backend
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDisablePaddleLogInfo(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief Set shape cache size while using Paddle Inference with mkldnn, by default it will cache all the difference shape
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] size cache size
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetPaddleMKLDNNCacheSize(__fd_keep FD_RuntimeOption *fd_runtime_option, int size);

/**
  * @brief Set device name for OpenVINO, default 'CPU', can also be 'AUTO', 'GPU', 'GPU.1'....
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] name device name
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetOpenVINODevice(__fd_keep FD_RuntimeOption *fd_runtime_option, const char* name);

/**
 * @brief Set shape info for OpenVINO
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 *
 */
// void SetOpenVINOShapeInfo(
//     __fd_keep FD_RuntimeOption *fd_runtime_option,
//     const std::map<std::string, std::vector<int64_t>>& shape_info);

// /**
//  * @brief While use OpenVINO backend with intel GPU, use this interface to specify operators run on CPU
//  */
// void SetOpenVINOCpuOperators(const std::vector<std::string>& operators) {
//   ov_cpu_operators = operators;
// }

/**
 * @brief Set optimzed model dir for Paddle Lite backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] optimized_model_dir optimzed model dir
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLiteOptimizedModelDir(__fd_keep FD_RuntimeOption *fd_runtime_option, const char* optimized_model_dir);

/**
 * @brief Set subgraph partition path for Paddle Lite backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] nnadapter_subgraph_partition_config_path subgraph partition path
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLiteSubgraphPartitionPath(
    __fd_keep FD_RuntimeOption *fd_runtime_option, const char* nnadapter_subgraph_partition_config_path);

/**
 * @brief Set subgraph partition path for Paddle Lite backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] nnadapter_subgraph_partition_config_buffer subgraph partition path
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLiteSubgraphPartitionConfigBuffer(
    __fd_keep FD_RuntimeOption *fd_runtime_option, const char* nnadapter_subgraph_partition_config_buffer);

/**
//  * @brief Set device name for Paddle Lite backend.
//  */
// void
// SetLiteDeviceNames(const std::vector<std::string>& nnadapter_device_names);

/**
 * @brief Set context properties for Paddle Lite backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] nnadapter_context_properties context properties
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLiteContextProperties(__fd_keep FD_RuntimeOption *fd_runtime_option, const char* nnadapter_context_properties);

/**
 * @brief Set model cache dir for Paddle Lite backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] nnadapter_model_cache_dir model cache dir
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLiteModelCacheDir(__fd_keep FD_RuntimeOption *fd_runtime_option, const char* nnadapter_model_cache_dir);

// /**
//  * @brief Set dynamic shape info for Paddle Lite backend.
//  */
// void SetLiteDynamicShapeInfo(
//     const std::map<std::string, std::vector<std::vector<int64_t>>>&
//         nnadapter_dynamic_shape_info);

/**
 * @brief Set mixed precision quantization config path for Paddle Lite backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] nnadapter_mixed_precision_quantization_config_path mixed precision quantization config path
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLiteMixedPrecisionQuantizationConfigPath(
    __fd_keep FD_RuntimeOption *fd_runtime_option, const char* nnadapter_mixed_precision_quantization_config_path);

/**
 * @brief enable half precision while use paddle lite backend
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnableLiteFP16(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief disable half precision, change to full precision(float32)
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDisableLiteFP16(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief enable int8 precision while use paddle lite backend
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnableLiteInt8(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
  * @brief disable int8 precision, change to full precision(float32)
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDisableLiteInt8(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief Set power mode while using Paddle Lite as inference backend, mode(0: LITE_POWER_HIGH; 1: LITE_POWER_LOW; 2: LITE_POWER_FULL; 3: LITE_POWER_NO_BIND, 4: LITE_POWER_RAND_HIGH; 5: LITE_POWER_RAND_LOW, refer [paddle lite](https://paddle-lite.readthedocs.io/zh/latest/api_reference/cxx_api_doc.html#set-power-mode) for more details)
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] mode power mode
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetLitePowerMode(__fd_keep FD_RuntimeOption *fd_runtime_option, FD_LitePowerMode mode);

// /** \brief Set shape range of input tensor for the model that contain dynamic input shape while using TensorRT backend
//  *
//  * \param[in] input_name The name of input for the model which is dynamic shape
//  * \param[in] min_shape The minimal shape for the input tensor
//  * \param[in] opt_shape The optimized shape for the input tensor, just set the most common shape, if set as default value, it will keep same with min_shape
//  * \param[in] max_shape The maximum shape for the input tensor, if set as default value, it will keep same with min_shape
//  */
// void SetTrtInputShape(
//     const char* input_name, const std::vector<int32_t>& min_shape,
//     const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
//     const std::vector<int32_t>& max_shape = std::vector<int32_t>());

/**
 * @brief Set max_workspace_size for TensorRT, default 1<<30
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 *
 * /
void SetTrtMaxWorkspaceSize(__fd_keep FD_RuntimeOption *fd_runtime_option, size_t trt_max_workspace_size);

/**
 * @brief Set max_batch_size for TensorRT, default 32
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 *
 * /
void SetTrtMaxBatchSize(__fd_keep FD_RuntimeOption *fd_runtime_option, size_t max_batch_size);

/**
 * @brief Enable FP16 inference while using TensorRT backend. Notice: not all the GPU device support FP16, on those device doesn't support FP16, FastDeploy will fallback to FP32 automaticly
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnableTrtFP16(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief Disable FP16 inference while using TensorRT backend
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDisableTrtFP16(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief Set cache file path while use TensorRT backend. Loadding a Paddle/ONNX model and initialize TensorRT will take a long time, by this interface it will save the tensorrt engine to `cache_file_path`, and load it directly while execute the code again
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] cache_file_path cache file path
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetTrtCacheFile(__fd_keep FD_RuntimeOption *fd_runtime_option, const char* cache_file_path);

/**
 * @brief Enable pinned memory. Pinned memory can be utilized to speedup the data transfer between CPU and GPU. Currently it's only suppurted in TRT backend and Paddle Inference backend.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnablePinnedMemory(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief Disable pinned memory
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDisablePinnedMemory(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief Enable to collect shape in paddle trt backend
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionEnablePaddleTrtCollectShape(__fd_keep FD_RuntimeOption *fd_runtime_option);

/**
 * @brief Disable to collect shape in paddle trt backend
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionDisablePaddleTrtCollectShape(__fd_keep FD_RuntimeOption *fd_runtime_option);

  // /**
  //  * @brief Prevent ops running in paddle trt backend
  //  */
  // void DisablePaddleTrtOPs(const std::vector<std::string>& ops);

/**
  * @brief Set number of streams by the OpenVINO backends
  *
  * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
  * \param[in] num_streams number of streams
  */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetOpenVINOStreams(__fd_keep FD_RuntimeOption *fd_runtime_option, int num_streams);

/**
 * @brief \Use Graphcore IPU to inference.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] device_num the number of IPUs.
 * \param[in] micro_batch_size the batch size in the graph, only work when graph has no batch shape info.
 * \param[in] enable_pipelining enable pipelining.
 * \param[in] batches_per_step the number of batches per run in pipelining.
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionUseIpu(__fd_keep FD_RuntimeOption *fd_runtime_option, int device_num, int micro_batch_size,
            FD_Bool enable_pipelining, int batches_per_step);

/** \brief Set IPU config.
 *
 * \param[in] fd_runtime_option pointer to FD_RuntimeOption object
 * \param[in] enable_fp16 enable fp16.
 * \param[in] replica_num the number of graph replication.
 * \param[in] available_memory_proportion the available memory proportion for matmul/conv.
 * \param[in] enable_half_partial enable fp16 partial for matmul, only work with fp16.
 */
FASTDEPLOY_CAPI_EXPORT extern void FD_RuntimeOptionSetIpuConfig(__fd_keep FD_RuntimeOption *fd_runtime_option, FD_Bool enable_fp16, int replica_num,
                  float available_memory_proportion,
                  FD_Bool enable_half_partial);

#ifdef __cplusplus
}  // extern "C"
#endif
