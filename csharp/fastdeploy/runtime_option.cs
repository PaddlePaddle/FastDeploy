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

using System;
using System.IO;
using System.Runtime.InteropServices;

namespace fastdeploy {

/*! @brief Option object used when create a new Runtime object
 */
public class RuntimeOption {

  public RuntimeOption() {
    fd_runtime_option_wrapper = FD_C_CreateRuntimeOptionWrapper();
  }

  ~RuntimeOption() {
    FD_C_DestroyRuntimeOptionWrapper(fd_runtime_option_wrapper);
  }

  /** \brief Set path of model file and parameter file
   *
   * \param[in] model_path Path of model file, e.g ResNet50/model.pdmodel for Paddle format model / ResNet50/model.onnx for ONNX format model
   * \param[in] params_path Path of parameter file, this only used when the model format is Paddle, e.g Resnet50/model.pdiparams
   * \param[in] format Format of the loaded model
   */
  public void SetModelPath(string model_path, string params_path = "",
                           ModelFormat format = ModelFormat.PADDLE) {
    FD_C_RuntimeOptionWrapperSetModelPath(fd_runtime_option_wrapper, model_path,
                                          params_path, format);
  }

  /** \brief Specify the memory buffer of model and parameter. Used when model and params are loaded directly from memory
   *
   * \param[in] model_buffer The string of model memory buffer
   * \param[in] params_buffer The string of parameters memory buffer
   * \param[in] format Format of the loaded model
   */
  public void SetModelBuffer(string model_buffer, string params_buffer = "",
                             ModelFormat format = ModelFormat.PADDLE) {
    FD_C_RuntimeOptionWrapperSetModelBuffer(
        fd_runtime_option_wrapper, model_buffer, params_buffer, format);
  }

  /// Use cpu to inference, the runtime will inference on CPU by default
  public void UseCpu() {
    FD_C_RuntimeOptionWrapperUseCpu(fd_runtime_option_wrapper);
  }

  /// Use Nvidia GPU to inference
  public void UseGpu(int gpu_id = 0) {
    FD_C_RuntimeOptionWrapperUseGpu(fd_runtime_option_wrapper, gpu_id);
  }

  /// Use RKNPU2 e.g RK3588/RK356X to inference
  public void
  UseRKNPU2(rknpu2_CpuName rknpu2_name = rknpu2_CpuName.RK3588,
            rknpu2_CoreMask rknpu2_core = rknpu2_CoreMask.RKNN_NPU_CORE_0) {
    FD_C_RuntimeOptionWrapperUseRKNPU2(fd_runtime_option_wrapper, rknpu2_name,
                                       rknpu2_core);
  }

  /// Use TimVX e.g RV1126/A311D to inference
  public void UseTimVX() {
    FD_C_RuntimeOptionWrapperUseTimVX(fd_runtime_option_wrapper);
  }

  /// Use Huawei Ascend to inference
  public void UseAscend() {
    FD_C_RuntimeOptionWrapperUseAscend(fd_runtime_option_wrapper);
  }

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
  /// \param gm_default_size The default size of context global memory of KunlunXin XPU.
  ///
  public void
  UseKunlunXin(int kunlunxin_id = 0, int l3_workspace_size = 0xfffc00,
               bool locked = false, bool autotune = true,
               string autotune_file = "", string precision = "int16",
               bool adaptive_seqlen = false, bool enable_multi_stream = false,
               int64_t gm_default_size = 0) {
    FD_C_RuntimeOptionWrapperUseKunlunXin(
        fd_runtime_option_wrapper, kunlunxin_id, l3_workspace_size, locked,
        autotune, autotune_file, precision,  adaptive_seqlen,
        enable_multi_stream, gm_default_size);
  }

  /// Use Sophgo to inference
  public void UseSophgo() {
    FD_C_RuntimeOptionWrapperUseSophgo(fd_runtime_option_wrapper);
  }

  public void SetExternalStream(IntPtr external_stream) {
    FD_C_RuntimeOptionWrapperSetExternalStream(fd_runtime_option_wrapper,
                                               external_stream);
  }

  /*
   * @brief Set number of cpu threads while inference on CPU, by default it will decided by the different backends
   */
  public void SetCpuThreadNum(int thread_num) {
    FD_C_RuntimeOptionWrapperSetCpuThreadNum(fd_runtime_option_wrapper,
                                             thread_num);
  }

  public void SetOrtGraphOptLevel(int level = -1) {
    FD_C_RuntimeOptionWrapperSetOrtGraphOptLevel(fd_runtime_option_wrapper,
                                                 level);
  }

  public void UsePaddleBackend() {
    FD_C_RuntimeOptionWrapperUsePaddleBackend(fd_runtime_option_wrapper);
  }

  /// Set Paddle Inference as inference backend, support CPU/GPU
  public void UsePaddleInferBackend() {
    FD_C_RuntimeOptionWrapperUsePaddleInferBackend(fd_runtime_option_wrapper);
  }

  /// Set ONNX Runtime as inference backend, support CPU/GPU
  public void UseOrtBackend() {
    FD_C_RuntimeOptionWrapperUseOrtBackend(fd_runtime_option_wrapper);
  }

  /// Set SOPHGO Runtime as inference backend, support SOPHGO
  public void UseSophgoBackend() {
    FD_C_RuntimeOptionWrapperUseSophgoBackend(fd_runtime_option_wrapper);
  }

  /// Set TensorRT as inference backend, only support GPU
  public void UseTrtBackend() {
    FD_C_RuntimeOptionWrapperUseTrtBackend(fd_runtime_option_wrapper);
  }

  /// Set Poros backend as inference backend, support CPU/GPU
  public void UsePorosBackend() {
    FD_C_RuntimeOptionWrapperUsePorosBackend(fd_runtime_option_wrapper);
  }

  /// Set OpenVINO as inference backend, only support CPU
  public void UseOpenVINOBackend() {
    FD_C_RuntimeOptionWrapperUseOpenVINOBackend(fd_runtime_option_wrapper);
  }

  /// Set Paddle Lite as inference backend, only support arm cpu
  public void UseLiteBackend() {
    FD_C_RuntimeOptionWrapperUseLiteBackend(fd_runtime_option_wrapper);
  }

  /// Set Paddle Lite as inference backend, only support arm cpu
  public void UsePaddleLiteBackend() {
    FD_C_RuntimeOptionWrapperUsePaddleLiteBackend(fd_runtime_option_wrapper);
  }

  
  public void SetPaddleMKLDNN(bool pd_mkldnn = true) {
    FD_C_RuntimeOptionWrapperSetPaddleMKLDNN(fd_runtime_option_wrapper,
                                             pd_mkldnn);
  }

  public void EnablePaddleToTrt() {
    FD_C_RuntimeOptionWrapperEnablePaddleToTrt(fd_runtime_option_wrapper);
  }

  public void DeletePaddleBackendPass(string delete_pass_name) {
    FD_C_RuntimeOptionWrapperDeletePaddleBackendPass(fd_runtime_option_wrapper,
                                                     delete_pass_name);
  }

  public void EnablePaddleLogInfo() {
    FD_C_RuntimeOptionWrapperEnablePaddleLogInfo(fd_runtime_option_wrapper);
  }

  public void DisablePaddleLogInfo() {
    FD_C_RuntimeOptionWrapperDisablePaddleLogInfo(fd_runtime_option_wrapper);
  }

  public void SetPaddleMKLDNNCacheSize(int size) {
    FD_C_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize(fd_runtime_option_wrapper,
                                                      size);
  }

  public void SetOpenVINODevice(string name = "CPU") {
    FD_C_RuntimeOptionWrapperSetOpenVINODevice(fd_runtime_option_wrapper, name);
  }

  public void SetLiteOptimizedModelDir(string optimized_model_dir) {
    FD_C_RuntimeOptionWrapperSetLiteOptimizedModelDir(fd_runtime_option_wrapper,
                                                      optimized_model_dir);
  }

  public void SetLiteSubgraphPartitionPath(
      string nnadapter_subgraph_partition_config_path) {
    FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionPath(
        fd_runtime_option_wrapper, nnadapter_subgraph_partition_config_path);
  }

  public void SetLiteSubgraphPartitionConfigBuffer(
      string nnadapter_subgraph_partition_config_buffer) {
    FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer(
        fd_runtime_option_wrapper, nnadapter_subgraph_partition_config_buffer);
  }

  public void SetLiteContextProperties(string nnadapter_context_properties) {
    FD_C_RuntimeOptionWrapperSetLiteContextProperties(
        fd_runtime_option_wrapper, nnadapter_context_properties);
  }

  public void SetLiteModelCacheDir(string nnadapter_model_cache_dir) {
    FD_C_RuntimeOptionWrapperSetLiteModelCacheDir(fd_runtime_option_wrapper,
                                                  nnadapter_model_cache_dir);
  }

  public void SetLiteMixedPrecisionQuantizationConfigPath(
      string nnadapter_mixed_precision_quantization_config_path) {
    FD_C_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath(
        fd_runtime_option_wrapper,
        nnadapter_mixed_precision_quantization_config_path);
  }

  public void EnableLiteFP16() {
    FD_C_RuntimeOptionWrapperEnableLiteFP16(fd_runtime_option_wrapper);
  }

  public void DisableLiteFP16() {
    FD_C_RuntimeOptionWrapperDisableLiteFP16(fd_runtime_option_wrapper);
  }

  public void EnableLiteInt8() {
    FD_C_RuntimeOptionWrapperEnableLiteInt8(fd_runtime_option_wrapper);
  }

  public void DisableLiteInt8() {
    FD_C_RuntimeOptionWrapperDisableLiteInt8(fd_runtime_option_wrapper);
  }

  public void SetLitePowerMode(LitePowerMode mode) {
    FD_C_RuntimeOptionWrapperSetLitePowerMode(fd_runtime_option_wrapper, mode);
  }

  public void EnableTrtFP16() {
    FD_C_RuntimeOptionWrapperEnableTrtFP16(fd_runtime_option_wrapper);
  }

  public void DisableTrtFP16() {
    FD_C_RuntimeOptionWrapperDisableTrtFP16(fd_runtime_option_wrapper);
  }

  public void SetTrtCacheFile(string cache_file_path) {
    FD_C_RuntimeOptionWrapperSetTrtCacheFile(fd_runtime_option_wrapper,
                                             cache_file_path);
  }

  public void EnablePinnedMemory() {
    FD_C_RuntimeOptionWrapperEnablePinnedMemory(fd_runtime_option_wrapper);
  }

  public void DisablePinnedMemory() {
    FD_C_RuntimeOptionWrapperDisablePinnedMemory(fd_runtime_option_wrapper);
  }

  public void EnablePaddleTrtCollectShape() {
    FD_C_RuntimeOptionWrapperEnablePaddleTrtCollectShape(
        fd_runtime_option_wrapper);
  }

  public void DisablePaddleTrtCollectShape() {
    FD_C_RuntimeOptionWrapperDisablePaddleTrtCollectShape(
        fd_runtime_option_wrapper);
  }

  public void SetOpenVINOStreams(int num_streams) {
    FD_C_RuntimeOptionWrapperSetOpenVINOStreams(fd_runtime_option_wrapper,
                                                num_streams);
  }

  public void UseIpu(int device_num = 1, int micro_batch_size = 1,
                     bool enable_pipelining = false, int batches_per_step = 1) {
    FD_C_RuntimeOptionWrapperUseIpu(fd_runtime_option_wrapper, device_num,
                                    micro_batch_size, enable_pipelining,
                                    batches_per_step);
  }

  public IntPtr GetWrapperPtr() { return fd_runtime_option_wrapper; }

  // Below are underlying C api
  private IntPtr fd_runtime_option_wrapper;

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateRuntimeOptionWrapper")]
  private static extern IntPtr FD_C_CreateRuntimeOptionWrapper();

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyRuntimeOptionWrapper")]
  private static extern void
  FD_C_DestroyRuntimeOptionWrapper(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetModelPath")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetModelPath(IntPtr fd_runtime_option_wrapper,
                                        string model_path, string params_path,
                                        ModelFormat format);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetModelBuffer")]
  private static extern void FD_C_RuntimeOptionWrapperSetModelBuffer(
      IntPtr fd_runtime_option_wrapper, string model_buffer,
      string params_buffer, ModelFormat format);

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_RuntimeOptionWrapperUseCpu")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseCpu(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_RuntimeOptionWrapperUseGpu")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseGpu(IntPtr fd_runtime_option_wrapper, int gpu_id);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseRKNPU2")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseRKNPU2(IntPtr fd_runtime_option_wrapper,
                                     rknpu2_CpuName rknpu2_name,
                                     rknpu2_CoreMask rknpu2_core);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseTimVX")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseTimVX(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseAscend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseAscend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseKunlunXin")]
  private static extern void FD_C_RuntimeOptionWrapperUseKunlunXin(
      IntPtr fd_runtime_option_wrapper, int kunlunxin_id, int l3_workspace_size,
      bool locked, bool autotune, string autotune_file, string precision,
      bool adaptive_seqlen, bool enable_multi_stream,
      Int64 gm_default_size);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseSophgo")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseSophgo(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetExternalStream")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetExternalStream(IntPtr fd_runtime_option_wrapper,
                                             IntPtr external_stream);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetCpuThreadNum")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetCpuThreadNum(IntPtr fd_runtime_option_wrapper,
                                           int thread_num);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetOrtGraphOptLevel")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetOrtGraphOptLevel(IntPtr fd_runtime_option_wrapper,
                                               int level);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUsePaddleBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUsePaddleBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUsePaddleInferBackend")]
  private static extern void FD_C_RuntimeOptionWrapperUsePaddleInferBackend(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseOrtBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseOrtBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseSophgoBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseSophgoBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseTrtBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseTrtBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUsePorosBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUsePorosBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseOpenVINOBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseOpenVINOBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUseLiteBackend")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseLiteBackend(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperUsePaddleLiteBackend")]
  private static extern void FD_C_RuntimeOptionWrapperUsePaddleLiteBackend(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetPaddleMKLDNN")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetPaddleMKLDNN(IntPtr fd_runtime_option_wrapper,
                                           bool pd_mkldnn);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperEnablePaddleToTrt")]
  private static extern void
  FD_C_RuntimeOptionWrapperEnablePaddleToTrt(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperDeletePaddleBackendPass")]
  private static extern void FD_C_RuntimeOptionWrapperDeletePaddleBackendPass(
      IntPtr fd_runtime_option_wrapper, string delete_pass_name);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperEnablePaddleLogInfo")]
  private static extern void FD_C_RuntimeOptionWrapperEnablePaddleLogInfo(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperDisablePaddleLogInfo")]
  private static extern void FD_C_RuntimeOptionWrapperDisablePaddleLogInfo(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize")]
  private static extern void FD_C_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize(
      IntPtr fd_runtime_option_wrapper, int size);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetOpenVINODevice")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetOpenVINODevice(IntPtr fd_runtime_option_wrapper,
                                             string name);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetLiteOptimizedModelDir")]
  private static extern void FD_C_RuntimeOptionWrapperSetLiteOptimizedModelDir(
      IntPtr fd_runtime_option_wrapper, string optimized_model_dir);

  [DllImport("fastdeploy.dll",
             EntryPoint =
                 "FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionPath")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionPath(
      IntPtr fd_runtime_option_wrapper,
      string nnadapter_subgraph_partition_config_path);

  [DllImport(
      "fastdeploy.dll",
      EntryPoint =
          "FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer(
      IntPtr fd_runtime_option_wrapper,
      string nnadapter_subgraph_partition_config_buffer);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetLiteContextProperties")]
  private static extern void FD_C_RuntimeOptionWrapperSetLiteContextProperties(
      IntPtr fd_runtime_option_wrapper, string nnadapter_context_properties);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetLiteModelCacheDir")]
  private static extern void FD_C_RuntimeOptionWrapperSetLiteModelCacheDir(
      IntPtr fd_runtime_option_wrapper, string nnadapter_model_cache_dir);

  [DllImport(
      "fastdeploy.dll",
      EntryPoint =
          "FD_C_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath(
      IntPtr fd_runtime_option_wrapper,
      string nnadapter_mixed_precision_quantization_config_path);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperEnableLiteFP16")]
  private static extern void
  FD_C_RuntimeOptionWrapperEnableLiteFP16(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperDisableLiteFP16")]
  private static extern void
  FD_C_RuntimeOptionWrapperDisableLiteFP16(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperEnableLiteInt8")]
  private static extern void
  FD_C_RuntimeOptionWrapperEnableLiteInt8(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperDisableLiteInt8")]
  private static extern void
  FD_C_RuntimeOptionWrapperDisableLiteInt8(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetLitePowerMode")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetLitePowerMode(IntPtr fd_runtime_option_wrapper,
                                            LitePowerMode mode);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperEnableTrtFP16")]
  private static extern void
  FD_C_RuntimeOptionWrapperEnableTrtFP16(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperDisableTrtFP16")]
  private static extern void
  FD_C_RuntimeOptionWrapperDisableTrtFP16(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetTrtCacheFile")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetTrtCacheFile(IntPtr fd_runtime_option_wrapper,
                                           string cache_file_path);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperEnablePinnedMemory")]
  private static extern void
  FD_C_RuntimeOptionWrapperEnablePinnedMemory(IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperDisablePinnedMemory")]
  private static extern void FD_C_RuntimeOptionWrapperDisablePinnedMemory(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint =
                 "FD_C_RuntimeOptionWrapperEnablePaddleTrtCollectShape")]
  private static extern void
  FD_C_RuntimeOptionWrapperEnablePaddleTrtCollectShape(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint =
                 "FD_C_RuntimeOptionWrapperDisablePaddleTrtCollectShape")]
  private static extern void
  FD_C_RuntimeOptionWrapperDisablePaddleTrtCollectShape(
      IntPtr fd_runtime_option_wrapper);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RuntimeOptionWrapperSetOpenVINOStreams")]
  private static extern void
  FD_C_RuntimeOptionWrapperSetOpenVINOStreams(IntPtr fd_runtime_option_wrapper,
                                              int num_streams);

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_RuntimeOptionWrapperUseIpu")]
  private static extern void
  FD_C_RuntimeOptionWrapperUseIpu(IntPtr fd_runtime_option_wrapper,
                                  int device_num, int micro_batch_size,
                                  bool enable_pipelining, int batches_per_step);
}
}
