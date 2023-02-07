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

namespace fastdeploy{


  public class RuntimeOption{


    public RuntimeOption(){
      fd_runtime_option_wrapper = FD_CreateRuntimeOptionWrapper();
    }

    ~RuntimeOption(){
      FD_DestroyRuntimeOptionWrapper(fd_runtime_option_wrapper);
    }


    public void SetModelPath(string model_path,
                      string params_path,
                      ModelFormat format){
      FD_RuntimeOptionWrapperSetModelPath(fd_runtime_option_wrapper, model_path, params_path, format);
    }


    public void SetModelBuffer(string model_buffer,
                          string params_buffer,
                          ModelFormat format){
      FD_RuntimeOptionWrapperSetModelBuffer(fd_runtime_option_wrapper, model_buffer, params_buffer, format);
    }



    public void UseCpu(){
      FD_RuntimeOptionWrapperUseCpu(fd_runtime_option_wrapper);
    }


    public void UseGpu(int gpu_id){
      FD_RuntimeOptionWrapperUseGpu(fd_runtime_option_wrapper, gpu_id);
    }



    public void UseRKNPU2(
                    rknpu2_CpuName rknpu2_name,
                    rknpu2_CoreMask rknpu2_core){
      FD_RuntimeOptionWrapperUseRKNPU2(fd_runtime_option_wrapper, rknpu2_name, rknpu2_core);
    }


    public void UseTimVX(){
      FD_RuntimeOptionWrapperUseTimVX(fd_runtime_option_wrapper);
    }


    public void UseAscend(){
      FD_RuntimeOptionWrapperUseAscend(fd_runtime_option_wrapper);
    }

    public void UseKunlunXin(
                        int kunlunxin_id, int l3_workspace_size,
                        bool locked, bool autotune,
                        string autotune_file,
                        string precision,
                        bool adaptive_seqlen,
                        bool enable_multi_stream){
      FD_RuntimeOptionWrapperUseKunlunXin(fd_runtime_option_wrapper, kunlunxin_id,
      l3_workspace_size, locked, autotune, autotune_file, precision, adaptive_seqlen, enable_multi_stream);
    }

    public void UseSophgo(){
      FD_RuntimeOptionWrapperUseSophgo(fd_runtime_option_wrapper);
    }


    public void SetExternalStream(void* external_stream){
      FD_RuntimeOptionWrapperSetExternalStream(fd_runtime_option_wrapper, external_stream);
    }


    public void SetCpuThreadNum(int thread_num){
      FD_RuntimeOptionWrapperSetCpuThreadNum(fd_runtime_option_wrapper, thread_num);
    }


    public void SetOrtGraphOptLevel(int level){
      FD_RuntimeOptionWrapperSetOrtGraphOptLevel(fd_runtime_option_wrapper, level);
    }


    public void UsePaddleBackend(){
      FD_RuntimeOptionWrapperUsePaddleBackend(fd_runtime_option_wrapper);
    }


    public void UsePaddleInferBackend(){
      FD_RuntimeOptionWrapperUsePaddleInferBackend(fd_runtime_option_wrapper);
    }


    public void UseOrtBackend(){
      FD_RuntimeOptionWrapperUseOrtBackend(fd_runtime_option_wrapper);
    }


    public void UseSophgoBackend(){
      FD_RuntimeOptionWrapperUseSophgoBackend(fd_runtime_option_wrapper);
    }


    public void UseTrtBackend(){
      FD_RuntimeOptionWrapperUseTrtBackend(fd_runtime_option_wrapper);
    }


    public void UsePorosBackend(){
      FD_RuntimeOptionWrapperUsePorosBackend(fd_runtime_option_wrapper);
    }


    public void UseOpenVINOBackend(){
      FD_RuntimeOptionWrapperUseOpenVINOBackend(fd_runtime_option_wrapper);
    }


    public void UseLiteBackend(){
      FD_RuntimeOptionWrapperUseLiteBackend(fd_runtime_option_wrapper);
    }


    public void UsePaddleLiteBackend(){
      FD_RuntimeOptionWrapperUsePaddleLiteBackend(fd_runtime_option_wrapper);
    }


    public void SetPaddleMKLDNN(bool pd_mkldnn){
      FD_RuntimeOptionWrapperSetPaddleMKLDNN(fd_runtime_option_wrapper, pd_mkldnn);
    }


    public void EnablePaddleToTrt(){
      FD_RuntimeOptionWrapperEnablePaddleToTrt(fd_runtime_option_wrapper);
    } 


    public void DeletePaddleBackendPass(string delete_pass_name){
      FD_RuntimeOptionWrapperDeletePaddleBackendPass(fd_runtime_option_wrapper, delete_pass_name);
    }


    public void EnablePaddleLogInfo(){
      FD_RuntimeOptionWrapperEnablePaddleLogInfo(fd_runtime_option_wrapper);
    }


    public void DisablePaddleLogInfo(){
      FD_RuntimeOptionWrapperDisablePaddleLogInfo(fd_runtime_option_wrapper);
    }


    public void SetPaddleMKLDNNCacheSize(int size){
      FD_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize(fd_runtime_option_wrapper, size);
    }


    public void SetOpenVINODevice(string name){
      FD_RuntimeOptionWrapperSetOpenVINODevice(fd_runtime_option_wrapper, name);
    }


    public void SetLiteOptimizedModelDir(string optimized_model_dir){
      FD_RuntimeOptionWrapperSetLiteOptimizedModelDir(fd_runtime_option_wrapper, optimized_model_dir);
    }


    public void SetLiteSubgraphPartitionPath(
      string nnadapter_subgraph_partition_config_path){
        FD_RuntimeOptionWrapperSetLiteSubgraphPartitionPath(fd_runtime_option_wrapper, nnadapter_subgraph_partition_config_path);
      }


    public void SetLiteSubgraphPartitionConfigBuffer(
      string nnadapter_subgraph_partition_config_buffer){
        FD_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer(fd_runtime_option_wrapper, nnadapter_subgraph_partition_config_buffer);
      }



    public void SetLiteContextProperties(string nnadapter_context_properties){
      FD_RuntimeOptionWrapperSetLiteContextProperties(fd_runtime_option_wrapper, nnadapter_context_properties);
    }


    public void SetLiteModelCacheDir(string nnadapter_model_cache_dir){
      FD_RuntimeOptionWrapperSetLiteModelCacheDir(fd_runtime_option_wrapper, nnadapter_model_cache_dir);
    }


    public void SetLiteMixedPrecisionQuantizationConfigPath(
        string nnadapter_mixed_precision_quantization_config_path){
      FD_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath(fd_runtime_option_wrapper, nnadapter_mixed_precision_quantization_config_path);
    }


    public void EnableLiteFP16(){
      FD_RuntimeOptionWrapperEnableLiteFP16(fd_runtime_option_wrapper);
    }


    public void DisableLiteFP16(){
      FD_RuntimeOptionWrapperDisableLiteFP16(fd_runtime_option_wrapper);
    }


    public void EnableLiteInt8(){
      FD_RuntimeOptionWrapperEnableLiteInt8(fd_runtime_option_wrapper);
    }


    public void DisableLiteInt8(){
      FD_RuntimeOptionWrapperDisableLiteInt8(fd_runtime_option_wrapper);
    }


    public void SetLitePowerMode(LitePowerMode mode){
      FD_RuntimeOptionWrapperSetLitePowerMode(fd_runtime_option_wrapper, mode);
    }

    public void EnableTrtFP16(){
      FD_RuntimeOptionWrapperEnableTrtFP16(fd_runtime_option_wrapper);
    }


    public void DisableTrtFP16(){
      FD_RuntimeOptionWrapperDisableTrtFP16(fd_runtime_option_wrapper);
    }


    public void SetTrtCacheFile(string cache_file_path){
      FD_RuntimeOptionWrapperSetTrtCacheFile(fd_runtime_option_wrapper,cache_file_path);
    }


    public void EnablePinnedMemory(){
      FD_RuntimeOptionWrapperEnablePinnedMemory(fd_runtime_option_wrapper);
    }


    public void DisablePinnedMemory(){
      FD_RuntimeOptionWrapperDisablePinnedMemory(fd_runtime_option_wrapper);
    }


    public void EnablePaddleTrtCollectShape(){
      FD_RuntimeOptionWrapperEnablePaddleTrtCollectShape(fd_runtime_option_wrapper);
    }


    public void DisablePaddleTrtCollectShape(){
      FD_RuntimeOptionWrapperDisablePaddleTrtCollectShape(fd_runtime_option_wrapper);
    }


    public void SetOpenVINOStreams(int num_streams){
      FD_RuntimeOptionWrapperSetOpenVINOStreams(fd_runtime_option_wrapper, num_streams);
    }


    public void UseIpu(int device_num, int micro_batch_size,
                bool enable_pipelining, int batches_per_step){
      FD_RuntimeOptionWrapperUseIpu(fd_runtime_option_wrapper, device_num, micro_batch_size, enable_pipelining, batches_per_step);
    }


    public void SetIpuConfig(bool enable_fp16, int replica_num,
                      float available_memory_proportion,
                      bool enable_half_partial){
      FD_RuntimeOptionWrapperSetIpuConfig(fd_runtime_option_wrapper, enable_fp16, replica_num, available_memory_proportion, enable_half_partial);
    }
    
    public IntPtr GetWrapperPtr(){
      return fd_runtime_option_wrapper;
    }

    // Below are underlying C api   
    private IntPtr fd_runtime_option_wrapper;

    [DllImport("fastdeploy.dll", EntryPoint = "FD_CreateRuntimeOptionWrapper")]
    private static extern  IntPtr FD_CreateRuntimeOptionWrapper();

    [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyRuntimeOptionWrapper")]
    private static extern  void FD_DestroyRuntimeOptionWrapper( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetModelPath")]
    private static extern void FD_RuntimeOptionWrapperSetModelPath(
                      IntPtr fd_runtime_option_wrapper,
                      string model_path,
                      string params_path,
                      ModelFormat format);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetModelBuffer")]
    private static extern void FD_RuntimeOptionWrapperSetModelBuffer( IntPtr fd_runtime_option_wrapper,
                          string model_buffer,
                          string params_buffer,
                          ModelFormat format);


    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseCpu")]
    private static extern void FD_RuntimeOptionWrapperUseCpu( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseGpu")]
    private static extern void FD_RuntimeOptionWrapperUseGpu( IntPtr fd_runtime_option_wrapper, int gpu_id);


    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseRKNPU2")]
    private static extern void FD_RuntimeOptionWrapperUseRKNPU2(
                      IntPtr fd_runtime_option_wrapper,
                    rknpu2_CpuName rknpu2_name,
                    rknpu2_CoreMask rknpu2_core);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseTimVX")]
    private static extern void FD_RuntimeOptionWrapperUseTimVX( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseAscend")]
    private static extern void FD_RuntimeOptionWrapperUseAscend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseKunlunXin")]
    private static extern void FD_RuntimeOptionWrapperUseKunlunXin(
                        IntPtr fd_runtime_option_wrapper,
                        int kunlunxin_id, int l3_workspace_size,
                        bool locked, bool autotune,
                        string autotune_file,
                        string precision,
                        bool adaptive_seqlen,
                        bool enable_multi_stream);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseSophgo")]
    private static extern void FD_RuntimeOptionWrapperUseSophgo( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetExternalStream")]
    private static extern void FD_RuntimeOptionWrapperSetExternalStream( IntPtr fd_runtime_option_wrapper, void* external_stream);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetCpuThreadNum")]
    private static extern void FD_RuntimeOptionWrapperSetCpuThreadNum( IntPtr fd_runtime_option_wrapper, int thread_num);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetOrtGraphOptLevel")]
    private static extern void FD_RuntimeOptionWrapperSetOrtGraphOptLevel( IntPtr fd_runtime_option_wrapper, int level);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUsePaddleBackend")]
    private static extern void FD_RuntimeOptionWrapperUsePaddleBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUsePaddleInferBackend")]
    private static extern void FD_RuntimeOptionWrapperUsePaddleInferBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseOrtBackend")]
    private static extern void FD_RuntimeOptionWrapperUseOrtBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseSophgoBackend")]
    private static extern void FD_RuntimeOptionWrapperUseSophgoBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseTrtBackend")]
    private static extern void FD_RuntimeOptionWrapperUseTrtBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUsePorosBackend")]
    private static extern void FD_RuntimeOptionWrapperUsePorosBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseOpenVINOBackend")]
    private static extern void FD_RuntimeOptionWrapperUseOpenVINOBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseLiteBackend")]
    private static extern void FD_RuntimeOptionWrapperUseLiteBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUsePaddleLiteBackend")]
    private static extern void FD_RuntimeOptionWrapperUsePaddleLiteBackend( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetPaddleMKLDNN")]
    private static extern void FD_RuntimeOptionWrapperSetPaddleMKLDNN( IntPtr fd_runtime_option_wrapper, bool pd_mkldnn);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnablePaddleToTrt")]
    private static extern void FD_RuntimeOptionWrapperEnablePaddleToTrt( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDeletePaddleBackendPass")]
    private static extern void FD_RuntimeOptionWrapperDeletePaddleBackendPass( IntPtr fd_runtime_option_wrapper, string delete_pass_name);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnablePaddleLogInfo")]
    private static extern void FD_RuntimeOptionWrapperEnablePaddleLogInfo( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDisablePaddleLogInfo")]
    private static extern void FD_RuntimeOptionWrapperDisablePaddleLogInfo( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize")]
    private static extern void FD_RuntimeOptionWrapperSetPaddleMKLDNNCacheSize( IntPtr fd_runtime_option_wrapper, int size);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetOpenVINODevice")]
    private static extern void FD_RuntimeOptionWrapperSetOpenVINODevice( IntPtr fd_runtime_option_wrapper, string name);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLiteOptimizedModelDir")]
    private static extern void FD_RuntimeOptionWrapperSetLiteOptimizedModelDir( IntPtr fd_runtime_option_wrapper, string optimized_model_dir);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLiteSubgraphPartitionPath")]
    private static extern void FD_RuntimeOptionWrapperSetLiteSubgraphPartitionPath(
        IntPtr fd_runtime_option_wrapper, string nnadapter_subgraph_partition_config_path);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer")]
    private static extern void FD_RuntimeOptionWrapperSetLiteSubgraphPartitionConfigBuffer(
        IntPtr fd_runtime_option_wrapper, string nnadapter_subgraph_partition_config_buffer);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLiteContextProperties")]
    private static extern void FD_RuntimeOptionWrapperSetLiteContextProperties( IntPtr fd_runtime_option_wrapper, string nnadapter_context_properties);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLiteModelCacheDir")]
    private static extern void FD_RuntimeOptionWrapperSetLiteModelCacheDir( IntPtr fd_runtime_option_wrapper, string nnadapter_model_cache_dir);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath")]
    private static extern void FD_RuntimeOptionWrapperSetLiteMixedPrecisionQuantizationConfigPath(
        IntPtr fd_runtime_option_wrapper, string nnadapter_mixed_precision_quantization_config_path);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnableLiteFP16")]
    private static extern void FD_RuntimeOptionWrapperEnableLiteFP16( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDisableLiteFP16")]
    private static extern void FD_RuntimeOptionWrapperDisableLiteFP16( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnableLiteInt8")]
    private static extern void FD_RuntimeOptionWrapperEnableLiteInt8( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDisableLiteInt8")]
    private static extern void FD_RuntimeOptionWrapperDisableLiteInt8( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetLitePowerMode")]
    private static extern void FD_RuntimeOptionWrapperSetLitePowerMode( IntPtr fd_runtime_option_wrapper, LitePowerMode mode);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnableTrtFP16")]
    private static extern void FD_RuntimeOptionWrapperEnableTrtFP16( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDisableTrtFP16")]
    private static extern void FD_RuntimeOptionWrapperDisableTrtFP16( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetTrtCacheFile")]
    private static extern void FD_RuntimeOptionWrapperSetTrtCacheFile( IntPtr fd_runtime_option_wrapper, string cache_file_path);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnablePinnedMemory")]
    private static extern void FD_RuntimeOptionWrapperEnablePinnedMemory( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDisablePinnedMemory")]
    private static extern void FD_RuntimeOptionWrapperDisablePinnedMemory( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperEnablePaddleTrtCollectShape")]
    private static extern void FD_RuntimeOptionWrapperEnablePaddleTrtCollectShape( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperDisablePaddleTrtCollectShape")]
    private static extern void FD_RuntimeOptionWrapperDisablePaddleTrtCollectShape( IntPtr fd_runtime_option_wrapper);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetOpenVINOStreams")]
    private static extern void FD_RuntimeOptionWrapperSetOpenVINOStreams( IntPtr fd_runtime_option_wrapper, int num_streams);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperUseIpu")]
    private static extern void FD_RuntimeOptionWrapperUseIpu( IntPtr fd_runtime_option_wrapper, int device_num, int micro_batch_size,
                bool enable_pipelining, int batches_per_step);

    [DllImport("fastdeploy.dll", EntryPoint = "FD_RuntimeOptionWrapperSetIpuConfig")]
    private static extern void FD_RuntimeOptionWrapperSetIpuConfig( IntPtr fd_runtime_option_wrapper, bool enable_fp16, int replica_num,
                      float available_memory_proportion,
                      bool enable_half_partial);


  }
}

