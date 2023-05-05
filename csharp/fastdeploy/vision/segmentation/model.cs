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
using System.Collections.Generic;
using OpenCvSharp;
using fastdeploy.types_internal_c;

namespace fastdeploy {
namespace vision {
namespace segmentation {

/*! @brief PaddleSeg serials model object used when to load a PaddleSeg model exported by PaddleSeg repository
 */
public class PaddleSegModel {

  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g unet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g unet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g unet/deploy.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  public PaddleSegModel(string model_file, string params_file,
                         string config_file, RuntimeOption custom_option = null,
                         ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleseg_model_wrapper = FD_C_CreatePaddleSegModelWrapper(
        model_file, params_file, config_file, custom_option.GetWrapperPtr(),
        model_format);
  }

  ~PaddleSegModel() {
    FD_C_DestroyPaddleSegModelWrapper(fd_paddleseg_model_wrapper);
  }

  /// Get model's name
  public string ModelName() {
    return "PaddleSeg";
  }

  /** \brief DEPRECATED Predict the segmentation result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * 
   * \return SegmentationResult
   */
  public SegmentationResult Predict(Mat img) {
    FD_SegmentationResult fd_segmentation_result = new FD_SegmentationResult();
    if(! FD_C_PaddleSegModelWrapperPredict(
        fd_paddleseg_model_wrapper, img.CvPtr,
        ref fd_segmentation_result))
    {
      return null;
    } // predict
    SegmentationResult segmentation_result =
        ConvertResult.ConvertCResultToSegmentationResult(fd_segmentation_result);
    FD_C_DestroySegmentationResult(ref fd_segmentation_result);
    return segmentation_result;
  }

  /** \brief Predict the segmentation results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * 
   * \return List<SegmentationResult>
   */
  public List<SegmentationResult> BatchPredict(List<Mat> imgs){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimSegmentationResult fd_segmentation_result_array =  new FD_OneDimSegmentationResult();
    if (!FD_C_PaddleSegModelWrapperBatchPredict(fd_paddleseg_model_wrapper, imgs_in, ref fd_segmentation_result_array)){
      return null;
    }
    List<SegmentationResult> results_out = new List<SegmentationResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_SegmentationResult fd_segmentation_result = (FD_SegmentationResult)Marshal.PtrToStructure(
          fd_segmentation_result_array.data + i * Marshal.SizeOf(new FD_SegmentationResult()),
          typeof(FD_SegmentationResult));
      results_out.Add(ConvertResult.ConvertCResultToSegmentationResult(fd_segmentation_result));
      FD_C_DestroySegmentationResult(ref fd_segmentation_result);
    }
    return results_out;
  }

  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_PaddleSegModelWrapperInitialized(fd_paddleseg_model_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_paddleseg_model_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreatePaddleSegModelWrapper")]
  private static extern IntPtr FD_C_CreatePaddleSegModelWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyPaddleSegModelWrapper")]
  private static extern void
  FD_C_DestroyPaddleSegModelWrapper(IntPtr fd_paddleseg_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleSegModelWrapperPredict")]
  private static extern bool
  FD_C_PaddleSegModelWrapperPredict(IntPtr fd_paddleseg_model_wrapper,
                                     IntPtr img,
                                     ref FD_SegmentationResult fd_segmentation_result);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateSegmentationResultWrapper")]
  private static extern IntPtr FD_C_CreateSegmentationResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroySegmentationResultWrapper")]
  private static extern void
  FD_C_DestroySegmentationResultWrapper(IntPtr fd_segmentation_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroySegmentationResult")]
  private static extern void
  FD_C_DestroySegmentationResult(ref FD_SegmentationResult fd_segmentation_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_SegmentationResultWrapperToCResult")]
  private static extern void
  FD_C_SegmentationResultWrapperToCResult(IntPtr fd_segmentation_result_wrapper, ref FD_SegmentationResult fd_segmentation_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateSegmentationResultWrapperFromCResult")]
  private static extern IntPtr
  FD_C_CreateSegmentationResultWrapperFromCResult(ref FD_SegmentationResult fd_segmentation_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleSegModelWrapperInitialized")]
  private static extern bool
  FD_C_PaddleSegModelWrapperInitialized(IntPtr fd_paddleseg_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleSegModelWrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleSegModelWrapperBatchPredict(IntPtr fd_paddleseg_model_wrapper,
                                         FD_OneDimMat imgs,
                                         ref FD_OneDimSegmentationResult results);

}

}
}
}