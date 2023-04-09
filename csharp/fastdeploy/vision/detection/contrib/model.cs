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
namespace detection {

// YOLOv5

/*! @brief YOLOv5 model
 */
public class YOLOv5 {

  public YOLOv5( string model_file, string params_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolov5_wrapper =
        FD_C_CreateYOLOv5Wrapper(model_file, params_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOv5() { FD_C_DestroyYOLOv5Wrapper(fd_yolov5_wrapper); }


  /** \brief Predict the detection result for an input image
    * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
    * 
    * \return DetectionResult
    */
  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLOv5WrapperPredict(fd_yolov5_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResult(ref fd_detection_result);
    return detection_result;
  }


  /** \brief Predict the detection result for an input image list
   * \param[in] im The input image list, all the elements come from cv::imread(), is a 3-D array with layout HWC, BGR format
   * 
   * \return List<DetectionResult>
   */
  public List<DetectionResult> BatchPredict(List<Mat> imgs){
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
    FD_OneDimDetectionResult fd_detection_result_array =  new FD_OneDimDetectionResult();
    if(!FD_C_YOLOv5WrapperBatchPredict(fd_yolov5_wrapper, imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
      FD_C_DestroyDetectionResult(ref fd_detection_result);
    }
    return results_out;
  }


  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_YOLOv5WrapperInitialized(fd_yolov5_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolov5_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLOv5Wrapper")]
  private static extern IntPtr FD_C_CreateYOLOv5Wrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLOv5Wrapper")]
  private static extern void
  FD_C_DestroyYOLOv5Wrapper(IntPtr fd_yolov5_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLOv5WrapperPredict")]
  private static extern bool
  FD_C_YOLOv5WrapperPredict(IntPtr fd_yolov5_wrapper, IntPtr img,
                             ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(ref FD_DetectionResult fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv5WrapperInitialized")]
  private static extern bool
  FD_C_YOLOv5WrapperInitialized(IntPtr fd_c_yolov5_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv5WrapperBatchPredict")]
  private static extern bool
  FD_C_YOLOv5WrapperBatchPredict(IntPtr fd_c_yolov5_wrapper,
                                 FD_OneDimMat imgs,
                                 ref FD_OneDimDetectionResult results);
}


// YOLOv7


/*! @brief YOLOv7 model
 */
public class YOLOv7 {

  public YOLOv7( string model_file, string params_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolov7_wrapper =
        FD_C_CreateYOLOv7Wrapper(model_file, params_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOv7() { FD_C_DestroyYOLOv7Wrapper(fd_yolov7_wrapper); }


  /** \brief Predict the detection result for an input image
    * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
    * 
    * \return DetectionResult
    */
  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLOv7WrapperPredict(fd_yolov7_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResult(ref fd_detection_result);
    return detection_result;
  }


  /** \brief Predict the detection result for an input image list
   * \param[in] im The input image list, all the elements come from cv::imread(), is a 3-D array with layout HWC, BGR format
   * 
   * \return List<DetectionResult>
   */
  public List<DetectionResult> BatchPredict(List<Mat> imgs){
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
    FD_OneDimDetectionResult fd_detection_result_array =  new FD_OneDimDetectionResult();
    if(!FD_C_YOLOv7WrapperBatchPredict(fd_yolov7_wrapper, imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
      FD_C_DestroyDetectionResult(ref fd_detection_result);
    }
    return results_out;
  }


  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_YOLOv7WrapperInitialized(fd_yolov7_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolov7_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLOv7Wrapper")]
  private static extern IntPtr FD_C_CreateYOLOv7Wrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLOv7Wrapper")]
  private static extern void
  FD_C_DestroyYOLOv7Wrapper(IntPtr fd_yolov7_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLOv7WrapperPredict")]
  private static extern bool
  FD_C_YOLOv7WrapperPredict(IntPtr fd_yolov7_wrapper, IntPtr img,
                             ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv7WrapperInitialized")]
  private static extern bool
  FD_C_YOLOv7WrapperInitialized(IntPtr fd_c_yolov7_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv7WrapperBatchPredict")]
  private static extern bool
  FD_C_YOLOv7WrapperBatchPredict(IntPtr fd_c_yolov7_wrapper,
                                 FD_OneDimMat imgs,
                                 ref FD_OneDimDetectionResult results);
}


// YOLOv8


/*! @brief YOLOv8 model
 */
public class YOLOv8 {

  public YOLOv8( string model_file, string params_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolov8_wrapper =
        FD_C_CreateYOLOv8Wrapper(model_file, params_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOv8() { FD_C_DestroyYOLOv8Wrapper(fd_yolov8_wrapper); }


  /** \brief Predict the detection result for an input image
    * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
    * 
    * \return DetectionResult
    */
  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLOv8WrapperPredict(fd_yolov8_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResult(ref fd_detection_result);
    return detection_result;
  }


  /** \brief Predict the detection result for an input image list
   * \param[in] im The input image list, all the elements come from cv::imread(), is a 3-D array with layout HWC, BGR format
   * 
   * \return List<DetectionResult>
   */
  public List<DetectionResult> BatchPredict(List<Mat> imgs){
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
    FD_OneDimDetectionResult fd_detection_result_array =  new FD_OneDimDetectionResult();
    if(!FD_C_YOLOv8WrapperBatchPredict(fd_yolov8_wrapper, imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
      FD_C_DestroyDetectionResult(ref fd_detection_result);
    }
    return results_out;
  }


  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_YOLOv8WrapperInitialized(fd_yolov8_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolov8_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLOv8Wrapper")]
  private static extern IntPtr FD_C_CreateYOLOv8Wrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLOv8Wrapper")]
  private static extern void
  FD_C_DestroyYOLOv8Wrapper(IntPtr fd_yolov8_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLOv8WrapperPredict")]
  private static extern bool
  FD_C_YOLOv8WrapperPredict(IntPtr fd_yolov8_wrapper, IntPtr img,
                             ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(ref FD_DetectionResult fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv8WrapperInitialized")]
  private static extern bool
  FD_C_YOLOv8WrapperInitialized(IntPtr fd_c_yolov8_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv8WrapperBatchPredict")]
  private static extern bool
  FD_C_YOLOv8WrapperBatchPredict(IntPtr fd_c_yolov8_wrapper,
                                 FD_OneDimMat imgs,
                                 ref FD_OneDimDetectionResult results);
}



// YOLOv6


/*! @brief YOLOv6 model
 */
public class YOLOv6 {

  public YOLOv6( string model_file, string params_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolov6_wrapper =
        FD_C_CreateYOLOv6Wrapper(model_file, params_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOv6() { FD_C_DestroyYOLOv6Wrapper(fd_yolov6_wrapper); }

  public DetectionResult Predict(Mat img, float conf_threshold,
                                 float nms_threshold) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLOv6WrapperPredict(fd_yolov6_wrapper, img.CvPtr,
                               ref fd_detection_result, conf_threshold,
                               nms_threshold))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResult(ref fd_detection_result);
    return detection_result;
  }


  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_YOLOv6WrapperInitialized(fd_yolov6_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolov6_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLOv6Wrapper")]
  private static extern IntPtr FD_C_CreateYOLOv6Wrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLOv6Wrapper")]
  private static extern void
  FD_C_DestroyYOLOv6Wrapper(IntPtr fd_yolov6_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLOv6WrapperPredict")]
  private static extern bool
  FD_C_YOLOv6WrapperPredict(IntPtr fd_yolov6_wrapper, IntPtr img,
                             ref FD_DetectionResult fd_detection_result,
                             float conf_threshold,
                             float nms_threshold);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv6WrapperInitialized")]
  private static extern bool
  FD_C_YOLOv6WrapperInitialized(IntPtr fd_c_yolov6_wrapper);

}

// YOLOR


/*! @brief YOLOR model
 */
public class YOLOR {

  public YOLOR( string model_file, string params_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolor_wrapper =
        FD_C_CreateYOLORWrapper(model_file, params_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOR() { FD_C_DestroyYOLORWrapper(fd_yolor_wrapper); }

  public DetectionResult Predict(Mat img, float conf_threshold,
                                 float nms_threshold) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLORWrapperPredict(fd_yolor_wrapper, img.CvPtr,
                               ref fd_detection_result, conf_threshold,
                               nms_threshold))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResult(ref fd_detection_result);
    return detection_result;
  }


  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_YOLORWrapperInitialized(fd_yolor_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolor_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLORWrapper")]
  private static extern IntPtr FD_C_CreateYOLORWrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLORWrapper")]
  private static extern void
  FD_C_DestroyYOLORWrapper(IntPtr fd_yolor_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLORWrapperPredict")]
  private static extern bool
  FD_C_YOLORWrapperPredict(IntPtr fd_yolor_wrapper, IntPtr img,
                             ref FD_DetectionResult fd_detection_result,
                             float conf_threshold,
                             float nms_threshold);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLORWrapperInitialized")]
  private static extern bool
  FD_C_YOLORWrapperInitialized(IntPtr fd_c_yolor_wrapper);

}


// YOLOX


/*! @brief YOLOX model
 */
public class YOLOX {

  public YOLOX( string model_file, string params_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolox_wrapper =
        FD_C_CreateYOLOXWrapper(model_file, params_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOX() { FD_C_DestroyYOLOXWrapper(fd_yolox_wrapper); }

  public DetectionResult Predict(Mat img, float conf_threshold,
                                 float nms_threshold) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLOXWrapperPredict(fd_yolox_wrapper, img.CvPtr,
                               ref fd_detection_result, conf_threshold,
                               nms_threshold))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResult(ref fd_detection_result);
    return detection_result;
  }


  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_YOLOXWrapperInitialized(fd_yolox_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolox_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLOXWrapper")]
  private static extern IntPtr FD_C_CreateYOLOXWrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLOXWrapper")]
  private static extern void
  FD_C_DestroyYOLOXWrapper(IntPtr fd_yolox_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLOXWrapperPredict")]
  private static extern bool
  FD_C_YOLOXWrapperPredict(IntPtr fd_yolox_wrapper, IntPtr img,
                             ref FD_DetectionResult fd_detection_result,
                             float conf_threshold,
                             float nms_threshold);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(ref FD_DetectionResult fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOXWrapperInitialized")]
  private static extern bool
  FD_C_YOLOXWrapperInitialized(IntPtr fd_c_yolox_wrapper);

}



}
}
}