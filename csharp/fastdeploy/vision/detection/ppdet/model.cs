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

// PPYOLOE
public class PPYOLOE {

  public PPYOLOE(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_ppyoloe_wrapper =
        FD_C_CreatePPYOLOEWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PPYOLOE() { FD_C_DestroyPPYOLOEWrapper(fd_ppyoloe_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PPYOLOEWrapperPredict(fd_ppyoloe_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PPYOLOEWrapperBatchPredict(fd_ppyoloe_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PPYOLOEWrapperInitialized(fd_ppyoloe_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_ppyoloe_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePPYOLOEWrapper")]
  private static extern IntPtr FD_C_CreatePPYOLOEWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPPYOLOEWrapper")]
  private static extern void
  FD_C_DestroyPPYOLOEWrapper(IntPtr fd_ppyoloe_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PPYOLOEWrapperPredict")]
  private static extern bool
  FD_C_PPYOLOEWrapperPredict(IntPtr fd_ppyoloe_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPYOLOEWrapperInitialized")]
  private static extern bool
  FD_C_PPYOLOEWrapperInitialized(IntPtr fd_c_ppyoloe_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPYOLOEWrapperBatchPredict")]
  private static extern bool
  FD_C_PPYOLOEWrapperBatchPredict(IntPtr fd_c_ppyoloe_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PicoDet
public class PicoDet {

  public PicoDet(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_picodet_wrapper =
        FD_C_CreatePicoDetWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PicoDet() { FD_C_DestroyPicoDetWrapper(fd_picodet_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PicoDetWrapperPredict(fd_picodet_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PicoDetWrapperBatchPredict(fd_picodet_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PicoDetWrapperInitialized(fd_picodet_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_picodet_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePicoDetWrapper")]
  private static extern IntPtr FD_C_CreatePicoDetWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPicoDetWrapper")]
  private static extern void
  FD_C_DestroyPicoDetWrapper(IntPtr fd_picodet_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PicoDetWrapperPredict")]
  private static extern bool
  FD_C_PicoDetWrapperPredict(IntPtr fd_picodet_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PicoDetWrapperInitialized")]
  private static extern bool
  FD_C_PicoDetWrapperInitialized(IntPtr fd_c_picodet_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PicoDetWrapperBatchPredict")]
  private static extern bool
  FD_C_PicoDetWrapperBatchPredict(IntPtr fd_c_picodet_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}


// PPYOLO

public class PPYOLO {

  public PPYOLO(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_ppyolo_wrapper =
        FD_C_CreatePPYOLOWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PPYOLO() { FD_C_DestroyPPYOLOWrapper(fd_ppyolo_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PPYOLOWrapperPredict(fd_ppyolo_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PPYOLOWrapperBatchPredict(fd_ppyolo_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PPYOLOWrapperInitialized(fd_ppyolo_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_ppyolo_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePPYOLOWrapper")]
  private static extern IntPtr FD_C_CreatePPYOLOWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPPYOLOWrapper")]
  private static extern void
  FD_C_DestroyPPYOLOWrapper(IntPtr fd_ppyolo_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PPYOLOWrapperPredict")]
  private static extern bool
  FD_C_PPYOLOWrapperPredict(IntPtr fd_ppyolo_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPYOLOWrapperInitialized")]
  private static extern bool
  FD_C_PPYOLOWrapperInitialized(IntPtr fd_c_ppyolo_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPYOLOWrapperBatchPredict")]
  private static extern bool
  FD_C_PPYOLOWrapperBatchPredict(IntPtr fd_c_ppyolo_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// YOLOv3

public class YOLOv3 {

  public YOLOv3(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_yolov3_wrapper =
        FD_C_CreateYOLOv3Wrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~YOLOv3() { FD_C_DestroyYOLOv3Wrapper(fd_yolov3_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_YOLOv3WrapperPredict(fd_yolov3_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_YOLOv3WrapperBatchPredict(fd_yolov3_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_YOLOv3WrapperInitialized(fd_yolov3_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_yolov3_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateYOLOv3Wrapper")]
  private static extern IntPtr FD_C_CreateYOLOv3Wrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyYOLOv3Wrapper")]
  private static extern void
  FD_C_DestroyYOLOv3Wrapper(IntPtr fd_yolov3_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_YOLOv3WrapperPredict")]
  private static extern bool
  FD_C_YOLOv3WrapperPredict(IntPtr fd_yolov3_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv3WrapperInitialized")]
  private static extern bool
  FD_C_YOLOv3WrapperInitialized(IntPtr fd_c_yolov3_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_YOLOv3WrapperBatchPredict")]
  private static extern bool
  FD_C_YOLOv3WrapperBatchPredict(IntPtr fd_c_yolov3_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PaddleYOLOX

public class PaddleYOLOX {

  public PaddleYOLOX(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleyolox_wrapper =
        FD_C_CreatePaddleYOLOXWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PaddleYOLOX() { FD_C_DestroyPaddleYOLOXWrapper(fd_paddleyolox_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PaddleYOLOXWrapperPredict(fd_paddleyolox_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PaddleYOLOXWrapperBatchPredict(fd_paddleyolox_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PaddleYOLOXWrapperInitialized(fd_paddleyolox_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_paddleyolox_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePaddleYOLOXWrapper")]
  private static extern IntPtr FD_C_CreatePaddleYOLOXWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPaddleYOLOXWrapper")]
  private static extern void
  FD_C_DestroyPaddleYOLOXWrapper(IntPtr fd_paddleyolox_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PaddleYOLOXWrapperPredict")]
  private static extern bool
  FD_C_PaddleYOLOXWrapperPredict(IntPtr fd_paddleyolox_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOXWrapperInitialized")]
  private static extern bool
  FD_C_PaddleYOLOXWrapperInitialized(IntPtr fd_c_paddleyolox_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOXWrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleYOLOXWrapperBatchPredict(IntPtr fd_c_paddleyolox_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// FasterRCNN

public class FasterRCNN {

  public FasterRCNN(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_fasterrcnn_wrapper =
        FD_C_CreateFasterRCNNWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~FasterRCNN() { FD_C_DestroyFasterRCNNWrapper(fd_fasterrcnn_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_FasterRCNNWrapperPredict(fd_fasterrcnn_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_FasterRCNNWrapperBatchPredict(fd_fasterrcnn_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_FasterRCNNWrapperInitialized(fd_fasterrcnn_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_fasterrcnn_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateFasterRCNNWrapper")]
  private static extern IntPtr FD_C_CreateFasterRCNNWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyFasterRCNNWrapper")]
  private static extern void
  FD_C_DestroyFasterRCNNWrapper(IntPtr fd_fasterrcnn_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_FasterRCNNWrapperPredict")]
  private static extern bool
  FD_C_FasterRCNNWrapperPredict(IntPtr fd_fasterrcnn_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_FasterRCNNWrapperInitialized")]
  private static extern bool
  FD_C_FasterRCNNWrapperInitialized(IntPtr fd_c_fasterrcnn_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_FasterRCNNWrapperBatchPredict")]
  private static extern bool
  FD_C_FasterRCNNWrapperBatchPredict(IntPtr fd_c_fasterrcnn_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// MaskRCNN

public class MaskRCNN {

  public MaskRCNN(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_maskrcnn_wrapper =
        FD_C_CreateMaskRCNNWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~MaskRCNN() { FD_C_DestroyMaskRCNNWrapper(fd_maskrcnn_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_MaskRCNNWrapperPredict(fd_maskrcnn_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_MaskRCNNWrapperBatchPredict(fd_maskrcnn_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_MaskRCNNWrapperInitialized(fd_maskrcnn_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_maskrcnn_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateMaskRCNNWrapper")]
  private static extern IntPtr FD_C_CreateMaskRCNNWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyMaskRCNNWrapper")]
  private static extern void
  FD_C_DestroyMaskRCNNWrapper(IntPtr fd_maskrcnn_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_MaskRCNNWrapperPredict")]
  private static extern bool
  FD_C_MaskRCNNWrapperPredict(IntPtr fd_maskrcnn_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_MaskRCNNWrapperInitialized")]
  private static extern bool
  FD_C_MaskRCNNWrapperInitialized(IntPtr fd_c_maskrcnn_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_MaskRCNNWrapperBatchPredict")]
  private static extern bool
  FD_C_MaskRCNNWrapperBatchPredict(IntPtr fd_c_maskrcnn_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// SSD

public class SSD {

  public SSD(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_ssd_wrapper =
        FD_C_CreateSSDWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~SSD() { FD_C_DestroySSDWrapper(fd_ssd_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_SSDWrapperPredict(fd_ssd_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_SSDWrapperBatchPredict(fd_ssd_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_SSDWrapperInitialized(fd_ssd_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_ssd_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateSSDWrapper")]
  private static extern IntPtr FD_C_CreateSSDWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroySSDWrapper")]
  private static extern void
  FD_C_DestroySSDWrapper(IntPtr fd_ssd_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_SSDWrapperPredict")]
  private static extern bool
  FD_C_SSDWrapperPredict(IntPtr fd_ssd_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_SSDWrapperInitialized")]
  private static extern bool
  FD_C_SSDWrapperInitialized(IntPtr fd_c_ssd_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_SSDWrapperBatchPredict")]
  private static extern bool
  FD_C_SSDWrapperBatchPredict(IntPtr fd_c_ssd_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PaddleYOLOv5

public class PaddleYOLOv5 {

  public PaddleYOLOv5(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleyolov5_wrapper =
        FD_C_CreatePaddleYOLOv5Wrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PaddleYOLOv5() { FD_C_DestroyPaddleYOLOv5Wrapper(fd_paddleyolov5_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PaddleYOLOv5WrapperPredict(fd_paddleyolov5_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PaddleYOLOv5WrapperBatchPredict(fd_paddleyolov5_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PaddleYOLOv5WrapperInitialized(fd_paddleyolov5_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_paddleyolov5_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePaddleYOLOv5Wrapper")]
  private static extern IntPtr FD_C_CreatePaddleYOLOv5Wrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPaddleYOLOv5Wrapper")]
  private static extern void
  FD_C_DestroyPaddleYOLOv5Wrapper(IntPtr fd_paddleyolov5_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PaddleYOLOv5WrapperPredict")]
  private static extern bool
  FD_C_PaddleYOLOv5WrapperPredict(IntPtr fd_paddleyolov5_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv5WrapperInitialized")]
  private static extern bool
  FD_C_PaddleYOLOv5WrapperInitialized(IntPtr fd_c_paddleyolov5_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv5WrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleYOLOv5WrapperBatchPredict(IntPtr fd_c_paddleyolov5_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PaddleYOLOv6

public class PaddleYOLOv6 {

  public PaddleYOLOv6(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleyolov6_wrapper =
        FD_C_CreatePaddleYOLOv6Wrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PaddleYOLOv6() { FD_C_DestroyPaddleYOLOv6Wrapper(fd_paddleyolov6_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PaddleYOLOv6WrapperPredict(fd_paddleyolov6_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PaddleYOLOv6WrapperBatchPredict(fd_paddleyolov6_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PaddleYOLOv6WrapperInitialized(fd_paddleyolov6_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_paddleyolov6_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePaddleYOLOv6Wrapper")]
  private static extern IntPtr FD_C_CreatePaddleYOLOv6Wrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPaddleYOLOv6Wrapper")]
  private static extern void
  FD_C_DestroyPaddleYOLOv6Wrapper(IntPtr fd_paddleyolov6_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PaddleYOLOv6WrapperPredict")]
  private static extern bool
  FD_C_PaddleYOLOv6WrapperPredict(IntPtr fd_paddleyolov6_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv6WrapperInitialized")]
  private static extern bool
  FD_C_PaddleYOLOv6WrapperInitialized(IntPtr fd_c_paddleyolov6_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv6WrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleYOLOv6WrapperBatchPredict(IntPtr fd_c_paddleyolov6_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PaddleYOLOv7

public class PaddleYOLOv7 {

  public PaddleYOLOv7(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleyolov7_wrapper =
        FD_C_CreatePaddleYOLOv7Wrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PaddleYOLOv7() { FD_C_DestroyPaddleYOLOv7Wrapper(fd_paddleyolov7_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PaddleYOLOv7WrapperPredict(fd_paddleyolov7_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PaddleYOLOv7WrapperBatchPredict(fd_paddleyolov7_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PaddleYOLOv7WrapperInitialized(fd_paddleyolov7_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_paddleyolov7_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePaddleYOLOv7Wrapper")]
  private static extern IntPtr FD_C_CreatePaddleYOLOv7Wrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPaddleYOLOv7Wrapper")]
  private static extern void
  FD_C_DestroyPaddleYOLOv7Wrapper(IntPtr fd_paddleyolov7_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PaddleYOLOv7WrapperPredict")]
  private static extern bool
  FD_C_PaddleYOLOv7WrapperPredict(IntPtr fd_paddleyolov7_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv7WrapperInitialized")]
  private static extern bool
  FD_C_PaddleYOLOv7WrapperInitialized(IntPtr fd_c_paddleyolov7_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv7WrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleYOLOv7WrapperBatchPredict(IntPtr fd_c_paddleyolov7_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PaddleYOLOv8

public class PaddleYOLOv8 {

  public PaddleYOLOv8(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleyolov8_wrapper =
        FD_C_CreatePaddleYOLOv8Wrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PaddleYOLOv8() { FD_C_DestroyPaddleYOLOv8Wrapper(fd_paddleyolov8_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PaddleYOLOv8WrapperPredict(fd_paddleyolov8_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PaddleYOLOv8WrapperBatchPredict(fd_paddleyolov8_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PaddleYOLOv8WrapperInitialized(fd_paddleyolov8_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_paddleyolov8_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePaddleYOLOv8Wrapper")]
  private static extern IntPtr FD_C_CreatePaddleYOLOv8Wrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPaddleYOLOv8Wrapper")]
  private static extern void
  FD_C_DestroyPaddleYOLOv8Wrapper(IntPtr fd_paddleyolov8_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PaddleYOLOv8WrapperPredict")]
  private static extern bool
  FD_C_PaddleYOLOv8WrapperPredict(IntPtr fd_paddleyolov8_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv8WrapperInitialized")]
  private static extern bool
  FD_C_PaddleYOLOv8WrapperInitialized(IntPtr fd_c_paddleyolov8_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleYOLOv8WrapperBatchPredict")]
  private static extern bool
  FD_C_PaddleYOLOv8WrapperBatchPredict(IntPtr fd_c_paddleyolov8_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// RTMDet

public class RTMDet {

  public RTMDet(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_rtmdet_wrapper =
        FD_C_CreateRTMDetWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~RTMDet() { FD_C_DestroyRTMDetWrapper(fd_rtmdet_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_RTMDetWrapperPredict(fd_rtmdet_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_RTMDetWrapperBatchPredict(fd_rtmdet_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_RTMDetWrapperInitialized(fd_rtmdet_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_rtmdet_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateRTMDetWrapper")]
  private static extern IntPtr FD_C_CreateRTMDetWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyRTMDetWrapper")]
  private static extern void
  FD_C_DestroyRTMDetWrapper(IntPtr fd_rtmdet_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_RTMDetWrapperPredict")]
  private static extern bool
  FD_C_RTMDetWrapperPredict(IntPtr fd_rtmdet_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RTMDetWrapperInitialized")]
  private static extern bool
  FD_C_RTMDetWrapperInitialized(IntPtr fd_c_rtmdet_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RTMDetWrapperBatchPredict")]
  private static extern bool
  FD_C_RTMDetWrapperBatchPredict(IntPtr fd_c_rtmdet_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// CascadeRCNN

public class CascadeRCNN {

  public CascadeRCNN(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_cascadercnn_wrapper =
        FD_C_CreateCascadeRCNNWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~CascadeRCNN() { FD_C_DestroyCascadeRCNNWrapper(fd_cascadercnn_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_CascadeRCNNWrapperPredict(fd_cascadercnn_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_CascadeRCNNWrapperBatchPredict(fd_cascadercnn_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_CascadeRCNNWrapperInitialized(fd_cascadercnn_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_cascadercnn_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateCascadeRCNNWrapper")]
  private static extern IntPtr FD_C_CreateCascadeRCNNWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyCascadeRCNNWrapper")]
  private static extern void
  FD_C_DestroyCascadeRCNNWrapper(IntPtr fd_cascadercnn_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CascadeRCNNWrapperPredict")]
  private static extern bool
  FD_C_CascadeRCNNWrapperPredict(IntPtr fd_cascadercnn_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CascadeRCNNWrapperInitialized")]
  private static extern bool
  FD_C_CascadeRCNNWrapperInitialized(IntPtr fd_c_cascadercnn_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CascadeRCNNWrapperBatchPredict")]
  private static extern bool
  FD_C_CascadeRCNNWrapperBatchPredict(IntPtr fd_c_cascadercnn_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// PSSDet

public class PSSDet {

  public PSSDet(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_pssdet_wrapper =
        FD_C_CreatePSSDetWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PSSDet() { FD_C_DestroyPSSDetWrapper(fd_pssdet_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_PSSDetWrapperPredict(fd_pssdet_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_PSSDetWrapperBatchPredict(fd_pssdet_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_PSSDetWrapperInitialized(fd_pssdet_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_pssdet_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatePSSDetWrapper")]
  private static extern IntPtr FD_C_CreatePSSDetWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPSSDetWrapper")]
  private static extern void
  FD_C_DestroyPSSDetWrapper(IntPtr fd_pssdet_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PSSDetWrapperPredict")]
  private static extern bool
  FD_C_PSSDetWrapperPredict(IntPtr fd_pssdet_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PSSDetWrapperInitialized")]
  private static extern bool
  FD_C_PSSDetWrapperInitialized(IntPtr fd_c_pssdet_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PSSDetWrapperBatchPredict")]
  private static extern bool
  FD_C_PSSDetWrapperBatchPredict(IntPtr fd_c_pssdet_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// RetinaNet

public class RetinaNet {

  public RetinaNet(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_retinanet_wrapper =
        FD_C_CreateRetinaNetWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~RetinaNet() { FD_C_DestroyRetinaNetWrapper(fd_retinanet_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_RetinaNetWrapperPredict(fd_retinanet_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_RetinaNetWrapperBatchPredict(fd_retinanet_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_RetinaNetWrapperInitialized(fd_retinanet_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_retinanet_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateRetinaNetWrapper")]
  private static extern IntPtr FD_C_CreateRetinaNetWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyRetinaNetWrapper")]
  private static extern void
  FD_C_DestroyRetinaNetWrapper(IntPtr fd_retinanet_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_RetinaNetWrapperPredict")]
  private static extern bool
  FD_C_RetinaNetWrapperPredict(IntPtr fd_retinanet_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RetinaNetWrapperInitialized")]
  private static extern bool
  FD_C_RetinaNetWrapperInitialized(IntPtr fd_c_retinanet_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RetinaNetWrapperBatchPredict")]
  private static extern bool
  FD_C_RetinaNetWrapperBatchPredict(IntPtr fd_c_retinanet_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// FCOS

public class FCOS {

  public FCOS(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_fcos_wrapper =
        FD_C_CreateFCOSWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~FCOS() { FD_C_DestroyFCOSWrapper(fd_fcos_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_FCOSWrapperPredict(fd_fcos_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_FCOSWrapperBatchPredict(fd_fcos_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_FCOSWrapperInitialized(fd_fcos_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_fcos_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateFCOSWrapper")]
  private static extern IntPtr FD_C_CreateFCOSWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyFCOSWrapper")]
  private static extern void
  FD_C_DestroyFCOSWrapper(IntPtr fd_fcos_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_FCOSWrapperPredict")]
  private static extern bool
  FD_C_FCOSWrapperPredict(IntPtr fd_fcos_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_FCOSWrapperInitialized")]
  private static extern bool
  FD_C_FCOSWrapperInitialized(IntPtr fd_c_fcos_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_FCOSWrapperBatchPredict")]
  private static extern bool
  FD_C_FCOSWrapperBatchPredict(IntPtr fd_c_fcos_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// TTFNet

public class TTFNet {

  public TTFNet(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_ttfnet_wrapper =
        FD_C_CreateTTFNetWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~TTFNet() { FD_C_DestroyTTFNetWrapper(fd_ttfnet_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_TTFNetWrapperPredict(fd_ttfnet_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_TTFNetWrapperBatchPredict(fd_ttfnet_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_TTFNetWrapperInitialized(fd_ttfnet_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_ttfnet_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateTTFNetWrapper")]
  private static extern IntPtr FD_C_CreateTTFNetWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyTTFNetWrapper")]
  private static extern void
  FD_C_DestroyTTFNetWrapper(IntPtr fd_ttfnet_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_TTFNetWrapperPredict")]
  private static extern bool
  FD_C_TTFNetWrapperPredict(IntPtr fd_ttfnet_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_TTFNetWrapperInitialized")]
  private static extern bool
  FD_C_TTFNetWrapperInitialized(IntPtr fd_c_ttfnet_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_TTFNetWrapperBatchPredict")]
  private static extern bool
  FD_C_TTFNetWrapperBatchPredict(IntPtr fd_c_ttfnet_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// TOOD

public class TOOD {

  public TOOD(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_tood_wrapper =
        FD_C_CreateTOODWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~TOOD() { FD_C_DestroyTOODWrapper(fd_tood_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_TOODWrapperPredict(fd_tood_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_TOODWrapperBatchPredict(fd_tood_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_TOODWrapperInitialized(fd_tood_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_tood_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateTOODWrapper")]
  private static extern IntPtr FD_C_CreateTOODWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyTOODWrapper")]
  private static extern void
  FD_C_DestroyTOODWrapper(IntPtr fd_tood_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_TOODWrapperPredict")]
  private static extern bool
  FD_C_TOODWrapperPredict(IntPtr fd_tood_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_TOODWrapperInitialized")]
  private static extern bool
  FD_C_TOODWrapperInitialized(IntPtr fd_c_tood_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_TOODWrapperBatchPredict")]
  private static extern bool
  FD_C_TOODWrapperBatchPredict(IntPtr fd_c_tood_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

// GFL

public class GFL {

  public GFL(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_gfl_wrapper =
        FD_C_CreateGFLWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~GFL() { FD_C_DestroyGFLWrapper(fd_gfl_wrapper); }

  public DetectionResult Predict(Mat img) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();
    if(! FD_C_GFLWrapperPredict(fd_gfl_wrapper, img.CvPtr,
                               ref fd_detection_result))
    {
      return null;
    } // predict
    
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    return detection_result;
  }

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
    if(!FD_C_GFLWrapperBatchPredict(fd_gfl_wrapper, ref imgs_in, ref fd_detection_result_array)){
      return null;
    }
    List<DetectionResult> results_out = new List<DetectionResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_DetectionResult fd_detection_result = (FD_DetectionResult)Marshal.PtrToStructure(
          fd_detection_result_array.data + i * Marshal.SizeOf(new FD_DetectionResult()),
          typeof(FD_DetectionResult));
      results_out.Add(ConvertResult.ConvertCResultToDetectionResult(fd_detection_result));
    }
    return results_out;
  }

  public bool Initialized() {
    return FD_C_GFLWrapperInitialized(fd_gfl_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_gfl_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateGFLWrapper")]
  private static extern IntPtr FD_C_CreateGFLWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyGFLWrapper")]
  private static extern void
  FD_C_DestroyGFLWrapper(IntPtr fd_gfl_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_GFLWrapperPredict")]
  private static extern bool
  FD_C_GFLWrapperPredict(IntPtr fd_gfl_wrapper, IntPtr img,
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
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_GFLWrapperInitialized")]
  private static extern bool
  FD_C_GFLWrapperInitialized(IntPtr fd_c_gfl_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_GFLWrapperBatchPredict")]
  private static extern bool
  FD_C_GFLWrapperBatchPredict(IntPtr fd_c_gfl_wrapper,
                                  ref FD_OneDimMat imgs,
                                  ref FD_OneDimDetectionResult results);
}

}
}
}