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
namespace classification {

class PaddleClasModel {

  public PaddleClasModel(string model_file, string params_file,
                         string config_file, RuntimeOption custom_option = null,
                         ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_paddleclas_model_wrapper = FD_C_CreatePaddleClasModelWrapper(
        model_file, params_file, config_file, custom_option.GetWrapperPtr(),
        model_format);
  }

  ~PaddleClasModel() {
    FD_C_DestroyPaddleClasModelWrapper(fd_paddleclas_model_wrapper);
  }

  public string ModelName() {
    return "PaddleClas/Model";
  }

  public ClassifyResult Predict(Mat img) {
    IntPtr fd_classify_result_wrapper_ptr = FD_C_CreateClassifyResultWrapper();
    FD_C_PaddleClasModelWrapperPredict(
        fd_paddleclas_model_wrapper, img.CvPtr,
        fd_classify_result_wrapper_ptr);  // predict
    IntPtr fd_classify_result_ptr = FD_C_ClassifyResultWrapperGetData(
        fd_classify_result_wrapper_ptr);  // get result from wrapper
    FD_ClassifyResult fd_classify_result =
        (FD_ClassifyResult)Marshal.PtrToStructure(fd_classify_result_ptr,
                                                  typeof(FD_ClassifyResult));
    ClassifyResult classify_result =
        ConvertResult.ConvertCResultToClassifyResult(fd_classify_result);
    FD_C_DestroyClassifyResultWrapper(
        fd_classify_result_wrapper_ptr);  // free fd_classify_result_wrapper_ptr
    FD_C_DestroyClassifyResult(
        fd_classify_result_ptr);  // free fd_classify_result_ptr
    return classify_result;
  }

  public List<ClassifyResult> BatchPredict(List<Mat> imgs){

  }

  public bool Initialized() {

  }

  // below are underlying C api
  private IntPtr fd_paddleclas_model_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreatePaddleClasModelWrapper")]
  private static extern IntPtr FD_C_CreatePaddleClasModelWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyPaddleClasModelWrapper")]
  private static extern void
  FD_C_DestroyPaddleClasModelWrapper(IntPtr fd_paddleclas_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PaddleClasModelWrapperPredict")]
  private static extern bool
  FD_C_PaddleClasModelWrapperPredict(IntPtr fd_paddleclas_model_wrapper,
                                     IntPtr img,
                                     IntPtr fd_classify_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreateClassifyResultWrapper")]
  private static extern IntPtr FD_C_CreateClassifyResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyClassifyResultWrapper")]
  private static extern void
  FD_C_DestroyClassifyResultWrapper(IntPtr fd_classify_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyClassifyResult")]
  private static extern void
  FD_C_DestroyClassifyResult(IntPtr fd_classify_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_ClassifyResultWrapperGetData")]
  private static extern IntPtr
  FD_C_ClassifyResultWrapperGetData(IntPtr fd_classify_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateClassifyResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateClassifyResultWrapperFromData(IntPtr fd_classify_result);
}

}
}
}