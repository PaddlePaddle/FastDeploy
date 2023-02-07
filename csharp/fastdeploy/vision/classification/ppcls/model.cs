using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using OpenCvSharp;
using fastdeploy.types_internal_c;

namespace fastdeploy{
  namespace vision{
    namespace classification {

    class PaddleClasModel{

      public PaddleClasModel(string model_file, string params_file,
                             string config_file, RuntimeOption custom_option,
                             ModelFormat model_format){
        fd_paddleclas_model_wrapper =  FD_CreatePaddleClasModelWrapper(model_file, params_file, config_file, custom_option.GetWrapperPtr(), model_format);   
      }

      ~PaddleClasModel(){
        FD_DestroyPaddleClasModelWrapper(fd_paddleclas_model_wrapper);
      }

      public ClassifyResult Predict(Mat img){
        IntPtr fd_classify_result_wrapper_ptr = FD_CreateClassifyResultWrapper();
        FD_PPYOLOEWrapperPredict(fd_paddleclas_model_wrapper, img.CvPtr, fd_classify_result_wrapper_ptr); // predict
        IntPtr fd_classify_result_ptr = FD_ClassifyResultWrapperGetData(fd_classify_result_wrapper_ptr); // get result from wrapper
        FD_ClassifyResult fd_classify_result = new FD_ClassifyResult();
        Marshal.PtrToStructure(fd_classify_result_ptr, fd_classify_result);
        ClassifyResult classify_result = ConvertResult.ConvertCResultToClassifyResult(fd_classify_result);
        FD_DestroyClassifyResultWrapper(fd_classify_result_wrapper_ptr); // free fd_classify_result_wrapper_ptr
        FD_DestroyClassifyResult(fd_classify_result_ptr); // free fd_classify_result_ptr
        return classify_result;
      }

      // below are underlying C api
      private IntPtr fd_paddleclas_model_wrapper;
      [DllImport("fastdeploy.dll", EntryPoint = "FD_CreatePaddleClasModelWrapper")]
      private static IntPtr FD_CreatePaddleClasModelWrapper(string model_file, string params_file,
                                                     string config_file, IntPtr fd_runtime_option_wrapper,
                                                     ModelFormat model_format);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyPaddleClasModelWrapper")]
      private static void FD_DestroyPaddleClasModelWrapper(IntPtr fd_paddleclas_model_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_PaddleClasModelWrapperPredict")]
      private static bool FD_PaddleClasModelWrapperPredict(IntPtr fd_paddleclas_model_wrapper,
                                    IntPtr img, IntPtr fd_classify_result_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_CreateClassifyResultWrapper")]
      IntPtr FD_CreateClassifyResultWrapper();
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyClassifyResultWrapper")]
      void FD_DestroyClassifyResultWrapper(IntPtr fd_classify_result_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyClassifyResult")]
      void FD_DestroyClassifyResult(IntPtr fd_classify_result);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_ClassifyResultWrapperGetData")]
      IntPtr FD_ClassifyResultWrapperGetData(IntPtr fd_classify_result_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_CreateClassifyResultWrapperFromData")]
      IntPtr FD_CreateClassifyResultWrapperFromData(IntPtr fd_classify_result);
    }

    }
  }
}