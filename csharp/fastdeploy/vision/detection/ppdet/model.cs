using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using OpenCvSharp;
using fastdeploy.types_internal_c;

namespace fastdeploy{
  namespace vision{
    namespace detection {

    class PPYOLOE{

      public PPYOLOE(string model_file, string params_file,
                             string config_file, RuntimeOption custom_option,
                             ModelFormat model_format){
        fd_ppyoloe_wrapper =  FD_CreatesPPYOLOEWrapper(model_file, params_file, config_file, custom_option.GetWrapperPtr(), model_format);   
      }

      ~PPYOLOE(){
        FD_DestroyPPYOLOEWrapper(fd_ppyoloe_wrapper);
      }

      public DetectionResult Predict(Mat img){
        IntPtr fd_detection_result_wrapper_ptr = FD_CreateDetectionResultWrapper();
        FD_PPYOLOEWrapperPredict(fd_ppyoloe_wrapper, img.CvPtr, fd_detection_result_wrapper_ptr); // predict
        IntPtr fd_detection_result_ptr = FD_DetectionResultWrapperGetData(fd_detection_result_wrapper_ptr); // get result from wrapper
        FD_DetectionResult fd_detection_result = new FD_DetectionResult();
        Marshal.PtrToStructure(fd_detection_result_ptr, fd_detection_result);
        DetectionResult detection_result = ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
        FD_DestroyDetectionResultWrapper(fd_detection_result_wrapper_ptr); // free fd_detection_result_wrapper_ptr
        FD_DestroyDetectionResult(fd_detection_result_ptr); // free fd_detection_result_ptr
        return detection_result;
      }

      // below are underlying C api
      private IntPtr fd_ppyoloe_wrapper;
      [DllImport("fastdeploy.dll", EntryPoint = "FD_CreatesPPYOLOEWrapper")]
      private static IntPtr FD_CreatesPPYOLOEWrapper(string model_file, string params_file,
                                                     string config_file, IntPtr fd_runtime_option_wrapper,
                                                     ModelFormat model_format);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyPPYOLOEWrapper")]
      private static void FD_DestroyPPYOLOEWrapper(IntPtr fd_ppyoloe_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_PPYOLOEWrapperPredict")]
      private static bool FD_PPYOLOEWrapperPredict(IntPtr fd_ppyoloe_wrapper,
                                    IntPtr img, IntPtr fd_detection_result_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_CreateDetectionResultWrapper")]
      IntPtr FD_CreateDetectionResultWrapper();
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyDetectionResultWrapper")]
      void FD_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DestroyDetectionResult")]
      void FD_DestroyDetectionResult(IntPtr fd_detection_result);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_DetectionResultWrapperGetData")]
      IntPtr FD_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
      [DllImport("fastdeploy.dll", EntryPoint = "FD_CreateDetectionResultWrapperFromData")]
      IntPtr FD_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);
    }
    }
  }
}