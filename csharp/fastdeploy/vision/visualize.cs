using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using OpenCvSharp;
using fastdeploy.types_internal_c;

namespace fastdeploy{
  namespace vision
    {

    class Visualize{

      public static Mat VisDetection(Mat im, DetectionResult detection_result, float score_threshold, int line_size, float font_size){
        FD_C_DetectionResult fd_detection_result = ConvertResult.ConvertDetectionResultToCResult(detection_result);
        IntPtr result_ptr = FD_C_VisDetection(im.CvPtr, ref fd_detection_result, score_threshold, line_size, font_size);
        return new Mat(result_ptr);
      }

      [DllImport("fastdeploy.dll", EntryPoint = "FD_C_VisDetection")]
      private static extern IntPtr FD_C_VisDetection(IntPtr im,  ref FD_C_DetectionResult fd_detection_result,
                        float score_threshold, int line_size, float font_size);
    }
   
  }
}