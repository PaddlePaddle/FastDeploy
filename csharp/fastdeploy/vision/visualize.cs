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

public class Visualize {

  public static Mat VisDetection(Mat im, DetectionResult detection_result,
                                 float score_threshold = 0.0f,
                                 int line_size = 1, float font_size = 0.5f) {
    FD_DetectionResult fd_detection_result =
        ConvertResult.ConvertDetectionResultToCResult(detection_result);
    IntPtr result_ptr =
        FD_C_VisDetection(im.CvPtr, ref fd_detection_result, score_threshold,
                          line_size, font_size);
    return new Mat(result_ptr);
  }

  public static Mat VisDetection(Mat im, DetectionResult detection_result,
                                 string[] labels, 
                                 float score_threshold = 0.0f,
                                 int line_size = 1, float font_size = 0.5f) {
    FD_DetectionResult fd_detection_result =
        ConvertResult.ConvertDetectionResultToCResult(detection_result);
    FD_OneDimArrayCstr labels_in = ConvertResult.ConvertStringArrayToCOneDimArrayCstr(labels);
    IntPtr result_ptr = 
        FD_C_VisDetection(im.CvPtr, ref fd_detection_result, 
                          ref labels_in, score_threshold,
                          line_size, font_size);
    return new Mat(result_ptr);
  }

  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_VisDetection")]
  private static extern IntPtr
  FD_C_VisDetection(IntPtr im, ref FD_DetectionResult fd_detection_result,
                    float score_threshold, int line_size, float font_size);
  
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_VisDetectionWithLabel")]
  private static extern IntPtr
  FD_C_VisDetectionWithLabel(IntPtr im, ref FD_DetectionResult fd_detection_result,
                    ref FD_OneDimArrayCstr labels,
                    float score_threshold, int line_size, float font_size);
}

}
}