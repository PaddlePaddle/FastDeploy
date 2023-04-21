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
using fastdeploy.vision;

namespace fastdeploy {
namespace types_internal_c {

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimArrayUint8 {
  public nuint size;
  public IntPtr data;  // byte[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimArrayInt32 {
  public nuint size;
  public IntPtr data;  // int[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimArraySize {
  public nuint size;
  public IntPtr data;  // nuint[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimArrayInt64 {
  public nuint size;
  public IntPtr data;  // long[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimArrayFloat {
  public nuint size;
  public IntPtr data;  // float[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_Cstr {
  public nuint size;
  public string data;
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimArrayCstr {
  public nuint size;
  public IntPtr data;  // FD_Cstr[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_TwoDimArraySize {
  public nuint size;
  public IntPtr data;  // FD_OneDimArraySize[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_TwoDimArrayFloat {
  public nuint size;
  public IntPtr data;  // FD_OneDimArrayFloat[]
}



[StructLayout(LayoutKind.Sequential)]
public struct FD_TwoDimArrayInt32 {
  public nuint size;
  public IntPtr data;  // FD_OneDimArrayInt32[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_ThreeDimArrayInt32 {
  public nuint size;
  public IntPtr data;  // FD_TwoDimArrayInt32[]
}


public enum FD_ResultType {
  UNKNOWN_RESULT,
  CLASSIFY,
  DETECTION,
  SEGMENTATION,
  OCR,
  MOT,
  FACE_DETECTION,
  FACE_ALIGNMENT,
  FACE_RECOGNITION,
  MATTING,
  MASK,
  KEYPOINT_DETECTION,
  HEADPOSE
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_ClassifyResult {
  public FD_OneDimArrayInt32 label_ids;
  public FD_OneDimArrayFloat scores;
  public FD_ResultType type;
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimClassifyResult {
  public nuint size;
  public IntPtr data; // FD_ClassifyResult[]
}


[StructLayout(LayoutKind.Sequential)]
public struct FD_Mask {
  public FD_OneDimArrayUint8 data;
  public FD_OneDimArrayInt64 shape;
  public FD_ResultType type;
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimMask {
  public nint size;
  public IntPtr data;  // FD_Mask*
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_DetectionResult {
  public FD_TwoDimArrayFloat boxes;
  public FD_TwoDimArrayFloat rotated_boxes;
  public FD_OneDimArrayFloat scores;
  public FD_OneDimArrayInt32 label_ids;
  public FD_OneDimMask masks;
  [MarshalAs(UnmanagedType.U1)]
  public bool contain_masks;
  public FD_ResultType type;
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimDetectionResult {
  public nuint size;
  public IntPtr data; // FD_DetectionResult[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OCRResult {
  public FD_TwoDimArrayInt32 boxes;
  public FD_OneDimArrayCstr text;
  public FD_OneDimArrayFloat rec_scores;
  public FD_OneDimArrayFloat cls_scores;
  public FD_OneDimArrayInt32 cls_labels;
  public FD_ResultType type;
}


[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimOCRResult {
  public nuint size;
  public IntPtr data; // FD_OCRResult[]
}

public struct FD_SegmentationResult {
  public FD_OneDimArrayUint8 label_map;
  public FD_OneDimArrayFloat score_map;
  public FD_OneDimArrayInt64 shape;
  [MarshalAs(UnmanagedType.U1)]
  public bool contain_score_map;
  public FD_ResultType type;
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimSegmentationResult {
  public nuint size;
  public IntPtr data; // FD_SegmentationResult[]
}

[StructLayout(LayoutKind.Sequential)]
public struct FD_OneDimMat {
  public nuint size;
  public IntPtr data; // Mat[]
}

}
}