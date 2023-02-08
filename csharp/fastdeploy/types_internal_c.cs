using System;
using System.IO;
using System.Runtime.InteropServices;
using fastdeploy.vision;

namespace fastdeploy{
  namespace types_internal_c {
  
  [StructLayout(LayoutKind.Sequential)]
  public struct FD_OneDimArrayUint8 
  {
    public nuint size;
    public IntPtr data;  // byte[]
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_OneDimArrayInt32 
  {
    public nuint size;
    public IntPtr data; // int[]
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_OneDimArraySize 
  {
    public nuint size;
    public IntPtr data;  // nuint[]
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_OneDimArrayInt64 
  {
    public nuint size;
    public IntPtr data; // long[]
  }  

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_OneDimArrayFloat {
    public nuint size;
    public IntPtr data; // float[]
  } 

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_Cstr 
  {
    public nuint size;
    public string data;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_OneDimArrayCstr
  {
    public nuint size;
    public IntPtr data; // FD_Cstr[]
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct FD_TwoDimArraySize
  {
    public nuint size;
    public IntPtr data; // FD_OneDimArraySize[]
  }

  [StructLayout(LayoutKind.Sequential)]
    public struct FD_TwoDimArrayFloat {
    public nuint size;
    public IntPtr data; // FD_OneDimArrayFloat[]
  }

  public enum FD_ResultType
  {
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
  public struct FD_ClassifyResult
  {
      public FD_OneDimArrayInt32 label_ids;
      public FD_OneDimArrayFloat scores;
      public FD_ResultType type;
  } 
 
 [StructLayout(LayoutKind.Sequential)]
  public struct FD_Mask
  {
      public FD_OneDimArrayUint8 data;
      public FD_OneDimArrayInt64 shape;
      public FD_ResultType type;
  }
 
 [StructLayout(LayoutKind.Sequential)]
    public struct FD_OneDimMask 
  {
    public nint size;
    public IntPtr data; // FD_Mask*
  }
 
 [StructLayout(LayoutKind.Sequential)]
  public struct FD_DetectionResult
  {
      public FD_TwoDimArrayFloat boxes;
      public FD_OneDimArrayFloat scores;
      public FD_OneDimArrayInt32 label_ids;
      public FD_OneDimMask masks;
      [MarshalAs(UnmanagedType.U1)]
      public bool contain_masks;
      public FD_ResultType type;
  }

  }
}

