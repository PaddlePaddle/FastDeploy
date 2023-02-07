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
  struct FD_OneDimArrayFloat {
    public nuint size;
    public IntPtr data; // float[]
  } 

  [StructLayout(LayoutKind.Sequential)]
  struct FD_Cstr 
  {
    public nuint size;
    public string data;
  }

  [StructLayout(LayoutKind.Sequential)]
  struct FD_OneDimArrayCstr
  {
    public nuint size;
    public IntPtr data; // FD_Cstr[]
  }

  [StructLayout(LayoutKind.Sequential)]
  struct FD_TwoDimArraySize
  {
    public nuint size;
    public IntPtr data; // FD_OneDimArraySize[]
  }

  [StructLayout(LayoutKind.Sequential)]
  struct FD_TwoDimArrayFloat {
    public nuint size;
    public IntPtr data; // FD_OneDimArrayFloat[]
  }  

  enum FD_ResultType{
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
  struct FD_ClassifyResult{
  FD_OneDimArrayInt32 label_ids;
  FD_OneDimArrayFloat scores;
  FD_ResultType type;
} 
 
 [StructLayout(LayoutKind.Sequential)]
 struct FD_Mask{
  FD_OneDimArrayUint8 data;
  FD_OneDimArrayInt64 shape;
  FD_ResultType type;
}
 
 [StructLayout(LayoutKind.Sequential)]
 struct FD_OneDimMask {
  size_t size;
  IntPtr data; // FD_Mask*
}
 
 [StructLayout(LayoutKind.Sequential)]
 struct FD_DetectionResult{
  FD_TwoDimArrayFloat  boxes;
  FD_OneDimArrayFloat scores;
  FD_OneDimArrayInt32 label_ids;
  FD_OneDimMask masks;
  bool contain_masks;
  FD_ResultType type;
}

  }
}

