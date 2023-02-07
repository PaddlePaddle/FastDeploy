using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using fastdeploy.types_internal_c;

namespace fastdeploy{
  namespace vision{
  
  enum ResultType{
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

  struct Mask{
     public List<byte> data;
     public List<long> shape;
     public ResultType type;
     public Mask() 
    {
        this.type = ResultType.MASK;
    }
  }

  struct ClassifyResult{
     public List<int> label_ids;
     public List<float> scores;
     public ResultType type;
     public ClassifyResult() 
    {
        this.type = ResultType.CLASSIFY;
    }
  }

  struct DetectionResult{
    public List<float[]> boxes;
    public List<float> scores;
    public List<int> label_ids;
    public List<Mask> masks;
    public bool contain_masks；
    public ResultType type;
    public DetectionResult() 
    {
        this.type = ResultType.DETECTION;
    }

  }

  class ConvertResult{

    public static FD_ClassifyResult ConvertClassifyResultToCResult(ClassifyResult classify_result){
      FD_ClassifyResult fd_classify_result = new FD_ClassifyResult();
      
      // copy label_ids
      // Create a managed array
      fd_classify_result.label_ids.size = classify_result.label_ids.Count;
      int [] label_ids = new int[fd_classify_result.label_ids.size];
      // Copy data from Link to Array
      classify_result.label_ids.CopyTo(label_ids);
      // Copy data to unmanaged memory
      int size = Marshal.SizeOf(label_ids[0]) * label_ids.Length;
      fd_classify_result.label_ids.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(label_ids, 0, fd_classify_result.label_ids.data, label_ids.Length);
      
      // copy scores
      // Create a managed array
      fd_classify_result.scores.size = classify_result.scores.Count;
      float [] scores = new float[fd_classify_result.scores.size];
      // Copy data from Link to Array
      classify_result.scores.CopyTo(scores);
      // Copy data to unmanaged memory
      size = Marshal.SizeOf(scores[0]) * scores.Length;
      fd_classify_result.scores.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(scores, 0, fd_classify_result.scores.data, scores.Length);
      
      fd_classify_result.type = (FD_ResultType)classify_result.type;

      return fd_classify_result;
    }

    public static ClassifyResult ConvertCResultToClassifyResult(FD_ClassifyResult fd_classify_result){
      ClassifyResult classify_result = new ClassifyResult();
      
      // copy label_ids
      int [] label_ids = new int[fd_classify_result.label_ids.size];
      Marshal.Copy(fd_classify_result.label_ids.data, label_ids, 0, label_ids.Length);
      classify_result.label_ids = new List<int>(label_ids);
      
      // copy scores
      float [] scores = new float[fd_classify_result.scores.size];
      Marshal.Copy(fd_classify_result.scores.data, scores, 0, scores.Length);
      classify_result.scores = new List<float>(scores);

      classify_result.type = (ResultType)fd_classify_result.type;
      return classify_result;
    }

    public static FD_DetectionResult ConvertDetectionResultToCResult(DetectionResult detection_result){
      FD_DetectionResult fd_detection_result = new FD_DetectionResult();

      // copy boxes
      int boxes_coordinate_dim = 4;
      fd_detection_result.boxes.size = detection_result.boxes.Count;
      FD_OneDimArraySize [] boxes = new FD_OneDimArraySize[fd_detection_result.boxes.size];
      // Copy each box
      for (size_t i = 0; i < fd_detection_result.boxes.size; i++) {
        boxes[i].size = detection_result.boxes[i].Length;
        float[] boxes_i = new float[boxes_coordinate_dim];
        detection_result.boxes[i].CopyTo(boxes_i, 0);
        int size = Marshal.SizeOf(boxes_i[0]) * boxes_i.Length;
        boxes[i].data = Marshal.AllocHGlobal(size);
        Marshal.Copy(boxes_i, 0, boxes[i].data, boxes_i.Length);
      }
      // Copy data to unmanaged memory
      int size = Marshal.SizeOf(boxes[0]) * boxes.Length;
      fd_detection_result.boxes.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(boxes, 0, fd_detection_result.boxes.data, boxes.Length);
      
      // copy scores
      fd_detection_result.scores.size = detection_result.scores.Count;
      float [] scores = new float[fd_detection_result.scores.size];
      // Copy data from Link to Array
      detection_result.scores.CopyTo(scores);
      // Copy data to unmanaged memory
      size = Marshal.SizeOf(scores[0]) * scores.Length;
      fd_detection_result.scores.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(scores, 0, fd_detection_result.scores.data, scores.Length);
      
      // copy label_ids
      fd_detection_result.label_ids.size = detection_result.label_ids.Count;
      int [] label_ids = new int[fd_detection_result.label_ids.size];
      // Copy data from Link to Array
      detection_result.label_ids.CopyTo(label_ids);
      // Copy data to unmanaged memory
      int size = Marshal.SizeOf(label_ids[0]) * label_ids.Length;
      fd_detection_result.label_ids.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(label_ids, 0, fd_detection_result.label_ids.data, label_ids.Length);

      // copy masks
      fd_detection_result.masks.size = detection_result.masks.Count;
      FD_Mask [] masks = new FD_Mask[fd_detection_result.masks.size];
      // copy each mask
      for (size_t i = 0; i < fd_detection_result.masks.size; i++) {
        // copy data in mask
        masks[i].data.size = detection_result.masks[i].data.Count;
        byte [] masks_data_i = new byte[masks[i].data.size];
        detection_result.masks[i].data.CopyTo(masks_data_i);
        size = Marshal.SizeOf(masks_data_i[0]) * masks_data_i.Length;
        masks[i].data.data = Marshal.AllocHGlobal(size);
        Marshal.Copy(masks_data_i, 0, masks[i].data.data, masks_data_i.Length);
        // copy shape in mask
        masks[i].shape.size = detection_result.masks[i].shape.Count;
        long [] masks_shape_i = new long[masks[i].shape.size];
        detection_result.masks[i].shape.CopyTo(masks_shape_i);
        size = Marshal.SizeOf(masks_shape_i[0]) * masks_shape_i.Length;
        masks[i].shape.data = Marshal.AllocHGlobal(size);
        Marshal.Copy(masks_shape_i, 0, masks[i].shape.data, masks_shape_i.Length);
        // copy type
        masks[i].type = (FD_ResultType)detection_result.masks[i].type;
      }
      size = Marshal.SizeOf(masks[0]) * masks.Length;
      fd_detection_result.masks.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(masks, 0, fd_detection_result.masks.data, masks.Length);

      fd_detection_result.contain_masks = detection_result.contain_masks;
      fd_detection_result.type = (FD_ResultType)detection_result.type;
      return fd_detection_result;
    }

    public static DetectionResult ConvertCResultToDetectionResult(FD_DetectionResult fd_detection_result){
      DetectionResult detection_result = new DetectionResult();
      
      // copy boxes
      detection_result.boxes = new List<float[]>();
      FD_OneDimArraySize [] boxes = new FD_OneDimArraySize[fd_detection_result.boxes.size];
      Marshal.Copy(fd_detection_result.boxes.data, boxes, 0, boxes.Length);
      for (size_t i = 0; i < fd_detection_result.boxes.size; i++) {
        float[] box_i = new float[boxes[i].size];
        Marshal.Copy(boxes[i].data, box_i, 0, box_i.Length);
        detection_result.boxes.Add(box_i);
      }
      
      // copy scores
      float [] scores = new float[fd_detection_result.scores.size];
      Marshal.Copy(fd_detection_result.scores.data, scores, 0, scores.Length);
      detection_result.scores = new List<float>(scores);

      // copy label_ids
      int [] label_ids = new int[fd_detection_result.label_ids.size];
      Marshal.Copy(fd_detection_result.label_ids.data, label_ids, 0, label_ids.Length);
      detection_result.label_ids = new List<int>(label_ids);

      // copy masks
      detection_result.masks = new List<Mask>();
      FD_Mask [] fd_masks = new FD_Mask[fd_detection_result.masks.size];
      Marshal.Copy(fd_detection_result.masks.data, fd_masks, 0, fd_masks.Length);
      for (size_t i = 0; i < fd_detection_result.masks.size; i++) {
        Mask mask_i = new Mask();
        byte[] mask_i_data = new byte[fd_masks[i].data.size];
        Marshal.Copy(fd_masks[i].data.data, mask_i_data, 0, mask_i_data.Length);
        long[] mask_i_shape = new long[fd_masks[i].shape.size];
        Marshal.Copy(fd_masks[i].shape.data, mask_i_shape, 0, mask_i_shape.Length);
        mask_i.type = (ResultType)fd_masks[i].type;
        detection_result.masks.Add(mask_i);
      }


      detection_result.contain_masks = fd_detection_result.contain_masks;
      detection_result.type = (ResultType)fd_detection_result.type;
      return classify_result;

    }
  }

  }

}

