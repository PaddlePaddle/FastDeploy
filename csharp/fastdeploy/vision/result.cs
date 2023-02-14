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
using fastdeploy.types_internal_c;

namespace fastdeploy {
namespace vision {

public enum ResultType {
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

public struct Mask {
  public List<byte> data;
  public List<long> shape;
  public ResultType type;
  public Mask() {
    this.data = new List<byte>();
    this.shape = new List<long>();
    this.type = ResultType.MASK;
  }
}

public struct ClassifyResult {
  public List<int> label_ids;
  public List<float> scores;
  public ResultType type;
  public ClassifyResult() {
    this.label_ids = new List<int>();
    this.scores = new List<float>();
    this.type = ResultType.CLASSIFY;
  }
}

public struct DetectionResult {
  public List<float[]> boxes;
  public List<float> scores;
  public List<int> label_ids;
  public List<Mask> masks;
  public bool contain_masks;
  public ResultType type;
  public DetectionResult() {
    this.boxes = new List<float[]>();
    this.scores = new List<float>();
    this.label_ids = new List<int>();
    this.masks = new List<Mask>();
    this.contain_masks = false;
    this.type = ResultType.DETECTION;
  }
}

public class ConvertResult {

  public static FD_ClassifyResult
  ConvertClassifyResultToCResult(ClassifyResult classify_result) {
    FD_ClassifyResult fd_classify_result = new FD_ClassifyResult();

    // copy label_ids
    // Create a managed array
    fd_classify_result.label_ids.size = (uint)classify_result.label_ids.Count;
    int[] label_ids = new int[fd_classify_result.label_ids.size];
    // Copy data from Link to Array
    classify_result.label_ids.CopyTo(label_ids);
    // Copy data to unmanaged memory
    int size = Marshal.SizeOf(label_ids[0]) * label_ids.Length;
    fd_classify_result.label_ids.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(label_ids, 0, fd_classify_result.label_ids.data,
                 label_ids.Length);

    // copy scores
    // Create a managed array
    fd_classify_result.scores.size = (uint)classify_result.scores.Count;
    float[] scores = new float[fd_classify_result.scores.size];
    // Copy data from Link to Array
    classify_result.scores.CopyTo(scores);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(scores[0]) * scores.Length;
    fd_classify_result.scores.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(scores, 0, fd_classify_result.scores.data, scores.Length);

    fd_classify_result.type = (FD_ResultType)classify_result.type;

    return fd_classify_result;
  }

  public static ClassifyResult
  ConvertCResultToClassifyResult(FD_ClassifyResult fd_classify_result) {
    ClassifyResult classify_result = new ClassifyResult();

    // copy label_ids
    int[] label_ids = new int[fd_classify_result.label_ids.size];
    Marshal.Copy(fd_classify_result.label_ids.data, label_ids, 0,
                 label_ids.Length);
    classify_result.label_ids = new List<int>(label_ids);

    // copy scores
    float[] scores = new float[fd_classify_result.scores.size];
    Marshal.Copy(fd_classify_result.scores.data, scores, 0, scores.Length);
    classify_result.scores = new List<float>(scores);

    classify_result.type = (ResultType)fd_classify_result.type;
    return classify_result;
  }

  public static FD_DetectionResult
  ConvertDetectionResultToCResult(DetectionResult detection_result) {
    FD_DetectionResult fd_detection_result = new FD_DetectionResult();

    // copy boxes
    int boxes_coordinate_dim = 4;
    int size;
    fd_detection_result.boxes.size = (uint)detection_result.boxes.Count;
    FD_OneDimArraySize[] boxes =
        new FD_OneDimArraySize[fd_detection_result.boxes.size];
    // Copy each box
    for (int i = 0; i < (int)fd_detection_result.boxes.size; i++) {
      boxes[i].size = (uint)detection_result.boxes[i].Length;
      float[] boxes_i = new float[boxes_coordinate_dim];
      detection_result.boxes[i].CopyTo(boxes_i, 0);
      size = Marshal.SizeOf(boxes_i[0]) * boxes_i.Length;
      boxes[i].data = Marshal.AllocHGlobal(size);
      Marshal.Copy(boxes_i, 0, boxes[i].data, boxes_i.Length);
    }
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(boxes[0]) * boxes.Length;
    fd_detection_result.boxes.data = Marshal.AllocHGlobal(size);
    for (int i = 0; i < boxes.Length; i++) {
      Marshal.StructureToPtr(
          boxes[i],
          fd_detection_result.boxes.data + i * Marshal.SizeOf(boxes[0]), true);
    }

    // copy scores
    fd_detection_result.scores.size = (uint)detection_result.scores.Count;
    float[] scores = new float[fd_detection_result.scores.size];
    // Copy data from Link to Array
    detection_result.scores.CopyTo(scores);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(scores[0]) * scores.Length;
    fd_detection_result.scores.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(scores, 0, fd_detection_result.scores.data, scores.Length);

    // copy label_ids
    fd_detection_result.label_ids.size = (uint)detection_result.label_ids.Count;
    int[] label_ids = new int[fd_detection_result.label_ids.size];
    // Copy data from Link to Array
    detection_result.label_ids.CopyTo(label_ids);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(label_ids[0]) * label_ids.Length;
    fd_detection_result.label_ids.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(label_ids, 0, fd_detection_result.label_ids.data,
                 label_ids.Length);

    // copy masks
    fd_detection_result.masks.size = detection_result.masks.Count;
    FD_Mask[] masks = new FD_Mask[fd_detection_result.masks.size];
    // copy each mask
    for (int i = 0; i < (int)fd_detection_result.masks.size; i++) {
      // copy data in mask
      masks[i].data.size = (uint)detection_result.masks[i].data.Count;
      byte[] masks_data_i = new byte[masks[i].data.size];
      detection_result.masks[i].data.CopyTo(masks_data_i);
      size = Marshal.SizeOf(masks_data_i[0]) * masks_data_i.Length;
      masks[i].data.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(masks_data_i, 0, masks[i].data.data, masks_data_i.Length);
      // copy shape in mask
      masks[i].shape.size = (uint)detection_result.masks[i].shape.Count;
      long[] masks_shape_i = new long[masks[i].shape.size];
      detection_result.masks[i].shape.CopyTo(masks_shape_i);
      size = Marshal.SizeOf(masks_shape_i[0]) * masks_shape_i.Length;
      masks[i].shape.data = Marshal.AllocHGlobal(size);
      Marshal.Copy(masks_shape_i, 0, masks[i].shape.data, masks_shape_i.Length);
      // copy type
      masks[i].type = (FD_ResultType)detection_result.masks[i].type;
    }
    if (fd_detection_result.masks.size != 0) {
      size = Marshal.SizeOf(masks[0]) * masks.Length;
      fd_detection_result.masks.data = Marshal.AllocHGlobal(size);
      for (int i = 0; i < masks.Length; i++) {
        Marshal.StructureToPtr(masks[i],
                               fd_detection_result.masks.data +
                                   i * Marshal.SizeOf(masks[0]),
                               true);
      }
    }

    fd_detection_result.contain_masks = detection_result.contain_masks;
    fd_detection_result.type = (FD_ResultType)detection_result.type;
    return fd_detection_result;
  }

  public static DetectionResult
  ConvertCResultToDetectionResult(FD_DetectionResult fd_detection_result) {
    DetectionResult detection_result = new DetectionResult();

    // copy boxes
    detection_result.boxes = new List<float[]>();
    FD_OneDimArraySize[] boxes =
        new FD_OneDimArraySize[fd_detection_result.boxes.size];
    Console.WriteLine(fd_detection_result.boxes.size);
    for (int i = 0; i < (int)fd_detection_result.boxes.size; i++) {
      boxes[i] = (FD_OneDimArraySize)Marshal.PtrToStructure(
          fd_detection_result.boxes.data + i * Marshal.SizeOf(boxes[0]),
          typeof(FD_OneDimArraySize));
      float[] box_i = new float[boxes[i].size];
      Marshal.Copy(boxes[i].data, box_i, 0, box_i.Length);
      detection_result.boxes.Add(box_i);
    }

    // copy scores
    float[] scores = new float[fd_detection_result.scores.size];
    Marshal.Copy(fd_detection_result.scores.data, scores, 0, scores.Length);
    detection_result.scores = new List<float>(scores);

    // copy label_ids
    int[] label_ids = new int[fd_detection_result.label_ids.size];
    Marshal.Copy(fd_detection_result.label_ids.data, label_ids, 0,
                 label_ids.Length);
    detection_result.label_ids = new List<int>(label_ids);

    // copy masks
    detection_result.masks = new List<Mask>();
    FD_Mask[] fd_masks = new FD_Mask[fd_detection_result.masks.size];
    for (int i = 0; i < (int)fd_detection_result.masks.size; i++) {
      fd_masks[i] = (FD_Mask)Marshal.PtrToStructure(
          fd_detection_result.masks.data + i * Marshal.SizeOf(fd_masks[0]),
          typeof(FD_Mask));
      Mask mask_i = new Mask();
      byte[] mask_i_data = new byte[fd_masks[i].data.size];
      Marshal.Copy(fd_masks[i].data.data, mask_i_data, 0, mask_i_data.Length);
      long[] mask_i_shape = new long[fd_masks[i].shape.size];
      Marshal.Copy(fd_masks[i].shape.data, mask_i_shape, 0,
                   mask_i_shape.Length);
      mask_i.type = (ResultType)fd_masks[i].type;
      detection_result.masks.Add(mask_i);
    }
    detection_result.contain_masks = fd_detection_result.contain_masks;
    detection_result.type = (ResultType)fd_detection_result.type;
    return detection_result;
  }

  

}

}

}
