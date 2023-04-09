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

/*! Mask structure, used in DetectionResult for instance segmentation models
 */
public class Mask {
  public List<byte> data; /// Mask data buffer
  public List<long> shape; /// Shape of mask
  public ResultType type;
  public Mask() {
    this.data = new List<byte>();
    this.shape = new List<long>();
    this.type = ResultType.MASK;
  }

  /// convert the result to string to print
  public override string ToString() {
    string information = "Mask(" ;
    int ndim = this.shape.Count;
    for (int i = 0; i < ndim; i++) {
    if (i < ndim - 1) {
      information += this.shape[i].ToString() + ",";
    } else {
      information += this.shape[i].ToString();
    }
  }
    information += ")\n";
    return information;
  }

}

/*! @brief Classify result structure for all the image classify models
 */
public class ClassifyResult {
  public List<int> label_ids; /// Classify result for an image
  public List<float> scores; /// The confidence for each classify result
  public ResultType type;   
  public ClassifyResult() {
    this.label_ids = new List<int>();
    this.scores = new List<float>();
    this.type = ResultType.CLASSIFY;
  }

  /// convert the result to string to print
  public string ToString() {  
    string information;
    information = "ClassifyResult(\nlabel_ids: ";
    for (int i = 0; i < label_ids.Count; i++) {
      information = information + label_ids[i].ToString() + ", ";
    }
    information += "\nscores: ";
    for (int i = 0; i < scores.Count; i++) {
      information = information + scores[i].ToString() + ", ";
    }
    information += "\n)";
    return information;
  
  }
}

/*! @brief Detection result structure for all the object detection models and instance segmentation models
 */
public class DetectionResult {
  public List<float[]> boxes;  /// Member variable which indicates the coordinates of all detected target boxes in a single image, each box is represented by 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
  public List<float> scores;   /// Member variable which indicates the confidence level of all targets detected in a single image
  public List<int> label_ids;  /// Member variable which indicates all target categories detected in a single image
  public List<Mask> masks;  ///  Member variable which indicates all detected instance masks of a single image
  public bool contain_masks; /// Member variable which indicates whether the detected result contains instance masks
  public ResultType type;
  public DetectionResult() {
    this.boxes = new List<float[]>();
    this.scores = new List<float>();
    this.label_ids = new List<int>();
    this.masks = new List<Mask>();
    this.contain_masks = false;
    this.type = ResultType.DETECTION;
  }

  /// convert the result to string to print
  public string ToString() {
    string information;
    if (!contain_masks) {
      information = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
    } else {
      information =
          "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id, mask_shape]\n";
    }
    for (int i = 0; i < boxes.Count; i++) {
      information = information + boxes[i][0].ToString() + "," +
            boxes[i][1].ToString() + ", " + boxes[i][2].ToString() +
            ", " + boxes[i][3].ToString() + ", " +
            scores[i].ToString() + ", " + label_ids[i].ToString();
      if (!contain_masks) {
        information += "\n";
      } else {
        information += ", " + masks[i].ToString();
      }
    }
    return information;
  }

}

/*! @brief OCR result structure for all the OCR models.
 */
public class OCRResult {
  public List<int[]> boxes;   /// Member variable which indicates the coordinates of all detected target boxes in a single image. Each box is represented by 8 int values to indicate the 4 coordinates of the box, in the order of lower left, lower right, upper right, upper left.
  public List<string> text;  /// Member variable which indicates the content of the recognized text in multiple text boxes
  public List<float> rec_scores; ///  Member variable which indicates the confidence level of the recognized text.
  public List<float> cls_scores;  ///  Member variable which indicates the confidence level of the classification result of the text box
  public List<int> cls_labels;  /// Member variable which indicates the directional category of the textbox
  public ResultType type;

  public OCRResult() {
    this.boxes = new List<int[]>();
    this.text = new List<string>();
    this.rec_scores = new List<float>();
    this.cls_scores = new List<float>();
    this.cls_labels = new List<int>();
    this.type = ResultType.OCR;
  }

  /// convert the result to string to print
  public string ToString() {
  string no_result = "";
  if (boxes.Count > 0) {
    string information = "";
    for (int n = 0; n < boxes.Count; n++) {
      information = information + "det boxes: [";
      for (int i = 0; i < 4; i++) {
        information = information + "[" + boxes[n][i * 2].ToString() + "," +
              boxes[n][i * 2 + 1].ToString() + "]";

        if (i != 3) {
          information = information + ",";
        }
      }
      information = information + "]";

      if (rec_scores.Count > 0) {
        information = information + "rec text: " + text[n] + " rec score:" +
              rec_scores[n].ToString() + " ";
      }
      if (cls_labels.Count > 0) {
        information = information + "cls label: " + cls_labels[n].ToString() +
              " cls score: " + cls_scores[n].ToString();
      }
      information = information + "\n";
    }
    return information;

  } else if (boxes.Count == 0 && rec_scores.Count > 0 &&
             cls_scores.Count > 0) {
    string information="";
    for (int i = 0; i < rec_scores.Count; i++) {
      information = information + "rec text: " + text[i] + " rec score:" +
            rec_scores[i].ToString() + " ";
      information = information + "cls label: " + cls_labels[i].ToString() +
            " cls score: " + cls_scores[i].ToString();
      information = information + "\n";
    }
    return information;
  } else if (boxes.Count == 0 && rec_scores.Count == 0 &&
             cls_scores.Count > 0) {
    string information="";
    for (int i = 0; i < cls_scores.Count; i++) {
      information = information + "cls label: " + cls_labels[i].ToString() +
            " cls score: " + cls_scores[i].ToString();
      information = information + "\n";
    }
    return information;
  } else if (boxes.Count == 0 && rec_scores.Count > 0 &&
             cls_scores.Count == 0) {
    string information="";
    for (int i = 0; i < rec_scores.Count; i++) {
      information = information + "rec text: " + text[i] + " rec score:" +
            rec_scores[i].ToString() + " ";
      information = information + "\n";
    }
    return information;
  }

  no_result = no_result + "No Results!";
  return no_result;
  }

}

public class OCRClassifierResult{
  public int cls_label;
  public float cls_score;
}

public class OCRDBDetectorResult{
  public List<int[]> boxes;
}

public class OCRRecognizerResult{
  public string text;
  public float rec_score;
}

/*! @brief Segmentation result structure for all the segmentation models
 */
public class SegmentationResult{
  public List<byte> label_map;  /// `label_map` stores the pixel-level category labels for input image. 
  public List<float> score_map;  /// `score_map` stores the probability of the predicted label for each pixel of input image.
  public List<long> shape;  /// The output shape, means [H, W]
  public bool contain_score_map;  /// SegmentationResult whether containing score_map
  public ResultType type;
  public SegmentationResult() {
    this.label_map = new List<byte>();
    this.score_map = new List<float>();
    this.shape = new List<long>();
    this.contain_score_map = false;
    this.type = ResultType.SEGMENTATION;
  }

  /// convert the result to string to print
  public string ToString() {
    string information;
    information = "SegmentationResult Image masks 10 rows x 10 cols: \n";
    for (int i = 0; i < 10; ++i) {
      information += "[";
      for (int j = 0; j < 10; ++j) {
        information = information + label_map[i * 10 + j].ToString() + ", ";
      }
      information += ".....]\n";
    }
    information += "...........\n";
    if (contain_score_map) {
      information += "SegmentationResult Score map 10 rows x 10 cols: \n";
      for (int i = 0; i < 10; ++i) {
        information += "[";
        for (int j = 0; j < 10; ++j) {
          information = information + score_map[i * 10 + j].ToString() + ", ";
        }
        information += ".....]\n";
      }
      information += "...........\n";
    }
    information += "result shape is: [" + shape[0].ToString() + " " +
          shape[1].ToString() + "]";
    return information;
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

  // OCRResult
  public static FD_OCRResult
  ConvertOCRResultToCResult(OCRResult ocr_result) {
    FD_OCRResult fd_ocr_result = new FD_OCRResult();

    // copy boxes
    int boxes_coordinate_dim = 8;
    int size;
    fd_ocr_result.boxes.size = (uint)ocr_result.boxes.Count;
    FD_OneDimArrayInt32[] boxes =
        new FD_OneDimArrayInt32[fd_ocr_result.boxes.size];
    // Copy each box
    for (int i = 0; i < (int)fd_ocr_result.boxes.size; i++) {
      boxes[i].size = (uint)ocr_result.boxes[i].Length;
      int[] boxes_i = new int[boxes_coordinate_dim];
      ocr_result.boxes[i].CopyTo(boxes_i, 0);
      size = Marshal.SizeOf(boxes_i[0]) * boxes_i.Length;
      boxes[i].data = Marshal.AllocHGlobal(size);
      Marshal.Copy(boxes_i, 0, boxes[i].data, boxes_i.Length);
    }
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(boxes[0]) * boxes.Length;
    fd_ocr_result.boxes.data = Marshal.AllocHGlobal(size);
    for (int i = 0; i < boxes.Length; i++) {
      Marshal.StructureToPtr(
          boxes[i],
          fd_ocr_result.boxes.data + i * Marshal.SizeOf(boxes[0]), true);
    }

    // copy text 
    fd_ocr_result.text = ConvertStringArrayToCOneDimArrayCstr(ocr_result.text.ToArray());

    // copy rec_scores
    fd_ocr_result.rec_scores.size = (uint)ocr_result.rec_scores.Count;
    float[] rec_scores = new float[fd_ocr_result.rec_scores.size];
    // Copy data from Link to Array
    ocr_result.rec_scores.CopyTo(rec_scores);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(rec_scores[0]) * rec_scores.Length;
    fd_ocr_result.rec_scores.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(rec_scores, 0, fd_ocr_result.rec_scores.data, rec_scores.Length);

    // copy cls_scores
    fd_ocr_result.cls_scores.size = (uint)ocr_result.cls_scores.Count;
    float[] cls_scores = new float[fd_ocr_result.cls_scores.size];
    // Copy data from Link to Array
    ocr_result.cls_scores.CopyTo(cls_scores);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(cls_scores[0]) * cls_scores.Length;
    fd_ocr_result.cls_scores.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(cls_scores, 0, fd_ocr_result.cls_scores.data, cls_scores.Length);

    // copy cls_labels
    fd_ocr_result.cls_labels.size = (uint)ocr_result.cls_labels.Count;
    int[] cls_labels = new int[fd_ocr_result.cls_labels.size];
    // Copy data from Link to Array
    ocr_result.cls_labels.CopyTo(cls_labels);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(cls_labels[0]) * cls_labels.Length;
    fd_ocr_result.cls_labels.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(cls_labels, 0, fd_ocr_result.cls_labels.data, cls_labels.Length);
    
    fd_ocr_result.type = (FD_ResultType)ocr_result.type;
    return fd_ocr_result;
  }

  public static OCRResult
  ConvertCResultToOCRResult(FD_OCRResult fd_ocr_result) {
    OCRResult ocr_result = new OCRResult();

    // copy boxes
    ocr_result.boxes = new List<int[]>();
    FD_OneDimArrayInt32[] boxes =
        new FD_OneDimArrayInt32[fd_ocr_result.boxes.size];
    for (int i = 0; i < (int)fd_ocr_result.boxes.size; i++) {
      boxes[i] = (FD_OneDimArrayInt32)Marshal.PtrToStructure(
          fd_ocr_result.boxes.data + i * Marshal.SizeOf(boxes[0]),
          typeof(FD_OneDimArrayInt32));
      int[] box_i = new int[boxes[i].size];
      Marshal.Copy(boxes[i].data, box_i, 0, box_i.Length);
      ocr_result.boxes.Add(box_i);
    }

    // copy text
    string[] texts = ConvertCOneDimArrayCstrToStringArray(fd_ocr_result.text);
    ocr_result.text = new List<string>(texts);

    // copy rec_scores
    float[] rec_scores = new float[fd_ocr_result.rec_scores.size];
    Marshal.Copy(fd_ocr_result.rec_scores.data, rec_scores, 0,
                 rec_scores.Length);
    ocr_result.rec_scores = new List<float>(rec_scores);

    // copy cls_scores
    float[] cls_scores = new float[fd_ocr_result.cls_scores.size];
    Marshal.Copy(fd_ocr_result.cls_scores.data, cls_scores, 0,
                 cls_scores.Length);
    ocr_result.cls_scores = new List<float>(cls_scores);

    // copy cls_labels
    int[] cls_labels = new int[fd_ocr_result.cls_labels.size];
    Marshal.Copy(fd_ocr_result.cls_labels.data, cls_labels, 0,
                 cls_labels.Length);
    ocr_result.cls_labels = new List<int>(cls_labels);

    ocr_result.type = (ResultType)fd_ocr_result.type;
    return ocr_result;
  }

  public static SegmentationResult
  ConvertCResultToSegmentationResult(FD_SegmentationResult fd_segmentation_result){
    SegmentationResult segmentation_result = new SegmentationResult();

    // copy label_map
    byte[] label_map = new byte[fd_segmentation_result.label_map.size];
    Marshal.Copy(fd_segmentation_result.label_map.data, label_map, 0,
                 label_map.Length);
    segmentation_result.label_map = new List<byte>(label_map);

    // copy score_map
    float[] score_map = new float[fd_segmentation_result.score_map.size];
    Marshal.Copy(fd_segmentation_result.score_map.data, score_map, 0,
                 score_map.Length);
    segmentation_result.score_map = new List<float>(score_map);

    // copy shape
    long[] shape = new long[fd_segmentation_result.shape.size];
    Marshal.Copy(fd_segmentation_result.shape.data, shape, 0,
                 shape.Length);
    segmentation_result.shape = new List<long>(shape);

    segmentation_result.contain_score_map = fd_segmentation_result.contain_score_map;
    segmentation_result.type = (ResultType)fd_segmentation_result.type;
    return segmentation_result;

  }
  public static FD_SegmentationResult
  ConvertSegmentationResultToCResult(SegmentationResult segmentation_result){
    FD_SegmentationResult fd_segmentation_result = new FD_SegmentationResult();
    // copy label_map
    // Create a managed array
    fd_segmentation_result.label_map.size = (uint)segmentation_result.label_map.Count;
    byte[] label_map = new byte[fd_segmentation_result.label_map.size];
    // Copy data from Link to Array
    segmentation_result.label_map.CopyTo(label_map);

    // Copy data to unmanaged memory
    int size = Marshal.SizeOf(label_map[0]) * label_map.Length;
    fd_segmentation_result.label_map.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(label_map, 0, fd_segmentation_result.label_map.data,
                 label_map.Length);
    
    // copy score_map
    // Create a managed array
    fd_segmentation_result.score_map.size = (uint)segmentation_result.score_map.Count;
    if(fd_segmentation_result.score_map.size != 0){
    float[] score_map = new float[fd_segmentation_result.score_map.size];
    // Copy data from Link to Array
    segmentation_result.score_map.CopyTo(score_map);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(score_map[0]) * score_map.Length;
    fd_segmentation_result.score_map.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(score_map, 0, fd_segmentation_result.score_map.data,
                 score_map.Length);
    }
    
    // copy shape
    // Create a managed array
    fd_segmentation_result.shape.size = (uint)segmentation_result.shape.Count;
    long[] shape = new long[fd_segmentation_result.shape.size];
    // Copy data from Link to Array
    segmentation_result.shape.CopyTo(shape);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(shape[0]) * shape.Length;
    fd_segmentation_result.shape.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(shape, 0, fd_segmentation_result.shape.data,
                 shape.Length);

    fd_segmentation_result.contain_score_map = segmentation_result.contain_score_map;
    fd_segmentation_result.type = (FD_ResultType)segmentation_result.type;

    return fd_segmentation_result;
  }


  public static FD_OneDimArrayCstr
  ConvertStringArrayToCOneDimArrayCstr(string[] strs){
    FD_OneDimArrayCstr fd_one_dim_cstr = new FD_OneDimArrayCstr();
    fd_one_dim_cstr.size = (nuint)strs.Length;
    
    // Copy data to unmanaged memory
    FD_Cstr[] c_strs = new FD_Cstr[strs.Length];
    int size = Marshal.SizeOf(c_strs[0]) * c_strs.Length;
    fd_one_dim_cstr.data = Marshal.AllocHGlobal(size);
    for (int i = 0; i < strs.Length; i++) {
      c_strs[i].size = (nuint)strs[i].Length;
      c_strs[i].data = strs[i];
      Marshal.StructureToPtr(
          c_strs[i],
          fd_one_dim_cstr.data + i * Marshal.SizeOf(c_strs[0]), true);
    }
    return fd_one_dim_cstr;
  }

  public static string[]
  ConvertCOneDimArrayCstrToStringArray(FD_OneDimArrayCstr c_strs){
    string[] strs = new string[c_strs.size];
    for(int i=0; i<(int)c_strs.size; i++){
      FD_Cstr cstr = (FD_Cstr)Marshal.PtrToStructure(
          c_strs.data + i * Marshal.SizeOf(new FD_Cstr()),
          typeof(FD_Cstr));
      strs[i] = cstr.data;
    }
    return strs;
  }

}

}

}