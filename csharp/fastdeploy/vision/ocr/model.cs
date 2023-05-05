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
using fastdeploy.vision;
using fastdeploy.vision.ocr;

namespace fastdeploy {
namespace vision {
namespace ocr {

// Recognizer

/*! @brief Recognizer object is used to load the recognition model provided by PaddleOCR.
 */
public class Recognizer {

  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ./ch_PP-OCRv3_rec_infer/model.pdmodel.
   * \param[in] params_file Path of parameter file, e.g ./ch_PP-OCRv3_rec_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
   * \param[in] label_path Path of label file used by OCR recognition model. e.g ./ppocr_keys_v1.txt
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
   * \param[in] model_format Model format of the loaded model, default is Paddle format.
   */
  public Recognizer(string model_file, string params_file,
                    string label_path,
                    RuntimeOption custom_option = null,
                    ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_recognizer_model_wrapper = FD_C_CreateRecognizerWrapper(
        model_file, params_file, label_path, custom_option.GetWrapperPtr(),
        model_format);
  }

  ~Recognizer() {
    FD_C_DestroyRecognizerWrapper(fd_recognizer_model_wrapper);
  }

  /// Get model's name
  public string ModelName() {
    return "ppocr/ocr_rec";
  }

  /** \brief Predict the input image and get OCR recognition model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return The output of OCR recognition model result
   */
  public OCRRecognizerResult Predict(Mat img) {
    OCRRecognizerResult ocr_recognizer_result = new OCRRecognizerResult();
    FD_Cstr text = new FD_Cstr();
    if(! FD_C_RecognizerWrapperPredict(
        fd_recognizer_model_wrapper, img.CvPtr,
        ref text, ref ocr_recognizer_result.rec_score))
    {
      return null;
    } // predict
    ocr_recognizer_result.text = text.data;
    FD_C_DestroyCstr(ref text);
    return ocr_recognizer_result;
  }

  /** \brief BatchPredict the input image and get OCR recognition model result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return The output of OCR recognition model result.
   */
  public List<OCRRecognizerResult> BatchPredict(List<Mat> imgs){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimArrayCstr fd_texts_list =  new FD_OneDimArrayCstr();
    FD_OneDimArrayFloat fd_rec_scores_list = new FD_OneDimArrayFloat();
    if (!FD_C_RecognizerWrapperBatchPredict(fd_recognizer_model_wrapper, imgs_in, ref fd_texts_list, ref fd_rec_scores_list)){
      return null;
    }

    // copy texts
    string[] texts =  ConvertResult.ConvertCOneDimArrayCstrToStringArray(fd_texts_list);
    // copy rec_scores
    float[] rec_scores = new float[fd_rec_scores_list.size];
    Marshal.Copy(fd_rec_scores_list.data, rec_scores, 0,
                 rec_scores.Length);

    List<OCRRecognizerResult> results_out = new List<OCRRecognizerResult>();
    
    for(int i=0;i < (int)imgs.Count; i++){
    OCRRecognizerResult result = new OCRRecognizerResult();
    result.text = texts[i];
    result.rec_score = rec_scores[i];
    results_out.Add(result);
    }
    FD_C_DestroyOneDimArrayCstr(ref fd_texts_list);
    FD_C_DestroyOneDimArrayFloat(ref fd_rec_scores_list);
    Marshal.FreeHGlobal(imgs_in.data);
    return results_out;
  }

    public List<OCRRecognizerResult> BatchPredict(List<Mat> imgs, int start_index, int end_index, List<int> indices){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimArrayCstr fd_texts_list =  new FD_OneDimArrayCstr();
    FD_OneDimArrayFloat fd_rec_scores_list = new FD_OneDimArrayFloat();
    FD_OneDimArrayInt32 indices_in = new FD_OneDimArrayInt32();
    indices_in.size = (uint)indices.Count;
    int[] indices_array = new int[indices_in.size];
    indices.CopyTo(indices_array);
    // Copy data to unmanaged memory
    size = Marshal.SizeOf(indices_array[0]) * indices_array.Length;
    indices_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(indices_array, 0, indices_in.data,
                 indices_array.Length);
    if (!FD_C_RecognizerWrapperBatchPredictWithIndex(fd_recognizer_model_wrapper, imgs_in, ref fd_texts_list, ref fd_rec_scores_list, start_index, end_index,  indices_in)){
      return null;
    }

    // copy texts
    string[] texts =  ConvertResult.ConvertCOneDimArrayCstrToStringArray(fd_texts_list);
    // copy rec_scores
    float[] rec_scores = new float[fd_rec_scores_list.size];
    Marshal.Copy(fd_rec_scores_list.data, rec_scores, 0,
                 rec_scores.Length);

    List<OCRRecognizerResult> results_out = new List<OCRRecognizerResult>();
    
    for(int i=0;i < (int)imgs.Count; i++){
    OCRRecognizerResult result = new OCRRecognizerResult();
    result.text = texts[i];
    result.rec_score = rec_scores[i];
    results_out.Add(result);
    }
    FD_C_DestroyOneDimArrayCstr(ref fd_texts_list);
    FD_C_DestroyOneDimArrayFloat(ref fd_rec_scores_list);
    Marshal.FreeHGlobal(imgs_in.data);
    Marshal.FreeHGlobal(indices_in.data);
    return results_out;
  }

  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_RecognizerWrapperInitialized(fd_recognizer_model_wrapper);
  }

  public IntPtr GetWrapperPtr(){
    return fd_recognizer_model_wrapper;
  }

  // below are underlying C api
  private IntPtr fd_recognizer_model_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateRecognizerWrapper")]
  private static extern IntPtr FD_C_CreateRecognizerWrapper(
      string model_file, string params_file,string label_path,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyRecognizerWrapper")]
  private static extern void
  FD_C_DestroyRecognizerWrapper(IntPtr fd_recognizer_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RecognizerWrapperPredict")]
  private static extern bool
  FD_C_RecognizerWrapperPredict(IntPtr fd_recognizer_model_wrapper,
                                IntPtr img,
                                ref FD_Cstr text,
                                ref float rec_score);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyCstr")]
  private static extern void
  FD_C_DestroyCstr(ref FD_Cstr fd_cstr);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOneDimArrayCstr")]
  private static extern void
  FD_C_DestroyOneDimArrayCstr(ref FD_OneDimArrayCstr fd_onedim_cstr);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOneDimArrayFloat")]
  private static extern void
  FD_C_DestroyOneDimArrayFloat(ref FD_OneDimArrayFloat fd_onedim_float);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RecognizerWrapperInitialized")]
  private static extern bool
  FD_C_RecognizerWrapperInitialized(IntPtr fd_recognizer_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RecognizerWrapperBatchPredict")]
  private static extern bool
  FD_C_RecognizerWrapperBatchPredict(IntPtr fd_recognizer_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_OneDimArrayCstr texts,
                                     ref FD_OneDimArrayFloat rec_scores);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_RecognizerWrapperBatchPredictWithIndex")]
  private static extern bool
  FD_C_RecognizerWrapperBatchPredictWithIndex(IntPtr fd_recognizer_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_OneDimArrayCstr texts,
                                     ref FD_OneDimArrayFloat rec_scores,
                                     int start_index,
                                     int end_index,
                                     FD_OneDimArrayInt32 indices);

}

// Classifier

/*! @brief Classifier object is used to load the classification model provided by PaddleOCR.
 */
public class Classifier {

  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
   * \param[in] params_file Path of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
   * \param[in] model_format Model format of the loaded model, default is Paddle format.
   */
  public Classifier(string model_file, string params_file,
                    RuntimeOption custom_option = null,
                    ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_classifier_model_wrapper = FD_C_CreateClassifierWrapper(
        model_file, params_file, custom_option.GetWrapperPtr(),
        model_format);
  }

  ~Classifier() {
    FD_C_DestroyClassifierWrapper(fd_classifier_model_wrapper);
  }

  /// Get model's name
  public string ModelName() {
    return "ppocr/ocr_cls";
  }

  /** \brief Predict the input image and get OCR classification model cls_result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return OCRClassifierResult
   */
  public OCRClassifierResult Predict(Mat img) {
    OCRClassifierResult ocr_classify_result = new OCRClassifierResult();
    if(! FD_C_ClassifierWrapperPredict(
        fd_classifier_model_wrapper, img.CvPtr,
        ref ocr_classify_result.cls_label, ref ocr_classify_result.cls_score))
    {
      return null;
    } // predict
    return ocr_classify_result;
  }

  /** \brief BatchPredict the input image and get OCR classification model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return List<OCRClassifierResult>
   */
  public List<OCRClassifierResult> BatchPredict(List<Mat> imgs){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimArrayInt32 fd_cls_labels_list =  new FD_OneDimArrayInt32();
    FD_OneDimArrayFloat fd_cls_scores_list = new FD_OneDimArrayFloat();
    if (!FD_C_ClassifierWrapperBatchPredict(fd_classifier_model_wrapper, imgs_in, ref fd_cls_labels_list, ref fd_cls_scores_list)){
      return null;
    }

    // copy cls_labels
    int[] cls_labels = new int[fd_cls_labels_list.size];
    Marshal.Copy(fd_cls_labels_list.data, cls_labels, 0,
                 cls_labels.Length);
    // copy cls_scores
    float[] cls_scores = new float[fd_cls_scores_list.size];
    Marshal.Copy(fd_cls_scores_list.data, cls_scores, 0,
                 cls_scores.Length);

    List<OCRClassifierResult> results_out = new List<OCRClassifierResult>();
    
    for(int i=0;i < (int)imgs.Count; i++){
    OCRClassifierResult result = new OCRClassifierResult();
    result.cls_label = cls_labels[i];
    result.cls_score = cls_scores[i];
    results_out.Add(result);
    }
    FD_C_DestroyOneDimArrayInt32(ref fd_cls_labels_list);
    FD_C_DestroyOneDimArrayFloat(ref fd_cls_scores_list);
    Marshal.FreeHGlobal(imgs_in.data);
    return results_out;
  }

    public List<OCRClassifierResult> BatchPredict(List<Mat> imgs, int start_index, int end_index){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimArrayInt32 fd_cls_labels_list =  new FD_OneDimArrayInt32();
    FD_OneDimArrayFloat fd_cls_scores_list = new FD_OneDimArrayFloat();
    if (!FD_C_ClassifierWrapperBatchPredictWithIndex(fd_classifier_model_wrapper, imgs_in, ref fd_cls_labels_list, ref fd_cls_scores_list, start_index, end_index)){
      return null;
    }

    // copy cls_labels
    int[] cls_labels = new int[fd_cls_labels_list.size];
    Marshal.Copy(fd_cls_labels_list.data, cls_labels, 0,
                 cls_labels.Length);
    // copy cls_scores
    float[] cls_scores = new float[fd_cls_scores_list.size];
    Marshal.Copy(fd_cls_scores_list.data, cls_scores, 0,
                 cls_scores.Length);

    List<OCRClassifierResult> results_out = new List<OCRClassifierResult>();
    
    for(int i=0;i < (int)imgs.Count; i++){
    OCRClassifierResult result = new OCRClassifierResult();
    result.cls_label = cls_labels[i];
    result.cls_score = cls_scores[i];
    results_out.Add(result);
    }
    FD_C_DestroyOneDimArrayInt32(ref fd_cls_labels_list);
    FD_C_DestroyOneDimArrayFloat(ref fd_cls_scores_list);
    Marshal.FreeHGlobal(imgs_in.data);
    return results_out;
  }

  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_ClassifierWrapperInitialized(fd_classifier_model_wrapper);
  }

  public IntPtr GetWrapperPtr(){
    return fd_classifier_model_wrapper;
  }

  // below are underlying C api
  private IntPtr fd_classifier_model_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateClassifierWrapper")]
  private static extern IntPtr FD_C_CreateClassifierWrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyClassifierWrapper")]
  private static extern void
  FD_C_DestroyClassifierWrapper(IntPtr fd_classifier_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_ClassifierWrapperPredict")]
  private static extern bool
  FD_C_ClassifierWrapperPredict(IntPtr fd_classifier_model_wrapper,
                                IntPtr img,
                                ref int cls_label,
                                ref float cls_score);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_ClassifierWrapperInitialized")]
  private static extern bool
  FD_C_ClassifierWrapperInitialized(IntPtr fd_classifier_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_ClassifierWrapperBatchPredict")]
  private static extern bool
  FD_C_ClassifierWrapperBatchPredict(IntPtr fd_classifier_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_OneDimArrayInt32 cls_labels,
                                     ref FD_OneDimArrayFloat cls_scores);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_ClassifierWrapperBatchPredictWithIndex")]
  private static extern bool
  FD_C_ClassifierWrapperBatchPredictWithIndex(IntPtr fd_classifier_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_OneDimArrayInt32 cls_labels,
                                     ref FD_OneDimArrayFloat cls_scores,
                                     int start_index,
                                     int end_index);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOneDimArrayFloat")]
  private static extern void
  FD_C_DestroyOneDimArrayFloat(ref FD_OneDimArrayFloat fd_onedim_float);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOneDimArrayInt32")]
  private static extern void
  FD_C_DestroyOneDimArrayInt32(ref FD_OneDimArrayInt32 fd_onedim_int32);

}

// DBDetector

/*! @brief DBDetector object is used to load the detection model provided by PaddleOCR.
 */
public class DBDetector {

  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ./ch_PP-OCRv3_det_infer/model.pdmodel.
   * \param[in] params_file Path of parameter file, e.g ./ch_PP-OCRv3_det_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`.
   * \param[in] model_format Model format of the loaded model, default is Paddle format.
   */
  public DBDetector(string model_file, string params_file,
                    RuntimeOption custom_option = null,
                    ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_dbdetector_model_wrapper = FD_C_CreateDBDetectorWrapper(
        model_file, params_file, custom_option.GetWrapperPtr(),
        model_format);
  }

  ~DBDetector() {
    FD_C_DestroyDBDetectorWrapper(fd_dbdetector_model_wrapper);
  }

  /// Get model's name
  public string ModelName() {
    return "ppocr/ocr_det";
  }

  /** \brief Predict the input image and get OCR detection model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return OCRDBDetectorResult
   */
  public OCRDBDetectorResult Predict(Mat img) {
    OCRDBDetectorResult ocr_detector_result = new OCRDBDetectorResult();
    FD_TwoDimArrayInt32 fd_box_result = new FD_TwoDimArrayInt32();
    if(! FD_C_DBDetectorWrapperPredict(
        fd_dbdetector_model_wrapper, img.CvPtr,
        ref fd_box_result))
    {
      return null;
    } // predict
    ocr_detector_result.boxes = new List<int[]>();
    FD_OneDimArrayInt32[] boxes =
        new FD_OneDimArrayInt32[fd_box_result.size];
    for (int i = 0; i < (int)fd_box_result.size; i++) {
      boxes[i] = (FD_OneDimArrayInt32)Marshal.PtrToStructure(
          fd_box_result.data + i * Marshal.SizeOf(boxes[0]),
          typeof(FD_OneDimArrayInt32));
      int[] box_i = new int[boxes[i].size];
      Marshal.Copy(boxes[i].data, box_i, 0, box_i.Length);
      ocr_detector_result.boxes.Add(box_i);
    }
    FD_C_DestroyTwoDimArrayInt32(ref fd_box_result);
    return ocr_detector_result;
  }

  /** \brief BatchPredict the input image and get OCR detection model result.
   *
   * \param[in] images The list input of image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return List<OCRDBDetectorResult>
   */
  public List<OCRDBDetectorResult> BatchPredict(List<Mat> imgs){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_ThreeDimArrayInt32 fd_det_results_list =  new FD_ThreeDimArrayInt32();
    if (!FD_C_DBDetectorWrapperBatchPredict(fd_dbdetector_model_wrapper, imgs_in, ref fd_det_results_list)){
      return null;
    }
    
    List<OCRDBDetectorResult> results_out = new List<OCRDBDetectorResult>();
    FD_TwoDimArrayInt32[] batch_boxes =
        new FD_TwoDimArrayInt32[fd_det_results_list.size];
    for(int i=0;i < (int)imgs.Count; i++){
    OCRDBDetectorResult result = new OCRDBDetectorResult();
    result.boxes = new List<int[]>();
    batch_boxes[i] = (FD_TwoDimArrayInt32)Marshal.PtrToStructure(
          fd_det_results_list.data + i * Marshal.SizeOf(batch_boxes[0]),
          typeof(FD_TwoDimArrayInt32));
    FD_OneDimArrayInt32[] boxes =
        new FD_OneDimArrayInt32[batch_boxes[i].size];
    for (int j = 0; j < (int)batch_boxes[i].size; j++) {
      boxes[j] = (FD_OneDimArrayInt32)Marshal.PtrToStructure(
          batch_boxes[i].data + j * Marshal.SizeOf(boxes[0]),
          typeof(FD_OneDimArrayInt32));
      int[] box_j = new int[boxes[j].size];
      Marshal.Copy(boxes[j].data, box_j, 0, box_j.Length);
      result.boxes.Add(box_j);
    }
    results_out.Add(result);
    }
    FD_C_DestroyThreeDimArrayInt32(ref fd_det_results_list);
    Marshal.FreeHGlobal(imgs_in.data);
    return results_out;
  }

  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_DBDetectorWrapperInitialized(fd_dbdetector_model_wrapper);
  }

  public IntPtr GetWrapperPtr(){
    return fd_dbdetector_model_wrapper;
  }

  // below are underlying C api
  private IntPtr fd_dbdetector_model_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDBDetectorWrapper")]
  private static extern IntPtr FD_C_CreateDBDetectorWrapper(
      string model_file, string params_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDBDetectorWrapper")]
  private static extern void
  FD_C_DestroyDBDetectorWrapper(IntPtr fd_dbdetector_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DBDetectorWrapperPredict")]
  private static extern bool
  FD_C_DBDetectorWrapperPredict(IntPtr fd_dbdetector_model_wrapper,
                                IntPtr img,
                                ref FD_TwoDimArrayInt32 boxes_result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DBDetectorWrapperInitialized")]
  private static extern bool
  FD_C_DBDetectorWrapperInitialized(IntPtr fd_dbdetector_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DBDetectorWrapperBatchPredict")]
  private static extern bool
  FD_C_DBDetectorWrapperBatchPredict(IntPtr fd_dbdetector_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_ThreeDimArrayInt32 det_results);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOneDimArrayInt32")]
  private static extern void
  FD_C_DestroyOneDimArrayInt32(ref FD_OneDimArrayInt32 fd_onedim_int32);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyTwoDimArrayInt32")]
  private static extern void
  FD_C_DestroyTwoDimArrayInt32(ref FD_TwoDimArrayInt32 fd_twodim_int32);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyThreeDimArrayInt32")]
  private static extern void
  FD_C_DestroyThreeDimArrayInt32(ref FD_ThreeDimArrayInt32 fd_threedim_int32);

}
}
}

namespace pipeline {

// PPOCRv2

/*! @brief PPOCRv2 is used to load PP-OCRv2 series models provided by PaddleOCR.
 */
public class PPOCRv2 {

  /** \brief Set up the detection model path, classification model path and recognition model path respectively.
   *
   * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv2_det_infer
   * \param[in] cls_model Path of classification model, e.g ./ch_ppocr_mobile_v2.0_cls_infer
   * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv2_rec_infer
   */
  public PPOCRv2(DBDetector ppocrv2, Classifier classifier,
                 Recognizer recognizer) {
    fd_ppocrv2_wrapper = FD_C_CreatePPOCRv2Wrapper(
        ppocrv2.GetWrapperPtr(), 
        classifier.GetWrapperPtr(), 
        recognizer.GetWrapperPtr());
  }

  ~PPOCRv2() {
    FD_C_DestroyPPOCRv2Wrapper(fd_ppocrv2_wrapper);
  }


  public string ModelName() {
    return "PPOCRv2";
  }

  /** \brief Predict the input image and get OCR result.
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return OCRResult
   */
  public OCRResult Predict(Mat img) {
    FD_OCRResult fd_ocr_result = new FD_OCRResult();
    if(! FD_C_PPOCRv2WrapperPredict(
        fd_ppocrv2_wrapper, img.CvPtr,
        ref fd_ocr_result))
    {
      return null;
    } // predict
    OCRResult ocr_detector_result = ConvertResult.ConvertCResultToOCRResult(fd_ocr_result);
    FD_C_DestroyOCRResult(ref fd_ocr_result);
    return ocr_detector_result;
  }

  /** \brief BatchPredict the input image and get OCR result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return List<OCRResult>
   */
  public List<OCRResult> BatchPredict(List<Mat> imgs){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimOCRResult fd_ocr_result_array =  new FD_OneDimOCRResult();
    if (!FD_C_PPOCRv2WrapperBatchPredict(fd_ppocrv2_wrapper, imgs_in, ref fd_ocr_result_array)){
      return null;
    }
    List<OCRResult> results_out = new List<OCRResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_OCRResult fd_ocr_result = (FD_OCRResult)Marshal.PtrToStructure(
          fd_ocr_result_array.data + i * Marshal.SizeOf(new FD_OCRResult()),
          typeof(FD_OCRResult));
      results_out.Add(ConvertResult.ConvertCResultToOCRResult(fd_ocr_result));
      FD_C_DestroyOCRResult(ref fd_ocr_result);
    }
    Marshal.FreeHGlobal(imgs_in.data);
    return results_out;
  }

  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_PPOCRv2WrapperInitialized(fd_ppocrv2_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_ppocrv2_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreatePPOCRv2Wrapper")]
  private static extern IntPtr FD_C_CreatePPOCRv2Wrapper(
      IntPtr det_model, IntPtr cls_model,
      IntPtr rec_model);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyPPOCRv2Wrapper")]
  private static extern void
  FD_C_DestroyPPOCRv2Wrapper(IntPtr fd_ppocrv2_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPOCRv2WrapperPredict")]
  private static extern bool
  FD_C_PPOCRv2WrapperPredict(IntPtr fd_ppocrv2_model_wrapper,
                                IntPtr img,
                                ref FD_OCRResult result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPOCRv2WrapperInitialized")]
  private static extern bool
  FD_C_PPOCRv2WrapperInitialized(IntPtr fd_ppocrv2_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPOCRv2WrapperBatchPredict")]
  private static extern bool
  FD_C_PPOCRv2WrapperBatchPredict(IntPtr fd_ppocrv2_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_OneDimOCRResult batch_result);
  
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOCRResult")]
  private static extern void
  FD_C_DestroyOCRResult(ref FD_OCRResult fd_ocr_result);

}

// PPOCRv3

/*! @brief PPOCRv3 is used to load PP-OCRv3 series models provided by PaddleOCR.
 */
public class PPOCRv3 {

  /** \brief Set up the detection model path, classification model path and recognition model path respectively.
   *
   * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv3_det_infer
   * \param[in] cls_model Path of classification model, e.g ./ch_ppocr_mobile_v2.0_cls_infer
   * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv3_rec_infer
   */
  public PPOCRv3(DBDetector ppocrv3, Classifier classifier,
                 Recognizer recognizer) {
    fd_ppocrv3_wrapper = FD_C_CreatePPOCRv3Wrapper(
        ppocrv3.GetWrapperPtr(), 
        classifier.GetWrapperPtr(), 
        recognizer.GetWrapperPtr());
  }

  ~PPOCRv3() {
    FD_C_DestroyPPOCRv3Wrapper(fd_ppocrv3_wrapper);
  }


  public string ModelName() {
    return "PPOCRv3";
  }

  /** \brief Predict the input image and get OCR result.
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return OCRResult
   */
  public OCRResult Predict(Mat img) {
    FD_OCRResult fd_ocr_result = new FD_OCRResult();
    if(! FD_C_PPOCRv3WrapperPredict(
        fd_ppocrv3_wrapper, img.CvPtr,
        ref fd_ocr_result))
    {
      return null;
    } // predict
    OCRResult ocr_detector_result = ConvertResult.ConvertCResultToOCRResult(fd_ocr_result);
    FD_C_DestroyOCRResult(ref fd_ocr_result);
    return ocr_detector_result;
  }

  /** \brief BatchPredict the input image and get OCR result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
   * 
   * \return List<OCRResult>
   */
  public List<OCRResult> BatchPredict(List<Mat> imgs){
    FD_OneDimMat imgs_in = new FD_OneDimMat();
    imgs_in.size = (nuint)imgs.Count;
    // Copy data to unmanaged memory
    IntPtr[] mat_ptrs = new IntPtr[imgs_in.size];
    for(int i=0;i < (int)imgs.Count; i++){
      mat_ptrs[i] = imgs[i].CvPtr;
    }
    int size = Marshal.SizeOf(new IntPtr()) * (int)imgs_in.size;
    imgs_in.data = Marshal.AllocHGlobal(size);
    Marshal.Copy(mat_ptrs, 0, imgs_in.data,
                 mat_ptrs.Length);
    FD_OneDimOCRResult fd_ocr_result_array =  new FD_OneDimOCRResult();
    if (!FD_C_PPOCRv3WrapperBatchPredict(fd_ppocrv3_wrapper, imgs_in, ref fd_ocr_result_array)){
      return null;
    }
    List<OCRResult> results_out = new List<OCRResult>();
    for(int i=0;i < (int)imgs.Count; i++){
      FD_OCRResult fd_ocr_result = (FD_OCRResult)Marshal.PtrToStructure(
          fd_ocr_result_array.data + i * Marshal.SizeOf(new FD_OCRResult()),
          typeof(FD_OCRResult));
      results_out.Add(ConvertResult.ConvertCResultToOCRResult(fd_ocr_result));
      FD_C_DestroyOCRResult(ref fd_ocr_result);
    }
    Marshal.FreeHGlobal(imgs_in.data);
    return results_out;
  }

  /// Check whether model is initialized successfully
  public bool Initialized() {
    return FD_C_PPOCRv3WrapperInitialized(fd_ppocrv3_wrapper);
  }

  // below are underlying C api
  private IntPtr fd_ppocrv3_wrapper;
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreatePPOCRv3Wrapper")]
  private static extern IntPtr FD_C_CreatePPOCRv3Wrapper(
      IntPtr det_model, IntPtr cls_model,
      IntPtr rec_model);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyPPOCRv3Wrapper")]
  private static extern void
  FD_C_DestroyPPOCRv3Wrapper(IntPtr fd_ppocrv3_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPOCRv3WrapperPredict")]
  private static extern bool
  FD_C_PPOCRv3WrapperPredict(IntPtr fd_ppocrv3_model_wrapper,
                                IntPtr img,
                                ref FD_OCRResult result);

  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPOCRv3WrapperInitialized")]
  private static extern bool
  FD_C_PPOCRv3WrapperInitialized(IntPtr fd_ppocrv3_model_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_PPOCRv3WrapperBatchPredict")]
  private static extern bool
  FD_C_PPOCRv3WrapperBatchPredict(IntPtr fd_ppocrv3_model_wrapper,
                                     FD_OneDimMat imgs,
                                     ref FD_OneDimOCRResult batch_result);
  
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyOCRResult")]
  private static extern void
  FD_C_DestroyOCRResult(ref FD_OCRResult fd_ocr_result);

}

}

}