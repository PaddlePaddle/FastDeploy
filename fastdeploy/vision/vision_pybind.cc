// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/pybind/main.h"

namespace fastdeploy {

void BindDetection(pybind11::module& m);
void BindClassification(pybind11::module& m);
void BindSegmentation(pybind11::module& m);
void BindMatting(pybind11::module& m);
void BindFaceDet(pybind11::module& m);
void BindFaceAlign(pybind11::module& m);
void BindFaceId(pybind11::module& m);
void BindOcr(pybind11::module& m);
void BindTracking(pybind11::module& m);
void BindKeyPointDetection(pybind11::module& m);
#ifdef ENABLE_VISION_VISUALIZE
void BindVisualize(pybind11::module& m);
#endif

void BindVision(pybind11::module& m) {
  pybind11::class_<vision::Mask>(m, "Mask")
      .def(pybind11::init())
      .def_readwrite("data", &vision::Mask::data)
      .def_readwrite("shape", &vision::Mask::shape)
      .def("__repr__", &vision::Mask::Str)
      .def("__str__", &vision::Mask::Str);

  pybind11::class_<vision::ClassifyResult>(m, "ClassifyResult")
      .def(pybind11::init())
      .def_readwrite("label_ids", &vision::ClassifyResult::label_ids)
      .def_readwrite("scores", &vision::ClassifyResult::scores)
      .def("__repr__", &vision::ClassifyResult::Str)
      .def("__str__", &vision::ClassifyResult::Str);

  pybind11::class_<vision::DetectionResult>(m, "DetectionResult")
      .def(pybind11::init())
      .def_readwrite("boxes", &vision::DetectionResult::boxes)
      .def_readwrite("scores", &vision::DetectionResult::scores)
      .def_readwrite("label_ids", &vision::DetectionResult::label_ids)
      .def_readwrite("masks", &vision::DetectionResult::masks)
      .def_readwrite("contain_masks", &vision::DetectionResult::contain_masks)
      .def("__repr__", &vision::DetectionResult::Str)
      .def("__str__", &vision::DetectionResult::Str);

  pybind11::class_<vision::OCRResult>(m, "OCRResult")
      .def(pybind11::init())
      .def_readwrite("boxes", &vision::OCRResult::boxes)
      .def_readwrite("text", &vision::OCRResult::text)
      .def_readwrite("rec_scores", &vision::OCRResult::rec_scores)
      .def_readwrite("cls_scores", &vision::OCRResult::cls_scores)
      .def_readwrite("cls_labels", &vision::OCRResult::cls_labels)
      .def("__repr__", &vision::OCRResult::Str)
      .def("__str__", &vision::OCRResult::Str);

  pybind11::class_<vision::MOTResult>(m, "MOTResult")
      .def(pybind11::init())
      .def_readwrite("boxes", &vision::MOTResult::boxes)
      .def_readwrite("ids", &vision::MOTResult::ids)
      .def_readwrite("scores", &vision::MOTResult::scores)
      .def_readwrite("class_ids", &vision::MOTResult::class_ids)
      .def("__repr__", &vision::MOTResult::Str)
      .def("__str__", &vision::MOTResult::Str);

  pybind11::class_<vision::FaceDetectionResult>(m, "FaceDetectionResult")
      .def(pybind11::init())
      .def_readwrite("boxes", &vision::FaceDetectionResult::boxes)
      .def_readwrite("scores", &vision::FaceDetectionResult::scores)
      .def_readwrite("landmarks", &vision::FaceDetectionResult::landmarks)
      .def_readwrite("landmarks_per_face",
                     &vision::FaceDetectionResult::landmarks_per_face)
      .def("__repr__", &vision::FaceDetectionResult::Str)
      .def("__str__", &vision::FaceDetectionResult::Str);

  pybind11::class_<vision::FaceAlignmentResult>(m, "FaceAlignmentResult")
      .def(pybind11::init())
      .def_readwrite("landmarks", &vision::FaceAlignmentResult::landmarks)
      .def("__repr__", &vision::FaceAlignmentResult::Str)
      .def("__str__", &vision::FaceAlignmentResult::Str);

  pybind11::class_<vision::FaceRecognitionResult>(m, "FaceRecognitionResult")
      .def(pybind11::init())
      .def_readwrite("embedding", &vision::FaceRecognitionResult::embedding)
      .def("__repr__", &vision::FaceRecognitionResult::Str)
      .def("__str__", &vision::FaceRecognitionResult::Str);

  pybind11::class_<vision::SegmentationResult>(m, "SegmentationResult")
      .def(pybind11::init())
      .def_readwrite("label_map", &vision::SegmentationResult::label_map)
      .def_readwrite("score_map", &vision::SegmentationResult::score_map)
      .def_readwrite("shape", &vision::SegmentationResult::shape)
      .def("__repr__", &vision::SegmentationResult::Str)
      .def("__str__", &vision::SegmentationResult::Str);

  pybind11::class_<vision::MattingResult>(m, "MattingResult")
      .def(pybind11::init())
      .def_readwrite("alpha", &vision::MattingResult::alpha)
      .def_readwrite("foreground", &vision::MattingResult::foreground)
      .def_readwrite("shape", &vision::MattingResult::shape)
      .def_readwrite("contain_foreground", &vision::MattingResult::shape)
      .def("__repr__", &vision::MattingResult::Str)
      .def("__str__", &vision::MattingResult::Str);

  pybind11::class_<vision::KeyPointDetectionResult>(m,
                                                    "KeyPointDetectionResult")
      .def(pybind11::init())
      .def_readwrite("keypoints", &vision::KeyPointDetectionResult::keypoints)
      .def_readwrite("scores", &vision::KeyPointDetectionResult::scores)
      .def_readwrite("num_joints", &vision::KeyPointDetectionResult::num_joints)
      .def("__repr__", &vision::KeyPointDetectionResult::Str)
      .def("__str__", &vision::KeyPointDetectionResult::Str);

  BindDetection(m);
  BindClassification(m);
  BindSegmentation(m);
  BindFaceDet(m);
  BindFaceAlign(m);
  BindFaceId(m);
  BindMatting(m);
  BindOcr(m);
  BindTracking(m);
  BindKeyPointDetection(m);
#ifdef ENABLE_VISION_VISUALIZE
  BindVisualize(m);
#endif
}
}  // namespace fastdeploy
