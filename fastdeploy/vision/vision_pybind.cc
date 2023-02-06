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

void BindProcessorManager(pybind11::module& m);
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
void BindHeadPose(pybind11::module& m);
void BindSR(pybind11::module& m);
void BindGeneration(pybind11::module& m);
void BindVisualize(pybind11::module& m);

void BindVision(pybind11::module& m) {
  pybind11::class_<vision::Mask>(m, "Mask")
      .def(pybind11::init())
      .def_readwrite("data", &vision::Mask::data)
      .def_readwrite("shape", &vision::Mask::shape)
      .def(pybind11::pickle(
          [](const vision::Mask& m) {
            return pybind11::make_tuple(m.data, m.shape);
          },
          [](pybind11::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error(
                  "vision::Mask pickle with invalid state!");

            vision::Mask m;
            m.data = t[0].cast<std::vector<uint8_t>>();
            m.shape = t[1].cast<std::vector<int64_t>>();

            return m;
          }))
      .def("__repr__", &vision::Mask::Str)
      .def("__str__", &vision::Mask::Str);

  pybind11::class_<vision::ClassifyResult>(m, "ClassifyResult")
      .def(pybind11::init())
      .def_readwrite("label_ids", &vision::ClassifyResult::label_ids)
      .def_readwrite("scores", &vision::ClassifyResult::scores)
      .def(pybind11::pickle(
          [](const vision::ClassifyResult& c) {
            return pybind11::make_tuple(c.label_ids, c.scores);
          },
          [](pybind11::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error(
                  "vision::ClassifyResult pickle with invalid state!");

            vision::ClassifyResult c;
            c.label_ids = t[0].cast<std::vector<int32_t>>();
            c.scores = t[1].cast<std::vector<float>>();

            return c;
          }))
      .def("__repr__", &vision::ClassifyResult::Str)
      .def("__str__", &vision::ClassifyResult::Str);

  pybind11::class_<vision::DetectionResult>(m, "DetectionResult")
      .def(pybind11::init())
      .def_readwrite("boxes", &vision::DetectionResult::boxes)
      .def_readwrite("scores", &vision::DetectionResult::scores)
      .def_readwrite("label_ids", &vision::DetectionResult::label_ids)
      .def_readwrite("masks", &vision::DetectionResult::masks)
      .def_readwrite("contain_masks", &vision::DetectionResult::contain_masks)
      .def(pybind11::pickle(
          [](const vision::DetectionResult& d) {
            return pybind11::make_tuple(d.boxes, d.scores, d.label_ids, d.masks,
                                        d.contain_masks);
          },
          [](pybind11::tuple t) {
            if (t.size() != 5)
              throw std::runtime_error(
                  "vision::DetectionResult pickle with Invalid state!");

            vision::DetectionResult d;
            d.boxes = t[0].cast<std::vector<std::array<float, 4>>>();
            d.scores = t[1].cast<std::vector<float>>();
            d.label_ids = t[2].cast<std::vector<int32_t>>();
            d.masks = t[3].cast<std::vector<vision::Mask>>();
            d.contain_masks = t[4].cast<bool>();

            return d;
          }))
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
      .def_readwrite("contain_score_map",
                     &vision::SegmentationResult::contain_score_map)
      .def(pybind11::pickle(
          [](const vision::SegmentationResult& s) {
            return pybind11::make_tuple(s.label_map, s.score_map, s.shape,
                                        s.contain_score_map);
          },
          [](pybind11::tuple t) {
            if (t.size() != 4)
              throw std::runtime_error(
                  "vision::SegmentationResult pickle with Invalid state!");

            vision::SegmentationResult s;
            s.label_map = t[0].cast<std::vector<uint8_t>>();
            s.score_map = t[1].cast<std::vector<float>>();
            s.shape = t[2].cast<std::vector<int64_t>>();
            s.contain_score_map = t[3].cast<bool>();

            return s;
          }))
      .def("__repr__", &vision::SegmentationResult::Str)
      .def("__str__", &vision::SegmentationResult::Str);

  pybind11::class_<vision::MattingResult>(m, "MattingResult")
      .def(pybind11::init())
      .def_readwrite("alpha", &vision::MattingResult::alpha)
      .def_readwrite("foreground", &vision::MattingResult::foreground)
      .def_readwrite("shape", &vision::MattingResult::shape)
      .def_readwrite("contain_foreground",
                     &vision::MattingResult::contain_foreground)
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

  pybind11::class_<vision::HeadPoseResult>(m, "HeadPoseResult")
      .def(pybind11::init())
      .def_readwrite("euler_angles", &vision::HeadPoseResult::euler_angles)
      .def("__repr__", &vision::HeadPoseResult::Str)
      .def("__str__", &vision::HeadPoseResult::Str);

  m.def("enable_flycv", &vision::EnableFlyCV,
        "Enable image preprocessing by FlyCV.");
  m.def("disable_flycv", &vision::DisableFlyCV,
        "Disable image preprocessing by FlyCV, change to use OpenCV.");

  BindProcessorManager(m);
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
  BindHeadPose(m);
  BindSR(m);
  BindGeneration(m);
  BindVisualize(m);
}
}  // namespace fastdeploy
