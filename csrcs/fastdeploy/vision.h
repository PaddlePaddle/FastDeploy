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
#pragma once

#include "fastdeploy/core/config.h"
#ifdef ENABLE_VISION
#include "fastdeploy/vision/biubug6/retinaface.h"
#include "fastdeploy/vision/deepcam/yolov5face.h"
#include "fastdeploy/vision/linzaer/ultraface.h"
#include "fastdeploy/vision/megvii/yolox.h"
#include "fastdeploy/vision/meituan/yolov6.h"
#include "fastdeploy/vision/ppcls/model.h"
#include "fastdeploy/vision/ppdet/model.h"
#include "fastdeploy/vision/ppogg/yolov5lite.h"
#include "fastdeploy/vision/ppseg/model.h"
#include "fastdeploy/vision/rangilyu/nanodet_plus.h"
#include "fastdeploy/vision/ultralytics/yolov5.h"
#include "fastdeploy/vision/wongkinyiu/scaledyolov4.h"
#include "fastdeploy/vision/wongkinyiu/yolor.h"
#include "fastdeploy/vision/wongkinyiu/yolov7.h"
#endif

#include "fastdeploy/vision/visualize/visualize.h"
