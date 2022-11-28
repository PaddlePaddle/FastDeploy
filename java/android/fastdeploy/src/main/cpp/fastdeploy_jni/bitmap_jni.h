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

#include <jni.h>  // NOLINT

#include "fastdeploy/vision.h"  // NOLINT

namespace fastdeploy {
namespace jni {

// Convert the android bitmap(ARGB8888) to the OpenCV RGBA image. Actually,
// the data layout of ARGB8888 is R, G, B, A, it's the same as CV RGBA image,
// so it is unnecessary to do the conversion of color format, check
// https://developer.android.com/reference/android/graphics/Bitmap.Config#ARGB_8888
// to get the more details about Bitmap.Config.ARGB8888
jboolean ARGB888Bitmap2RGBA(JNIEnv *env, jobject j_argb8888_bitmap,
                            cv::Mat *c_rgba);

jboolean RGBA2ARGB888Bitmap(JNIEnv *env, jobject j_argb8888_bitmap,
                            const cv::Mat &c_rgba);

jboolean ARGB888Bitmap2BGR(JNIEnv *env, jobject j_argb8888_bitmap,
                           cv::Mat *c_bgr);

jboolean BGR2ARGB888Bitmap(JNIEnv *env, jobject j_argb8888_bitmap,
                           const cv::Mat &c_bgr);

}  // namespace jni
}  // namespace fastdeploy
