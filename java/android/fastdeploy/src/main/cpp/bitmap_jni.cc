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

#include "bitmap_jni.h"  // NOLINT

#include <android/bitmap.h>  // NOLINT

#include "utils_jni.h"  // NOLINT

namespace fastdeploy {
namespace jni {

jboolean ARGB888Bitmap2RGBA(JNIEnv *env, jobject j_argb8888_bitmap,
                            cv::Mat *c_rgba) {
  // Convert the android bitmap(ARGB8888) to the OpenCV RGBA image. Actually,
  // the data layout of ARGB8888 is R, G, B, A, it's the same as CV RGBA image,
  // so it is unnecessary to do the conversion of color format, check
  // https://developer.android.com/reference/android/graphics/Bitmap.Config#ARGB_8888
  // to get the more details about Bitmap.Config.ARGB8888
  AndroidBitmapInfo j_bitmap_info;
  if (AndroidBitmap_getInfo(env, j_argb8888_bitmap, &j_bitmap_info) < 0) {
    LOGE("Invoke AndroidBitmap_getInfo() failed!");
    return JNI_FALSE;
  }
  if (j_bitmap_info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("Only Bitmap.Config.ARGB8888 color format is supported!");
    return JNI_FALSE;
  }
  void *j_bitmap_pixels;
  if (AndroidBitmap_lockPixels(env, j_argb8888_bitmap, &j_bitmap_pixels) < 0) {
    LOGE("Invoke AndroidBitmap_lockPixels() failed!");
    return JNI_FALSE;
  }
  cv::Mat j_bitmap_im(static_cast<int>(j_bitmap_info.height),
                      static_cast<int>(j_bitmap_info.width), CV_8UC4,
                      j_bitmap_pixels);
  j_bitmap_im.copyTo(*(c_rgba));
  if (AndroidBitmap_unlockPixels(env, j_argb8888_bitmap) < 0) {
    LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean ARGB888Bitmap2BGR(JNIEnv *env, jobject j_argb8888_bitmap,
                           cv::Mat *c_bgr) {
  cv::Mat c_rgba;
  if (!ARGB888Bitmap2RGBA(env, j_argb8888_bitmap, &c_rgba)) {
    return JNI_FALSE;
  }
  cv::cvtColor(c_rgba, *(c_bgr), cv::COLOR_RGBA2BGR);
  return JNI_TRUE;
}

jboolean RGBA2ARGB888Bitmap(JNIEnv *env, jobject j_argb8888_bitmap,
                            const cv::Mat &c_rgba) {
  AndroidBitmapInfo j_bitmap_info;
  if (AndroidBitmap_getInfo(env, j_argb8888_bitmap, &j_bitmap_info) < 0) {
    LOGE("Invoke AndroidBitmap_getInfo() failed!");
    return JNI_FALSE;
  }
  void *j_bitmap_pixels;
  if (AndroidBitmap_lockPixels(env, j_argb8888_bitmap, &j_bitmap_pixels) < 0) {
    LOGE("Invoke AndroidBitmap_lockPixels() failed!");
    return JNI_FALSE;
  }
  cv::Mat j_bitmap_im(static_cast<int>(j_bitmap_info.height),
                      static_cast<int>(j_bitmap_info.width), CV_8UC4,
                      j_bitmap_pixels);
  c_rgba.copyTo(j_bitmap_im);
  if (AndroidBitmap_unlockPixels(env, j_argb8888_bitmap) < 0) {
    LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean BGR2ARGB888Bitmap(JNIEnv *env, jobject j_argb8888_bitmap,
                           const cv::Mat &c_bgr) {
  if (c_bgr.empty()) {
    return JNI_FALSE;
  }
  cv::Mat c_rgba;
  cv::cvtColor(c_bgr, c_rgba, cv::COLOR_BGR2RGBA);
  return RGBA2ARGB888Bitmap(env, j_argb8888_bitmap, c_rgba);
}

}  // namespace jni
}  // namespace fastdeploy
