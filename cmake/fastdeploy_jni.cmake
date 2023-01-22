# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(WITH_JAVA)
  if(NOT ANDROID)
    message(FATAL_ERROR "Only support jni lib for Android now!")
  else()
    set(JNI_SRCS_DIR "${PROJECT_SOURCE_DIR}/java/android/fastdeploy/src/main/cpp")
    include_directories(${JNI_SRCS_DIR})
    file(GLOB JNI_SRCS ${JNI_SRCS_DIR}/fastdeploy_jni/*.cc)
    file(GLOB JNI_VISION_SRCS ${JNI_SRCS_DIR}/fastdeploy_jni/vision/*.cc)
    file(GLOB JNI_PIPELINE_SRCS ${JNI_SRCS_DIR}/fastdeploy_jni/pipeline/*.cc)
    file(GLOB JNI_TEXT_SRCS ${JNI_SRCS_DIR}/fastdeploy_jni/text/*.cc)
    list(APPEND JNI_SRCS ${JNI_VISION_SRCS} ${JNI_PIPELINE_SRCS} ${JNI_TEXT_SRCS})
    set(JNI_SRCS_Found TRUE CACHE BOOL "JNI SRCS Flags" FORCE)
  endif()
endif()
