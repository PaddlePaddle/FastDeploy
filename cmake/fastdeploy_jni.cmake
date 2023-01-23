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
  set(JNI_SRCS_FOUND OFF)
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
    set(JNI_SRCS_FOUND ON CACHE BOOL "JNI SRCS Flags" FORCE)
  endif()
  if(JNI_SRCS_FOUND)
    # Here, we use a dummy target (fastdeploy_jni_bundled) 
    # to form a build dependency tree for fastdeploy_jni lib.
    add_library(fastdeploy_jni_bundled STATIC ${ALL_DEPLOY_SRCS})
    add_library(fastdeploy_jni SHARED ${JNI_SRCS})
    target_link_libraries(fastdeploy_jni fastdeploy_jni_bundled ${DEPEND_LIBS} 
                          jnigraphics GLESv2 EGL)
    add_dependencies(fastdeploy_jni fastdeploy_jni_bundled)  
    # Build fastdeploy_jni_bundled when the third-party 
    # libraries (opencv, paddle lite, flycv) are ready.
    add_dependencies(fastdeploy_jni_bundled ${LIBRARY_NAME})
    # Strip debug C++ symbol table
    set_target_properties(fastdeploy_jni PROPERTIES COMPILE_FLAGS 
        "-fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections")
    set_target_properties(fastdeploy_jni PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
    set_target_properties(fastdeploy_jni PROPERTIES LINK_FLAGS_RELEASE ${COMMON_LINK_FLAGS_REL})
    set_target_properties(fastdeploy_jni PROPERTIES LINK_FLAGS_MINSIZEREL ${COMMON_LINK_FLAGS_REL}) 
  else()
    message(FATAL_ERROR "Can not found Android JNI_SRCS!")  
  endif()
endif()
