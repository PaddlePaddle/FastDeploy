# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
############################# Options for Android cross compiling #########################


# Options only for FastDeploy Android lib.
if(ANDROID)
  # These options are only support for Android now. Some options, such as 
  # WITH_OPENMP/WITH_JAVA/WITH_STATIC_LIB may export to main CMakeLists.txt
  # to support IOS/NON Android JAVA API/..., etc.
  option(WITH_ANDROID_OPENCV_STATIC "Whether to use OpenCV static lib for Android." OFF)
  option(WITH_ANDROID_FLYCV_STATIC "Whether to use FlyCV static lib for Android." OFF)
  option(WITH_ANDROID_LITE_STATIC "Whether to use Paddle Lite static lib for Android." OFF)
  option(WITH_ANDROID_OPENMP "Whether to use OpenMP support for Android." OFF)
  option(WITH_ANDROID_JAVA "Whether to build JNI lib for Android." OFF)
  option(WITH_ANDROID_STATIC_LIB "Whether to build FastDeploy static lib." OFF)
  option(WITH_ANDROID_TENSOR_FUNCS "Whether to build FastDeploy tensor function." ON)
else()
  message(FATAL_ERROR "WITH_ANDROID_xxx options only support for Android!")  
endif()

# Check Android ABI policy.
function(check_android_options_policy)
  if((NOT ANDROID_ABI MATCHES "armeabi-v7a") AND (NOT ANDROID_ABI MATCHES "arm64-v8a"))
    message(FATAL_ERROR "FastDeploy with FlyCV only support armeabi-v7a, arm64-v8a now.")
  endif()
  if(ENABLE_FLYCV OR ENABLE_TEXT OR ENABLE_LITE_BACKEND OR WITH_ANDROID_OPENMP)
    if(NOT ANDROID_TOOLCHAIN MATCHES "clang")
      message(FATAL_ERROR "Currently, only support clang toolchain while cross compiling FastDeploy for Android with Paddle Lite/FlyCV/FastTokenizer/OpenMP, but found ${ANDROID_TOOLCHAIN}.")
    endif()  
  endif()
  if(WITH_ANDROID_STATIC_LIB)
    message(STATUS "Found WITH_ANDROID_STATIC_LIB=ON:")
    if(ENABLE_LITE_BACKEND AND (NOT WITH_ANDROID_LITE_STATIC))
      set(WITH_ANDROID_LITE_STATIC ON CACHE BOOL "\tForce WITH_ANDROID_LITE_STATIC=ON" FORCE)
    endif()
    if(ENABLE_VISION AND (NOT WITH_ANDROID_OPENCV_STATIC))
      set(WITH_ANDROID_OPENCV_STATIC ON CACHE BOOL "\tForce WITH_ANDROID_LITE_STATIC=ON" FORCE)
    endif()
    if(ENABLE_FLYCV AND (NOT WITH_ANDROID_FLYCV_STATIC))
      set(WITH_ANDROID_FLYCV_STATIC ON CACHE BOOL "\tForce WITH_ANDROID_LITE_STATIC=ON" FORCE)
    endif()  
    if(ENABLE_TEXT)
      message(FATAL_ERROR "Not support to build FastDeploy static lib with Text API now!")
    endif()
  endif()
  # Add WITH_LITE_STATIC/WITH_ANDROID_LITE_STATIC compile definitions, see lite_backend.cc.
  if(WITH_ANDROID_LITE_STATIC)
    add_definitions(-DWITH_LITE_STATIC)
    add_definitions(-DWITH_ANDROID_LITE_STATIC)
  endif()
endfunction()

# Compile flags for FastDeploy Android lib.
function(set_android_cxx_complie_flags)
  if(ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g0 -Os -Ofast -ffast-math -ffunction-sections -fdata-sections" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0 -Os -Ofast -ffast-math -ffunction-sections -fdata-sections" PARENT_SCOPE)
  endif()
endfunction()

# Extra depend libs for FastDeploy Android lib (OMP&log lib).
function(set_android_openmp_compile_policy)
  if(ANDROID)
    find_library(log-lib log)
    list(APPEND DEPEND_LIBS ${log-lib})
    set(DEPEND_LIBS ${DEPEND_LIBS} PARENT_SCOPE)
    if(WITH_ANDROID_LITE_STATIC)
      # Need omp for static Paddle Lite lib
      set(WITH_ANDROID_OPENMP ON CACHE BOOL "Force WITH_ANDROID_OPENMP=ON while WITH_ANDROID_LITE_STATIC=ON" FORCE)
      message(STATUS "Force WITH_ANDROID_OPENMP=${WITH_ANDROID_OPENMP} while WITH_ANDROID_LITE_STATIC=ON")
    endif()
    if(WITH_ANDROID_OPENMP)
      include(${PROJECT_SOURCE_DIR}/cmake/openmp.cmake)
    endif()
  endif()
endfunction()

# Processing tensor function source for Android.
function(set_android_tensor_funcs_compile_policy)
  if(ANDROID AND (NOT WITH_ANDROID_TENSOR_FUNCS))
    if(ENABLE_VISION OR ENABLE_TEXT OR ENABLE_TRT_BACKEND)
      message(FATAL_ERROR "WITH_ANDROID_TENSOR_FUNCS must be set as ON, while ENABLE_VISION/ENABLE_TEXT/ENABLE_TRT_BACKEND is ON.")
    endif()
    file(GLOB_RECURSE DEPLOY_FUNCS_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/function/*.cc)
    list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_FUNCS_SRCS})
    set(ALL_DEPLOY_SRCS ${ALL_DEPLOY_SRCS} PARENT_SCOPE)
  endif()
endfunction()

# Link flags for FastDeploy Android lib.
function(set_android_library_cxx_link_flags)
  if(ANDROID)
    set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    # Strip debug C++ symbol table
    set(COMMON_LINK_FLAGS "-Wl,-exclude-libs,ALL")
    set(COMMON_LINK_FLAGS_REL "-Wl,-s,--gc-sections,-exclude-libs,ALL")
    if(WITH_ANDROID_OPENCV_STATIC OR WITH_ANDROID_LITE_STATIC)
      set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS},--allow-multiple-definition" CACHE INTERNAL "" FORCE)
      set(COMMON_LINK_FLAGS_REL "${COMMON_LINK_FLAGS_REL},--allow-multiple-definition" CACHE INTERNAL "" FORCE)
    endif()
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_RELEASE ${COMMON_LINK_FLAGS_REL})
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_MINSIZEREL ${COMMON_LINK_FLAGS_REL})
  endif()
endfunction()

# FastDeploy Android JNI lib & FastDeploy static lib.
function(set_android_extra_libraries_target)
  if(ANDROID AND WITH_ANDROID_JAVA)
    include(${PROJECT_SOURCE_DIR}/cmake/fastdeploy_jni.cmake)
  endif()

  if(ANDROID AND WITH_ANDROID_STATIC_LIB)
  # Here, we use a dummy target (fastdelpoy_dummy)
  # to form a build dependency tree for fastdeploy_static lib.
  add_library(fastdelpoy_dummy STATIC ${ALL_DEPLOY_SRCS})
  # Still add ${DEPEND_LIBS} for cmake to form link_libraries
  # property tree for a static library.
  target_link_libraries(fastdelpoy_dummy ${DEPEND_LIBS})
  # Build fastdelpoy_dummy when the third-party
  # libraries (opencv, paddle lite, flycv) are ready.
  add_dependencies(fastdelpoy_dummy ${LIBRARY_NAME})
  # Add WITH_STATIC_LIB/WITH_ANDROID_STATIC_LIB compile definitions, see lite_backend.cc.
  target_compile_definitions(fastdelpoy_dummy PRIVATE WITH_STATIC_LIB WITH_ANDROID_STATIC_LIB)
  target_compile_definitions(fastdelpoy_dummy PRIVATE WITH_STATIC_LIB_AT_COMPILING 
                                                      WITH_ANDROID_STATIC_LIB_AT_COMPILING)
  bundle_static_library(fastdelpoy_dummy fastdeploy_static bundle_fastdeploy)
  endif()
endfunction()

# Install FastDepploy Android lib.
function(set_android_libraries_installation)
  if(ANDROID)
    if(WITH_ANDROID_STATIC_LIB)
    install(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/libfastdeploy_static.a
      DESTINATION lib/${ANDROID_ABI}
    )
    else()
    install(
      TARGETS ${LIBRARY_NAME}
      LIBRARY DESTINATION lib/${ANDROID_ABI}
    )
    endif()
    # Install omp into fastdeploy lib dir if WITH_OPENMP=ON
    # and WITH_LITE_STATIC=OFF.
    if(WITH_ANDROID_OPENMP AND (NOT WITH_ANDROID_LITE_STATIC) 
        AND OpenMP_CXX_FOUND AND ENABLE_OPENMP_SHARED)
    install(
      FILES
      ${OpenMP_CXX_LIBRARIES}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${ANDROID_ABI}
    )
    endif()
    # install Android JNI lib
    if(WITH_ANDROID_JAVA)
    install(
      TARGETS fastdeploy_jni
      LIBRARY DESTINATION jni/${ANDROID_ABI}
    )
    endif()
  endif()
endfunction()

# Install third_libs
function(set_android_third_libs_installation)
  if(ANDROID)
    # opencv/flycv always needs to be provided to users because our api
    # explicitly depends on opencv's and flycv's api in headers.
    # The headers and libs of opencv must be install.
    if(ENABLE_VISION)
      if(WITH_ANDROID_OPENCV_STATIC AND WITH_ANDROID_STATIC_LIB)
        # Only need to install headers while building
        # FastDeploy static lib. (TODO:qiuyanjun)
        install(
          DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/opencv/sdk/native/jni/include
          DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install/opencv/sdk/native/jni
        )
      else()
        install(
          DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/opencv
          DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install
        )
      endif()
      # Only need flycv's headers (may also install libs? TODO:qiuyanjun)
      if(ENABLE_FLYCV)
        if(WITH_ANDROID_FLYCV_STATIC)
          install(
            DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/flycv/include
            DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install/flycv
          )
        else()
          install(
            DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/flycv
            DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install
          )
        endif()
      endif()
    endif(ENABLE_VISION)
    # fast_tokenizer's static lib is not avaliable now!
    # may support some days later(TODO:qiuyanjun)
    if(ENABLE_TEXT)
      install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/fast_tokenizer
        DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install
      )
    endif()
    # Some libs may not to install while in static mode
    if(ENABLE_LITE_BACKEND)
      if(WITH_ANDROID_LITE_STATIC)
        if(WITH_ANDROID_STATIC_LIB)
          install(
            DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/paddlelite/include
            DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install/paddlelite
          )
        endif()
      else()
        install(
          DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/paddlelite
          DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs/install
        )
      endif()
    endif()
  endif()
endfunction()