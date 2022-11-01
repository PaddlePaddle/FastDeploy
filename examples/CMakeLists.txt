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

function(get_fastdeploy_example_names NAME_SPACE CLASS_NAME CC_FILE)
  get_filename_component(CPP_DIR ${CC_FILE} DIRECTORY)
  get_filename_component(CLASS_DIR ${CPP_DIR} DIRECTORY)
  get_filename_component(NAME_SPACE_DIR ${CLASS_DIR} DIRECTORY)
  get_filename_component(_CLASS_NAME ${CLASS_DIR} NAME)
  get_filename_component(_NAME_SPACE ${NAME_SPACE_DIR} NAME)
  set(${NAME_SPACE} ${_NAME_SPACE} PARENT_SCOPE)
  set(${CLASS_NAME} ${_CLASS_NAME} PARENT_SCOPE)
endfunction()

function(config_fastdeploy_executable_link_flags TARGET_NAME)
  if(ANDROID)
    # some specific flags for Android.
    set(COMMON_LINK_FLAGS "-Wl,-exclude-libs,ALL")
    set(COMMON_LINK_FLAGS_REL "-Wl,-s,--gc-sections,-exclude-libs,ALL")
    if(WITH_OPENCV_STATIC OR WITH_LITE_STATIC)
      set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS},--allow-multiple-definition")
      set(COMMON_LINK_FLAGS_REL "${COMMON_LINK_FLAGS_REL},--allow-multiple-definition")
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE ${COMMON_LINK_FLAGS_REL})
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_MINSIZEREL ${COMMON_LINK_FLAGS_REL})
  endif()
endfunction()

set(EXAMPLES_NUM 0)
function(add_fastdeploy_executable FIELD CC_FILE)
  # temp target name/file var in function scope
  set(TEMP_TARGET_FILE ${CC_FILE})
  get_filename_component(FILE_NAME ${CC_FILE} NAME)
  string(REGEX REPLACE ".cc" "" FILE_NAME ${FILE_NAME})
  get_fastdeploy_example_names(NAME_SPACE CLASS_NAME ${CC_FILE})
  set(TEMP_TARGET_NAME ${FIELD}_${NAME_SPACE}_${CLASS_NAME}_${FILE_NAME})
  if(EXISTS ${TEMP_TARGET_FILE} AND TARGET fastdeploy)
    add_executable(${TEMP_TARGET_NAME} ${TEMP_TARGET_FILE})
    target_link_libraries(${TEMP_TARGET_NAME} PUBLIC fastdeploy)
    if(TARGET gflags)
      target_link_libraries(${TEMP_TARGET_NAME} PRIVATE gflags pthread)
    endif()
    config_fastdeploy_executable_link_flags(${TEMP_TARGET_NAME})
    math(EXPR _EXAMPLES_NUM "${EXAMPLES_NUM} + 1")
    set(EXAMPLES_NUM ${_EXAMPLES_NUM} PARENT_SCOPE)
    string(LENGTH ${EXAMPLES_NUM} len)
    set(MESSAGE_HEAD "[${EXAMPLES_NUM}]")
    if(${len} EQUAL 1)
      set(MESSAGE_HEAD "[00${EXAMPLES_NUM}]")
    elseif(${len} EQUAL 2)
      set(MESSAGE_HEAD "[0${EXAMPLES_NUM}]")
    endif()
    message(STATUS "  ${MESSAGE_HEAD} Added FastDeploy Executable   : ${TEMP_TARGET_NAME}")
  endif()
  unset(TEMP_TARGET_FILE)
  unset(TEMP_TARGET_NAME)
endfunction()

# vision examples
if(BUILD_EXAMPLES AND ENABLE_VISION)
  if(NOT ENABLE_VISION_VISUALIZE)
    message(FATAL_ERROR "ENABLE_VISION_VISUALIZE must be ON while BUILD_EXAMPLES and ENABLE_VISION both ON.")
  endif()
  if(EXISTS ${PROJECT_SOURCE_DIR}/examples/vision)
    message(STATUS "")
    message(STATUS "*************FastDeploy Vision Examples Summary**********")
    file(GLOB_RECURSE ALL_VISION_EXAMPLE_SRCS ${PROJECT_SOURCE_DIR}/examples/vision/*/*/cpp/*.cc)
    foreach(_CC_FILE ${ALL_VISION_EXAMPLE_SRCS})
      add_fastdeploy_executable(vision ${_CC_FILE})
    endforeach()
    message(STATUS "  [FastDeploy Executable Path]        : ${EXECUTABLE_OUTPUT_PATH}")
  endif()
endif()

# text examples
if(BUILD_EXAMPLES AND ENABLE_TEXT)
  if(EXISTS ${PROJECT_SOURCE_DIR}/examples/text)
    message(STATUS "")
    message(STATUS "*************FastDeploy Text Examples Summary**********")
    file(GLOB_RECURSE ALL_TEXT_EXAMPLE_SRCS ${PROJECT_SOURCE_DIR}/examples/text/*.cc)
    foreach(_CC_FILE ${ALL_TEXT_EXAMPLE_SRCS})
      add_fastdeploy_executable(text ${_CC_FILE})
    endforeach()
    message(STATUS "  [FastDeploy Executable Path]        : ${EXECUTABLE_OUTPUT_PATH}")
  endif()
endif()

# other examples ...
