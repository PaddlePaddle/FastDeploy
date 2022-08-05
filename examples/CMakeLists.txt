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

function(add_fastdeploy_executable FIELD CC_FILE)
  # temp target name/file var in function scope
  set(TEMP_TARGET_FILE ${CC_FILE})
  string(REGEX MATCHALL "[0-9A-Za-z_]*.cc" FILE_NAME ${CC_FILE})
  string(REGEX REPLACE ".cc" "" FILE_PREFIX ${FILE_NAME})
  set(TEMP_TARGET_NAME ${FIELD}_${FILE_PREFIX})
  if (EXISTS ${TEMP_TARGET_FILE} AND TARGET fastdeploy)
    add_executable(${TEMP_TARGET_NAME} ${TEMP_TARGET_FILE})
    target_link_libraries(${TEMP_TARGET_NAME} PUBLIC fastdeploy)
    message(STATUS "  Added FastDeploy Executable       : ${TEMP_TARGET_NAME}")
  endif()
  unset(TEMP_TARGET_FILE)
  unset(TEMP_TARGET_NAME)
endfunction()

# vision examples
if(WITH_VISION_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples/vision)
  message(STATUS "")
  message(STATUS "*************FastDeploy Examples Summary**********")
  file(GLOB ALL_VISION_EXAMPLE_SRCS ${PROJECT_SOURCE_DIR}/examples/vision/*.cc)
  foreach(_CC_FILE ${ALL_VISION_EXAMPLE_SRCS})
    add_fastdeploy_executable(vision ${_CC_FILE})
  endforeach()
endif()

# text examples
if(WITH_TEXT_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples/text)
  message(STATUS "")
  message(STATUS "*************FastDeploy Examples Summary**********")
  file(GLOB ALL_VISION_EXAMPLE_SRCS ${PROJECT_SOURCE_DIR}/examples/text/*.cc)
  foreach(_CC_FILE ${ALL_VISION_EXAMPLE_SRCS})
    add_fastdeploy_executable(text ${_CC_FILE})
  endforeach()
endif()

# other examples ...
