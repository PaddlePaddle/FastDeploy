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

function(cc_test_build TARGET_NAME)
  if(WITH_TESTING)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bin)
    if(WIN32)
      if("${cc_test_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_test_DEPS python)
        target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
      endif()
      set(EXTERNAL_LIB "")
    else(WIN32)
      set(EXTERNAL_LIB "-lrt -ldl -lpthread")
    endif(WIN32)
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)

    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} fastdeploy_gtest_main gtest glog ${EXTERNAL_LIB})
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} gtest)
  endif()
endfunction()

function(cc_test TARGET_NAME)
  if(WITH_TESTING)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cc_test_build(${TARGET_NAME}
      SRCS ${cc_test_SRCS}
      DEPS ${cc_test_DEPS})
    add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_COMMAND})
  endif()
endfunction(cc_test)

function(add_fastdeploy_unittest CC_FILE)
  set(TEMP_TARGET_FILE ${CC_FILE})
  string(REGEX MATCHALL "[0-9A-Za-z_]*.cc" FILE_NAME ${CC_FILE})
  string(REGEX REPLACE ".cc" "" FILE_PREFIX ${FILE_NAME})
  set(TEMP_TARGET_NAME ${FILE_PREFIX})
  if (EXISTS ${TEMP_TARGET_FILE} AND TARGET fastdeploy)
    cc_test(${TEMP_TARGET_NAME} SRCS ${TEMP_TARGET_FILE} DEPS fastdeploy)
    message(STATUS "  Added FastDeploy unittest       : ${TEMP_TARGET_NAME}")
  endif()
  unset(TEMP_TARGET_FILE)
  unset(TEMP_TARGET_NAME)
endfunction()

if(WITH_TESTING)
  add_library(fastdeploy_gtest_main STATIC gtest_main)
  target_link_libraries(fastdeploy_gtest_main PUBLIC gtest gflags)
  message(STATUS "")
  message(STATUS "*************FastDeploy Unittest Summary**********")
  file(GLOB ALL_TEST_SRCS ${PROJECT_SOURCE_DIR}/tests/test_*.cc)
  foreach(_CC_FILE ${ALL_TEST_SRCS})
    add_fastdeploy_unittest(${_CC_FILE})
  endforeach()
endif()
