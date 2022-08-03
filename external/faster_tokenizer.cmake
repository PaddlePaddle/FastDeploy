

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
include(ExternalProject)

set(FASTERTOKENIZER_PROJECT "extern_faster_tokenizer")
set(FASTERTOKENIZER_PREFIX_DIR ${THIRD_PARTY_PATH}/faster_tokenizer)
set(FASTERTOKENIZER_SOURCE_DIR
    ${THIRD_PARTY_PATH}/faster_tokenizer/src/${FASTERTOKENIZER_PROJECT})
set(FASTERTOKENIZER_INSTALL_DIR ${THIRD_PARTY_PATH}/install/faster_tokenizer)
set(FASTERTOKENIZER_INC_DIR
    "${FASTERTOKENIZER_INSTALL_DIR}/faster_tokenizer/include"
    CACHE PATH "faster_tokenizer include directory." FORCE)
set(FASTERTOKENIZER_LIB_DIR
    "${FASTERTOKENIZER_INSTALL_DIR}/faster_tokenizer/lib/"
    CACHE PATH "faster_tokenizer lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${FASTERTOKENIZER_LIB_DIR}")

include_directories(${FASTERTOKENIZER_INC_DIR})

# Set lib path
if(WIN32)
elseif(APPLE)
# Not support apple so far.
else()

set(FASTERTOKENIZER_COMPILE_LIB
"${FASTERTOKENIZER_INSTALL_DIR}/paddle/lib/libcore_tokenizers.so"
CACHE FILEPATH "faster_tokenizer compile library." FORCE)
set(ICUDT_LIB "")
set(ICUUC_LIB "")
endif(WIN32)

set(FASTERTOKENIZER_URL_BASE "https://bj.bcebos.com/paddlenlp/faster_tokenizer/")
set(FASTERTOKENIZER_VERSION "dev")

# Set download url
if(WIN32)
elseif(APPLE)
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(FASTERTOKENIZER_FILE "faster_tokenizer-linux-aarch64-${FASTERTOKENIZER_VERSION}.tgz")
  else()
    set(FASTERTOKENIZER_FILE "faster_tokenizer-linux-x64-${FASTERTOKENIZER_VERSION}.tgz")
  endif()
endif()
set(FASTERTOKENIZER_URL "${FASTERTOKENIZER_URL_BASE}${FASTERTOKENIZER_FILE}")

ExternalProject_Add(
  ${FASTERTOKENIZER_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${FASTERTOKENIZER_URL}
  PREFIX ${FASTERTOKENIZER_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${FASTERTOKENIZER_SOURCE_DIR} ${FASTERTOKENIZER_INSTALL_DIR}
  BUILD_BYPRODUCTS ${FASTERTOKENIZER_COMPILE_LIB})

add_library(faster_tokenizer SHARED IMPORTED GLOBAL)
set_property(TARGET extern_faster_tokenizer PROPERTY IMPORTED_LOCATION ${FASTERTOKENIZER_COMPILE_LIB})
add_dependencies(faster_tokenizer ${FASTERTOKENIZER_PROJECT})