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

SET(OPENSSL_URL_PREFIX "https://bj.bcebos.com/paddlex/tools")
IF(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(OPENSSL_FILENAME "windows_openssl-1.1.0k")
  set(COMPRESSED_SUFFIX ".zip")
  add_definitions(-DWIN32)
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(OPENSSL_FILENAME "openssl-1.1.0k")
  set(COMPRESSED_SUFFIX ".tar.gz")
  add_definitions(-DLINUX)
ENDIF()
set(OPENSSL_URL ${OPENSSL_URL_PREFIX}/${OPENSSL_FILENAME}${COMPRESSED_SUFFIX})
if(THIRD_PARTY_PATH)
        SET(OPENSSL_INSTALL_DIR  ${THIRD_PARTY_PATH})
        SET(OPENSSL_ROOT_DIR ${THIRD_PARTY_PATH}/openssl-1.1.0k/install-${CMAKE_SYSTEM_PROCESSOR})
else()
        SET(OPENSSL_INSTALL_DIR  ${FASTDEPLOY_INSTALL_DIR}/installed_fastdeploy/cmake)
        SET(OPENSSL_ROOT_DIR ${FASTDEPLOY_INSTALL_DIR}/installed_fastdeploy/cmake/openssl-1.1.0k/install-${CMAKE_SYSTEM_PROCESSOR})
endif()
download_and_decompress(${OPENSSL_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENSSL_FILENAME}${COMPRESSED_SUFFIX} ${OPENSSL_INSTALL_DIR})
SET(OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include" CACHE PATH "openssl include directory." FORCE)
include_directories(${OPENSSL_INCLUDE_DIR})
IF(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(OPENSSL_LIBRARIES
                        "${OPENSSL_ROOT_DIR}/lib/libssl_static.lib"
                        "${OPENSSL_ROOT_DIR}/lib/libcrypto_static.lib"
                        ${GFLAGS_LIBRARIES}
                        shlwapi
                        CACHE FILEPATH "OPENSSL_LIBRARIES" FORCE)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(OPENSSL_LIBRARIES
                        "${OPENSSL_ROOT_DIR}/lib/libssl.a"
                        "${OPENSSL_ROOT_DIR}/lib/libcrypto.a"
                        ${GFLAGS_LIBRARIES}
                        -ldl -lpthread
                        CACHE FILEPATH "OPENSSL_LIBRARIES" FORCE)
ENDIF()