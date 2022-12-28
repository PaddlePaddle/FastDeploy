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
function(download_and_extract url filename tar_suffix dirname)
        SET(tar_name "${filename}${tar_suffix}")
        SET(tar_file "${dirname}/${tar_name}")
        SET(lib_name "${dirname}/${filename}")
        if( (NOT EXISTS ${tar_file}) AND (NOT EXISTS ${lib_name}) )
                file(DOWNLOAD ${url} ${tar_file} SHOW_PROGRESS STATUS ERR)
                IF(ERR EQUAL 0)
                        message(STATUS "download ${tar_name} success")
                ELSE()
                        message(STATUS "download ${tar_name} failed")
                ENDIF()
        endif()

        if(EXISTS ${lib_name})
                message(STATUS "${lib_name} is already exist.")
        elseif(NOT EXISTS ${tar_file})
                message(STATUS "download ${tar_name} failed, please check it.")
        else()
                execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xzvf ${tar_name}
                        WORKING_DIRECTORY ${dirname} RESULT_VARIABLE tar_result)
                if (tar_result MATCHES 0)
                        message(STATUS "extract ${lib_name} success")
                else()
                        message(STATUS "extract ${lib_name} failed")
                endif()
        endif()
        file(REMOVE_RECURSE ${tar_file})
endfunction()

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        if(THIRD_PARTY_PATH)
                SET(OPENSSL_PREFIX_DIR  ${THIRD_PARTY_PATH})
                SET(OPENSSL_INSTALL_DIR ${THIRD_PARTY_PATH}/openssl-1.1.0k/install-${CMAKE_SYSTEM_PROCESSOR})
        else()
        # For example cmake
                SET(OPENSSL_PREFIX_DIR  ${FASTDEPLOY_INSTALL_DIR}/installed_fastdeploy/cmake)
                SET(OPENSSL_INSTALL_DIR ${FASTDEPLOY_INSTALL_DIR}/installed_fastdeploy/cmake/openssl-1.1.0k/install-${CMAKE_SYSTEM_PROCESSOR})
        endif()
	download_and_extract("https://bj.bcebos.com/paddlex/tools/openssl-1.1.0k.tar.gz" "openssl-1.1.0k" ".tar.gz" ${OPENSSL_PREFIX_DIR})
        SET(OPENSSL_INCLUDE_DIR "${OPENSSL_INSTALL_DIR}/include" CACHE PATH "openssl include directory." FORCE)
	include_directories(${OPENSSL_INCLUDE_DIR})
        set(OPENSSL_LIBRARIES
                "${OPENSSL_INSTALL_DIR}/lib/libssl.a"
                "${OPENSSL_INSTALL_DIR}/lib/libcrypto.a"
                ${GFLAGS_LIBRARIES}
                -ldl -lpthread
                CACHE FILEPATH "OPENSSL_LIBRARIES" FORCE)
ENDIF ()
