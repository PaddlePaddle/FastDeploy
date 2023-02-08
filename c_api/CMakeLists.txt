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

##################################### Building: FastDeploy C API #######################################
message("----start--CAPI-------")

if(NOT WITH_CAPI)
  return()
endif()

file(GLOB_RECURSE DEPLOY_CAPI_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/c_api/fastdeploy_capi/*.cc)
if(NOT ENABLE_VISION)
    file(GLOB_RECURSE DEPLOY_VISION_CAPI_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/c_api/fastdeploy_capi/vision/*.cc)
    list(REMOVE_ITEM DEPLOY_CAPI_SRCS ${DEPLOY_VISION_CAPI_SRCS})
endif()
list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_CAPI_SRCS})
include_directories(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/c_api)
