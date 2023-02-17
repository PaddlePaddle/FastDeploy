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

##################################### Building: FastDeploy C# API #######################################
PROJECT(fastdeploy_csharp CSharp)

CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(ENABLE_VISION "Whether to enable vision models usage." OFF)

message("fastdeploy_csharp_SOURCE_DIR: ${fastdeploy_csharp_SOURCE_DIR}")
file(GLOB_RECURSE DEPLOY_CSHARPAPI_SRCS ${fastdeploy_csharp_SOURCE_DIR}/fastdeploy/*.cs)
if(NOT ENABLE_VISION)
    file(GLOB_RECURSE DEPLOY_VISION_CSHARPAPI_SRCS ${fastdeploy_csharp_SOURCE_DIR}/fastdeploy/vision/*.cs)
    list(REMOVE_ITEM DEPLOY_CSHARPAPI_SRCS ${DEPLOY_VISION_CSHARPAPI_SRCS})
endif()

# Define the DLL target, including all relevant project files.
add_library(${PROJECT_NAME} SHARED ${DEPLOY_CSHARPAPI_SRCS})

# Set the C# language version (defaults to 3.0 if not set).
set(CMAKE_CSharp_FLAGS "/langversion:10")
# Add in some .NET reference libraries.
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_REFERENCES
    "Microsoft.CSharp"
    "System"
    "System.Core"
    "System.Data"
    "System.Deployment"
    "System.Drawing"
    "System.Net.Http"
    "System.Xml"
)


set_property(TARGET ${PROJECT_NAME}
    PROPERTY VS_PACKAGE_REFERENCES  "OpenCvSharp4_4.7.0.20230115"
)

##################################### Installing: FastDeploy C# API #######################################

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION csharp_lib
    ARCHIVE DESTINATION csharp_lib
    RUNTIME DESTINATION csharp_lib
)
