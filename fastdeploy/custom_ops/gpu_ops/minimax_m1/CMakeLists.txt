# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
project(minimax_m1_custom_ops)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    lightning_attention.cpp
)

# Create shared library
add_library(minimax_m1_ops SHARED ${SOURCES})

# Set CUDA properties
set_target_properties(minimax_m1_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
)

# Link libraries
target_link_libraries(minimax_m1_ops
    ${CUDA_LIBRARIES}
    cublas
    curand
)

# Install targets
install(TARGETS minimax_m1_ops
    LIBRARY DESTINATION fastdeploy/custom_ops/gpu_ops/minimax_m1
    ARCHIVE DESTINATION fastdeploy/custom_ops/gpu_ops/minimax_m1
)

# Install Python bindings
install(FILES __init__.py
    DESTINATION fastdeploy/custom_ops/gpu_ops/minimax_m1
)

install(FILES lightning_attention.h
    DESTINATION fastdeploy/custom_ops/gpu_ops/minimax_m1/include
)