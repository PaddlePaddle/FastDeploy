PROJECT(fastdeploy_ppshitu C CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# Options for PP-ShiTu
option(WITH_FAISS_STATIC "Whether to build PP-ShiTu with Faiss static lib" ON)
option(WITH_FAISS_GPU "Whether to build PP-ShiTu with Faiss gpu lib" OFF)

set(THIRD_LIBS_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)

# Path to FastDeploy_DIR
set(FastDeploy_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fastdeploy-linux-x64-gpu-0.0.0)
find_package(FastDeploy REQUIRED)
if(FastDeploy_FOUND) 
  message(STATUS "Found FastDeploy: ${FastDeploy_FOUND} ")
  message(STATUS "Found FastDeploy_INCLUDE_DIRS: ${FastDeploy_INCLUDE_DIRS} ")
  message(STATUS "Found FastDeploy_LIBS: ${FastDeploy_LIBS} ")
endif()

# Add Faiss lib for vector search
include(cmake/faiss.cmake)
if(FAISS_FOUND)
  message(STATUS "Found FAISS: ${FAISS_FOUND}")
  message(STATUS "Found FAISS_INCLUDE_DIRS: ${FAISS_INCLUDE_DIRS}")
  message(STATUS "Found FAISS_LIBS: ${FAISS_LIBS}")
endif()

# Add FastDeploy & FAISS & PP-ShiTu headers
include_directories(${FastDeploy_INCLUDE_DIRS})
include_directories(${FAISS_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add PP-ShiTu srcs
file(GLOB PP_ShiTu_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

# Compile libfastdeploy_ppshitu.so
add_library(fastdeploy_ppshitu SHARED ${PP_ShiTu_SRCS})
target_link_libraries(fastdeploy_ppshitu ${FAISS_LIBS} ${FastDeploy_LIBS})

# Compile infer demo
add_executable(infer_demo ${CMAKE_CURRENT_SOURCE_DIR}/infer.cc)
target_link_libraries(infer_demo fastdeploy_ppshitu)

