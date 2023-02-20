cmake_minimum_required(VERSION 3.10)
project(silero_vad)

set(CMAKE_CXX_STANDARD 11)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_onnx_silero_vad ${PROJECT_SOURCE_DIR}/infer_onnx_silero_vad.cc wav.h vad.cc vad.h)

# 添加FastDeploy库依赖
target_link_libraries(infer_onnx_silero_vad ${FASTDEPLOY_LIBS})
