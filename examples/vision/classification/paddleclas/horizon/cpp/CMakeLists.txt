CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(sunrise_test)

set(CMAKE_CXX_STANDARD 14)

# 指定下载解压后的fastdeploy库路径
set(FASTDEPLOY_INSTALL_DIR "/home/xuhao/Hackathon/FastDeploy/build/fastdeploy-0.0.0")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeployConfig.cmake)
include_directories(${FastDeploy_INCLUDE_DIRS})
add_executable(sunrise_test infer.cc)
target_link_libraries(sunrise_test
        ${FastDeploy_LIBS} pthread rt dl
        )

