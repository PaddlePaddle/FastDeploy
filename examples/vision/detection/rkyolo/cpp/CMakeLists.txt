CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(rknpu2_test)

set(CMAKE_CXX_STANDARD 14)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
add_executable(infer_rkyolo infer_rkyolo.cc)
target_link_libraries(infer_rkyolo ${FASTDEPLOY_LIBS})
