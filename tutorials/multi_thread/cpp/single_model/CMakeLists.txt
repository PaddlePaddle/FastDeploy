PROJECT(multi_thread_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(multi_thread_demo ${PROJECT_SOURCE_DIR}/multi_thread.cc)
# 添加FastDeploy库依赖
target_link_libraries(multi_thread_demo ${FASTDEPLOY_LIBS} pthread)
