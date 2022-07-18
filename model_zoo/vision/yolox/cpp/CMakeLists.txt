PROJECT(yolox_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

# 在低版本ABI环境中，通过如下代码进行兼容性编译
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 指定下载解压后的fastdeploy库路径
set(FASTDEPLOY_INSTALL_DIR ${PROJECT_SOURCE_DIR}/fastdeploy-linux-x64-0.0.3/)

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(yolox_demo ${PROJECT_SOURCE_DIR}/yolox.cc)
# 添加FastDeploy库依赖
target_link_libraries(yolox_demo ${FASTDEPLOY_LIBS})
