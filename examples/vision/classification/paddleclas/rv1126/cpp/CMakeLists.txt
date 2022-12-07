PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
include_directories(${FastDeploy_INCLUDE_DIRS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/install)

install(TARGETS infer_demo DESTINATION ./)

install(DIRECTORY models DESTINATION ./)
install(DIRECTORY images DESTINATION ./)
# install(DIRECTORY run_with_adb.sh DESTINATION ./)

file(GLOB FASTDEPLOY_LIBS ${FASTDEPLOY_INSTALL_DIR}/lib/*)
install(PROGRAMS ${FASTDEPLOY_LIBS} DESTINATION lib)

file(GLOB OPENCV_LIBS ${FASTDEPLOY_INSTALL_DIR}/third_libs/install/opencv/lib/lib*)
install(PROGRAMS ${OPENCV_LIBS} DESTINATION lib)

file(GLOB PADDLELITE_LIBS ${FASTDEPLOY_INSTALL_DIR}/third_libs/install/paddlelite/lib/lib*)
install(PROGRAMS ${PADDLELITE_LIBS} DESTINATION lib)

file(GLOB TIMVX_LIBS ${FASTDEPLOY_INSTALL_DIR}/third_libs/install/paddlelite/lib/verisilicon_timvx/*)
install(PROGRAMS ${TIMVX_LIBS} DESTINATION lib)

file(GLOB ADB_TOOLS run_with_adb.sh)
install(PROGRAMS ${ADB_TOOLS} DESTINATION ./)
