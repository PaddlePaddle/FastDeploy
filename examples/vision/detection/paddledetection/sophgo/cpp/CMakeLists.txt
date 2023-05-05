PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)
# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

set(ENABLE_LITE_BACKEND OFF)
#set(FDLIB ${FASTDEPLOY_INSTALL_DIR})

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
include_directories(${FastDeploy_INCLUDE_DIRS})

add_executable(infer_ppyoloe ${PROJECT_SOURCE_DIR}/infer_ppyoloe.cc)
add_executable(infer_ppyoloe_r ${PROJECT_SOURCE_DIR}/infer_ppyoloe_r.cc)
add_executable(infer_picodet ${PROJECT_SOURCE_DIR}/infer_picodet.cc)
add_executable(infer_yolov8 ${PROJECT_SOURCE_DIR}/infer_yolov8.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_ppyoloe ${FASTDEPLOY_LIBS})
target_link_libraries(infer_ppyoloe_r ${FASTDEPLOY_LIBS})
target_link_libraries(infer_picodet ${FASTDEPLOY_LIBS})
target_link_libraries(infer_yolov8 ${FASTDEPLOY_LIBS})
