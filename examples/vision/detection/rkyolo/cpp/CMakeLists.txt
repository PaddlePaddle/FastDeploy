CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(rknpu2_test)

set(CMAKE_CXX_STANDARD 14)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_rkyolov5 infer_rkyolov5.cc)
target_link_libraries(infer_rkyolov5 ${FASTDEPLOY_LIBS})

add_executable(infer_rkyolov7 infer_rkyolov7.cc)
target_link_libraries(infer_rkyolov7 ${FASTDEPLOY_LIBS})

add_executable(infer_rkyolox infer_rkyolox.cc)
target_link_libraries(infer_rkyolox ${FASTDEPLOY_LIBS})
