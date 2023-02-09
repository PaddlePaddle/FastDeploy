PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# specify the decompress directory of FastDeploy SDK
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
include(${FASTDEPLOY_INSTALL_DIR}/utils/gflags.cmake)
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

add_executable(benchmark_yolov5 ${PROJECT_SOURCE_DIR}/benchmark_yolov5.cc)
add_executable(benchmark_ppyolov8 ${PROJECT_SOURCE_DIR}/benchmark_ppyolov8.cc)

if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
  target_link_libraries(benchmark_yolov5 ${FASTDEPLOY_LIBS} gflags pthread)
  target_link_libraries(benchmark_ppyolov8 ${FASTDEPLOY_LIBS} gflags pthread)
else()
  target_link_libraries(benchmark_yolov5 ${FASTDEPLOY_LIBS} gflags)
  target_link_libraries(benchmark_ppyolov8 ${FASTDEPLOY_LIBS} gflags)
endif()
