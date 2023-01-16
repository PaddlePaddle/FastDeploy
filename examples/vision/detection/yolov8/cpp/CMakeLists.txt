PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Specify the fastdeploy library path after downloading and decompression
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# Add FastDeploy dependent header files
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
# Add FastDeploy library dependencies
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
