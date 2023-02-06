PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Specifies the path to the fastdeploy library after you have downloaded it
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(../../../../../FastDeploy.cmake)

# Add the FastDeploy dependency header
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
# Add the FastDeploy library dependency
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
