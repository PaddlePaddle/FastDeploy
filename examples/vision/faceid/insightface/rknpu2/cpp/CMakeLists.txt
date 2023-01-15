PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

add_executable(infer_arcface_demo ${PROJECT_SOURCE_DIR}/infer_arcface.cc)
target_link_libraries(infer_arcface_demo ${FASTDEPLOY_LIBS})
