PROJECT(preprocessor_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

add_executable(preprocessor_demo ${PROJECT_SOURCE_DIR}/main.cc)
target_link_libraries(preprocessor_demo ${FASTDEPLOY_LIBS})
