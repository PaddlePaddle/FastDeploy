PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

add_executable(infer_ppshituv2_det_demo ${PROJECT_SOURCE_DIR}/infer_ppshituv2_det.cc)
add_executable(infer_ppshituv2_rec_demo ${PROJECT_SOURCE_DIR}/infer_ppshituv2_rec.cc)

target_link_libraries(infer_ppshituv2_det_demo ${FASTDEPLOY_LIBS})
target_link_libraries(infer_ppshituv2_rec_demo ${FASTDEPLOY_LIBS})
