PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_arcface_demo ${PROJECT_SOURCE_DIR}/infer_arcface.cc)
target_link_libraries(infer_arcface_demo ${FASTDEPLOY_LIBS})

add_executable(infer_cosface_demo ${PROJECT_SOURCE_DIR}/infer_cosface.cc)
target_link_libraries(infer_cosface_demo ${FASTDEPLOY_LIBS})

add_executable(infer_vpl_demo ${PROJECT_SOURCE_DIR}/infer_vpl.cc)
target_link_libraries(infer_vpl_demo ${FASTDEPLOY_LIBS})

add_executable(infer_partial_fc_demo ${PROJECT_SOURCE_DIR}/infer_partial_fc.cc)
target_link_libraries(infer_partial_fc_demo ${FASTDEPLOY_LIBS})
