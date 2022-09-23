PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

find_package(Threads)

if(WIN32)
    add_definitions(/DGOOGLE_GLOG_DLL_DECL=)
endif()
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/gflags.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS} ${GFLAGS_INCLUDE_DIR})

add_executable(infer_ppyoloe_demo ${PROJECT_SOURCE_DIR}/infer_ppyoloe.cc)
target_link_libraries(infer_ppyoloe_demo ${FASTDEPLOY_LIBS} gflags ${CMAKE_THREAD_LIBS_INIT} Shlwapi.lib)
