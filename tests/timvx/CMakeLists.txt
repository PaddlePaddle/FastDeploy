PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
include_directories(${FastDeploy_INCLUDE_DIRS})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/timvx_tests)

# add test for yolov5
add_executable(test_yolov5 ${PROJECT_SOURCE_DIR}/test_yolov5.cc)
target_link_libraries(test_yolov5 ${FASTDEPLOY_LIBS})
install(TARGETS test_yolov5 DESTINATION ./)

# add test for ppyoloe
add_executable(test_ppyoloe ${PROJECT_SOURCE_DIR}/test_ppyoloe.cc)
target_link_libraries(test_ppyoloe ${FASTDEPLOY_LIBS})
install(TARGETS test_ppyoloe DESTINATION ./)

# add test for paddleclas
add_executable(test_clas ${PROJECT_SOURCE_DIR}/test_clas.cc)
target_link_libraries(test_clas ${FASTDEPLOY_LIBS})
install(TARGETS test_clas DESTINATION ./)

# add test for pp-liteseg
add_executable(test_ppliteseg ${PROJECT_SOURCE_DIR}/test_ppliteseg.cc)
target_link_libraries(test_ppliteseg ${FASTDEPLOY_LIBS})
install(TARGETS test_ppliteseg DESTINATION ./)


install(DIRECTORY models DESTINATION ./)
install(DIRECTORY images DESTINATION ./)
install(DIRECTORY results DESTINATION ./)

file(GLOB RUN_TEST run_test.sh)
install(PROGRAMS ${RUN_TEST} DESTINATION ./)

file(GLOB_RECURSE FASTDEPLOY_LIBS ${FASTDEPLOY_INSTALL_DIR}/lib/lib*.so*)
file(GLOB_RECURSE ALL_LIBS ${FASTDEPLOY_INSTALL_DIR}/third_libs/install/lib*.so*)
list(APPEND ALL_LIBS ${FASTDEPLOY_LIBS})
install(PROGRAMS ${ALL_LIBS} DESTINATION lib)
