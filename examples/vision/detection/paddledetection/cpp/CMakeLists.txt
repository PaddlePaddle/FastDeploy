PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_faster_rcnn_demo ${PROJECT_SOURCE_DIR}/infer_faster_rcnn.cc)
target_link_libraries(infer_faster_rcnn_demo ${FASTDEPLOY_LIBS})

add_executable(infer_ppyoloe_demo ${PROJECT_SOURCE_DIR}/infer_ppyoloe.cc)
target_link_libraries(infer_ppyoloe_demo ${FASTDEPLOY_LIBS})

add_executable(infer_picodet_demo ${PROJECT_SOURCE_DIR}/infer_picodet.cc)
target_link_libraries(infer_picodet_demo ${FASTDEPLOY_LIBS})

add_executable(infer_yolox_demo ${PROJECT_SOURCE_DIR}/infer_yolox.cc)
target_link_libraries(infer_yolox_demo ${FASTDEPLOY_LIBS})

add_executable(infer_yolov3_demo ${PROJECT_SOURCE_DIR}/infer_yolov3.cc)
target_link_libraries(infer_yolov3_demo ${FASTDEPLOY_LIBS})

add_executable(infer_ppyolo_demo ${PROJECT_SOURCE_DIR}/infer_ppyolo.cc)
target_link_libraries(infer_ppyolo_demo ${FASTDEPLOY_LIBS})

add_executable(infer_mask_rcnn_demo ${PROJECT_SOURCE_DIR}/infer_mask_rcnn.cc)
target_link_libraries(infer_mask_rcnn_demo ${FASTDEPLOY_LIBS})

add_executable(infer_ssd_demo ${PROJECT_SOURCE_DIR}/infer_ssd.cc)
target_link_libraries(infer_ssd_demo ${FASTDEPLOY_LIBS})

add_executable(infer_yolov5_demo ${PROJECT_SOURCE_DIR}/infer_yolov5.cc)
target_link_libraries(infer_yolov5_demo ${FASTDEPLOY_LIBS})

add_executable(infer_yolov6_demo ${PROJECT_SOURCE_DIR}/infer_yolov6.cc)
target_link_libraries(infer_yolov6_demo ${FASTDEPLOY_LIBS})

add_executable(infer_yolov7_demo ${PROJECT_SOURCE_DIR}/infer_yolov7.cc)
target_link_libraries(infer_yolov7_demo ${FASTDEPLOY_LIBS})

add_executable(infer_rtmdet_demo ${PROJECT_SOURCE_DIR}/infer_rtmdet.cc)
target_link_libraries(infer_rtmdet_demo ${FASTDEPLOY_LIBS})
