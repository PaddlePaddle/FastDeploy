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

add_executable(infer_ppyoloe_r_demo ${PROJECT_SOURCE_DIR}/infer_ppyoloe_r.cc)
target_link_libraries(infer_ppyoloe_r_demo ${FASTDEPLOY_LIBS})

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

add_executable(infer_yolov8_demo ${PROJECT_SOURCE_DIR}/infer_yolov8.cc)
target_link_libraries(infer_yolov8_demo ${FASTDEPLOY_LIBS})

add_executable(infer_rtmdet_demo ${PROJECT_SOURCE_DIR}/infer_rtmdet.cc)
target_link_libraries(infer_rtmdet_demo ${FASTDEPLOY_LIBS})

add_executable(infer_cascadercnn_demo ${PROJECT_SOURCE_DIR}/infer_cascadercnn.cc)
target_link_libraries(infer_cascadercnn_demo ${FASTDEPLOY_LIBS})

add_executable(infer_pssdet_demo ${PROJECT_SOURCE_DIR}/infer_pssdet.cc)
target_link_libraries(infer_pssdet_demo ${FASTDEPLOY_LIBS})

add_executable(infer_retinanet_demo ${PROJECT_SOURCE_DIR}/infer_retinanet.cc)
target_link_libraries(infer_retinanet_demo ${FASTDEPLOY_LIBS})

add_executable(infer_ppyoloesod_demo ${PROJECT_SOURCE_DIR}/infer_ppyoloesod.cc)
target_link_libraries(infer_ppyoloesod_demo ${FASTDEPLOY_LIBS})

add_executable(infer_fcos_demo ${PROJECT_SOURCE_DIR}/infer_fcos.cc)
target_link_libraries(infer_fcos_demo ${FASTDEPLOY_LIBS})

add_executable(infer_ttfnet_demo ${PROJECT_SOURCE_DIR}/infer_ttfnet.cc)
target_link_libraries(infer_ttfnet_demo ${FASTDEPLOY_LIBS})

add_executable(infer_tood_demo ${PROJECT_SOURCE_DIR}/infer_tood.cc)
target_link_libraries(infer_tood_demo ${FASTDEPLOY_LIBS})

add_executable(infer_gfl_demo ${PROJECT_SOURCE_DIR}/infer_gfl.cc)
target_link_libraries(infer_gfl_demo ${FASTDEPLOY_LIBS})

add_executable(infer_solov2_demo ${PROJECT_SOURCE_DIR}/infer_solov2.cc)
target_link_libraries(infer_solov2_demo ${FASTDEPLOY_LIBS})
