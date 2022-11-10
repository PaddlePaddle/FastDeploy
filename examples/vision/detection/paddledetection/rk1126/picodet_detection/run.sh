#!/bin/bash
#run

TARGET_ABI=armv8 # for 64bit
#TARGET_ABI=armv7hf # for 32bit
if [ -n "$1" ]; then
    TARGET_ABI=$1
fi
export LD_LIBRARY_PATH=../Paddle-Lite/libs/$TARGET_ABI/
export GLOG_v=0
export VSI_NN_LOG_LEVEL=0
export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1
export VIV_VX_SET_PER_CHANNEL_ENTROPY=100
export TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION=30000
build/object_detection_demo models/picodetv2_relu6_coco_no_fuse ../../assets/labels/coco_label_list.txt models/picodetv2_relu6_coco_no_fuse/subgraph.txt models/picodetv2_relu6_coco_no_fuse/picodet.yml 
