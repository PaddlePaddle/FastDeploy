#!/bin/bash
HOST_SPACE=${PWD}
rm -r ${HOST_SPACE}/libs
mkdir ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/libfastdeploy.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/verisilicon_timvx/viv_sdk_6_4_6_5/lib/libgomp.so.1 ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/verisilicon_timvx/libnnadapter.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/opencv/lib/libopencv_core.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/opencv/lib/libopencv_imgcodecs.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/opencv/lib/libopencv_imgproc.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/libpaddle_full_api_shared.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/verisilicon_timvx/viv_sdk_6_4_6_5/lib/libtim-vx.so ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/verisilicon_timvx/libverisilicon_timvx.so* ${HOST_SPACE}/libs

DEMO_NAME=image_classification_demo
if [ -n "$1" ]; then
  DEMO_NAME=$1
fi
# copy libs and demo
cp -r ${HOST_SPACE}/../../build/bin/${DEMO_NAME} ${HOST_SPACE}/
