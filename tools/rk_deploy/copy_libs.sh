#!/bin/bash
HOST_SPACE=${PWD}
rm -r ${HOST_SPACE}/libs
mkdir ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/libfastdeploy.so* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/lib* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/paddlelite/lib/verisilicon_timvx/* ${HOST_SPACE}/libs
cp -r ${HOST_SPACE}/../../build/third_libs/install/opencv/lib/libopencv* ${HOST_SPACE}/libs

DEMO_NAME=image_classification_demo
if [ -n "$1" ]; then
  DEMO_NAME=$1
fi
# copy libs and demo
cp -r ${HOST_SPACE}/../../build/bin/${DEMO_NAME} ${HOST_SPACE}/
