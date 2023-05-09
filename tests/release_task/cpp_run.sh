#!/bin/bash
export no_proxy=bcebos.com,xly.bce.baidu.com,aliyun.com
CURRENT_DIR=$(cd $(dirname $0); pwd)
PLATFORM=$1
DEVICE=$2
VERSION=$3
CUSTOM_CPP_FASTDEPLOY_PACKAGE=$4
if [ "$VERSION" = "0.0.0" ];then
       DOWNLOAD_DIR=dev
else
       DOWNLOAD_DIR=rel_tmp
fi
if [ "$DEVICE" = "gpu" ];then
       CPP_FASTDEPLOY_PACKAGE=fastdeploy-$PLATFORM-$DEVICE-$VERSION
else
       CPP_FASTDEPLOY_PACKAGE=fastdeploy-$PLATFORM-$VERSION
fi
# Use custom package name if not empty
if [ ! "$CUSTOM_CPP_FASTDEPLOY_PACKAGE" = "" ];then
  CPP_FASTDEPLOY_PACKAGE=$CUSTOM_CPP_FASTDEPLOY_PACKAGE
  echo "using custom package: $CUSTOM_CPP_FASTDEPLOY_PACKAGE"
fi
echo "current package: $CPP_FASTDEPLOY_PACKAGE"

LINUX_X64_GPU_CASE=('ort' 'paddle' 'trt')
LINUX_X64_CPU_CASE=('ort' 'paddle' 'openvino')
# LINUX_AARCH_CPU_CASE=('ort' 'openvino')
LINUX_AARCH_CPU_CASE=('ort')
MACOS_INTEL_CPU_CASE=('ort' 'openvino')
MACOS_ARM64_CPU_CASE=('default')
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget -q https://fastdeploy.bj.bcebos.com/resource/images/000000014439.jpg
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/release_task_groud_truth_result.txt
tar -xvf ppyoloe_crn_l_300e_coco.tgz
IMAGE_PATH=$CURRENT_DIR/000000014439.jpg
MODEL_PATH=$CURRENT_DIR/ppyoloe_crn_l_300e_coco
GROUND_TRUTH_PATH=$CURRENT_DIR/release_task_groud_truth_result.txt
COMPARE_SHELL=$CURRENT_DIR/compare_with_gt.py

RUN_CASE=()
CONF_THRESHOLD=0
if [ "$DEVICE" = "gpu" ] && [ "$PLATFORM" = "linux-x64" ];then
	RUN_CASE=(${LINUX_X64_GPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "linux-x64" ];then
	RUN_CASE=(${LINUX_X64_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "linux-aarch64" ];then
	RUN_CASE=(${LINUX_AARCH_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "osx-x86_64" ];then
	RUN_CASE=(${MACOS_INTEL_CPU_CASE[*]})
	CONF_THRESHOLD=0.5
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "osx-arm64" ];then
	RUN_CASE=(${MACOS_ARM64_CPU_CASE[*]})
	CONF_THRESHOLD=0.5
fi

case_number=${#RUN_CASE[@]}

wget -q  https://fastdeploy.bj.bcebos.com/$DOWNLOAD_DIR/cpp/$CPP_FASTDEPLOY_PACKAGE.tgz

tar xvf $CPP_FASTDEPLOY_PACKAGE.tgz
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../$CPP_FASTDEPLOY_PACKAGE -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER
make -j
ret=0
for((i=0;i<case_number;i+=1))
do
       backend=${RUN_CASE[i]}
       echo "Cpp Backend:" $backend
       if [ "$backend" != "trt" ];then
               ./infer_ppyoloe_demo --model_dir=$MODEL_PATH --image_file=$IMAGE_PATH --device=cpu --backend=$backend >> cpp_$backend\_cpu_result.txt
               python $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path cpp_$backend\_cpu_result.txt --platform $PLATFORM --device cpu --conf_threshold $CONF_THRESHOLD
       fi
       if [ "$DEVICE" = "gpu" ];then

	       if [ "$backend" = "trt" ];then
                       ./infer_ppyoloe_demo --model_dir=$MODEL_PATH --image_file=$IMAGE_PATH --device=gpu --backend=$backend >> cpp_trt_result.txt
                       python $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path cpp_trt_result.txt --platform $PLATFORM --device trt --conf_threshold $CONF_THRESHOLD
	       else
                       ./infer_ppyoloe_demo --model_dir=$MODEL_PATH --image_file=$IMAGE_PATH --device=gpu --backend=$backend >> cpp_$backend\_gpu_result.txt
                       python $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path cpp_$backend\_gpu_result.txt --platform $PLATFORM --device gpu --conf_threshold $CONF_THRESHOLD
               fi
       fi
       if [ $? -ne 0 ];then
               ret=-1
       fi
done

res_file="result.txt"
if [ ! -f $res_file ];then
       if [ $ret -ne 0 ];then
               exit -1
       fi
       exit 0
else
       cat $res_file
       exit -1
fi
