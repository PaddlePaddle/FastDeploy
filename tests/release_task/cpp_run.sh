#!/bin/bash
export no_proxy=bcebos.com,xly.bce.baidu.com,aliyun.com
CURRENT_DIR=$(cd $(dirname $0); pwd)
PLATFORM=$1
DEVICE=$2
VERSION=$3
if [ "$DEVICE" = "gpu" ];then
       CPP_FASTDEPLOY_PACKAGE=fastdeploy-$PLATFORM-$DEVICE-$VERSION
else
       CPP_FASTDEPLOY_PACKAGE=fastdeploy-$PLATFORM-$VERSION
fi
echo $CPP_FASTDEPLOY_PACKAGE
LINUX_X64_GPU_CASE=('ort' 'paddle' 'trt')
LINUX_X64_CPU_CASE=('ort' 'paddle' 'openvino')
LINUX_AARCH_CPU_CASE=('ort' 'openvino')
MACOS_INTEL_CPU_CASE=('ort' 'paddle' 'openvino')
MACOS_ARM64_CPU_CASE=('default')
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget -q https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/release_task_groud_truth_result.txt
tar -xvf ppyoloe_crn_l_300e_coco.tgz
IMAGE_PATH=$CURRENT_DIR/000000014439.jpg
MODEL_PATH=$CURRENT_DIR/ppyoloe_crn_l_300e_coco
GROUND_TRUTH_PATH=$CURRENT_DIR/release_task_groud_truth_result.txt
COMPARE_SHELL=$CURRENT_DIR/compare_with_gt.py

RUN_CASE=()
if [ "$DEVICE" = "gpu" ] && [ "$PLATFORM" = "linux-x64" ];then
	RUN_CASE=(${LINUX_X64_GPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "linux-x64" ];then
	RUN_CASE=(${LINUX_X64_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "linux-aarch64" ];then
	RUN_CASE=(${LINUX_AARCH_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "osx-x86_64" ];then
	RUN_CASE=(${MACOS_INTEL_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "osx-arm64" ];then
	RUN_CASE=(${MACOS_ARM64_CPU_CASE[*]})
fi

case_number=${#RUN_CASE[@]}

wget -q  https://fastdeploy.bj.bcebos.com/dev/cpp/$CPP_FASTDEPLOY_PACKAGE.tgz

tar xvf $CPP_FASTDEPLOY_PACKAGE.tgz
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../$CPP_FASTDEPLOY_PACKAGE
make -j

for((i=0;i<case_number;i+=1))
do
       backend=${RUN_CASE[i]}
       echo "Cpp Backend:" $backend
       if [ "$backend" != "trt" ];then
               ./infer_ppyoloe_demo --model_dir=$MODEL_PATH --image_file=$IMAGE_PATH --device=cpu --backend=$backend >> cpp_cpu_result.txt
               python $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path cpp_cpu_result.txt --platform $PLATFORM --device cpu
       fi
       if [ "$DEVICE" = "gpu" ];then

	       if [ "$backend" = "trt" ];then
                       ./infer_ppyoloe_demo --model_dir=$MODEL_PATH --image_file=$IMAGE_PATH --device=gpu --backend=$backend >> cpp_trt_result.txt
                       python $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path cpp_trt_result.txt --platform $PLATFORM --device trt
	       else
                       ./infer_ppyoloe_demo --model_dir=$MODEL_PATH --image_file=$IMAGE_PATH --device=gpu --backend=$backend >> cpp_gpu_result.txt
                       python $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path cpp_gpu_result.txt --platform $PLATFORM --device gpu
               fi
       fi
done

ret=$?

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

