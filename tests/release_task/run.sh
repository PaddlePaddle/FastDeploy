#!/bin/bash
export no_proxy=bcebos.com
CURRENT_DIR=$(cd $(dirname $0); pwd)
PLATFORM=$1
DEVICE=$2
VERSION=$3
if [ "$DEVICE" = "gpu" ];then
       PY_FASTDEPLOY_PACKAGE=fastdeploy-$DEVICE-python
       CPP_FASTDEPLOY_PACKAGE=fastdeploy-$PLATFORM-$DEVICE-$VERSION
else
       PY_FASTDEPLOY_PACKAGE=fastdeploy-python
       CPP_FASTDEPLOY_PACKAGE=fastdeploy-$PLATFORM-$VERSION
fi
echo $CPP_FASTDEPLOY_PACKAGE
echo $PY_FASTDEPLOY_PACKAGE
PY_VERSION_CASE=('python3.6' 'python3.7' 'python3.8' 'python3.9' 'python3.10')
LINUX_X64_GPU_CASE=('ort' 'paddle' 'trt')
LINUX_X64_CPU_CASE=('ort' 'paddle')
LINUX_AARCH_CPU_CASE=('ort')
WIN_10_X64_GPU_CASE=('ort' 'paddle' 'trt')
WIN_10_X64_CPU_CASE=('ort' 'paddle')
MACOS_INTEL_CPU_CASE=('ort' 'paddle')
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
elif [ "$DEVICE" = "gpu" ] && [ "$PLATFORM" = "win-x64" ];then
	RUN_CASE=(${WIN_10_X64_GPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "win-x64" ];then
	RUN_CASE=(${WIN_10_X64_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "osx-x86_64" ];then
	RUN_CASE=(${MACOS_INTEL_CPU_CASE[*]})
elif [ "$DEVICE" = "cpu" ] && [ "$PLATFORM" = "osx-arm64" ];then
	RUN_CASE=(${MACOS_ARM64_CPU_CASE[*]})
fi

py_version_case_number=${#PY_VERSION_CASE[@]}
case_number=${#RUN_CASE[@]}
for((i=0;i<py_version_case_number;i+=1))
do
	py_version=${PY_VERSION_CASE[i]}
        echo "py_version:" $py_version
        $py_version -m pip freeze | grep fastdeploy | xargs pip uninstall -y
        $py_version -m pip install $PY_FASTDEPLOY_PACKAGE -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
        for((j=0;j<case_number;j+=1))
        do
               backend=${RUN_CASE[j]}
               echo "Python Backend:" $backend
               if [ "$backend" != "trt" ];then
                       $py_version infer_ppyoloe.py --model_dir $MODEL_PATH --image $IMAGE_PATH --device cpu --backend $backend >> py_cpu_result.txt
                       $py_version $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path py_cpu_result.txt --platform $PLATFORM --device cpu
               fi
               if [ "$DEVICE" = "gpu" ];then

        	       if [ "$backend" = "trt" ];then
                               $py_version infer_ppyoloe.py --model_dir $MODEL_PATH --image $IMAGE_PATH --device gpu --backend $backend >> py_trt_result.txt
                               $py_version $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path py_trt_result.txt --platform $PLATFORM --device trt
        	       else
        		       $py_version infer_ppyoloe.py --model_dir $MODEL_PATH --image $IMAGE_PATH --device gpu --backend $backend >> py_gpu_result.txt
                               $py_version $COMPARE_SHELL --gt_path $GROUND_TRUTH_PATH --result_path py_gpu_result.txt --platform $PLATFORM --device gpu
                       fi
               fi
        done
done
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

res_file="result.txt"
if [ ! -f $res_file ]; then
       exit 0
else
       cat $res_file
       exit -1
fi
