@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
set NO_PROXY=bj.bcebos.com,xly.bce.baidu.com,aliyun.com

set PLATFORM=$1
set DEVICE=$2
set VERSION=$3
set CURRENT_DIR='%~dp0'


if %DEVICE%=="gpu" (
    set CPP_FASTDEPLOY_PACKAGE=fastdeploy-%PLATFORM%-%DEVICE%-%VERSION%
    set RUN_CASES='ort' 'paddle' 'trt'
) else (
    set CPP_FASTDEPLOY_PACKAGE=fastdeploy-python
    set RUN_CASES='ort' 'paddle' 'openvino'
)

echo "CPP_FASTDEPLOY_PACKAGE: " %CPP_FASTDEPLOY_PACKAGE%

@REM set WIN_10_X64_GPU_CASE='ort' 'paddle' 'trt'
@REM set WIN_10_X64_CPU_CASE='ort' 'paddle' 'openvino'

wget -q https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget -q https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
wget -q https://bj.bcebos.com/paddlehub/fastdeploy/release_task_groud_truth_result.txt
tar -xvf ppyoloe_crn_l_300e_coco.tgz

set IMAGE_PATH=%CURRENT_DIR%/000000014439.jpg
set MODEL_PATH=%CURRENT_DIR%/ppyoloe_crn_l_300e_coco
set GROUND_TRUTH_PATH=%CURRENT_DIR%/release_task_groud_truth_result.txt
set COMPARE_SHELL=%CURRENT_DIR%/compare_with_gt.py

wget -q  https://fastdeploy.bj.bcebos.com/dev/cpp/$CPP_FASTDEPLOY_PACKAGE.tgz

tar xvf $CPP_FASTDEPLOY_PACKAGE.tgz
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../$CPP_FASTDEPLOY_PACKAGE
make -j

for %%b in (%RUN_CASES%) do (
    echo "Cpp Backend:" %b%
    if %b% != "trt" (
        ./infer_ppyoloe_demo --model_dir=%MODEL_PATH% --image_file=%IMAGE_PATH% --device=cpu --backend=%b% >> cpp_cpu_result.txt
        python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path cpp_cpu_result.txt --platform %PLATFORM% --device cpu
    )
    if  %DEVICE% == "gpu" (
        if %b% != "trt" (
            ./infer_ppyoloe_demo --model_dir=%MODEL_PATH% --image_file=%IMAGE_PATH% --device=gpu --backend=%b% >> cpp_trt_result.txt
            python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path cpp_trt_result.txt --platform %PLATFORM% --device trt
        ) else (
            ./infer_ppyoloe_demo --model_dir=%MODEL_PATH% --image_file=%IMAGE_PATH% --device=gpu --backend=%b% >> cpp_gpu_result.txt
            python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path cpp_gpu_result.txt --platform %PLATFORM% --device gpu
        )
    ) 
)

set res_file="*result.txt"

if exist %res_file% (
    for /f %%i in (%res_file%) do echo %%i
    exit -1
) else (
    exit 0
)
:END
ENDLOCAL
