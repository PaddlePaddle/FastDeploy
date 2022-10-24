@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
set NO_PROXY=bj.bcebos.com,xly.bce.baidu.com,aliyun.com

set PLATFORM=%1
set DEVICE=%2
set VERSION=%3
set CURRENT_DIR=%cd%

echo "CURRENT_DIR" %CURRENT_DIR%

echo "PLATFORM: " %PLATFORM%
echo "DEVICE: " %DEVICE%
echo "VERSION: " %VERSION%

for /f "delims=" %%a in ('python -V') do set py_version=%%a
echo "py_version:" %py_version%

if "%DEVICE%" == "gpu" (
    set PY_FASTDEPLOY_PACKAGE=fastdeploy-%DEVICE%-python
    set RUN_CASES=ort paddle trt
) else (
    set PY_FASTDEPLOY_PACKAGE=fastdeploy-python
    set RUN_CASES=ort paddle openvino
)

echo "PY_FASTDEPLOY_PACKAGE: " %PY_FASTDEPLOY_PACKAGE%
echo "RUN_CASES" %RUN_CASES%

python -c "import wget; wget.download('https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg')"
python -c "import wget; wget.download('https://bj.bcebos.com/paddlehub/fastdeploy/release_task_groud_truth_result.txt')"
python -c "from download import *; download_and_decompress('https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz', './')"

set IMAGE_PATH=%CURRENT_DIR%\000000014439.jpg
set MODEL_PATH=%CURRENT_DIR%\ppyoloe_crn_l_300e_coco
set GROUND_TRUTH_PATH=%CURRENT_DIR%\release_task_groud_truth_result.txt
set COMPARE_SHELL=%CURRENT_DIR%\compare_with_gt.py

for %%b in (%RUN_CASES%) do (
    echo "Python Backend:" %%b
    if %%b  NEQ trt (
        python infer_ppyoloe.py --model_dir=%MODEL_PATH% --image=%IMAGE_PATH% --device=cpu --backend=%%b >> py_%%b_cpu_result.txt
        python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path py_%%b_cpu_result.txt --platform %PLATFORM% --device cpu --conf_threshold 0.5
    )
    if  "%DEVICE%" == "gpu" (
        if %%b == trt (
            python infer_ppyoloe.py --model_dir=%MODEL_PATH% --image=%IMAGE_PATH% --device=gpu --backend=%%b >> py_%%b_trt_result.txt 
            python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path py_%%b_trt_result.txt --platform %PLATFORM% --device trt --conf_threshold 0.5
        ) else (
            python infer_ppyoloe.py --model_dir=%MODEL_PATH% --image=%IMAGE_PATH% --device=gpu --backend=%%b >> py_%%b_gpu_result.txt 
            python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path py_%%b_gpu_result.txt --platform %PLATFORM% --device gpu  --conf_threshold 0.5
        )
    ) 
)


set res_file=%cd%\result.txt

if exist %res_file% (
    for /f "delims=" %%i in (%res_file%) do echo %%i
    exit -1
) else (
    if %errorlevel% NEQ 0 (
        exit -1
    )
    exit 0
)
:END
ENDLOCAL
