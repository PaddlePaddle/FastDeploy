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

if "%DEVICE%" == "gpu" (
    set CPP_FASTDEPLOY_PACKAGE=fastdeploy-%PLATFORM%-%DEVICE%-%VERSION%
    set RUN_CASES=ort paddle trt
) else (
    set CPP_FASTDEPLOY_PACKAGE=fastdeploy-python
    set RUN_CASES=ort paddle openvino
)

echo "CPP_FASTDEPLOY_PACKAGE: " %CPP_FASTDEPLOY_PACKAGE%
echo "RUN_CASES" %RUN_CASES%

python -c "import wget; wget.download('https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg')"
python -c "import wget; wget.download('https://bj.bcebos.com/paddlehub/fastdeploy/release_task_groud_truth_result.txt')"
python -c "from download import *; download_and_decompress('https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz', './')"

set IMAGE_PATH=%CURRENT_DIR%\000000014439.jpg
set MODEL_PATH=%CURRENT_DIR%\ppyoloe_crn_l_300e_coco
set GROUND_TRUTH_PATH=%CURRENT_DIR%\release_task_groud_truth_result.txt
set COMPARE_SHELL=%CURRENT_DIR%\compare_with_gt.py

python -c "from download import *; download_and_decompress('https://fastdeploy.bj.bcebos.com/dev/cpp/%CPP_FASTDEPLOY_PACKAGE%.zip', './')"

mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=%cd%\..\%CPP_FASTDEPLOY_PACKAGE% -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"  -DCMAKE_CXX_COMPILER=%CMAKE_CXX_COMPILER%

msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64


set FASTDEPLOY_HOME=%cd%\..\%CPP_FASTDEPLOY_PACKAGE%
echo "FASTDEPLOY_HOME" %FASTDEPLOY_HOME%

copy /Y %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\onnxruntime* Release\
set PATH=%FASTDEPLOY_HOME%\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\openvino\bin;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\bin;%PATH%

echo "set path done"
cd %cd%\Release
for %%b in (%RUN_CASES%) do (
    echo "Cpp Backend:" %%b
    if %%b  NEQ trt (
        infer_ppyoloe_demo.exe --model_dir=%MODEL_PATH% --image_file=%IMAGE_PATH% --device=cpu --backend=%%b >> cpp_%%b_cpu_result.txt
        python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path cpp_%%b_cpu_result.txt --platform %PLATFORM% --device cpu --conf_threshold 0.5
    )
    if  "%DEVICE%" == "gpu" (
        if %%b == trt (
            infer_ppyoloe_demo.exe --model_dir=%MODEL_PATH% --image_file=%IMAGE_PATH% --device=gpu --backend=%%b >> cpp_%%b_trt_result.txt
            python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path cpp_%%b_trt_result.txt --platform %PLATFORM% --device trt --conf_threshold 0.5
        ) else (
            infer_ppyoloe_demo.exe --model_dir=%MODEL_PATH% --image_file=%IMAGE_PATH% --device=gpu --backend=%%b >> cpp_%%b_gpu_result.txt
            python %COMPARE_SHELL% --gt_path %GROUND_TRUTH_PATH% --result_path cpp_%%b_gpu_result.txt --platform %PLATFORM% --device gpu --conf_threshold 0.5
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
