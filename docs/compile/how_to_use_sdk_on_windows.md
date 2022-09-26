# 在 Windows 使用 FastDeploy C++ SDK

在 Windows 下使用 FastDeploy C++ SDK 与在 Linux 下使用稍有不同。以下以 PPYOLOE 为例进行演示在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。在部署前，需确认以下两个步骤：  
- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../quick_start)

## 目录
- [环境依赖](#Environment)  
- [下载 FastDeploy Windows 10 C++ SDK](#Download)  
- [Windows下多种方式使用 C++ SDK 的方式](#CommandLine)
  - [方式一：命令行方式使用 C++ SDK](#CommandLine)  
    - [步骤一：在 Windows 命令行终端 上编译 example](#CommandLine)  
    - [步骤二：运行可执行文件获得推理结果](#CommandLine)  
  - [方式二：Visual Studio 2019 IDE 方式使用 C++ SDK](#VisualStudio2019)
    - [步骤一：Visual Studio 2019 创建CMake工程项目](#VisualStudio20191)  
    - [步骤二：在CMakeLists中配置 FastDeploy C++ SDK](#VisualStudio20192)  
    - [步骤三：生成工程缓存并修改CMakeSetting.json配置](#VisualStudio20193)  
    - [步骤四：生成可执行文件，运行获取结果](#VisualStudio20194)  
  - [方式三：CLion IDE 方式使用 C++ SDK](#CLion)  
  - [方式四：Visual Studio Code IDE 方式使用 C++ SDK](#VisualStudioCode)
- [多种方法配置exe运行时所需的依赖库](#CommandLineDeps1)
  - [方式一：修改CMakeLists.txt，一行命令配置（推荐）](#CommandLineDeps1)  
  - [方式二：命令行设置环境变量](#CommandLineDeps2)  
  - [方法三：手动拷贝依赖库到exe的目录下](#CommandLineDeps3)  


## 1. 环境依赖
<div id="Environment"></div>  

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 8.0 (当WITH_GPU=ON)
- TensorRT >= 8.4 (当ENABLE_TRT_BACKEND=ON)

## 2. 下载 FastDeploy Windows 10 C++ SDK
<div id="Download"></div>  

可以从以下链接下载编译好的 FastDeploy Windows 10 C++ SDK，SDK中包含了examples代码。
```text
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip
```
## 3. 准备模型文件和测试图片
可以从以下链接下载模型文件和测试图片，并解压缩
```text
https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz # (下载后解压缩)
https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

## 4. SDK使用方式一：命令行方式使用 C++ SDK
<div id="CommandLine"></div>  

### 4.1 在 Windows 上编译 PPYOLOE
Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，cd到ppyoloe的demo路径  
```bat  
cd fastdeploy-win-x64-gpu-0.2.0\examples\vision\detection\paddledetection\cpp
```
```bat
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.1 -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
```
然后执行
```bat
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

### 4.2 运行 demo
```bat
cd Release
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0  # CPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 1  # GPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 2  # GPU + TensorRT
```  

特别说明，exe运行时所需要的依赖库配置方法，请参考章节: [多种方法配置exe运行时所需的依赖库](#CommandLineDeps)


## 5. SDK使用方式二：Visual Studio 2019 IDE 方式使用 C++ SDK
<div id="VisualStudio2019"></div>  

### 5.1 步骤一：Visual Studio 2019 创建“CMake”工程项目  

<div id="VisualStudio20191"></div>  

（1）打开Visual Studio 2019，点击"创建新项目"->点击"CMake"，从而创建CMake工程项目。以PPYOLOE为例，来说明如何在Visual Studio 2019 IDE中使用FastDeploy C++ SDK.

![image](https://user-images.githubusercontent.com/31974251/192143543-9f29e4cb-2307-45ca-a61a-bcfba5df19ff.png)

![image](https://user-images.githubusercontent.com/31974251/192143640-39e79c65-8b50-4254-8da6-baa21bb23e3c.png)  


![image](https://user-images.githubusercontent.com/31974251/192143713-be2e6490-4cab-4151-8463-8c367dbc451a.png)

（2）打开工程发现，Visual Stuio 2019已经为我们生成了一些基本的文件，其中包括CMakeLists.txt。infer_ppyoloe.h头文件这里实际上用不到，我们可以直接删除。  

![image](https://user-images.githubusercontent.com/31974251/192143930-db1655c2-66ee-448c-82cb-0103ca1ca2a0.png)  

### 5.2 步骤二：在CMakeLists中配置 FastDeploy C++ SDK

<div id="VisualStudio20192"></div>  

（1）在工程创建完成后，我们需要添加infer_ppyoloe推理源码，并修改CMakeLists.txt，修改如下：

![image](https://user-images.githubusercontent.com/31974251/192144782-79bccf8f-65d0-4f22-9f41-81751c530319.png)

（2）其中infer_ppyoloe.cpp的代码可以直接从examples中的代码拷贝过来：  
- [examples/vision/detection/paddledetection/cpp/infer_ppyoloe.cc](../../examples/vision/detection/paddledetection/cpp/infer_ppyoloe.cc)

（3）CMakeLists.txt主要包括配置FastDeploy C++ SDK的路径，如果是GPU版本的SDK，还需要配置CUDA_DIRECTORY为CUDA的安装路径，CMakeLists.txt的配置如下：

```cmake
project(infer_ppyoloe_demo C CXX)
cmake_minimum_required(VERSION 3.12)

# Only support "Release" mode now  
set(CMAKE_BUILD_TYPE "Release")

# Set FastDeploy install dir
set(FASTDEPLOY_INSTALL_DIR "D:/qiuyanjun/fastdeploy-win-x64-gpu-0.2.1"
    CACHE PATH "Path to downloaded or built fastdeploy sdk.")

# Set CUDA_DIRECTORY (CUDA 11.x) for GPU SDK
set(CUDA_DIRECTORY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7"
    CACHE PATH "Path to installed CUDA Toolkit.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

add_executable(infer_ppyoloe_demo ${PROJECT_SOURCE_DIR}/infer_ppyoloe.cpp)
target_link_libraries(infer_ppyoloe_demo ${FASTDEPLOY_LIBS})  

# Optional: install all DLLs to binary dir.
install_fastdeploy_libraries(${CMAKE_CURRENT_BINARY_DIR}/Release)
```

### 5.3 步骤三：生成工程缓存并修改CMakeSetting.json配置

<div id="VisualStudio20193"></div>  

（1）点击"CMakeLists.txt"->右键点击"生成缓存":  

![image](https://user-images.githubusercontent.com/31974251/192145349-c78b110a-0e41-4ee5-8942-3bf70bd94a75.png)

发现已经成功生成缓存了，但是由于打开工程时，默认是Debug模式，我们发现exe和缓存保存路径还是Debug模式下的。 我们可以先修改CMake的设置为Release.

（2）点击"CMakeLists.txt"->右键点击"infer_ppyoloe_demo的cmake设置"，进入CMakeSettings.json的设置面板，把其中的Debug设置修改为Release.  

![image](https://user-images.githubusercontent.com/31974251/192145242-01d37b44-e2fa-47df-82c1-c11c2ccbff99.png)  

同时设置CMake生成器为 "Visual Studio 16 2019 Win64"

![image](https://user-images.githubusercontent.com/31974251/192147961-ac46d0f6-7349-4126-a123-914af2b63d95.jpg)

（3）点击保存CMake缓存以切换为Release配置：  

![image](https://user-images.githubusercontent.com/31974251/192145974-b5a63341-9143-49a2-8bfe-94ac641b1670.png)

（4）：（4.1）点击"CMakeLists.txt"->右键"CMake缓存仅限x64-Release"->"点击删除缓存"；（4.2）点击"CMakeLists.txt"->"生成缓存"；（4.3）如果在步骤一发现删除缓存的选项是灰色的可以直接点击"CMakeLists.txt"->"生成"，若生成失败则可以重复尝试（4.1）和（4。2）

![image](https://user-images.githubusercontent.com/31974251/192146394-51fbf2b8-1cba-41ca-bb45-5f26890f64ce.jpg)  

最终可以看到，配置已经成功生成Relase模式下的CMake缓存了。  

![image](https://user-images.githubusercontent.com/31974251/192146239-a1eacd9e-034d-4373-a262-65b18ce25b87.png)  


### 5.4 步骤四：生成可执行文件，运行获取结果。

<div id="VisualStudio20194"></div>  

（1）点击"CMakeLists.txt"->"生成"。可以发现已经成功生成了infer_ppyoloe_demo.exe，并保存在`out/build/x64-Release/Release`目录下。  

![image](https://user-images.githubusercontent.com/31974251/192146852-c64d2252-8c8f-4309-a950-908a5cb258b8.png)

（2）执行可执行文件，获得推理结果。 首先需要拷贝所有的dll到exe所在的目录下，这里我们可以在CMakeLists.txt添加一下命令，可将FastDeploy中所有的dll安装到指定的目录。  

```cmake  
install_fastdeploy_libraries(${CMAKE_CURRENT_BINARY_DIR}/Release)
```
（3）同时，也需要把ppyoloe的模型文件和测试图片下载解压缩后，拷贝到exe所在的目录。 准备完成后，目录结构如下：  

![image](https://user-images.githubusercontent.com/31974251/192147505-054edb77-564b-405e-89ee-fd0d2e413e78.png)

（4）最后，执行以下命令获得推理结果：  

```bat  
D:\xxxinfer_ppyoloe\out\build\x64-Release\Release>infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0
[INFO] fastdeploy/runtime.cc(304)::fastdeploy::Runtime::Init    Runtime initialized with Backend::OPENVINO in Device::CPU.
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
415.047180,89.311569, 506.009613, 283.863098, 0.950423, 0
163.665710,81.914932, 198.585342, 166.760895, 0.896433, 0
581.788635,113.027618, 612.623474, 198.521713, 0.842596, 0
267.217224,89.777306, 298.796051, 169.361526, 0.837951, 0
......
153.301407,123.233757, 177.130539, 164.558350, 0.066697, 60
505.887604,140.919601, 523.167236, 151.875336, 0.084912, 67

Visualized result saved in ./vis_result.jpg
```  

打开保存的图片查看可视化结果：  

<div  align="center">  
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

特别说明，exe运行时所需要的依赖库配置方法，请参考章节: [多种方法配置exe运行时所需的依赖库](#CommandLineDeps)

## 6. 多种方法配置exe运行时所需的依赖库  
<div id="CommandLineDeps"></div>  

### 6.1 方式一：修改CMakeLists.txt，一行命令配置(推荐)  
<div id="CommandLineDeps1"></div>  

考虑到Windows下C++开发的特殊性，如经常需要拷贝所有的lib或dll文件到某个指定的目录，FastDeploy提供了`install_fastdeploy_libraries`的cmake函数，方便用户快速配置所有的dll。修改ppyoloe的CMakeLists.txt，添加：  
```cmake
install_fastdeploy_libraries(${CMAKE_CURRENT_BINARY_DIR}/Release)
```

### 6.2 方式二：命令行设置环境变量  
<div id="CommandLineDeps2"></div>  

编译好的exe保存在Release目录下，在运行demo前，需要将模型和测试图片拷贝至该目录。另外，需要在终端指定DLL的搜索路径。请在build目录下执行以下命令。
```bat
set FASTDEPLOY_HOME=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.1
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
```  
注意，需要拷贝onnxruntime.dll到exe所在的目录。
```bat
copy /Y %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\onnxruntime* Release\
```  
由于较新的Windows在System32系统目录下自带了onnxruntime.dll，因此就算设置了PATH，系统依然会出现onnxruntime的加载冲突。因此需要先拷贝demo用到的onnxruntime.dll到exe所在的目录。如下
```bat
where onnxruntime.dll
C:\Windows\System32\onnxruntime.dll  # windows自带的onnxruntime.dll
```  
可以把上述命令拷贝并保存到build目录下的某个bat脚本文件中(包含copy onnxruntime)，如`setup_fastdeploy_dll.bat`，方便多次使用。
```bat
setup_fastdeploy_dll.bat
```

### 6.3 方式三：手动拷贝依赖库到exe的目录下  

<div id="CommandLineDeps3"></div>  

手动拷贝，或者在build目录下执行以下命令：
```bat
set FASTDEPLOY_HOME=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.1
copy /Y %FASTDEPLOY_HOME%\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\bin\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\paddle\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mkldnn\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle_inference\third_party\install\mklml\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\paddle2onnx\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\tensorrt\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\faster_tokenizer\third_party\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\yaml-cpp\lib\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\bin\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\bin\*.xml Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\3rdparty\tbb\bin\*.dll Release\
```
可以把上述命令拷贝并保存到build目录下的某个bat脚本文件中，如`copy_fastdeploy_dll.bat`，方便多次使用。
```bat
copy_fastdeploy_dll.bat
```
特别说明：上述的set和copy命令对应的依赖库路径，需要用户根据自己使用SDK中的依赖库进行适当地修改。比如，若是CPU版本的SDK，则不需要TensorRT相关的设置。



## 7. CLion 2022 IDE 方式使用 C++ SDK
<div id="CLion"></div>  

- TODO  


## 8. Visual Studio Code IDE 方式使用 C++ SDK
<div id="VisualStudioCode"></div>  

- TODO  
