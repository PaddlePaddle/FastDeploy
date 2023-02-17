# FastDeploy C++库在Windows上的多种使用方式 

## 目录
- [1. 环境依赖](#Environment)  
- [2. 下载 FastDeploy Windows 10 C++ SDK](#Download)  
- [3. Windows下多种方式使用 C++ SDK 的方式](#CommandLine)
  - [3.1 命令行方式使用 C++ SDK](#CommandLine)  
    - [3.1.1 在 Windows 命令行终端 上编译 example](#CommandLine)  
    - [3.1.2 运行可执行文件获得推理结果](#CommandLine)  
  - [3.2 Visual Studio 2019 创建sln工程使用 C++ SDK](#VisualStudio2019Sln)  
    - [3.2.1 Visual Studio 2019 创建 sln 工程项目](#VisualStudio2019Sln1)  
    - [3.2.2 从examples中拷贝infer_ppyoloe.cc的代码到工程](#VisualStudio2019Sln2)  
    - [3.2.3 将工程配置设置成"Release x64"配置](#VisualStudio2019Sln3)  
    - [3.2.4 配置头文件include路径](#VisualStudio2019Sln4)  
    - [3.2.5 配置lib路径和添加库文件](#VisualStudio2019Sln5)  
    - [3.2.6 编译工程并运行获取结果](#VisualStudio2019Sln6)
  - [3.3 Visual Studio 2019 创建CMake工程使用 C++ SDK](#VisualStudio2019)
    - [3.3.1 Visual Studio 2019 创建CMake工程项目](#VisualStudio20191)  
    - [3.3.2 在CMakeLists中配置 FastDeploy C++ SDK](#VisualStudio20192)  
    - [3.3.3 生成工程缓存并修改CMakeSetting.json配置](#VisualStudio20193)  
    - [3.3.4 生成可执行文件，运行获取结果](#VisualStudio20194)  


## 1. 环境依赖
<div id="Environment"></div>  

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 8.0 (当WITH_GPU=ON)

## 2. 下载 FastDeploy Windows 10 C++ SDK
<div id="Download"></div>  

### 2.1 下载预编译库或者从源码编译最新的SDK
可以从以下链接下载编译好的 FastDeploy Windows 10 C++ SDK，SDK中包含了examples代码。
```text
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip
```
源码编译请参考: [build_and_install](../build_and_install)
### 2.2 准备模型文件和测试图片
可以从以下链接下载模型文件和测试图片，并解压缩
```text
https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz # (下载后解压缩)
https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

## 3. Windows下多种方式使用 C++ SDK 的方式
### 3.1 SDK使用方式一：命令行方式使用 C++ SDK
<div id="CommandLine"></div>  

#### 3.1.1 在 Windows 上编译 PPYOLOE
Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，cd到ppyoloe的demo路径  
```bat  
cd fastdeploy-win-x64-gpu-0.2.1\examples\vision\detection\paddledetection\cpp
```
```bat
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.1 -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
```
然后执行
```bat
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

#### 3.1.2 运行 demo
```bat
cd Release
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0  # CPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 1  # GPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 2  # GPU + TensorRT
```  

特别说明，exe运行时所需要的依赖库配置方法，请参考章节: [多种方法配置exe运行时所需的依赖库](#CommandLineDeps)

### 3.2 SDK使用方式二：Visual Studio 2019 创建 sln 工程使用 C++ SDK

本章节针对非CMake用户，介绍如何在Visual Studio 2019 中创建 sln 工程使用 FastDeploy C++ SDK. CMake用户请直接看下一章节。另外，本章节内容特别感谢“梦醒南天”同学关于FastDeploy使用的文档教程：[如何在 Windows 上使用 FastDeploy C++ 部署 PaddleDetection 目标检测模型](https://www.bilibili.com/read/cv18807232)

<div id="VisualStudio2019Sln"></div>  

#### 3.2.1 步骤一：Visual Studio 2019 创建 sln 工程项目

<div id="VisualStudio2019Sln1"></div>  

（1） 打开Visual Studio 2019，点击"创建新项目"->点击"控制台程序"，从而创建新的sln工程项目.

![image](https://user-images.githubusercontent.com/31974251/192813386-cf9a93e0-ee42-42b3-b8bf-d03ae7171d4e.png)

![image](https://user-images.githubusercontent.com/31974251/192816516-a4965b9c-21c9-4a01-bbb2-c648a8256fc9.png)

（2）点击“创建”，便创建了一个空的sln工程。我们直接从examples里面拷贝infer_ppyoloe的代码这里。

![image](https://user-images.githubusercontent.com/31974251/192817382-643c8ca2-1f2a-412e-954e-576c22b4ea62.png)

#### 3.2.2 步骤二：从examples中拷贝infer_ppyoloe.cc的代码到工程

<div id="VisualStudio2019Sln2"></div>  

（1）从examples中拷贝infer_ppyoloe.cc的代码到工程，直接替换即可，拷贝代码的路径为：  
```bat
fastdeploy-win-x64-gpu-0.2.1\examples\vision\detection\paddledetection\cpp
```

![image](https://user-images.githubusercontent.com/31974251/192818456-21ca846c-ab52-4001-96d2-77c8174bff6b.png)  

#### 3.2.3 步骤三：将工程配置设置成"Release x64"配置

<div id="VisualStudio2019Sln3"></div>  

![image](https://user-images.githubusercontent.com/31974251/192818918-98d7a54c-4a60-4760-a3cb-ecacc38b7e7a.png)

#### 3.2.4 步骤四：配置头文件include路径

<div id="VisualStudio2019Sln4"></div>  


（1）配置头文件include路径：鼠标选择项目，然后单击右键即可弹出下来菜单，在其中单击“属性”。

![image](https://user-images.githubusercontent.com/31974251/192820573-23096aea-046c-4bb4-9929-c412718805cb.png)


（2）在弹出来的属性页中选择：C/C++ —> 常规 —> 附加包含目录，然后在添加 fastdeploy 和 opencv 的头文件路径。如：  

```bat  

D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\include
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv-win-x64-3.4.16\build\include  
```  
注意，如果是自行编译最新的SDK或版本>0.2.1，依赖库目录结构有所变动，opencv路径需要做出适当的修改。如：  
```bat  
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv\build\include  
```

![image](https://user-images.githubusercontent.com/31974251/192824445-978c06ed-cc14-4d6a-8ccf-d4594ca11533.png)

用户需要根据自己实际的sdk路径稍作修改。


#### 3.2.5 步骤五：配置lib路径和添加库文件

<div id="VisualStudio2019Sln5"></div>  

（1）属性页中选择：链接器—>常规—> 附加库目录，然后在添加 fastdeploy 和 opencv 的lib路径。如：  
```bat  
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\lib
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\lib
```
注意，如果是自行编译最新的SDK或版本>0.2.1，依赖库目录结构有所变动，opencv路径需要做出适当的修改。如：  
```bat  
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv\build\include  
```  

![image](https://user-images.githubusercontent.com/31974251/192826130-fe28791f-317c-4e66-a6a5-133e60b726f0.png)

（2）添加库文件：只需要 fastdeploy.lib 和 opencv_world3416.lib  

 ![image](https://user-images.githubusercontent.com/31974251/192826884-44fc84a1-c57a-45f1-8ee2-30b7eaa3dce9.png)

#### 3.2.6 步骤六：编译工程并运行获取结果

<div id="VisualStudio2019Sln6"></div>  


（1）点击菜单栏“生成”->“生成解决方案”

![image](https://user-images.githubusercontent.com/31974251/192827608-beb53685-2f94-44dc-aa28-49b09a4ab864.png)

![image](https://user-images.githubusercontent.com/31974251/192827842-1f05d435-8a3e-492b-a3b7-d5e88f85f814.png)  

编译成功，可以看到exe保存在：  
```bat  
D:\qiuyanjun\fastdeploy_test\infer_ppyoloe\x64\Release\infer_ppyoloe.exe  
```  

（2）执行可执行文件，获得推理结果。 首先需要拷贝所有的dll到exe所在的目录下。同时，也需要把ppyoloe的模型文件和测试图片下载解压缩后，拷贝到exe所在的目录。 特别说明，exe运行时所需要的依赖库配置方法，请参考章节: [多种方法配置exe运行时所需的依赖库](#CommandLineDeps)  

![image](https://user-images.githubusercontent.com/31974251/192829545-3ea36bfc-9a54-492b-984b-2d5d39094d47.png)  


### 3.3 SDK使用方式三：Visual Studio 2019 创建 CMake 工程使用 C++ SDK
<div id="VisualStudio2019"></div>  

本章节针对CMake用户，介绍如何在Visual Studio 2019 中创建 CMake 工程使用 FastDeploy C++ SDK.

#### 3.3.1 步骤一：Visual Studio 2019 创建“CMake”工程项目

<div id="VisualStudio20191"></div>  

（1）打开Visual Studio 2019，点击"创建新项目"->点击"CMake"，从而创建CMake工程项目。以PPYOLOE为例，来说明如何在Visual Studio 2019 IDE中使用FastDeploy C++ SDK.

![image](https://user-images.githubusercontent.com/31974251/192143543-9f29e4cb-2307-45ca-a61a-bcfba5df19ff.png)

![image](https://user-images.githubusercontent.com/31974251/192143640-39e79c65-8b50-4254-8da6-baa21bb23e3c.png)  


![image](https://user-images.githubusercontent.com/31974251/192143713-be2e6490-4cab-4151-8463-8c367dbc451a.png)

（2）打开工程发现，Visual Stuio 2019已经为我们生成了一些基本的文件，其中包括CMakeLists.txt。infer_ppyoloe.h头文件这里实际上用不到，我们可以直接删除。  

![image](https://user-images.githubusercontent.com/31974251/192143930-db1655c2-66ee-448c-82cb-0103ca1ca2a0.png)  

#### 3.3.2 步骤二：在CMakeLists中配置 FastDeploy C++ SDK

<div id="VisualStudio20192"></div>  

（1）在工程创建完成后，我们需要添加infer_ppyoloe推理源码，并修改CMakeLists.txt，修改如下：

![image](https://user-images.githubusercontent.com/31974251/192144782-79bccf8f-65d0-4f22-9f41-81751c530319.png)

（2）其中infer_ppyoloe.cpp的代码可以直接从examples中的代码拷贝过来：  
- [examples/vision/detection/paddledetection/cpp/infer_ppyoloe.cc](../../../examples/vision/detection/paddledetection/cpp/infer_ppyoloe.cc)

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
注意，`install_fastdeploy_libraries`函数仅在最新的代码编译的SDK或版本>0.2.1下有效。  

#### 3.3.3 步骤三：生成工程缓存并修改CMakeSetting.json配置

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


#### 3.3.4 步骤四：生成可执行文件，运行获取结果。

<div id="VisualStudio20194"></div>  

（1）点击"CMakeLists.txt"->"生成"。可以发现已经成功生成了infer_ppyoloe_demo.exe，并保存在`out/build/x64-Release/Release`目录下。  

![image](https://user-images.githubusercontent.com/31974251/192146852-c64d2252-8c8f-4309-a950-908a5cb258b8.png)

（2）执行可执行文件，获得推理结果。 首先需要拷贝所有的dll到exe所在的目录下，这里我们可以在CMakeLists.txt添加一下命令，可将FastDeploy中所有的dll安装到指定的目录。注意，该方式仅在最新的代码编译的SDK或版本>0.2.1下有效。其他配置方式，请参考章节: [多种方法配置exe运行时所需的依赖库](#CommandLineDeps)  

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
