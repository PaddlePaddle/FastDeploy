English | [中文](../../cn/faq/use_sdk_on_windows.md)

# Using the FastDeploy C++ SDK on Windows Platform

## Contents

- [Using the FastDeploy C++ SDK on Windows Platform](#using-the-fastdeploy-c-sdk-on-windows-platform)
  - [Contents](#contents)
  - [1. Environment Dependent](#1-environment-dependent)
  - [2. Download FastDeploy Windows 10 C++ SDK](#2-download-fastdeploy-windows-10-c-sdk)
    - [2.1 Download the Pre-built Library or Build the Latest SDK from Source](#21-download-the-pre-built-library-or-build-the-latest-sdk-from-source)
    - [2.2 Prepare Model Files and Test Images](#22-prepare-model-files-and-test-images)
  - [3. Various ways to use C++ SDK on Windows Platform](#3-various-ways-to-use-c-sdk-on-windows-platform)
    - [3.1 SDK usage method 1：Using the C++ SDK from the Command Line](#31-sdk-usage-method-1using-the-c-sdk-from-the-command-line)
      - [3.1.1 Build PPYOLOE on Windows Platform](#311-build-ppyoloe-on-windows-platform)
      - [3.1.2 Run Demo](#312-run-demo)
    - [3.2 SDK usage method 2: Visual Studio 2019 creates sln project using C++ SDK](#32-sdk-usage-method-2-visual-studio-2019-creates-sln-project-using-c-sdk)
      - [3.2.1 Step 1：Visual Studio 2019 creates sln project project](#321-step-1visual-studio-2019-creates-sln-project-project)
      - [3.2.2 Step 2：Copy the code of infer\_ppyoloe.cc from examples to the project](#322-step-2copy-the-code-of-infer_ppyoloecc-from-examples-to-the-project)
      - [3.2.3 Step 3：Set the project configuration to "Release x64" configuration](#323-step-3set-the-project-configuration-to-release-x64-configuration)
      - [3.2.4 Step 4：Configure Include Header File Path](#324-step-4configure-include-header-file-path)
      - [3.2.5 Step 5：Configure Lib Path and Add Library Files](#325-step-5configure-lib-path-and-add-library-files)
      - [3.2.6 Step 6：Build the Project and Run to Get the Result](#326-step-6build-the-project-and-run-to-get-the-result)
    - [3.3 Visual Studio 2019 Create CMake project using C++ SDK](#33-visual-studio-2019-create-cmake-project-using-c-sdk)
      - [3.3.1 Step 1： Visual Studio 2019 Creates a CMake Project](#331-step-1-visual-studio-2019-creates-a-cmake-project)
      - [3.3.2 Step 2：Configure FastDeploy C++ SDK in CMakeLists](#332-step-2configure-fastdeploy-c-sdk-in-cmakelists)
      - [3.3.3 Step 3：Generate project cache and Modify CMakeSetting.json Configuration](#333-step-3generate-project-cache-and-modify-cmakesettingjson-configuration)
      - [3.3.4 Step 4：Generate executable file, Run to Get the Result](#334-step-4generate-executable-file-run-to-get-the-result)
  - [4. Multiple methods to Configure the Required Dependencies for the Exe Runtime](#4-multiple-methods-to-configure-the-required-dependencies-for-the-exe-runtime)
    - [4.1  Use method 1：Use Fastdeploy\_init.bat for Configuration (Recommended)](#41--use-method-1use-fastdeploy_initbat-for-configuration-recommended)
      - [4.1.1 fastdeploy\_init.bat User's Manual](#411-fastdeploy_initbat-users-manual)
      - [4.1.2 fastdeploy\_init.bat View all dll, lib and include paths in the SDK](#412-fastdeploy_initbat-view-all-dll-lib-and-include-paths-in-the-sdk)
      - [4.1.3 fastdeploy\_init.bat Installs all the dlls in the SDK to the specified directory](#413-fastdeploy_initbat-installs-all-the-dlls-in-the-sdk-to-the-specified-directory)
      - [4.1.4 fastdeploy\_init.bat Configures SDK Environment Variables](#414-fastdeploy_initbat-configures-sdk-environment-variables)
    - [4.2  Use method 2：Modify CMakeLists.txt, One Line of Command Configuration (Recommended)](#42--use-method-2modify-cmakeliststxt-one-line-of-command-configuration-recommended)
    - [4.3  Use method 3：Command Line Setting Environment Variables](#43--use-method-3command-line-setting-environment-variables)
    - [4.4 Use method 4：Manually Copy the Dependency Library to the Exe Directory](#44-use-method-4manually-copy-the-dependency-library-to-the-exe-directory)


## 1. Environment Dependent
<div id="Environment"></div>  

- cmake >= 3.12
- Visual Studio 2019
- cuda >= 11.2 (WITH_GPU=ON)
- cudnn >= 8.0 (WITH_GPU=ON)

## 2. Download FastDeploy Windows 10 C++ SDK
<div id="Download"></div>  

### 2.1 Download the Pre-built Library or Build the Latest SDK from Source
The compiled FastDeploy Windows 10 C++ SDK can be downloaded from the link below, and the examples code is included in the SDK.
```text
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-win-x64-gpu-0.2.1.zip
```
Please refer to source code compilation: [build_and_install](../build_and_install)
### 2.2 Prepare Model Files and Test Images
Model files and test images can be downloaded from the link below and unzipped
```text
https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz # (please unzip it after downloading)
https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

## 3. Various ways to use C++ SDK on Windows Platform
### 3.1 SDK usage method 1：Using the C++ SDK from the Command Line
<div id="CommandLine"></div>  

#### 3.1.1 Build PPYOLOE on Windows Platform
Open `x64 Native Tools Command Prompt for VS 2019` command tool on Winodws, cd to the demo path of ppyoloe:
```bat  
cd fastdeploy-win-x64-gpu-0.2.1\examples\vision\detection\paddledetection\cpp
```
```bat
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=%cd%\..\..\..\..\..\..\..\fastdeploy-win-x64-gpu-0.2.1 -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
```
Then Run
```bat
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

#### 3.1.2 Run Demo
```bat
cd Release
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 0  # CPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 1  # GPU
infer_ppyoloe_demo.exe ppyoloe_crn_l_300e_coco 000000014439.jpg 2  # GPU + TensorRT
```  

In particular, for the configuration method of the dependency library required by the exe runtime, please refer to the chapter: [Multiple methods to configure the dependency library required by the exe runtime](#CommandLineDeps)


### 3.2 SDK usage method 2: Visual Studio 2019 creates sln project using C++ SDK

This section is for non-CMake users and describes how to create a sln project in Visual Studio 2019 to use FastDeploy C++ SDK. CMake users please read the next section directly. In addition, this section is a special thanks to "Awake to the Southern Sky" for his tutorial on FastDeploy: [How to deploy PaddleDetection target detection model on Windows using FastDeploy C++](https://www.bilibili.com/read/cv18807232).

<div id="VisualStudio2019Sln"></div>  

#### 3.2.1 Step 1：Visual Studio 2019 creates sln project project

<div id="VisualStudio2019Sln1"></div>  

（1） Open Visual Studio 2019 and click on "Create New Project" -> click on "Console Program" to create a new sln project.

![image](https://user-images.githubusercontent.com/31974251/192813386-cf9a93e0-ee42-42b3-b8bf-d03ae7171d4e.png)

![image](https://user-images.githubusercontent.com/31974251/192816516-a4965b9c-21c9-4a01-bbb2-c648a8256fc9.png)

（2）Click "Create" and an empty sln project is created. We copy the code of infer_ppyoloe directly from examples here.

![image](https://user-images.githubusercontent.com/31974251/192817382-643c8ca2-1f2a-412e-954e-576c22b4ea62.png)

#### 3.2.2 Step 2：Copy the code of infer_ppyoloe.cc from examples to the project

<div id="VisualStudio2019Sln2"></div>  

（1）Copy the code of infer_ppyoloe.cc from examples to the project and replace it directly, the path to copy the code is：  
```bat
fastdeploy-win-x64-gpu-0.2.1\examples\vision\detection\paddledetection\cpp
```

![image](https://user-images.githubusercontent.com/31974251/192818456-21ca846c-ab52-4001-96d2-77c8174bff6b.png)  

#### 3.2.3 Step 3：Set the project configuration to "Release x64" configuration

<div id="VisualStudio2019Sln3"></div>  

![image](https://user-images.githubusercontent.com/31974251/192818918-98d7a54c-4a60-4760-a3cb-ecacc38b7e7a.png)

#### 3.2.4 Step 4：Configure Include Header File Path

<div id="VisualStudio2019Sln4"></div>  


（1）Configure the header file include path: select the project with the mouse, and then right-click to pop down the menu, in which click "Properties"。

![image](https://user-images.githubusercontent.com/31974251/192820573-23096aea-046c-4bb4-9929-c412718805cb.png)


（2）In the pop-up property page, select: C/C++ -> General -> Additional Include Directories, and then add the paths to the fastdeploy and opencv headers. As：  

```bat  

D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\include
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv-win-x64-3.4.16\build\include  
```  
Note that if you compile the latest SDK or version >0.2.1, the directory structure of the dependency library has changed, and the opencv path needs to be modified appropriately. For example：  

```bat  
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv\build\include  
```

![image](https://user-images.githubusercontent.com/31974251/192824445-978c06ed-cc14-4d6a-8ccf-d4594ca11533.png)

Developers need to make slight modifications according to their actual sdk path.


#### 3.2.5 Step 5：Configure Lib Path and Add Library Files

<div id="VisualStudio2019Sln5"></div>  

（1）In the property page, select: Linker -> General -> Additional Libraries Directory, then add the lib paths for fastdeploy and opencv. As：  
```bat  
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\lib
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv-win-x64-3.4.16\build\x64\vc15\lib
```
Note that if you compile the latest SDK or version >0.2.1, the directory structure of the dependency library has changed, and the opencv path needs to be modified appropriately. For example：  
```bat  
D:\qiuyanjun\fastdeploy_build\built\fastdeploy-win-x64-gpu-0.2.1\third_libs\install\opencv\build\include  
```  

![image](https://user-images.githubusercontent.com/31974251/192826130-fe28791f-317c-4e66-a6a5-133e60b726f0.png)

（2）Adding library files: only fastdeploy.lib and opencv_world3416.lib are needed  

 ![image](https://user-images.githubusercontent.com/31974251/192826884-44fc84a1-c57a-45f1-8ee2-30b7eaa3dce9.png)

#### 3.2.6 Step 6：Build the Project and Run to Get the Result

<div id="VisualStudio2019Sln6"></div>  


（1）Click on the menu bar "Generate" -> "Generate Solution"

![image](https://user-images.githubusercontent.com/31974251/192827608-beb53685-2f94-44dc-aa28-49b09a4ab864.png)

![image](https://user-images.githubusercontent.com/31974251/192827842-1f05d435-8a3e-492b-a3b7-d5e88f85f814.png)  

Compile successfully, you can see the exe saved in：

```bat  
D:\qiuyanjun\fastdeploy_test\infer_ppyoloe\x64\Release\infer_ppyoloe.exe  
```  

（2）Execute the executable file and get the inference result. First you need to copy all the dlls to the directory where the exe is located. At the same time, you also need to download and extract the pyoloe model files and test images, and then copy them to the directory where the exe is located. Special note, the exe needs to run when the dependency library configuration method, please refer to the section: [various methods to configure the exe to run the required dependency library](#CommandLineDeps).  

![image](https://user-images.githubusercontent.com/31974251/192829545-3ea36bfc-9a54-492b-984b-2d5d39094d47.png)  


### 3.3 Visual Studio 2019 Create CMake project using C++ SDK
<div id="VisualStudio2019"></div>  

This section is for CMake users and describes how to create CMake projects in Visual Studio 2019 using the FastDeploy C++ SDK.

#### 3.3.1 Step 1： Visual Studio 2019 Creates a CMake Project

<div id="VisualStudio20191"></div>  

（1）Open Visual Studio 2019, click "Create New Project" -> click "CMake" to create a CMake project. Take PPYOLOE as an example to illustrate how to use FastDeploy C++ SDK in Visual Studio 2019 IDE.

![image](https://user-images.githubusercontent.com/31974251/192143543-9f29e4cb-2307-45ca-a61a-bcfba5df19ff.png)

![image](https://user-images.githubusercontent.com/31974251/192143640-39e79c65-8b50-4254-8da6-baa21bb23e3c.png)  


![image](https://user-images.githubusercontent.com/31974251/192143713-be2e6490-4cab-4151-8463-8c367dbc451a.png)

（2）Open the project and find that Visual Stuio 2019 has generated some basic files for us, including CMakeLists.txt. infer_ppyoloe.h header file is not actually used here, we can just delete it.  

![image](https://user-images.githubusercontent.com/31974251/192143930-db1655c2-66ee-448c-82cb-0103ca1ca2a0.png)  

#### 3.3.2 Step 2：Configure FastDeploy C++ SDK in CMakeLists

<div id="VisualStudio20192"></div>  

（1）After the project is created, we need to add the infer_ppyoloe inference source code and modify CMakeLists.txt as follows:

![image](https://user-images.githubusercontent.com/31974251/192144782-79bccf8f-65d0-4f22-9f41-81751c530319.png)

（2）The code of infer_ppyoloe.cpp can be copied directly from the code in examples：  
- [examples/vision/detection/paddledetection/cpp/infer_ppyoloe.cc](../../../examples/vision/detection/paddledetection/cpp/infer_ppyoloe.cc)

（3）CMakeLists.txt mainly includes the configuration of the path of FastDeploy C++ SDK, if it is the GPU version of the SDK, you also need to configure CUDA_DIRECTORY as the installation path of CUDA, the configuration of CMakeLists.txt is as follows：

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

Note that the `install_fastdeploy_libraries` function is only valid with the latest code compiled for the SDK or version >0.2.1.

#### 3.3.3 Step 3：Generate project cache and Modify CMakeSetting.json Configuration

<div id="VisualStudio20193"></div>  

（1）Click on "CMakeLists.txt" -> right click on "Generate Cache":  

![image](https://user-images.githubusercontent.com/31974251/192145349-c78b110a-0e41-4ee5-8942-3bf70bd94a75.png)

We found that the cache has been successfully generated, but since the default is Debug mode when opening the project, we found that the exe and cache save path is still in Debug mode. We can first modify the CMake settings to Release.

（2）Click "CMakeLists.txt"->right-click "cmake settings for infer_ppyoloe_demo", enter the CMakeSettings.json settings panel, change the Debug setting to Release.  

![image](https://user-images.githubusercontent.com/31974251/192145242-01d37b44-e2fa-47df-82c1-c11c2ccbff99.png)  

Also set CMake Builder to "Visual Studio 16 2019 Win64".

![image](https://user-images.githubusercontent.com/31974251/192147961-ac46d0f6-7349-4126-a123-914af2b63d95.jpg)

（3）Click Save CMake Cache to switch to Release configuration：  

![image](https://user-images.githubusercontent.com/31974251/192145974-b5a63341-9143-49a2-8bfe-94ac641b1670.png)

（4）：（4.1）Click "CMakeLists.txt"->right click "CMake Cache for x64-Release only"->"Click to delete cache"; (4.2) Click "CMakeLists.txt"->"Generate cache"; (4.3) If you find the option to delete cache is grayed out in step 1, you can directly click "CMakeLists. txt"->"Generate", if it fails, you can try again (4.1) and (4.2)

![image](https://user-images.githubusercontent.com/31974251/192146394-51fbf2b8-1cba-41ca-bb45-5f26890f64ce.jpg)  

Finally, you can see that the configuration has successfully generated the CMake cache in Relase mode.  

![image](https://user-images.githubusercontent.com/31974251/192146239-a1eacd9e-034d-4373-a262-65b18ce25b87.png)  


#### 3.3.4 Step 4：Generate executable file, Run to Get the Result

<div id="VisualStudio20194"></div>  

（1）Click "CMakeLists.txt"->"Generate". You can find that infer_ppyoloe_demo.exe has been successfully generated and saved in the `out/build/x64-Release/Release` directory.  

![image](https://user-images.githubusercontent.com/31974251/192146852-c64d2252-8c8f-4309-a950-908a5cb258b8.png)

（2）Execute the executable file and get the inference result. First you need to copy all the dlls to the directory where the exe is located, here we can add a command in CMakeLists.txt to install all the dlls in FastDeploy to the specified directory. Note that this method is only valid for the latest code compiled SDK or version >0.2.1. For other configuration methods, please refer to the section: [Multiple Methods to Configure Dependencies for exe Runtime].(#CommandLineDeps)  

```cmake  
install_fastdeploy_libraries(${CMAKE_CURRENT_BINARY_DIR}/Release)
```  
（3）At the same time, you also need to download and unzip the pyoloe model files and test images and copy them to the directory where the exe is located. After preparation, the directory structure is as follows：  

![image](https://user-images.githubusercontent.com/31974251/192147505-054edb77-564b-405e-89ee-fd0d2e413e78.png)

（4）Finally, the following command is executed to obtain the inference results：  

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

Open the saved image to view the visualization results at：  

<div  align="center">  
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

Special note, the exe needs to run when the dependency library configuration method, please refer to the section: [a variety of methods to configure the exe to run the required dependency library](#CommandLineDeps).

## 4. Multiple methods to Configure the Required Dependencies for the Exe Runtime
<div id="CommandLineDeps"></div>  
Note: For users using the latest source compiled SDK or SDK version >0.2.1, we recommend configuring the runtime dependencies in the way in (4.1) and (4.2). If you are using SDK version <= 0.2.1, please refer to (4.3) and (4.4) for configuration.

### 4.1  Use method 1：Use Fastdeploy_init.bat for Configuration (Recommended)  
<div id="CommandLineDeps1"></div>  

For SDK versions higher than 0.2.1, we provide the **fastdeploy_init.bat** tool to manage all the dependent libraries in FastDeploy. This scripting tool allows you to view (show), copy (install) and set up (init and setup) all the dlls in the SDK, allowing you to quickly configure the runtime environment.


#### 4.1.1 fastdeploy_init.bat User's Manual
<div id="CommandLineDeps11"></div>  

First go to the root directory of the SDK and run the following command, you can see the usage description of fastdeploy_init.bat：
```bat
D:\path-to-your-fastdeploy-sdk-dir>fastdeploy_init.bat help
------------------------------------------------------------------------------------------------------------------------------------------------------------
[1] [help]    print help information:                      fastdeploy_init.bat help
[2] [show]    show all dlls/libs/include paths:            fastdeploy_init.bat show fastdeploy-sdk-dir
[3] [init]    init all dlls paths for current terminal:    fastdeploy_init.bat init fastdeploy-sdk-dir  [WARNING: need copy onnxruntime.dll manually]
[4] [setup]   setup path env for current terminal:         fastdeploy_init.bat setup fastdeploy-sdk-dir [WARNING: need copy onnxruntime.dll manually]
[5] [install] install all dlls to a specific dir:          fastdeploy_init.bat install fastdeploy-sdk-dir another-dir-to-install-dlls **[RECOMMEND]**
[6] [install] install all dlls with logging infos:         fastdeploy_init.bat install fastdeploy-sdk-dir another-dir-to-install-dlls info
------------------------------------------------------------------------------------------------------------------------------------------------------------
```  
A brief description of the usage is as follows.  
- help:  prints all usage notes  
- show:  view all dlls, libs and include paths in the SDK
- init:  initialize all dll paths and subsequently set the terminal environment variables (not recommended, please refer to 4.3 for onnxruntime)
- setup: run after init to set the terminal environment (not recommended, please refer to 4.3 for onnxruntime)  
- install: installs all dlls in the SDK into a specified directory (recommended)

#### 4.1.2 fastdeploy_init.bat View all dll, lib and include paths in the SDK  
<div id="CommandLineDeps12"></div>  

Go to the root directory of the SDK and run the show command to view all the dll, lib and include paths in the SDK. In the following command, %cd% means the current directory (the root directory of the SDK).

```bat
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat show %cd%
------------------------------------------------------------------------------------------------------------------------------------------------------------
[SDK] D:\path-to-fastdeploy-sdk-dir
------------------------------------------------------------------------------------------------------------------------------------------------------------
[DLL] D:\path-to-fastdeploy-sdk-dir\lib\fastdeploy.dll **[NEEDED]**
[DLL] D:\path-to-fastdeploy-sdk-dir\third_libs\install\faster_tokenizer\lib\core_tokenizers.dll  **[NEEDED]**
[DLL] D:\path-to-fastdeploy-sdk-dir\third_libs\install\opencv\build\x64\vc15\bin\opencv_ffmpeg3416_64.dll  **[NEEDED]**
......
------------------------------------------------------------------------------------------------------------------------------------------------------------
[Lib] D:\path-to-fastdeploy-sdk-dir\lib\fastdeploy.lib **[NEEDED][fastdeploy]**
[Lib] D:\path-to-fastdeploy-sdk-dir\third_libs\install\faster_tokenizer\lib\core_tokenizers.lib  **[NEEDED][fastdeploy::text]**
[Lib] D:\path-to-fastdeploy-sdk-dir\third_libs\install\opencv\build\x64\vc15\lib\opencv_world3416.lib  **[NEEDED][fastdeploy::vision]**
......
------------------------------------------------------------------------------------------------------------------------------------------------------------
[Include] D:\path-to-fastdeploy-sdk-dir\include **[NEEDED][fastdeploy]**
[Include] D:\path-to-fastdeploy-sdk-dir\third_libs\install\faster_tokenizer\include  **[NEEDED][fastdeploy::text]**
[Include] D:\path-to-fastdeploy-sdk-dir\third_libs\install\opencv\build\include  **[NEEDED][fastdeploy::vision]**
......
------------------------------------------------------------------------------------------------------------------------------------------------------------
[XML] D:\path-to-fastdeploy-sdk-dir\third_libs\install\openvino\runtime\bin\plugins.xml **[NEEDED]**
------------------------------------------------------------------------------------------------------------------------------------------------------------
```

You can see that this command will output the corresponding information according to your current SDK, including the path information of dll, lib and include. For dll, those marked as `[NEEDED]` are required for runtime, and if it contains OpenVINO backend, you also need to copy its plugins.xml to the directory where the exe is located; for lib and include, those marked as `[NEEDED]` are the minimum dependencies that need to be configured for development. And, we also added the corresponding API Tag, if you only use the vision API, you only need to configure the lib and include paths marked as `[NEEDED][fastdeploy::vision]`.  

#### 4.1.3 fastdeploy_init.bat Installs all the dlls in the SDK to the specified directory
<div id="CommandLineDeps13"></div>  

Go to the root directory of the SDK and run the install command to install all the dlls in the SDK to the specified directory (such as the directory where the exe is located). We recommend this way to configure the dependent libraries needed for the exe to run. For example, you can create a temporary bin directory in the root directory of the SDK to backup all dll files. The following command %cd% indicates the current directory (the root directory of the SDK).

```bat  
% info Parameters are optional, adding info parameters will print detailed installation information %
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat install %cd% bin
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat install %cd% bin info
```
```bat
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat install %cd% bin
[INFO] Do you want to install all FastDeploy dlls ?
[INFO] From: D:\path-to-fastdeploy-sdk-dir
[INFO]   To: bin
Choose y means YES, n means NO: [y/n]y
YES.
Please press any key to continue. . .
[INFO] Created bin done!
1 file has been copied.
1 file has been copied.
1 file has been copied.
1 file has been copied.
.....
1 file has been copied.
1 file has been copied.
Copied 1 file.
1 file has been copied.
.....
```  
#### 4.1.4 fastdeploy_init.bat Configures SDK Environment Variables
<div id="CommandLineDeps14"></div>  

Optionally, you can set the runtime dependency library environment by configuring environment variables, which is only valid for the current terminal. If you are using an SDK that includes an onnxruntime inference backend, we do not recommend this approach, for detailed reasons please refer to the description of onnxruntime configuration in (4.3) (you need to manually copy all dlls of onnxruntime to the directory where the exe is located). Configure the SDK environment variables in the following way. In the following command %cd% means the current directory (root directory of the SDK).

```bat
% First run init to initialize all the dll file paths of the current SDK %
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat init %cd%
% Run setup again to complete the SDK environment variable configuration  %
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat setup %cd%
```

### 4.2  Use method 2：Modify CMakeLists.txt, One Line of Command Configuration (Recommended)
<div id="CommandLineDeps2"></div>  

Considering the special characteristics of C++ development under Windows,if the frequent need to copy all lib or dll files to a specified directory, FastDeploy provides the `install_fastdeploy_libraries` cmake function to facilitate users to quickly configure all dlls. modify PP-YOLOE's CMakeLists.txt。 As follows：

```cmake
install_fastdeploy_libraries(${CMAKE_CURRENT_BINARY_DIR}/Release)
```
Note that this method is only valid with the latest code compiled for the SDK or version > 0.2.1.  

### 4.3  Use method 3：Command Line Setting Environment Variables
<div id="CommandLineDeps3"></div>  

The compiled exe is saved in the Release directory, and you need to copy the model and test images to this directory before running the demo. In addition, you need to specify the search path of the DLL in the terminal. Please execute the following command in the build directory.

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
Note that you need to copy onnxruntime.dll to the directory where the exe is located.
```bat
copy /Y %FASTDEPLOY_HOME%\third_libs\install\onnxruntime\lib\onnxruntime* Release\
```  
Since newer Windows comes with onnxruntime.dll in System32 system directory, even if PATH is set, the system will still have a loading conflict with onnxruntime. Therefore, you need to copy the onnxruntime.dll used in the demo to the directory where the exe is located. As follows：
```bat
where onnxruntime.dll
C:\Windows\System32\onnxruntime.dll  # windows comes with onnxruntime.dll
```  
Note that if you compile the latest SDK or version >0.2.1 by yourself, the opencv and openvino directory structure has changed and the path needs to be modified appropriately. For example：  
```bat  
set PATH=%FASTDEPLOY_HOME%\third_libs\install\opencv\build\x64\vc15\bin;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\openvino\runtime\bin;%PATH%
set PATH=%FASTDEPLOY_HOME%\third_libs\install\openvino\runtime\3rdparty\tbb\bin;%PATH%
```
You can copy the above command and save it to some bat script file (containing copy onnxruntime) in the build directory, such as `setup_fastdeploy_dll.bat`, for multiple use.

```bat
setup_fastdeploy_dll.bat
```

### 4.4 Use method 4：Manually Copy the Dependency Library to the Exe Directory

<div id="CommandLineDeps4"></div>  

Copy manually, or execute the following command in the build directory：
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
Note that if you compile the latest SDK or version >0.2.1 by yourself, the opencv and openvino directory structure has changed and the path needs to be modified appropriately. For example：
```bat  
copy /Y %FASTDEPLOY_HOME%\third_libs\install\opencv\build\x64\vc15\bin\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\runtime\bin\*.dll Release\
copy /Y %FASTDEPLOY_HOME%\third_libs\install\openvino\runtime\3rdparty\tbb\bin\*.dll Release\
```
You can copy the above command and save it to some bat script file in the build directory, such as `copy_fastdeploy_dll.bat`, for multiple use.
```bat
copy_fastdeploy_dll.bat
```
Special Note: The dependency library paths corresponding to the set and copy commands above need to be modified appropriately by the user according to the dependency libraries in the SDK they are using. For example, if it is the CPU version of the SDK, the TensorRT-related settings are not required.
