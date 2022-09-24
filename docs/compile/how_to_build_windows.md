# FastDeploy Windows SDK 编译

## 目录
- [环境依赖](#Environment)
- [命令行方式编译C++ SDK](#CommandLineCpp)
  - [编译CPU版本 C++ SDK](#CommandLineCppCPU)
  - [编译GPU版本 C++ SDK](#CommandLineCppGPU)
- [命令行方式编译Python Wheel包](#CommandLinePython)
  - [编译CPU版本 Python Wheel包](#CommandLinePythonCPU)
  - [编译GPU版本 Python Wheel包](#CommandLinePythonGPU)
- [CMake GUI + Visual Studio 2019 IDE方式编译C++ SDK](#CMakeGuiAndVS2019)
  - [使用CMake GUI进行基础配置](#CMakeGuiAndVS2019Basic)
  - [编译CPU版本 C++ SDK设置](#CMakeGuiAndVS2019CPU)
  - [编译GPU版本 C++ SDK设置](#CMakeGuiAndVS2019GPU)
  - [使用Visual Studio 2019 IDE进行编译](#CMakeGuiAndVS2019Build)


## 1. 环境依赖
<div id="Environment"></div>  

- cmake >= 3.12
- Visual Studio 16 2019
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 8.0 (当WITH_GPU=ON)
- TensorRT >= 8.4 (当ENABLE_TRT_BACKEND=ON)

## 2. 命令行方式编译C++ SDK
<div id="CommandLineCpp"></div>  

### 编译CPU版本 C++ SDK
<div id="CommandLineCppCPU"></div>  

Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，其中`CMAKE_INSTALL_PREFIX`用于指定编译后生成的SDK路径

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64 -DENABLE_ORT_BACKEND=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_VISION_VISUALIZE=ON
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```
编译后，FastDeploy CPU C++ SDK即在`D:\Paddle\FastDeploy\build\fastdeploy-win-x64`目录下

### 编译GPU版本 C++ SDK
<div id="CommandLineCppGPU"></div>  

Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，其中`CMAKE_INSTALL_PREFIX`用于指定编译后生成的SDK路径

```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64-gpu -DWITH_GPU=ON -DENABLE_ORT_BACKEND=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_VISION_VISUALIZE=ON -DCUDA_DIRECTORY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64  

% 附加说明：%
% (1) -DCUDA_DIRECTORY指定CUDA所在的目录 %
% (2) 若编译Paddle后端，设置-DENABLE_PADDLE_BACKEND=ON %
% (3) 若编译TensorRT后端，需要设置-DENABLE_TRT_BACKEND=ON，并指定TRT_DIRECTORY %
% (4) 如-DTRT_DIRECTORY=D:\x64\third_party\TensorRT-8.4.1.5 %
```
编译后，FastDeploy GPU C++ SDK即在`D:\Paddle\FastDeploy\build\fastdeploy-win-x64-gpu`目录下

## 命令行方式编译Python Wheel包
<div id="CommandLinePython"></div>  

### 编译CPU版本 Python Wheel包
<div id="CommandLinePythonCPU"></div>  

Windows菜单打开x64 Native Tools Command Prompt for VS 2019命令工具。Python编译时，通过环境变量获取编译选项，在命令行终端运行以下命令
```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python && git checkout develop

set ENABLE_ORT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_VISION_VISUALIZE=ON

% 这里指定用户自己的python解释器 以python3.8为例 %
C:\Python38\python.exe setup.py build
C:\Python38\python.exe setup.py bdist_wheel
```
编译好的wheel文件在dist目录下，pip安装编译好的wheel包，命令如下
```bat
C:\Python38\python.exe -m pip install dist\fastdeploy_python-0.2.1-cp38-cp38-win_amd64.whl
```

### 编译GPU版本 Python Wheel包
<div id="CommandLinePythonGPU"></div>  

Windows菜单打开x64 Native Tools Command Prompt for VS 2019命令工具。Python编译时，通过环境变量获取编译选项，在命令行终端运行以下命令
```bat
% 说明：CUDA_DIRECTORY 为用户自己的CUDA目录 以下为示例 %
set CUDA_DIRECTORY=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
% 说明：TRT_DIRECTORY 为下载的TensorRT库所在的目录 以下为示例 如果不编译TensorRT后端 可以不设置 %
set TRT_DIRECTORY=D:\x64\third_party\TensorRT-8.4.1.5
set WITH_GPU=ON
set ENABLE_ORT_BACKEND=ON
% 说明：如果不编译TensorRT后端 此项为OFF %
set ENABLE_TRT_BACKEND=ON
set ENABLE_PADDLE_BACKEND=ON
set ENABLE_VISION=ON
set ENABLE_VISION_VISUALIZE=ON

git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy && git checkout develop

% 说明：这里指定用户自己的python解释器 以python3.8为例 %
C:\Python38\python.exe setup.py build
C:\Python38\python.exe setup.py bdist_wheel
```
编译好的wheel文件在dist目录下，pip安装编译好的wheel包，命令如下
```bat
C:\Python38\python.exe -m pip install dist\fastdeploy_gpu_python-0.2.1-cp38-cp38-win_amd64.whl
```
更多编译选项说明参考[编译指南](./README.md)

## 3. CMake GUI + Visual Studio 2019 IDE方式编译C++ SDK
<div id="CMakeGuiAndVS2019"></div>    

### 使用CMake GUI进行基础配置
<div id="CMakeGuiAndVS2019Basic"></div> 

步骤一：首先，打开CMake GUI，先初始化FastDeploy工程：

![image](https://user-images.githubusercontent.com/31974251/192094881-c5beb0e5-82ae-4a62-a88c-73f3d80f7936.png)  

步骤二：点击Configure后，在弹窗中设置编译"x64"架构： 

![image](https://user-images.githubusercontent.com/31974251/192094951-958a0a22-2090-4ab6-84f5-3573164d0835.png)

初始化完成后，显示如下：  

![image](https://user-images.githubusercontent.com/31974251/192095053-874b9c73-fc0d-4325-b555-ac94ab9a9f38.png)

步骤三：由于FastDeploy目前只支持Release版本，因此，先将"CMAKE_CONFIGURATION_TYPES"修改成"Release"  

![image](https://user-images.githubusercontent.com/31974251/192095175-3aeede95-a633-4b3c-81f8-067f0a0a44a3.png)

接下来，用户可根据自己实际的开发需求开启对应的编译选项，并生成sln解决方案。以下，针对编译CPU和GPU版本SDK各举一个例子。

### 编译CPU版本 C++ SDK设置

<div id="CMakeGuiAndVS2019CPU"></div>    

步骤一：勾选CPU版本对应的编译选项。注意CPU版本，请`不要`勾选WITH_GPU和ENABLE_TRT_BACKEND

![image](https://user-images.githubusercontent.com/31974251/192095848-b3cfdf19-e378-41e0-b44e-5edb49461eeb.png)

这个示例中，我们开启ORT、Paddle、OpenVINO等推理后端，并且选择了需要编译TEXT和VISION的API


步骤二：自定义设置SDK安装路径，修改CMAKE_INSTALL_PREFIX

![image](https://user-images.githubusercontent.com/31974251/192095961-5f6e348a-c30b-4473-8331-8beefb7cd2e6.png)

由于默认的安装路径是C盘，我们可以修改CMAKE_INSTALL_PREFIX来指定自己的安装路径，这里我们将安装路径修改到`build\fastdeploy-win-x64-0.2.1`目录下。  

![image](https://user-images.githubusercontent.com/31974251/192096055-8a276a9e-6017-4447-9ded-b95c5579d663.png)



### 编译GPU版本 C++ SDK设置
<div id="CMakeGuiAndVS2019GPU"></div>  

步骤一：勾选GPU版本对应的编译选项。注意GPU版本，请`需要`勾选WITH_GPU

![image](https://user-images.githubusercontent.com/31974251/192099254-9f82abb0-8a29-41ce-a0ce-da6aacf23582.png) 

这个示例中，我们开启ORT、Paddle、OpenVINO和TRT等推理后端，并且选择了需要编译TEXT和VISION的API。并且，由于开启了GPU和TensorRT，此时需要额外指定CUDA_DIRECTORY和TRT_DIRECTORY，在GUI界面中找到这两个变量，点击右侧的选项框，分别选择您安装CUDA的路径和TensorRT的路径    


![image](https://user-images.githubusercontent.com/31974251/192098907-9dd9a49c-4a3e-4641-8e68-f25da1cafbba.png)


![image](https://user-images.githubusercontent.com/31974251/192098984-7fefd824-7e3b-4185-abba-bae5d8765e2a.png)


步骤二：自定义设置SDK安装路径，修改CMAKE_INSTALL_PREFIX

![image](https://user-images.githubusercontent.com/31974251/192099125-81fc8217-e51f-4039-9421-ba7a09c0027c.png)


由于默认的安装路径是C盘，我们可以修改CMAKE_INSTALL_PREFIX来指定自己的安装路径，这里我们将安装路径修改到`build\fastdeploy-win-x64-gpu-0.2.1`目录下。  


### 使用Visual Studio 2019 IDE进行编译

<div id="CMakeGuiAndVS2019Build"></div>  

步骤一：点击"Generate"，生成sln解决方案，并用Visual Studio 2019打开  

![image](https://user-images.githubusercontent.com/31974251/192096162-c05cbb11-f96e-4c82-afde-c7fc02cddf68.png)

这个过程默认会从下载一些编译需要的资源，cmake的dev警告可以不用管。生成完成之后可以看到以下界面：  

CPU版本SDK:  

![image](https://user-images.githubusercontent.com/31974251/192096478-faa570bd-7569-43c3-ad79-cc6be5b605e3.png)

GPU版本SDK:  

![image](https://user-images.githubusercontent.com/31974251/192099583-300e4680-1089-45cf-afaa-d2afda8fd436.png)


左侧界面，可以看到所有编译需要的include路径和lib路径已经被设置好了，用户可以考虑把这些路径记录下来方便后续的开发。右侧界面，可以看到已经生成fastdeploy.sln解决方案文件。接下来，我们使用Visual Studio 2019打开这个解决方案文件（理论上VS2022也可以编译，但目前建议使用VS2019）。  

![image](https://user-images.githubusercontent.com/31974251/192096765-2aeadd68-47fb-4cd6-b083-4a478cf5e584.jpg)


步骤二：在Visual Studio 2019点击"ALL BUILD"->右键点击"生成"开始编译  

![image](https://user-images.githubusercontent.com/31974251/192096893-5d6bc428-b824-4ffe-8930-0ec6d4dcfd02.png)   

CPU版本SDK编译成功！

![image](https://user-images.githubusercontent.com/31974251/192097020-979bd7a3-1cdd-4fb5-a931-864c5372933d.png)

GPU版本SDK编译成功！  

![image](https://user-images.githubusercontent.com/31974251/192099902-4b661f9a-7691-4f7f-b573-92ca9397a890.png)


步骤三：编译完成后，在Visual Studio 2019点击"INSTALL"->右键点击"生成"将编译好的SDK安装到先前指定的目录  


![image](https://user-images.githubusercontent.com/31974251/192097073-ce5236eb-1ed7-439f-8098-fef7a2d02779.png)

![image](https://user-images.githubusercontent.com/31974251/192097122-d675ae39-35fb-4dbb-9c75-eefb0597ec2e.png)  

SDK成功安装到指定目录！   

### 编译所有examples（可选）
可以在CMake GUI中勾选BUILD_EXAMPLES选项，连带编译所有的examples，编译完成后所有example的可执行文件保存在build/bin/Release目录下 

![image](https://user-images.githubusercontent.com/31974251/192110769-a4f0940d-dea3-4524-831b-1c2a6ab8e871.png)

![image](https://user-images.githubusercontent.com/31974251/192110930-e7e49bc6-c271-4076-be74-3d103f27bc78.png)



## 4. 特别提示

如果是用户自行编译SDK，理论上支持Windows 10/11，VS 2019/2022，CUDA 11.x 以及 TensorRT 8.x等配置，但建议使用我们推荐的默认配置，即：Windows 10, VS 2019, CUDA 11.2 和 TensorRT 8.4.x版本。另外，如果编译过程中遇到中文字符的编码问题（如UIE example必须传入中文字符进行预测），可以参考Visual Studio的官方文档，设置源字符集为`/utf-8`解决：  
- [/utf-8（将源字符集和执行字符集设置为 UTF-8）](https://learn.microsoft.com/zh-cn/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8?view=msvc-170)

