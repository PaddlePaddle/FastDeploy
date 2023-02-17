[English](../../en/build_and_install/directml.md) | 简体中文

# DirectML部署库编译
Direct Machine Learning (DirectML) 是Windows系统上用于机器学习的一款高性能, 提供硬件加速的 DirectX 12 库.
目前, Fastdeploy的ONNX Runtime后端已集成DirectML,让用户可以在支持DirectX 12的 AMD/Intel/Nvidia/Qualcomm的GPU上部署模型.

更多详细介绍可见:
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

# DirectML使用需求
- 编译需求: Visuald Studio 2017 及其以上工具链.
- 操作系统: Windows10, 1903 版本, 及其更新版本. (DirectML为操作系统的组成部分, 无需单独安装)
- 硬件需求: 支持DirectX 12的显卡, 例如, AMD GCN 第一代及以上版本/ Intel Haswell HD集成显卡及以上版本/Nvidia Kepler架构及以上版本/ Qualcomm Adreno 600及以上版本.

# 编译DirectML部署库
DirectML是基于ONNX Runtime后端集成, 所以要使用DirectML, 用户需要打开编译ONNX Runtime的选项. 同时, FastDeploy的DirectML支持x64/x86(Win32)架构的程序构建.


x64示例, 在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，执行如下命令
```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A x64 ^
         -DWITH_DIRECTML=ON ^
         -DENABLE_ORT_BACKEND=ON ^
         -DENABLE_VISION=ON ^
         -DCMAKE_INSTALL_PREFIX="D:\Paddle\compiled_fastdeploy" ^

msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```
编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库.
如您使用CMake GUI可参考文档[Windows使用CMakeGUI + Visual Studio 2019 IDE编译](../faq/build_on_win_with_gui.md)


x86(Win32)示例, 在Windows菜单中，找到`x86 Native Tools Command Prompt for VS 2019`打开，执行如下命令
```bat
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build

cmake .. -G "Visual Studio 16 2019" -A Win32 ^
         -DWITH_DIRECTML=ON ^
         -DENABLE_ORT_BACKEND=ON ^
         -DENABLE_VISION=ON ^
         -DCMAKE_INSTALL_PREFIX="D:\Paddle\compiled_fastdeploy" ^

msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=Win32
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=Win32
```
编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库.
如您使用CMake GUI可参考文档[Windows使用CMakeGUI + Visual Studio 2019 IDE编译](../faq/build_on_win_with_gui.md)

# 使用DirectML库
DirectML编译库的使用方式, 和其他硬件在Windows上使用的方式一样, 参考以下链接.
- [FastDeploy C++库在Windows上的多种使用方式](../faq/use_sdk_on_windows_build.md)
- [在 Windows 使用 FastDeploy C++ SDK](../faq/use_sdk_on_windows.md)
