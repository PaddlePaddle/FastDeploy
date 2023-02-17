English | [中文](../../cn/build_and_install/directml.md)

# How to Build DirectML Deployment Environment
Direct Machine Learning (DirectML) is a high-performance, hardware-accelerated DirectX 12 library for machine learning on Windows systems.
Currently, Fastdeploy's ONNX Runtime backend has DirectML integrated, allowing users to deploy models on AMD/Intel/Nvidia/Qualcomm GPUs with DirectX 12 support.

More details:
- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

# DirectML requirements
- Compilation requirements: Visual Studio 2017 toolchain and above.
- Operating system: Windows 10, version 1903, and newer. (DirectML is part of the operating system and does not need to be installed separately)
- Hardware requirements: DirectX 12 supported graphics cards, e.g., AMD GCN 1st generation and above/ Intel Haswell HD integrated graphics and above/ Nvidia Kepler architecture and above/ Qualcomm Adreno 600 and above.

# How to Build and Install DirectML C++ SDK
The DirectML is integrated with the ONNX Runtime backend, so to use DirectML, users need to turn on the option to compile ONNX Runtime. Also, FastDeploy's DirectML supports building programs for x64/x86 (Win32) architectures.

For the x64 example, in the Windows menu, find `x64 Native Tools Command Prompt for VS 2019` and open it by executing the following command
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
Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`
If you use CMake GUI, please refer to [How to Compile with CMakeGUI + Visual Studio 2019 IDE on Windows](../faq/build_on_win_with_gui.md)


For the x86(Win32) example, in the Windows menu, find `x86 Native Tools Command Prompt for VS 2019` and open it by executing the following command
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
Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`
If you use CMake GUI, please refer to [How to Compile with CMakeGUI + Visual Studio 2019 IDE on Windows](../faq/build_on_win_with_gui.md)

# How to use compiled DirectML SDK.
The DirectML compiled library can be used in the same way as any other hardware on Windows, see the following link.
- [Using the FastDeploy C++ SDK on Windows Platform](../faq/use_sdk_on_windows.md)
