# Windows编译

## 获取代码
```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
git checkout develop
```

## 编译C++ SDK

Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，其中`CMAKE_INSTALL_PREFIX`用于指定编译后生成的SDK路径

```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\FastDeploy\build\fastdeploy-win-x64-0.2.0 -DENABLE_ORT_BACKEND=ON -DENABLE_VISION=ON .. 
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```
编译后，C++ SDK即在`D:\Paddle\FastDeploy\build\fastdeploy-win-x64-0.2.0`目录下

## 编译Python Wheel包

Python编译时，通过环境变量获取编译选项
```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
git checkout develop

set ENABLE_ORT_BACKEND=ON
set ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel
```
