
FastDeploy中集成了多种推理后端，分别用于在不同硬件上的部署。例如CPU上支持OpenVINO/Paddle Inference/ONNX Runtime后端。

1. 在常见的硬件上，可能会存在一种硬件被多个后端支持，例如CPU被OpenVINO/Paddle Inference/ONNX Runtime后端支持，GPU被Paddle Inference/ONNX Runtime/TensorRT支持；
2. 也存在一个后端支持多种硬件，例如Paddle Inference同时支持CPU/GPU， Paddle Lite支持 ARM CPU/昆仑芯/昇腾等；
3. 也存在一个后端只支持1种硬件，例如瑞芯微3588则只对应1种后端。

本文档接下来介绍在FastDeploy中接入一个新的后端流程，一种后端的接入代码修改都在
```
FastDeploy/fastdeploy/runtime/
```
模块下。

## 后端接入流程

### 1. 各类枚举变量的创建

主要用于在FastDeploy中声明新加入的后端，以及后端对应支持的硬件。修改的内容需要包括

1. 如新增硬件支持，修改枚举类型`fastdeploy/runtime/enum_variables.h::Device`
2. 新增后端支持，修改枚举类型`fastdeploy/runtime/enum_variables.h::Backend`
3. 如若引入的新的模型格式，修改枚举类型`fastdeploy/runtime/enum_variables.h::ModelFormat`
4. 修改全局变量`fastdeploy/runtime/enum_variables.h::s_default_backends_by_format`，该变量表示各模型格式被哪些后端直接支持加载
5. 修改全局变量`fastdeploy/runtime/enum_variables.h::s_default_backends_by_device`，该变量表示各硬件被哪些后端支持

除上述5处，另外也同时修改`fastdeploy/runtime/enum_variables.cc`，加入各枚举类型被输出运算符重载的实现，参照其它后端或硬件即可。

### 2. 实现backend接口

此步骤的代码实现则需创建`fastdeploy/runtime/backends/new_backend`目录，建议在此目录下实现以下代码文件（不强制要求命名或文件数量，具体根据实现需求来加）
1. `fastdeploy/runtime/backends/new_backend/option.h` 此文件请按照这个命名方式，并在代码中添加`NewBackendOption`结构体的实现，用于配置后端运行时的参数（注意此结构体所有函数实现均在这个头文件中，只使用C++内置数据结构，不引入后端自定义类型）；
2. `fastdeploy/runtime/backends/new_backend/new_backend.h`在此代码中添加`NewBackend`类定义，并继承`fastdeploy/runtime/backends/backend.h::BaseBackend`，实现相应的接口

### 3. Backend集成进FastDeploy Runtime
在第2步后，已经可以通过Backend进行模型的加载、预测，并获取输出结果。接下来则是将Backend集成进Runtime，实现统一的推理接口。

1. 修改`fastdeploy/runtime/runtime_option.h`，添加新的接口
	1. 如有新硬件，增加`UseHardware()`接口，可以在此接口中添加对硬件配置的参数，如device_id等，但都给定默认参数
	2. 增加`UseNewBackend()`，可以在此接口中添加对后端配置的参数，但都给定默认参数
	3. 增加成员变量`NewBackendOption new_backend_option`, 对于更高级的配置，可以让开发者通过`RuntimeOption.new_backend_option`的访问方式来修改
2. 修改`fastdeploy/runtime/runtime.h`
	1. 创建私有函数`CreateNewBackend()`函数，并仿照其它`CreateXXXBackend()`函数完成实现
	2. 在`Init()`函数实现根据backend字段调用`CreateNewBackend()`的逻辑

## 4. 编译CMake配置
在完成1，2，3步骤后，我们已经完成了代码的开发，接下来是开发编译流程。这个步骤中涉及到修改3个文件，`FastDeploy/CMakeLists.txt`，`FastDeploy/cmake/new_backend.cmake`， `FastDeploy/FastDeploy.cmake.in`，分别是编译的主入口，依赖库的配置入口，以及FastDeploy提供经开发者依赖的配置入口。

1. 创建`FastDeploy/cmake/new_backend.cmake`，可参照同目录下的`rknpu2.cmake`，用于配置第三方库的下载，头文件的引入，以及库的引入
2. 修改`FastDeploy/CMakeLists.txt`，注意有以下几处修改
	1. 参照其它后端代码，添加`option(ENABLE_NEW_BACKEND)`用于控制编译FastDeploy时是否集成此后端
	2. 参照其它后端代码，添加`file(GLOB_RECURSE DEPLOY_BACKEND_SRCS)`用于将新后端集成的代码查找到，并以列表的形式存储到`DEPLOY_NEW_BACKEND_SRCS`中，并在CMakeLists.txt中的`list(REMOVE_ITEM ALL_DEPLOY_SRCS)`中移除所有的`DEPLOY_NEW_BACKEND_SRCS`
	3. 参照其它后端代码，添加`if(ENABLE_NEW_BACKEND)`的代码逻辑，用于将代码加入进来，添加相应库链接
3. 修改`FastDeploy/FastDeploy.cmake.in`（此文件为生成模版，会根据编译时配置，生成FastDeploy.cmake文件，用于提供给编译好FastDeploy库后，帮助开发者快速依赖编译好的FD库进行C++部署），在开始处获取编译参数，同时添加相应逻辑
- - https://github.com/PaddlePaddle/FastDeploy/blob/21b1cb87423d779f5c732e63322e168c8acd330d/FastDeploy.cmake.in#L24
- - https://github.com/PaddlePaddle/FastDeploy/blob/21b1cb87423d779f5c732e63322e168c8acd330d/FastDeploy.cmake.in#L118-L127

最后还需要再修改`FastDeploy/fastdeploy/core/config.h.in`文件，加入宏定义
https://github.com/PaddlePaddle/FastDeploy/blob/21b1cb87423d779f5c732e63322e168c8acd330d/fastdeploy/core/config.h.in#L24-L26

至此我们已经完成了C++集成FastDeploy的后端，可以进行C++的后端测试工作

### 5. C++后端测试

在完成上述部署后，即可编译FastDeploy库
```
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_NEW_BACKEND=ON -DCMAKE_INSTALL_PREFIX=${PWD}/installed_fd
make -j
make install
```
即在当前目录生成FastDeploy C++部署库`installed_fd`

我们可参照 https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime/cpp 此处的示例，写一个新增后端加载模型并推理测试，验证无问题即可。

### 6. Python接口绑定

在完成C++后端开发后，如有进一步Python接口支持的需求，仅需在FastDeploy快速绑定几个基础的接口即可。

1. `NewBackendOption`的绑定，可参照OrtBackend中Option的绑定方式，在`runtime/backends/new_backend`下新增`option_pybind.cc`来绑定自己的数据结构
2. 在`runtime/option_pybind.cc`中参照`BindOrtOption`方式，进行声明和调用
3. 由于C++开发中，我们修改了`runtime/option.h`新增了`UseXXX`接口，因此也需要修改`runtime/option_pybind.cc`中`RuntimeOption`的绑定，新增`use_xxx`函数的绑定，同时修改`FastDeploy/python/fastdeploy/runtime.py`，增加`use_xxx`python函数调用
4. Python编译时，我们无法通过命令行传参，所以FastDeploy通过获取环境变量来修改编译参数。 修改`FastDeploy/python/setup.py`，增加从环境变量中获取`ENABLE_NEW_BACKEND`的代码逻辑

至此，就完成了Python接口的绑定。

### 7.  Python 后端测试

在完成上述部署后，即可编译安装FastDeploy python包
```
cd FastDeploy/python
export ENABLE_NEW_BACKEND=ON
python setup.py build
python setup.py bdist_wheel
```
即在`python/dist`生成FastDeploy python wheel包
```
pip install dist/fastdeploy-xxxxx.whl
```

接下来，我们可参照 https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime/python 此处的示例，写一个新增后端加载模型并推理测试，验证无问题即可。
