[English](../../en/faq/use_sdk_on_linux.md) | 中文


# Linux上使用C++在华为昇腾部署

在完成部署示例的编译之后, 在运行程序之前, 由于我们需要借助华为昇腾工具包的功能, 所以还需要导入一些环境变量来初始化部署环境.
用户可以直接使用如下脚本(位于编译好的FastDeploy库的目录下), 来初始化华为昇腾部署的环境.


```
# 我们默认的昇腾工具包的路径如下,
# HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
# HUAWEI_ASCEND_DRIVER_PATH="/usr/local/Ascend/driver"
# 如果用户的安装目录与他不同, 需要自己先手动export.
# export HUAWEI_ASCEND_TOOLKIT_HOME="Your_ascend_toolkit_path"
# export HUAWEI_ASCEND_DRIVER_PATH="Your_ascend_driver_path"

source fastdeploy-ascend/fastdeploy_init.sh
```

注意此命令执行后仅在当前的命令环境中生效（切换一个新的终端窗口，或关闭窗口重新打开后会无效），如若需要在系统中持续生效，可将这些环境变量加入到`~/.bashrc`文件中。

# 昇腾部署时开启FlyCV
[FlyCV](https://github.com/PaddlePaddle/FlyCV) 是一款高性能计算机图像处理库, 针对ARM架构做了很多优化, 相比其他图像处理库性能更为出色.
FastDeploy现在已经集成FlyCV, 用户可以在支持的硬件平台上使用FlyCV, 实现模型端到端推理性能的加速.

模型端到端推理中, 预处理和后处理阶段为CPU计算, 当用户使用ARM CPU + 昇腾的硬件平台时, 我们推荐用户使用FlyCV, 可以实现端到端的推理性能加速, 详见以下使用文档.

- [FLyCV使用文档](./boost_cv_by_flycv.md)
