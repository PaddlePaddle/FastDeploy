[English](../../en/faq/use_sdk_on_ascend.md) | 中文


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
