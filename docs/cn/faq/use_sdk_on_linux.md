[English](../../en/faq/use_sdk_on_linux.md) | 中文


# Linux上C++部署

1. 编译完成运行，提示找不到.so文件 "cannot open shared object file: No such file or directory"


在执行二进制文件时，需要能够在环境变量中找到FastDeploy相关的库文件。FastDeploy提供了辅助脚本来帮助完成。

执行如下命令，即可将库路径导入到LD_LIBRARY_PATH中

```
source /Downloads/fastdeploy-linux-x64-1.0.0/fastdeploy_init.sh
```

再重新执行即可。 注意此命令执行后仅在当前的命令环境中生效（切换一个新的终端窗口，或关闭窗口重新打开后会无效），如若需要在系统中持续生效，可将这些环境变量加入到`~/.bashrc`文件中。
