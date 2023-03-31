[English](../../../en/faq/rknpu2/issues.md) | 中文
# RKNPU2常见问题合集

在使用FastDeploy的过程中大家可能会碰到很多的问题，这个文档用来记录已经解决的共性问题，方便大家查阅。

## 导航

- [动态链接库链接问题](#动态链接库链接问题)

## 动态链接库链接问题

### 关联issue

- [Issue 870](https://github.com/PaddlePaddle/FastDeploy/issues/870)

### 问题描述

在编译的过程中没有出现问题，但是运行程序时出现以下报错
```text
error while loading shared libraries: libfastdeploy.so.0.0.0: cannot open shared object file: No such file or directory
```

### 分析原因

链接器ld提示找不到库文件。ld默认的目录是/lib和/usr/lib，如果放在其他路径也可以，需要让ld知道库文件所在的路径。

### 解决方案

**临时解决方法**

临时解决方法对系统没有影响，但是仅在当前打开的终端时生效，关闭终端后，这个配置会失效。

```bash
source PathToFastDeploySDK/fastdeploy_init.sh
```

**永久解决方案**

临时解决方法解决方案在开关终端后会失效，因为每一次重新打开终端运行程序时，都需要重新输入命令。如果您不想每一次运行程序都需要运行一次代码，可以执行以下代码:
```bash
source PathToFastDeploySDK/fastdeploy_init.sh
sudo cp PathToFastDeploySDK/fastdeploy_libs.conf /etc/ld.so.conf.d/
sudo ldconfig
```
执行后配置文件将写入系统，刷新后即可让系统找到库的位置。
