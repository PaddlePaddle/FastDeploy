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

编译FastDeploy C++ SDK后，SDK目录下有一个fastdeploy_init.sh文件，运行这个文件即可。

```bash
source PathToFastDeploySDK/fastdeploy_init.sh
```

**永久解决方案**

运行以下代码:
```bash
source PathToFastDeploySDK/fastdeploy_init.sh
sudo cp PathToFastDeploySDK/fastdeploy_libs.conf /etc/ld.so.conf.d/
sudo ldconfig
```
