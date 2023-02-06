English | [中文](../../../cn/faq/rknpu2/issues.md) 
# RKNPU2 FAQs

This document collects the common problems when using FastDeploy.

## Navigation

- [Link issues in dynamic link library](#动态链接库链接问题)

## Link issues in dynamic link library

### Association issue

- [Issue 870](https://github.com/PaddlePaddle/FastDeploy/issues/870)

### Problem Description 

No problem during compiling, but the following error is reported when running the program
```text
error while loading shared libraries: libfastdeploy.so.0.0.0: cannot open shared object file: No such file or directory
```

### Analysis

The linker ld indicates that the library file cannot be found. The default directories for ld are /lib and /usr/lib.
Other directories are also OK, but you need to let ld know where the library files are located. 


### Solutions

**Temporary solution**

This solution has no influence on the system, but it only works on the current terminal and fails when closing this terminal.

```bash
source PathToFastDeploySDK/fastdeploy_init.sh
```

**Permanent solution**

The temporary solution fails because users need to retype the command each time they reopen the terminal to run the program. If you don't want to constantly run the code, execute the following code: 
```bash
source PathToFastDeploySDK/fastdeploy_init.sh
sudo cp PathToFastDeploySDK/fastdeploy_libs.conf /etc/ld.so.conf.d/
sudo ldconfig
```
After execution, the configuration file is written to the system. Refresh to let the system find the library location.
