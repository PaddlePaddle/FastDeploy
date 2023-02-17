# 常见问题

1. Windows安装fastdeploy-python或fastdeploy-gpu-python后，执行`import fastdeploy`时，出现提示"DLL Load failed: 找不到指定模块"
- **解决方式** 此问题原因可能在于系统没有安装VS动态库，在此页面根据个人环境下载安装后，重新import解决 https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
