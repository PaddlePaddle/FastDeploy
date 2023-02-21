[English](README.md) | 中文


# 使用C和C# API进行模型部署

FastDeploy提供了C API以及C# API满足多语言部署的需求，这一特性从FastDeploy v1.0.4版本开始进行了支持。默认情况下，C API被编译进入了fastdeploy的动态库。
如果希望手动编译FastDeploy，并且集成C和C# API进去，只需要打开编译开关-DWITH_CAPI=ON, -DWITH_CSHARPAPI=ON。关于如何编译FastDeploy可以参考文档[FastDeploy安装](../../docs/cn/build_and_install/README.md)。

关于如何使用C和C# API进行模型部署可以参考下列示例，更多示例请查看[examples目录](../../examples)

- [使用C API进行检测模型部署](c)
- [使用C# API进行检测模型部署](csharp)
