[English](README.md) | 中文


# 使用C或C# API进行模型部署

FastDeploy提供了C API以及C# API满足多语言部署的需求，这一特性从FastDeploy v1.0.4版本开始进行了支持。默认情况下，C API被编译进入了fastdeploy的动态库。
如果希望手动编译FastDeploy，并且集成C和C# API进去，只需要打开编译开关-DWITH_CAPI=ON, -DWITH_CSHARPAPI=ON。关于如何安装和使用C和C# API，可以参考下列文档：

- [C API指南](../../c_api/README_CN.md)
- [C# API指南](../../csharp/README_CN.md)
