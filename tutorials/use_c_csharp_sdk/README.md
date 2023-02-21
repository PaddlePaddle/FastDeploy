English | [中文](README_CN.md)

# Model Deployment with C or C# API

Fastdeploy provideds C and C# API to support deployment with multiple programming languages, and this feature is included since version 1.0.4. C API is compiled and included in fastdeploy dynamic library by default. If you want to compile fastdeploy with C and C# API manually, compile it with options -DWITH_CAPI=ON and -DWITH_CSHARPAPI=ON. For more information about how to compile fastdeploy, please refer to [Install FastDeploy](../../docs/en/build_and_install/README.md)。

We provide two demos to show how to use C and C# API to deploy models. For more examples, please refer to [examples](../../examples)

- [Detection model deployment with C API](c)
- [Detection model deployment with C# API](csharp)
