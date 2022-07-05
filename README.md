# FastDeploy

模型推理就用FastDeploy!

## 环境要求
- python >= 3.6
- cmake >= 3.18
- gcc >= 8.2
- cuda >= 11.0（如若需要启用GPU）
- tensorrt >= 8.4（如若需要启用TensorRT后端）

## 如何利用FastDeploy快速完成模型部署

- [C++部署指南](docs/cpp/README.md)
- [Python部署指南](docs/python/README.md)

## 如何自行编译FastDeploy

- [FastDeploy编译指南](docs/compile/README.md)

## 代码提交

提交代码前，先初始化代码环境，在clone代码后，执行
```
sh commit-prepare.sh
```

在之后commit代码时，会自动进行代码格式的检查。
