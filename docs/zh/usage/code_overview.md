[English](../../usage/code_overview.md)

# 代码说明
下边按照目录结构来介绍一下每个 FastDeploy 的代码结构及代码功能。

- ```custom_ops```：存放 FastDeploy 运行大模型所使用到的 C++ 算子，不同硬件下的算子放置到对应的目录下（cpu_ops/gpu_ops），根目录下的 setup_*.py 文件用来编译上述 C++ 代码的算子。
- ```dockerfiles```：存放运行 FastDeploy 的环境镜像 dockerfile。
- ```docs```：FastDeploy 代码库有关的说明文档。
- ```fastdeploy```
  - ```agent```：大模型服务启动使用到的脚本
  - ```cache_manager```：大模型缓存管理模块
  - ```engine```：管理大模型整体执行引擎类有关代码
  - ```entrypoints```：用户入口调用接口
  - ```input```：用户输入处理模块，包括预处理，多模态输入处理，tokenize 等功能
  - ```model_executor```
    - ```layers```：大模型组网需要用到的 layer 模块
    - ```model_runner```：模型推理执行模块
    - ```models```：FastDeploy 内置的大模型类模块
    - ```ops```：由 custom_ops 编译后可供 python 调用的算子模块，不同硬件平台的算子放置到对应的目录里
  - ```output```：大模型输出有关处理
  - ```platforms```：与底层硬件功能支持有关的平台模块
  - ```scheduler```：大模型请求调度模块
  - ```metrics```：用于收集、管理和导出 Prometheus 指标的核心组件，负责记录系统运行时的关键性能数据（如请求延迟、资源使用率、成功请求数等）
  - ```splitwise```: 分离式部署相关模块
- ```scripts```/```tools```：FastDeploy 用于执行功能的辅助脚本，比如编译，单测执行，代码风格纠正等
- ```test```：项目单测验证使用到的代码
