# SplitWise Scheduler 测试套件

本目录包含 `splitwise_scheduler.py` 模块的完整单元测试套件。

## 测试文件结构

```
tests/scheduler/
├── __init__.py                           # 包初始化文件
├── test_utils.py                         # 测试工具和辅助函数
├── test_splitwise_scheduler_config.py    # SplitWiseSchedulerConfig 类测试
├── test_node_info.py                     # NodeInfo 类测试
├── test_result_reader.py                 # ResultReader 类测试
├── test_result_writer.py                 # ResultWriter 类测试
├── test_api_scheduler.py                 # APIScheduler 类测试
├── test_infer_scheduler.py               # InferScheduler 类测试
├── test_splitwise_scheduler.py           # SplitWiseScheduler 主类测试
├── test_all.py                           # 综合测试运行器
└── README.md                             # 本文件
```

## 测试覆盖范围

### SplitWiseSchedulerConfig 类
- ✅ 默认值初始化
- ✅ 自定义值初始化
- ✅ 参数验证和断言
- ✅ 长预填充令牌阈值计算
- ✅ 配置打印功能

### NodeInfo 类
- ✅ 节点信息初始化
- ✅ 序列化和反序列化
- ✅ 过期检查
- ✅ 请求管理（添加、更新、完成）
- ✅ 线程安全性
- ✅ 比较操作

### ResultReader 类
- ✅ 结果读取器初始化
- ✅ 请求添加和跟踪
- ✅ 结果读取和分组
- ✅ Redis 同步
- ✅ 过期请求处理
- ✅ 异常处理

### ResultWriter 类
- ✅ 结果写入器初始化
- ✅ 数据批处理
- ✅ Redis 管道操作
- ✅ 线程安全性
- ✅ 异常处理

### APIScheduler 类
- ✅ API 调度器初始化
- ✅ 请求队列管理
- ✅ 集群状态同步
- ✅ 节点选择算法
- ✅ 请求调度逻辑
- ✅ 过期节点清理

### InferScheduler 类
- ✅ 推理调度器初始化
- ✅ 节点信息报告
- ✅ 请求获取和处理
- ✅ 结果写入
- ✅ 分块预填充支持
- ✅ 异常处理

### SplitWiseScheduler 主类
- ✅ 主调度器初始化
- ✅ 启动和停止
- ✅ 请求和结果管理
- ✅ 资源检查
- ✅ 集成工作流

## 运行测试

### 运行所有测试
```bash
cd /Users/jiangjiajun/Documents/FastDeploy
python -m pytest tests/scheduler/ -v
```

### 运行特定测试文件
```bash
python -m pytest tests/scheduler/test_splitwise_scheduler_config.py -v
```

### 运行特定测试类
```bash
python -m pytest tests/scheduler/test_splitwise_scheduler_config.py::TestSplitWiseSchedulerConfig -v
```

### 运行特定测试方法
```bash
python -m pytest tests/scheduler/test_splitwise_scheduler_config.py::TestSplitWiseSchedulerConfig::test_init_with_default_values -v
```

### 使用综合测试运行器
```bash
cd /Users/jiangjiajun/Documents/FastDeploy/tests/scheduler
python test_all.py
```

## 测试特性

### 模拟和存根
- 使用 `unittest.mock` 模拟 Redis 客户端
- 模拟网络请求和响应
- 模拟时间相关操作
- 模拟线程操作

### 边界条件测试
- 空输入处理
- 无效参数处理
- 资源不足情况
- 网络错误处理

### 并发测试
- 线程安全性验证
- 竞态条件测试
- 锁机制测试

### 异常处理测试
- 网络异常
- 数据解析异常
- 资源异常
- 超时异常

## 测试覆盖率

本测试套件旨在提供高代码覆盖率，包括：

- **函数覆盖率**: 测试所有公共方法
- **分支覆盖率**: 测试所有条件分支
- **异常覆盖率**: 测试异常处理路径
- **集成覆盖率**: 测试组件间交互

## 依赖项

测试依赖以下包：
- `unittest` (Python 标准库)
- `unittest.mock` (Python 标准库)
- `threading` (Python 标准库)
- `time` (Python 标准库)
- `collections` (Python 标准库)
- `orjson` (需要安装)
- `redis` (需要安装)

## 注意事项

1. 某些测试需要 Redis 服务器运行，但大部分测试使用模拟对象
2. 时间相关的测试使用模拟时间以避免测试不稳定
3. 线程相关的测试使用适当的同步机制
4. 测试数据使用工厂函数生成，确保测试的独立性

## 贡献指南

添加新测试时请遵循以下原则：

1. 测试名称应该清晰描述测试内容
2. 每个测试应该独立运行
3. 使用适当的断言验证结果
4. 包含边界条件和异常情况测试
5. 添加必要的文档和注释
