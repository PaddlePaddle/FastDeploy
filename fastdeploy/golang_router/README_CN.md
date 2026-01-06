# fd-router

【正在开发迭代中】
fd-router是一个高性能的 Go 语言路由框架，提供灵活的请求路由、中间件支持和健康检查功能。

可独立于 FastDeploy GPU 推理进程运行, 通过 HTTP 协议与推理进程通信

## 功能特性

- 高性能 HTTP/HTTPS 服务器
- RESTful API 路由支持
- 可扩展的中间件系统
- 动态配置管理
- 内置健康检查和监控
- 负载均衡
- 日志记录和指标收集

## 快速开始

### 前置要求

- Go 1.21
- 构建不依赖特定系统环境
- 可直接在 FastDeploy 官方 Docker 环境中编译与运行

### 编译

```bash
./build.sh
```

### 配置

1. 配置文件准备（可选）
如需修改默认配置，可复制配置模板并进行调整（示例可参考 examples/run_with_config）：

```bash
cp config/config.example.yaml config/config.yaml
```

2. 主要配置项说明：

```yaml
server:
  port: "8080" # 监听端口
  host: "0.0.0.0" # 监听地址
  mode: "debug" # 启动模式: debug, release, test
  splitwise: true # true代表开启pd分离模式,false代表开启非pd分离模式

scheduler:
  policy: "request_num" # 调度策略(可选): random, power_of_two, round_robin, process_tokens, request_num
  prefill-policy: "process_tokens" # pd分离模式下prefill节点调度策略
  decode-policy: "request_num" # pd分离模式下decode节点调度策略
  interval-cleanup-secs: 60 # cache-aware策略清理过期cache的间隔时间

manager:
  health-failure-threshold: 3 # 健康检查失败次数,超过次数后认为节点不健康
  health-success-threshold: 2 # 健康检查成功次数,超过次数后认为节点健康
  health-check-timeout-secs: 5 # 健康检查超时时间
  health-check-interval-secs: 5 # 健康检查间隔时间
  health-check-endpoint: /health # 健康检查接口

log:
  level: "info"  # 日志打印级别: debug / info / warn / error
  output: "file" # 日志输出方式: stdout / file

```

3. 启动时注册实例（可选）
支持通过配置文件在启动阶段注册推理实例（示例可参考 examples/run_with_default_workers）：

```bash
cp config/config.example.yaml config/config.yaml
cp config/register.example.yaml config/register.yaml
```

### 运行
本项目支持两种运行方式：直接运行源码 或 构建二进制文件后运行。
方式一：直接运行源码
在项目根目录下，使用 go run 启动服务：
```bash
go run cmd/main.go
```
该方式适用于本地开发与调试场景。
方式二：构建并运行二进制文件
1. 构建二进制文件
通过构建脚本生成可执行文件：
```bash
./build.sh
```
构建完成后，二进制文件将被安装到指定目录（默认为 /usr/local/bin，可通过修改 Makefile 中的 OUTDIR 进行调整）。
2. 运行二进制文件
可以通过运行脚本启动服务：
```bash
./run.sh
```
也可以直接执行构建生成的二进制文件，并手动指定启动参数（其中 --port 为必填参数）：
```bash
/usr/local/bin/fd-router \
  --port 8080 \
  --splitwise \
  --config_path ./config/config.yaml
```
该方式更适合用于部署及生产环境。

## 项目结构

```
.
├── cmd/              # 主程序入口
├── config/           # 配置文件
├── internal/         # 核心实现代码
│   ├── common/       # 公共接口定义
│   ├── config/       # 配置处理
│   ├── gateway/      # API网关实现
│   ├── manager/      # 路由管理
│   ├── middleware/   # 中间件实现
│   ├── router/       # 路由核心逻辑
│   └── scheduler/    # 调度器实现
├── logs/             # 日志目录
├── output/           # 构建输出
├── pkg/              # 可复用组件
│   ├── logger/       # 日志组件
│   └── metrics/      # 监控指标
├── build.sh          # 构建脚本
├── go.mod            # Go模块定义
├── go.sum            # 依赖校验
├── Makefile          # 构建管理
├── README.md         # 项目说明
└── run.sh          # 启动脚本
```

### 运行测试

```bash
make test
```

## 贡献

欢迎提交 Issue 和 Pull Request！


