# Golang-Router
## 关于
【正在开发迭代中】
Golang-Router 是一个面向大语言模型推理系统的高性能 Golang 路由框架，作为系统的**控制与调度平面**运行，负责请求接入、实例选择与流量转发，设计上适配 Prefill–Decode（PD）分离推理架构。

Golang-Router 可独立部署运行，也可通过 HTTP 接口与 FastDeploy 推理实例协同工作。框架提供基础而稳定的路由、中间件扩展与健康检查能力，适用于单点推理部署场景，并在架构层面为后续的水平扩展与调度能力演进预留空间。

### 背景与动机
在大语言模型推理系统中，路由组件已从传统的流量转发层演进为影响系统性能与资源利用效率的关键基础设施。随着 Prefill–Decode 分离推理架构的广泛采用，不同推理阶段在计算特征、显存占用与缓存行为方面呈现出明显差异，仅依赖请求级静态信息进行调度已难以满足稳定性与效率需求。

在保持请求级调度模型不变的前提下，引入更细粒度的运行时信号辅助调度决策，成为提升调度能力与系统可预测性的工程共识。Golang-Router 正是在这一背景下构建，作为独立的路由与调度组件，为推理系统提供清晰、可扩展的控制平面。

### 设计目标
Golang-Router 聚焦解决以下核心问题：
- **调度决策信息不足**
  传统 Router 通常仅基于请求级元信息或粗粒度实例状态进行调度，难以利用推理过程中产生的细粒度缓存相关信号，从而限制了 cache-aware 策略的实际效果。
- **调度逻辑与推理执行强耦合**
  路由与调度逻辑内嵌于推理框架内部，增加了系统复杂度，限制了调度策略的独立演进与复用能力。
- **高并发场景下的可扩展性挑战**
  在高并发推理负载下，实例状态维护与实例选择逻辑对路由组件的并发模型、性能与稳定性提出更高要求。

### 核心特性
- 基于 Golang 实现的高性能路由与调度组件，适用于高并发、低延迟推理场景
- 请求级调度模型，保持接口语义清晰与系统复杂度可控
- 利用token级缓存相关运行时信息作为调度策略的辅助输入，用于提升实例选择的准确性与稳定性
- 模块化架构设计（Gateway / Scheduler / Manager），职责边界清晰，便于扩展与维护
- 面向 Prefill–Decode 分离推理架构设计，为复杂调度策略与能力演进提供结构性支持

### 与现有方案的差异
与 sglang 等推理框架内置 Router 相比，Golang-Router 以**独立 Golang 服务**的形式运行，将路由、调度与实例状态管理能力从推理执行逻辑中解耦。

Golang-Router 已支持 cache-aware 调度，在请求级调度框架内引入 token 级缓存相关运行时信号，辅助调度决策制定，以更稳定地适配 Prefill–Decode 分离推理架构下的缓存利用需求。

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
  policy: "power_of_two" # 调度策略(可选): random, power_of_two, round_robin, process_tokens, request_num
  prefill-policy: "cache_aware" # pd分离模式下prefill节点调度策略
  decode-policy: "fd_metrics_score" # pd分离模式下decode节点调度策略
  eviction-interval-secs: 60 # cache-aware策略清理过期cache的间隔时间
  balance-abs-threshold: 1 # cache-aware策略绝对阈值
  balance-rel-threshold: 0.2 # cache-aware策略相对阈值
  hit-ratio-weight: 1.0 # cache-aware策略命中率权重
  load-balance-weight: 0.05 # cache-aware策略负载均衡权重
  cache-block-size: 4 # cache-aware策略cache block大小
  tokenizer-url: "http://0.0.0.0:8098" # tokenizer服务地址(可选)
  tokenizer-timeout-secs: 2 # tokenizer服务超时时间
  waiting-weight: 10 # cache-aware策略等待权重

manager:
  health-failure-threshold: 3 # 健康检查失败次数,超过次数后认为节点不健康
  health-success-threshold: 2 # 健康检查成功次数,超过次数后认为节点健康
  health-check-timeout-secs: 5 # 健康检查超时时间
  health-check-interval-secs: 5 # 健康检查间隔时间
  health-check-endpoint: /health # 健康检查接口
  register-path: "config/register.yaml" # 推理实例注册配置文件路径(可选)

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
此外，也可以在项目根目录下手动构建二进制文件：
```bash
go build -o ./fd-router ./cmd
```
该方式便于本地测试或将二进制文件与配置文件一并分发。
2. 运行二进制文件
可以通过运行脚本启动服务：
```bash
./run.sh
```
运行脚本会自动处理常见启动参数及日志目录，适合标准化部署场景。
也可以直接运行二进制文件，在项目根目录或二进制所在目录下执行：
```bash
./fd-router \
  --port 8080 \
  --splitwise \
  --config_path ./config/config.yaml
```
其中：
- --port 为必填参数
- 其他参数可根据实际需求配置

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
