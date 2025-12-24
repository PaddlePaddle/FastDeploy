# fd-router

一个高性能的 Go 语言路由框架，提供灵活的请求路由、中间件支持和健康检查功能。

## 功能特性

- 高性能 HTTP/HTTPS 服务器
- RESTful API 路由支持
- 可扩展的中间件系统
- 动态配置管理
- 内置健康检查和监控
- 请求限流和负载均衡
- 详细的日志记录和指标收集

## 快速开始

### 前置要求

- Go 1.21 或更高版本

### 编译

```bash
./build.sh
```

### 配置

1. 复制配置文件模板并修改：

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
  level: "info"  # 日志打印级别
  output: "file" # 日志输出方式: stdout, file

```

参考examples/run_with_config使用配置文件启动router

### 运行

```bash
go run cmd/main.go

# 或者使用二进制运行
./run.sh
```

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


