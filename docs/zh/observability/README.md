## Observability 示例配置 (`examples/observability`)

该目录提供了一套完整的、基于 Docker Compose 的可观测性（Observability）示例，包括：

- Prometheus：指标收集
- Grafana：指标可视化
- OpenTelemetry Collector：分布式追踪数据接收与处理

开发者可以使用此示例环境 一键启动本地监控与追踪系统。

### 先决条件

需要确保提前安装以下组件：

- Docker
- Docker Compose（或新版 Docker CLI 支持 `docker compose`）

### 使用方法

#### 整体启动

进入目录：

```shell
cd examples/observability
```

在 `examples/observability` 目录下执行以下命令即可启动完整的监控和追踪服务：

```bash
docker compose -f docker-compose.yaml up -d
```

启动完成后可访问：

- Prometheus 访问: http://localhost:9090﻿
- Grafana 访问: http://localhost:3000﻿
- OTLP 接收端: 应用程序应将 Traces 发送到 OTel Collector 的默认端口（通常是 `4317` 或 `4318`）。
  - grpc: 4317端口
  - http: 4318端口
- Jeager 访问：http://localhost:16886﻿

【注意事项】：

- Prometheus的抓取地址换成自己的地址

- Grafana的展示端口映射成自己可以访问的端口

- Jaeger的展示端口映射成自己可以访问的端口

- 如果启动了整体服务就不需要再单独去启动子服务了

#### metrics启动

进入目录：

```shell
cd examples/observability/metrics
```

在 `examples/observability` 目录下执行以下命令即可启动完整的监控和追踪服务：

```bash
docker compose -f prometheus_compose.yaml up -d
```

启动完成后可访问：

- Grafana 访问: http://localhost:3000﻿

#### trace启动

进入目录：

```shell
cd examples/observability/tracing
```

在 `examples/observability` 目录下执行以下命令即可启动完整的监控和追踪服务：

```bash
docker compose -f tracing_compose.yaml up -d
```

启动完成后可访问：

- OTLP 接收端:应用程序应将 Traces 发送到 OTel Collector 的默认端口（通常是 `4317` 或 `4318`）。
  - grpc: 4317端口
  - http: 4318端口
- Jeager 访问：http://localhost:16886﻿

### 目录结构与文件说明

#### 核心启动文件

| 文件名              | 作用       | 详情                                                         |
| ------------------- | ---------- | ------------------------------------------------------------ |
| docker-compose.yaml | 主启动文件 | 定义并启动完整的可观测性组件（Prometheus、Grafana、OTel Collector、Jaeger）。这是启动整个 Observability 环境的唯一入口。 |

#### 指标 (Metrics) 与监控配置

| 文件/目录                                         | 作用                   | 详情                                                         |
| ------------------------------------------------- | ---------------------- | ------------------------------------------------------------ |
| metrics                                           | 指标配置根目录         | 包含所有与指标收集和 Prometheus 相关的配置。                 |
| prometheus.yaml                                   | Prometheus 主配置      | 定义抓取目标（scrape targets）、全局采集参数，并可选地配置记录规则（recording rules）。所有监控端点都在此定义。 |
| prometheus_compose.yaml                           | Prometheus Docker 配置 | 定义 Prometheus 容器、卷挂载和网络设置。                     |
| grafana/datasources/datasource.yaml               | 数据源配置             | 定义 Grafana 连接 Prometheus 的方式。                        |
| grafana/dashboards/config/dashboard.yaml          | 仪表板加载配置         | 指定仪表板 JSON 文件所在路径。                               |
| grafana/dashboards/json/fastdeploy-dashboard.json | 仪表板                 | 包含 `fastdeploy`监控指标的可视化布局与查询定义。            |

#### 分布式追踪 (Tracing) 配置

| 文件/目录            | 作用                | 详情                                                         |
| -------------------- | ------------------- | ------------------------------------------------------------ |
| tracing              | 追踪配置根目录      | 包含所有与分布式追踪相关的配置。                             |
| opentelemetry.yaml   | OTel Collector 配置 | 定义 Collector 的数据管道：<br />• receivers：接收 OTLP 数据（traces, metrics, logs）<br />• processors：处理与批次化数据<br />• exporters：将数据导出到追踪后端（如 Jaeger）或文件<br />• extensions：健康检查、pprof 和 zpages<br />• pipelines：定义 traces、metrics 和 logs 的完整处理流程 |
| tracing_compose.yaml | Tracing Docker 配置 | 定义 OTel Collector 和 Jaeger 的容器配置。                   |

### 4. 如何定制

#### 4.1 修改指标抓取目标

若应用程序端口、路径更改，请编辑：

```plain
metrics/prometheus.yaml
```

#### 4.2 调整追踪采样率或处理逻辑

编辑：

```plain
tracing/opentelemetry.yaml
```

#### 4.3 添加自定义 Grafana 仪表盘

1. 新增 JSON 仪表盘至：

```plain
grafana/dashboards/json/
```

1. 在下方文件中注册该仪表盘，使 Grafana 自动加载：

```plain
grafana/dashboards/config/dashboard.yaml
```
