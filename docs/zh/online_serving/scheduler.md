[English](../../online_serving/scheduler.md)

# 调度器

FastDeploy 目前支持两种调度器: **本地调度器** 和 **全局调度器** 。 全局调度器专为大规模集群设计，能够基于实时工作负载指标在节点间实现二级负载均衡。

## 调度策略

### 本地调度器
本地调度器可以等效于内存管理器，根据 **任务队列长度** 和 **TTL** 的配置进行内存淘汰。

### 全局调度器
全局调度器基于 Redis 实现，各个节点根据自身 GPU 负载情况，空闲时主动从其他节点窃取任务，然后将任务的执行结果推送回原节点。

### PD分离调度器
基于全局调度器，FastDeploy 引入了专为大语言模型推理场景优化的 **PD 分离调度策略**。该策略将推理流程解耦为两个独立阶段：
- **Prefill 阶段** ：构建 KV 缓存，该过程计算密集度高、显存占用大，但延迟低；

- **Decode 阶段**：进行自回归解码，该过程串行执行、时延高，但显存占用低。

通过角色分离（prefill 节点负责接收并处理请求，decode节点完成后续生成），可以更细粒度地控制资源分配、提高吞吐量与 GPU 利用率。

## 配置参数
| 字段名                               | 字段类型 | 是否必填 | 默认值    | 生效范围                   | 说明                                |
| ------------------------------------ | -------- | -------- | --------- |------------------------|-----------------------------------|
| scheduler_name                       | str      | 否       | local     | local,global,splitwise | 调度器名：`local`，`global`，`splitwise` |
| scheduler_max_size                   | int      | 否       | -1        | local                  | 任务最大队列长度                          |
| scheduler_ttl                        | int      | 否       | 900       | local,global,splitwise | 任务最大存活时间(秒)                       |
| scheduler_host                       | str      | 否       | 127.0.0.1 | global,splitwise       | redis服务地址                         |
| scheduler_port                       | int      | 否       | 6379      | global,splitwise       | redis服务端口                         |
| scheduler_db                         | int      | 否       | 0         | global,splitwise                | redis数据库序号                        |
| scheduler_password                   | str      | 否       | ""        | global,splitwise       | redis访问密码                         |
| scheduler_topic                      | str      | 否       | default   | global,splitwise                | 在同一个主题下的节点之间才会发生任务调度              |
| scheduler_min_load_score             | float    | 否       | 3         | global                 | 当节点的负载大于最小阈值时，若其他节点空闲则可以窃取该节点的任务  |
| scheduler_load_shards_num            | int      | 否       | 1         | global                 | 集群负载信息表的分片数                       |
| scheduler_sync_period                | int      | 否       | 5         | splitwise              | 节点负载同步周期(秒)                       |
| scheduler_expire_period              | int      | 否       | 3000      | splitwise              | 节点信息失效时间(秒)，用于心跳机制                |
| scheduler_release_load_expire_period | int      | 否       | 600       | splitwise              | 请求失效释放负载时间(秒)                     |
| scheduler_reader_parallel            | int      | 否       | 4         | splitwise              | 输出读取线程数                           |
| scheduler_writer_parallel            | int      | 否       | 4         | splitwise              | 写入线程数                             |
| scheduler_reader_batch_size          | int      | 否       | 200       | splitwise              | 每次从 Redis 获取结果的 batch 大小          |
| scheduler_writer_batch_size          | int      | 否       | 200       | splitwise              | 每次向 Redis 写入结果的 batch 大小          |
