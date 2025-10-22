[简体中文](../zh/online_serving/scheduler.md)

# Scheduler

FastDeploy currently supports two types of schedulers: **Local Scheduler** and **Global Scheduler**. The Global Scheduler is designed for large-scale clusters, enabling secondary load balancing across nodes based on real-time workload metrics.

## Scheduling Strategies

### Local Scheduler
The Local Scheduler functions similarly to a memory manager, performing eviction policies based on **task queue length** and **TTL** configurations.

### Global Scheduler
The Global Scheduler is implemented using Redis. Each node actively steals tasks from others when its GPU is idle, then pushes the execution results back to the originating node.

### PD-Separated Scheduler
Building upon the Global Scheduler, FastDeploy introduces the **PD-Separated Scheduling Strategy**, specifically optimized for large language model inference scenarios. It decouples the inference pipeline into two distinct phases:
- **Prefill Phase**: Builds KV cache, which is compute-intensive with high memory usage but low latency.
- **Decode Phase**: Performs autoregressive decoding, which is sequential and time-consuming but requires less memory.

By separating roles (prefill nodes handle request processing while decode nodes manage generation), this strategy enables finer-grained resource allocation, improving throughput and GPU utilization.

## Configuration Parameters
| Parameter Name                       | Type     | Required | Default   | Scope                  | Description                                                                 |
| ------------------------------------ | -------- | -------- | --------- | ---------------------- | --------------------------------------------------------------------------- |
| scheduler_name                       | str      | No       | local     | local,global,splitwise | Scheduler type: `local`, `global`, or `splitwise`                          |
| scheduler_max_size                   | int      | No       | -1        | local                  | Maximum task queue length                                                  |
| scheduler_ttl                        | int      | No       | 900       | local,global,splitwise | Maximum task time-to-live (seconds)                                        |
| scheduler_host                       | str      | No       | 127.0.0.1 | global,splitwise       | Redis server host                                                          |
| scheduler_port                       | int      | No       | 6379      | global,splitwise       | Redis server port                                                          |
| scheduler_db                         | int      | No       | 0         | global,splitwise       | Redis database index                                                       |
| scheduler_password                   | str      | No       | ""        | global,splitwise       | Redis access password                                                      |
| scheduler_topic                      | str      | No       | default   | global,splitwise       | Nodes under the same topic participate in task scheduling                  |
| scheduler_min_load_score             | float    | No       | 3         | global                 | Minimum load threshold for task stealing (idle nodes steal from busy ones) |
| scheduler_load_shards_num            | int      | No       | 1         | global                 | Number of shards for cluster load tracking                                 |
| scheduler_sync_period                | int      | No       | 5         | splitwise              | Node load synchronization interval (seconds)                               |
| scheduler_expire_period              | int      | No       | 3000      | splitwise              | Node heartbeat expiration time (seconds)                                   |
| scheduler_release_load_expire_period | int      | No       | 600       | splitwise              | Request expiration time for load release (seconds)                         |
| scheduler_reader_parallel            | int      | No       | 4         | splitwise              | Number of output reader threads                                            |
| scheduler_writer_parallel            | int      | No       | 4         | splitwise              | Number of writer threads                                                   |
| scheduler_reader_batch_size          | int      | No       | 200       | splitwise              | Batch size for fetching results from Redis                                 |
| scheduler_writer_batch_size          | int      | No       | 200       | splitwise              | Batch size for writing results to Redis                                    |
