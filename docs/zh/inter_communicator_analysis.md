# FastDeploy `inter_communicator` 模块详细分析文档

## 一、模块总览

`inter_communicator` 是 FastDeploy 大模型推理引擎中的 **跨进程通信层**，位于 `fastdeploy/inter_communicator/` 目录下。该模块提供三大核心通信组件，用于 Engine（调度引擎）、Worker（推理执行进程）、CacheManager（KV Cache 管理器）等多进程之间的数据交换与状态同步。

| 组件 | 通信机制 | 核心用途 |
|------|----------|----------|
| **EngineWorkerQueue** | `multiprocessing.managers.BaseManager` | Engine ↔ Worker 之间的 **推理任务下发与结果回收** |
| **EngineCacheQueue** | `multiprocessing.managers.BaseManager` | PrefixCacheManager ↔ CacheTransferManager 之间的 **KV Cache 搬运任务分发与完成通知** |
| **IPCSignal** | `multiprocessing.shared_memory.SharedMemory` + numpy | 多进程间的 **轻量级状态信号**（无锁快速读写） |

此外模块中还有辅助组件：
- **FMQ / FMQFactory**：基于 ZeroMQ 的异步消息队列（API Server ↔ Engine ↔ Worker 的另一通道）
- **ZmqIpcServer / ZmqIpcClient**：基于 ZeroMQ 的 IPC/TCP 通信客户端和服务端
- **ipc_signal_const**：IPCSignal 使用的常量状态码定义

### 模块目录结构

```
fastdeploy/inter_communicator/
├── __init__.py                # 导出所有公开组件
├── engine_worker_queue.py     # EngineWorkerQueue 实现
├── engine_cache_queue.py      # EngineCacheQueue 实现
├── ipc_signal.py              # IPCSignal 实现
├── ipc_signal_const.py        # 状态常量定义
├── fmq.py                     # FMQ (Fast Message Queue) 基于 ZeroMQ 的异步队列
├── fmq_factory.py             # FMQ 工厂类
├── zmq_server.py              # ZeroMQ 服务端
└── zmq_client.py              # ZeroMQ 客户端
```

---

## 二、IPCSignal 详细分析

### 2.1 实现原理

`IPCSignal` 是对 `multiprocessing.shared_memory.SharedMemory` 的封装，底层使用 **POSIX 共享内存**。核心特点：

- 将 numpy 数组映射到共享内存缓冲区，多个进程可 **无锁直接读写同一块内存**
- 用 `name` 作为全局唯一标识符，支持 `suffix` 后缀避免多实例冲突
- `create=True` 时创建共享内存，`create=False` 时附加到已存在的共享内存

```
进程A (create=True)  →  SharedMemory("/dev/shm/signal_name")  ←  进程B (create=False)
        ↓                        ↓                                    ↓
   self.value[0] = 1       POSIX Shared Memory                  读取 value[0] == 1
```

> **源码位置**: `fastdeploy/inter_communicator/ipc_signal.py`

### 2.2 构造参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 共享内存块的唯一名称 |
| `array` | `np.ndarray` | numpy 数组模板，定义 shape 和数据类型 |
| `dtype` | `np.dtype` | 数据类型（必须与 `array.dtype` 一致） |
| `suffix` | `int` | 追加到 name 的后缀，避免多引擎实例名称冲突 |
| `create` | `bool` | `True` 创建新内存块；`False` 连接已存在的内存块 |
| `shm_size` | `int` | 共享内存块字节大小（当 `array` 和 `dtype` 为 None 时使用） |

### 2.3 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `shm` | `SharedMemory` | 底层共享内存对象 |
| `value` | `np.ndarray` 或 `None` | 映射到共享内存的 numpy 数组，用于读写信号值。如果仅指定 `shm_size` 而不传 `array/dtype`，则 `value=None`，需手动操作 `shm.buf` |

### 2.4 ipc_signal_const 状态常量

> **源码位置**: `fastdeploy/inter_communicator/ipc_signal_const.py`

| 常量类 | 值 | 含义 |
|--------|-----|------|
| **ModelWeightsStatus** | `NORMAL=0, UPDATING=1, CLEARING=-1, CLEARED=-2` | 模型权重状态（正常/更新中/清理中/已清理） |
| **PrefixTreeStatus** | `NORMAL=0, UPDATING=1, CLEARING=-1, CLEARED=-2` | 前缀树状态 |
| **KVCacheStatus** | `NORMAL=0, UPDATING=1, CLEARING=-1, CLEARED=-2` | KV Cache 状态 |
| **ExistTaskStatus** | `EMPTY=0, EXIST=1, REFUSE=2` | 任务队列状态（空/有任务/拒绝） |
| **RearrangeExpertStatus** | `FREE=0, DOING=1, LOAD_SUCC=2, DONE=3` | 专家重排状态（MoE EPLB 专用） |

### 2.5 IPCSignal 在各模块中的实例化汇总

以下是系统中创建的所有 IPCSignal 实例及其作用：

#### 2.5.1 CommonEngine 创建的信号（`fastdeploy/engine/common_engine.py`）

| 信号名称 | 数据类型 | 消费方 | 用途 |
|----------|----------|--------|------|
| `exist_task_signal` | int32[1] | WorkerProcess | 通知 Worker 是否有新推理任务 |
| `exist_swapped_task_signal` | int32[1] | WorkerProcess | 通知是否有需要 swap 回 GPU 的任务 |
| `exist_prefill_task_signal` | int32[1] | WorkerProcess | 通知是否有新的 prefill 阶段任务 |
| `engine_forward_signal` | int32[1] | WorkerProcess | Engine 通知 Worker 执行一次 forward 推理 |
| `worker_healthy_live_signal` | int32[1] | WorkerProcess / EngineClient / CacheTransferManager | Worker 健康心跳信号 |
| `cache_ready_signal` | int32[1] | Worker(GPU/XPU/Metax) / MTP / CacheMessager | KV Cache 初始化完成信号 |
| `swap_space_ready_signal` | int32[1] | PrefixCacheManager / CacheTransferManager | CPU swap 空间就绪信号 |
| `cache_transfer_inited_signal` | int32[1] | PrefixCacheManager / CacheTransferManager | Cache 传输管理器初始化完成信号 |
| `model_weights_status_signal` | int32[1] | WorkerProcess / EngineClient / DynamicWeightManager | 模型权重状态（使用 `ModelWeightsStatus` 常量） |
| `prefix_tree_status_signal` | int32[1] | PrefixCacheManager / EngineClient | 前缀树状态信号 |
| `kv_cache_status_signal` | int32[1] | CacheTransferManager / EngineClient / DynamicWeightManager | KV Cache 状态信号 |
| `worker_ready_signal` | int32[1] | Engine 自身 | Worker 进程就绪信号 |
| `launched_cache_manager_signal` | int32[1] | Engine 自身 | CacheManager 启动完成信号 |
| `launched_expert_service_signal` | int32[1] | Engine 自身 / WorkerProcess | ExpertService 启动完成信号 |
| `loaded_model_signal` | int32[1] | AsyncLLM / WorkerProcess | 模型加载完成信号 |
| `get_profile_block_num_signal` | int32[1] | WorkerProcess | profile 阶段 block 数量通信 |

#### 2.5.2 WorkerProcess 连接的信号（`fastdeploy/worker/worker_process.py`）

WorkerProcess 以 `create=False` 连接上述 CommonEngine 创建的信号，此外还创建/连接以下额外信号：

| 信号名称 | 数据类型 | 用途 |
|----------|----------|------|
| `signal_update_weight_from_tensor_array` | int32[N] | 从 tensor 更新权重的信号数组（EPLB） |
| `rearrange_experts_signal` | int32[N] | 专家重排状态信号数组（EPLB） |
| `local_experts_token_stats_array` | — | 本地专家 token 统计信息 |
| `signal_clear_experts_token_stats` | — | 清除专家 token 统计的信号 |

#### 2.5.3 PrefixCacheManager 创建的信号（`fastdeploy/cache_manager/prefix_cache_manager.py`）

| 信号名称 | 数据类型 | 消费方 | 用途 |
|----------|----------|--------|------|
| `shm_cache_task_flag_broadcast` | int32[N] | CacheTransferManager | Cache 任务广播标志 |
| `cache_task_is_paused_signal` | int32[1] | CacheTransferManager | Cache 任务暂停信号 |
| `cache_task_inflight_signal` | int32[1] | CacheTransferManager → PrefixCacheManager | 是否有正在执行的 cache 传输 |

#### 2.5.4 CacheTransferManager 连接的信号（`fastdeploy/cache_manager/cache_transfer_manager.py`）

| 信号名称 | 数据类型 | 用途 |
|----------|----------|------|
| `cache_ready_signal` | int32[1] | 读取 — 等待 Cache 初始化完成 |
| `swap_space_ready_signal` | int32[1] | 读取 — 等待 swap 空间就绪 |
| `cache_task_broadcast_signal` | int32[N] | 读取 — 接收 cache 任务广播 |
| `cache_task_is_paused_signal` | int32[1] | 读取 — 检查 cache 任务是否暂停 |
| `cache_task_inflight_signal` | int32[1] | 写入 — 报告正在执行的传输 |
| `worker_healthy_live_signal` | int32[1] | 读取 — 检查 worker 健康状态 |
| `kv_cache_status_signal` | int32[1] | 读取 — 检查 KV Cache 状态 |
| `cache_transfer_inited_signal` | int32[1] | 写入 — 标记传输管理器初始化完成 |

#### 2.5.5 ExpertsManager 创建的信号（`fastdeploy/eplb/experts_manager.py`）

| 信号名称 | 数据类型 | 用途 |
|----------|----------|------|
| `rearrange_experts_ips_size_signal` | int32[1] | 专家重排 IP 列表大小 |
| `shm_rearrange_experts_ips_list` | — | 专家重排 IP 列表共享内存 |
| `rearrange_experts_signal` | int32[N] | 专家重排状态 |
| `signal_update_weight_from_tensor_array` | int32[N] | 从 tensor 更新权重的信号 |
| `signal_update_weight_from_disk_array` | int32[N] | 从磁盘更新权重的信号 |
| `shm_all_experts_token_stats` | — | 所有专家的 token 统计信息 |
| `update_weight_from_disk_result` | int32[N] | 磁盘权重更新结果 |

#### 2.5.6 其他模块的信号

| 模块 | 信号名称 | 用途 |
|------|----------|------|
| EngineClient (`entrypoints/engine_client.py`) | 多个状态信号 (create=False) | API 客户端监控系统健康和就绪状态 |
| AsyncLLM (`engine/async_llm.py`) | `loaded_model_signal` (create=False) | 异步引擎监控模型加载进度 |
| ResourceManagerV1 (`engine/sched/resource_manager_v1.py`) | `need_block_num_signal` | V1 资源调度器 block 需求通信 |
| GPUModelRunner (`worker/gpu_model_runner.py`) | `cache_ready_signal` (create=False) | GPU Worker 等待 cache 就绪 |
| XPUModelRunner (`worker/xpu_model_runner.py`) | `cache_ready_signal` (create=False) | XPU Worker 等待 cache 就绪 |
| MetaxModelRunner (`worker/metax_model_runner.py`) | `cache_ready_signal` (create=False) | Metax Worker 等待 cache 就绪 |
| MTP (`spec_decode/mtp.py`) | `cache_ready_signal` (create=False) | 多 token 预测等待 cache 就绪 |
| CacheMessager (`cache_manager/cache_messager.py`) | `step_shm_value`, `layer_shm_value`, `cache_ready_signal` | Splitwise 模式下 prefill 步骤/层进度 |
| DynamicWeightManager (`rl/dynamic_weight_manager.py`) | 通过参数接收 `model_weights_status` / `kv_cache_status` | RL 场景下动态权重更新的状态协调 |

---

## 三、EngineWorkerQueue 详细分析

### 3.1 架构设计

> **源码位置**: `fastdeploy/inter_communicator/engine_worker_queue.py`

`EngineWorkerQueue` 基于 Python `multiprocessing.managers.BaseManager`，采用 **Server-Client 模式**：

- **Server 端**（在 CommonEngine 中创建）：管理所有共享数据结构
- **Client 端**（在 WorkerProcess 中连接）：通过代理对象访问共享数据

支持 **多机部署**（address 可配置为实际 IP）和 **数据并行**（每个 DP 有独立的数据结构副本）。

### 3.2 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `address` | `Tuple[str, int]` | `("0.0.0.0", 5000)` | 网络地址，`0.0.0.0` 表示单机部署 |
| `authkey` | `bytes` | `b"secret_key"` | 认证密钥 |
| `is_server` | `bool` | `False` | 是否作为服务端 |
| `num_client` | `int` | `1` | 客户端总数（= tensor parallel size） |
| `client_id` | `int` | `-1` | 客户端唯一 ID（= tensor parallel rank） |
| `local_data_parallel_size` | `int` | `1` | 本地数据并行大小 |
| `local_data_parallel_id` | `int` | `0` | 本地数据并行 ID |

### 3.3 注册变量详解

#### 3.3.1 核心推理任务队列

| 变量名 | 代理类型 | 初始值 | 作用 |
|--------|----------|--------|------|
| `tasks` (via `get_tasks`) | `ListProxy` | `[]` | **主任务队列**：存放 Engine 下发给 Worker 的推理任务。每个元素是 `(tasks_list, batch_size)` 元组，tasks_list 包含 Request 对象 |
| `client_read_flag` (via `get_client_read_flag`) | `ListProxy` | `[1]*num_client` | **客户端读取标志**：长度 = TP 数，每个位置表示对应 TP rank 是否已读取当前任务。初始全 1 表示"空闲可接收" |
| `lock` (via `get_lock`) | `AcquirerProxy` | `threading.Lock()` | **主锁**：保护 tasks 和 client_read_flag 的并发访问 |
| `read_finish_flag` (via `get_read_finish_flag`) | `ValueProxy` | `Value("i", 0)` | **读取完成标志**：所有 client 读完后设置 |
| `exist_tasks_inter_signal` (via `get_exist_tasks_inter_signal`) | `ValueProxy` | `Value("i", 0)` | **跨进程任务存在信号**：多机部署时使用，1=有任务，0=无任务 |
| `connected_client_counter` (via `get_connected_client_counter`) | `ValueProxy` | `Value("i", 0)` | **已连接客户端计数**：记录有多少 Worker 已连接 |

#### 3.3.2 Cache 信息传递（Engine → Worker）

| 变量名 | 代理类型 | 初始值 | 作用 |
|--------|----------|--------|------|
| `cache_infos` (via `get_cache_infos`) | `ListProxy` | `[]` | **Cache 信息队列**：Engine 向 Worker 传递 KV Cache 分配/释放信息 |
| `client_read_info_flag` (via `get_client_read_info_flag`) | `ListProxy` | `[0]*num_client` | **Cache 信息读取标志**：跟踪每个 TP rank 是否已读取 cache_info |
| `lock_info` (via `get_lock_info`) | `AcquirerProxy` | `threading.Lock()` | **Cache 信息锁**：保护 cache_infos 的并发访问 |

#### 3.3.3 PD 分离（Prefill-Decode Disaggregation）相关

| 变量名 | 代理类型 | 初始值 | 作用 |
|--------|----------|--------|------|
| `disaggregate_requests` (via `get_disaggregate_requests`) | `Queue` | `Queue()` | **分离请求队列**：PD 分离模式下存放分离的推理请求 |
| `connect_rdma_tasks` (via `get_connect_rdma_tasks`) | `ListProxy` | `[]` | **RDMA 连接任务队列**：存放需要建立 RDMA 连接的任务 |
| `connect_rdma_task_responses` (via `get_connect_rdma_tasks_responses`) | `ListProxy` | `[]` | **RDMA 连接响应队列**：存放 RDMA 连接建立结果 |
| `finished_send_cache_list` (via `get_finish_request_queue`) | `ListProxy` | `[]` | **Cache 发送完成列表**：记录已完成 cache 发送的请求 |
| `finished_add_cache_task_list` (via `get_finish_add_cache_task_queue`) | `ListProxy` | `[]` | **Cache 任务添加完成列表**：记录已完成 cache 任务添加的请求 |

#### 3.3.4 PD 分离同步标志

| 变量名 | 代理类型 | 初始值 | 作用 |
|--------|----------|--------|------|
| `client_get_connect_task_flag` | `ListProxy` | `[0]*num_client` | 跟踪每个 TP rank 是否已读取 RDMA 连接任务 |
| `client_get_connect_task_response_flag` | `ListProxy` | `[0]*num_client` | 跟踪每个 TP rank 是否已写入 RDMA 连接响应 |
| `client_get_finished_add_cache_task_flag` | `ListProxy` | `[0]*num_client` | 跟踪每个 TP rank 是否已写入 cache 任务完成信号 |
| `client_get_finish_send_cache_flag` | `ListProxy` | `[0]*num_client` | 跟踪每个 TP rank 是否已写入 cache 发送完成信号 |
| `can_put_next_connect_task_response_flag` | `ValueProxy` | `Value("i", 1)` | 控制是否可以放入下一批 RDMA 连接响应（防止覆盖未读数据） |
| `can_put_next_add_task_finished_flag` | `ValueProxy` | `Value("i", 1)` | 控制是否可以放入下一批 cache 任务完成信号 |
| `can_put_next_send_cache_finished_flag` | `ValueProxy` | `Value("i", 1)` | 控制是否可以放入下一批 cache 发送完成信号 |

#### 3.3.5 PD 分离锁

| 变量名 | 代理类型 | 作用 |
|--------|----------|------|
| `connect_task_lock` | `AcquirerProxy` | RDMA 连接任务队列的并发保护锁 |
| `connect_task_response_lock` | `AcquirerProxy` | RDMA 连接响应队列的并发保护锁 |
| `finish_add_cache_task_lock` | `AcquirerProxy` | Cache 任务完成队列的并发保护锁 |
| `finish_send_cache_lock` | `AcquirerProxy` | Cache 发送完成队列的并发保护锁 |

#### 3.3.6 Barrier 同步屏障

| 变量名 | Barrier 参与数 | 作用 |
|--------|---------------|------|
| `finish_request_barrier` | num_client (TP) | 所有 TP rank 完成请求处理后同步 |
| `worker_process_tp_barrier` | num_client (TP) | Worker 进程 TP 级同步 |
| `connect_task_barrier` | num_client (TP) | RDMA 连接任务读取同步 |
| `connect_task_response_barrier` | num_client (TP) | RDMA 连接响应写入同步 |
| `finish_add_cache_task_barrier` | num_client (TP) | Cache 任务添加完成同步 |
| `begin_send_cache_barrier` | num_client (TP) | 开始 cache 发送同步 |
| `finish_send_cache_barrier` | num_client (TP) | Cache 发送完成同步 |
| `cache_info_barrier` | num_client (TP) | Cache 信息获取同步 |

#### 3.3.7 本地共享内存信号

| 变量名 | 类型 | 作用 |
|--------|------|------|
| `exist_tasks_intra_signal` | `IPCSignal(int32[1])` | **仅单机部署**：通过 POSIX 共享内存快速检测任务是否存在，性能优于 BaseManager 跨进程 ValueProxy |

### 3.4 核心方法说明

#### 任务下发与获取

| 方法 | 调用方 | 说明 |
|------|--------|------|
| `put_tasks(tasks)` | Engine | 向队列写入推理任务，等待所有 client 读完上一批后才写入 |
| `get_tasks()` | Worker | 读取任务并标记已读，返回 `(tasks, all_client_read)` |
| `exist_tasks()` | Worker | 无锁快速检查是否有任务（单机用共享内存，多机用 ValueProxy） |
| `set_exist_tasks(flag)` | Engine | 设置任务存在标志 |
| `num_tasks()` | Engine | 获取当前任务数 |
| `clear_data()` | Engine | 清空任务队列和标志位 |

#### Cache 信息传递

| 方法 | 调用方 | 说明 |
|------|--------|------|
| `put_cache_info(cache_info)` | Engine | 写入 KV Cache 分配/释放信息 |
| `get_cache_info()` | Worker | 读取 cache 信息并标记已读 |

#### PD 分离相关

| 方法 | 调用方 | 说明 |
|------|--------|------|
| `put_connect_rdma_task(task)` | Engine | 下发 RDMA 连接任务 |
| `get_connect_rdma_task()` | Worker | 获取 RDMA 连接任务 |
| `put_connect_rdma_task_response(resp)` | Worker | 写入 RDMA 连接结果 |
| `get_connect_rdma_task_response()` | Engine | 获取并合并所有 TP rank 的 RDMA 连接结果 |
| `put_finished_req(result)` | Worker | 写入 cache 发送完成结果 |
| `get_finished_req()` | Engine | 获取 cache 发送完成结果 |
| `put_finished_add_cache_task_req(req_ids)` | Worker | 写入 cache 任务添加完成 |
| `get_finished_add_cache_task_req()` | Engine | 获取 cache 任务添加完成 |
| `put_disaggregated_tasks(item)` | Engine | 写入分离请求 |
| `get_disaggregated_tasks()` | Worker | 获取分离请求 |

### 3.5 任务下发工作流

```
Engine 调用 put_tasks(tasks):
  1. 获取锁
  2. 等待所有 client_read_flag == 1（即所有 Worker 都已读完上一批）
  3. 清空队列 → 重置 client_read_flag 为全 0
  4. 写入新任务 → 设置 exist_tasks 信号为 True
  5. 释放锁

Worker 检测到 exist_tasks() == True:
  1. 调用 get_tasks()
  2. 获取锁 → 读取任务
  3. 设置 client_read_flag[client_id] = 1
  4. 如果所有 client 都读完 → 清空队列 → exist_tasks 设为 False
  5. 释放锁
```

---

## 四、EngineCacheQueue 详细分析

### 4.1 架构设计

> **源码位置**: `fastdeploy/inter_communicator/engine_cache_queue.py`

与 EngineWorkerQueue 类似，`EngineCacheQueue` 也基于 `BaseManager`，但专门服务于 **Cache 传输子系统**。Server 由 CommonEngine 创建，Client 由 PrefixCacheManager 和 CacheTransferManager 连接。

### 4.2 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `address` | `Tuple[str, int]` | `("127.0.0.1", 56666)` | 网络地址 |
| `authkey` | `bytes` | `b"cache_queue_service"` | 认证密钥 |
| `is_server` | `bool` | `False` | 是否作为服务端 |
| `num_client` | `int` | `1` | 客户端总数（= tensor parallel size） |
| `client_id` | `int` | `-1` | 客户端唯一 ID |
| `local_data_parallel_size` | `int` | `1` | 数据并行大小 |
| `local_data_parallel_id` | `int` | `0` | 数据并行 ID |

### 4.3 注册变量详解

#### 4.3.1 Cache 传输任务队列

| 变量名 | 代理类型 | 初始值 | 作用 |
|--------|----------|--------|------|
| `transfer_task_queue` (via `get_transfer_task_queue`) | `ListProxy` | `[]` | **Cache 传输任务队列**：存放 GPU↔CPU/Storage 的传输指令，如 `(CacheStatus.GPU2STORAGE, task)` |
| `tansfer_done_queue` (via `get_tansfer_done_queue`) | `ListProxy` | `[]` | **传输完成队列**：存放已完成的传输结果通知 |
| `task_sync_value` (via `get_cache_sync_value`) | `ValueProxy` | `Value("i", 0)` | **位掩码同步值**：用位掩码跟踪每个 TP rank 的读取状态（详见 4.4） |
| `task_lock` (via `get_transfer_task_lock`) | `AcquirerProxy` | `threading.Lock()` | **传输任务锁**：保护 transfer_task_queue 的并发访问 |
| `task_done_lock` (via `get_transfer_task_done_lock`) | `AcquirerProxy` | `threading.Lock()` | **完成队列锁**：保护 tansfer_done_queue 的并发访问 |

#### 4.3.2 Barrier 同步屏障

| 变量名 | Barrier 参与数 | 作用 |
|--------|---------------|------|
| `barrier` | num_client (TP) | 通用同步屏障 |
| `barrier0` ~ `barrier3` | num_client (TP) | Cache 传输各阶段同步屏障 |
| `pause_barrier` | num_client (TP) | Cache 传输暂停同步 |
| `resume_barrier` | num_client (TP) | Cache 传输恢复同步 |
| `swap_to_cpu_barrier1` | num_client (TP) | GPU→CPU 交换第一阶段同步 |
| `swap_to_cpu_barrier2` | num_client (TP) | GPU→CPU 交换第二阶段同步 |
| `swap_to_gpu_barrier1` | num_client (TP) | CPU→GPU 交换第一阶段同步 |
| `swap_to_gpu_barrier2` | num_client (TP) | CPU→GPU 交换第二阶段同步 |
| `swap_storage_to_gpu_barrier` | num_client (TP) | 外存→GPU 传输同步 |
| `swap_to_storage_barrier` | num_client (TP) | GPU→外存 传输同步 |

### 4.4 核心同步机制：位掩码读取

EngineCacheQueue 的 `put_transfer_task` / `get_transfer_task` 使用 **位掩码**（bitmask）来追踪多个 TP rank 的读取状态：

```python
total_num = (1 << num_client) - 1   # 例如 4 个 client → 0b1111 = 15
position = 1 << client_id           # 例如 client_id=2 → 0b0100 = 4

# put 时：等待 task_sync_value 回到 0 或达到 total_num
# get 时：set_value = task_sync_value | position
#   - 如果 set_value >= total_num，说明所有 client 都已读取，清空队列
```

### 4.5 核心方法说明

| 方法 | 调用方 | 说明 |
|------|--------|------|
| `put_transfer_task(item)` | PrefixCacheManager | 发布 cache 搬运指令（如 GPU→Storage, Storage→GPU） |
| `get_transfer_task()` | CacheTransferManager | 获取搬运指令，每个 client 读一次，全部读完后清空 |
| `clear_transfer_task()` | PrefixCacheManager | 清空任务队列 |
| `put_transfer_done_signal(item)` | CacheTransferManager | 报告搬运完成 |
| `get_transfer_done_signal()` | PrefixCacheManager | 获取搬运完成通知 |
| `empty()` | — | 检查任务队列是否为空 |
| `result_queue_empty()` | — | 检查结果队列是否为空 |

### 4.6 Cache 传输工作流

```
PrefixCacheManager 调用 put_transfer_task((CacheStatus.GPU2STORAGE, task)):
  1. 获取锁
  2. 等待 task_sync_value 完成或归零
  3. 写入任务 → 重置 task_sync_value = 0
  4. 释放锁

CacheTransferManager (每个 TP rank) 调用 get_transfer_task():
  1. 获取锁
  2. 检查 task_sync_value 的对应 bit 是否为 0（未读）
  3. 读取任务 → 设置 bit（task_sync_value |= position）
  4. 如果所有 bit 都已设置 → 清空队列
  5. 释放锁

[各 TP rank 独立执行 GPU↔CPU/Storage 数据传输]

CacheTransferManager 调用 put_transfer_done_signal(result):
  1. 获取完成队列锁
  2. 写入完成结果
  3. 释放锁

PrefixCacheManager 调用 get_transfer_done_signal():
  1. 获取完成队列锁
  2. 弹出并返回完成结果
  3. 释放锁
```

---

## 五、通信拓扑总图

```
                           ┌─────────────────────────────────┐
                           │         API Server               │
                           └──────┬──────────────▲────────────┘
                                  │ ZMQ/FMQ      │ ZMQ/FMQ
                           ┌──────▼──────────────┴────────────┐
                           │       Engine (CommonEngine)       │
                           │                                   │
                           │  EngineWorkerQueue (server)       │
                           │  EngineCacheQueue (server)        │
                           │  IPCSignal (creator)              │
                           └──┬────────┬─────────┬────────────┘
               ┌──────────────┘        │         └──────────────────┐
               │                       │                            │
    ┌──────────▼──────────┐  ┌─────────▼───────────────┐  ┌────────▼──────────────┐
    │  Worker Process      │  │  PrefixCacheManager      │  │  CacheTransfer        │
    │  (TP rank 0..N)      │  │                          │  │  Manager (TP rank)    │
    │                      │  │  EngineCacheQueue        │  │                       │
    │  EngineWorkerQueue   │  │    (client)              │  │  EngineCacheQueue     │
    │    (client)          │  │  IPCSignal (reader)      │  │    (client)           │
    │  IPCSignal (reader)  │  └─────┬────────────────────┘  │  IPCSignal (reader)   │
    └──────────────────────┘        │                       └───────────▲────────────┘
                                    │   put_transfer_task              │
                                    └─────────────────────────────────→┘
                                          get_transfer_done_signal
```

### 数据流方向

1. **推理任务流**（EngineWorkerQueue）：
   - `Engine → put_tasks() → [Queue] → get_tasks() → Worker (all TP ranks)`

2. **Cache 信息流**（EngineWorkerQueue）：
   - `Engine → put_cache_info() → [Queue] → get_cache_info() → Worker (all TP ranks)`

3. **Cache 搬运任务流**（EngineCacheQueue）：
   - `PrefixCacheManager → put_transfer_task() → [Queue] → get_transfer_task() → CacheTransferManager`
   - `CacheTransferManager → put_transfer_done_signal() → [Queue] → get_transfer_done_signal() → PrefixCacheManager`

4. **PD 分离流**（EngineWorkerQueue）：
   - RDMA 连接：`Engine → put_connect_rdma_task() → Worker → put_connect_rdma_task_response() → Engine`
   - Cache 发送：`Worker → put_finished_req() → Engine`
   - 分离请求：`Engine → put_disaggregated_tasks() → Worker`

5. **状态信号流**（IPCSignal）：
   - 单向写读：`Writer 进程 → SharedMemory → Reader 进程`（无锁，纳秒级延迟）

---

## 六、同步模式总结

| 模式 | 使用场景 | 示例 |
|------|----------|------|
| **共享内存轮询** | 高频状态检测 | `IPCSignal.exist_task_signal` — Worker 轮询是否有新任务 |
| **Lock + ListProxy** | 任务入队/出队 | `EngineWorkerQueue.put_tasks/get_tasks` |
| **位掩码同步** | 多 rank 读取确认 | `EngineCacheQueue.task_sync_value` — 确认所有 TP rank 读完同一任务 |
| **Barrier 屏障** | 多 rank 阶段同步 | `EngineCacheQueue.swap_to_cpu_barrier1/2` — 所有 rank 同步进入/退出 swap 阶段 |
| **Flag 数组** | 多 rank 写入确认 | `EngineWorkerQueue.client_get_connect_task_response_flag` — 确认所有 rank 都已提交响应 |
| **ValueProxy 门控** | 写入流量控制 | `can_put_next_connect_task_response_flag` — 防止新数据覆盖未读数据 |

---

## 七、Golang Router 中的引用

在 `fastdeploy/golang_router/internal/manager/` 中，`EngineWorkerQueuePort` 作为分布式部署注册信息的一部分：

- `instance.go`：实例配置中包含 `EngineWorkerQueuePort` 字段
- `handler.go`：注册请求的 JSON 结构中传递该端口
- `register.go`：注册时将端口信息发送给路由管理器

这使得 Golang Router 能够感知各个引擎实例的 EngineWorkerQueue 端口，支持 **跨机 PD 分离** 场景下的实例发现与连接。

---

## 八、设计要点与注意事项

1. **单机 vs 多机优化**：EngineWorkerQueue 在单机部署时使用 IPCSignal（POSIX 共享内存）作为快速任务存在检测，多机部署时退化为 BaseManager 的 ValueProxy。

2. **所有共享数据按 DP 分组**：EngineWorkerQueue 和 EngineCacheQueue 的所有共享变量都按 `local_data_parallel_size` 创建独立副本，避免多 DP 之间的干扰。

3. **TP 同步模式**：所有需要多 TP rank 同步的操作都使用 `client_read_flag` 数组 + 求和检查的方式，确保每个 rank 恰好读取一次。

4. **位掩码 vs Flag 数组**：EngineCacheQueue 使用位掩码（更紧凑），EngineWorkerQueue 使用 Flag 数组（更直观），两者功能等价。

5. **IPCSignal 的命名冲突处理**：如果创建时发现同名共享内存已存在，会先 `unlink` 再重建，避免残留数据导致错误。
