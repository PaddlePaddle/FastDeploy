# FastDeploy vs vLLM vs SGLang：进程间通信与状态同步机制对比分析

## 一、背景与动机

在大模型推理服务中，**Engine（调度器）** 与 **Worker（模型执行器）** 通常运行在不同进程中，需要高效的进程间通信（IPC）来：

1. **下发推理任务**：Engine 把 batch 调度结果传给 Worker
2. **回收推理结果**：Worker 把 model output 返回给 Engine
3. **同步状态信号**：各进程间协调就绪状态、健康心跳、资源状态等
4. **TP 同步**：多个 Tensor Parallel rank 之间的调度数据广播

FastDeploy、vLLM、SGLang 三个项目采用了**截然不同的架构方案**，下面逐一分析。

---

## 二、FastDeploy 的 IPCSignal + BaseManager 方案

### 2.1 架构概览

FastDeploy 采用 **多进程多组件** 架构，Engine、Worker、PrefixCacheManager、CacheTransferManager 都是独立进程。通信依赖三套机制：

```
┌──────────────────────────────────────────────────────────┐
│                    FastDeploy 通信层                       │
├──────────────┬──────────────────┬────────────────────────┤
│  IPCSignal   │ EngineWorkerQueue│ EngineCacheQueue       │
│  (SharedMem) │ (BaseManager)    │ (BaseManager)          │
├──────────────┼──────────────────┼────────────────────────┤
│ 状态信号      │ 推理任务下发/回收 │ Cache 搬运任务/结果    │
│ 无锁快速读写  │ Lock+ListProxy   │ 位掩码+Barrier同步     │
└──────────────┴──────────────────┴────────────────────────┘
```

### 2.2 IPCSignal — POSIX SharedMemory 封装

> 源码：`fastdeploy/inter_communicator/ipc_signal.py`

**实现原理**：

- 基于 Python `multiprocessing.shared_memory.SharedMemory`（底层 POSIX `/dev/shm`）
- 将 numpy 数组映射到共享内存缓冲区
- 进程 A 写入 `signal.value[0] = 1`，进程 B 直接读取 `signal.value[0]` 即可感知
- **无锁**、**纳秒级延迟**，但无内存屏障保证（依赖 x86 TSO 内存模型）

**使用方式**：

```python
# Engine 端（创建）
signal = IPCSignal(name="exist_task_signal", array=np.zeros([1], dtype=np.int32),
                   dtype=np.int32, suffix=pid, create=True)
signal.value[0] = 1  # 通知 Worker 有新任务

# Worker 端（连接）
signal = IPCSignal(name="exist_task_signal", array=np.zeros([1], dtype=np.int32),
                   dtype=np.int32, suffix=pid, create=False)
while signal.value[0] == 0:
    time.sleep(0.001)  # 轮询等待
```

**系统中的信号实例** — FastDeploy 创建了 **20+ 个 IPCSignal** 用于各种状态同步：

| 类别 | 信号名 | 写入方 | 读取方 | 用途 |
|------|--------|--------|--------|------|
| **任务调度** | `exist_task_signal` | Engine | Worker | 是否有新推理任务 |
| | `exist_swapped_task_signal` | Engine | Worker | 是否有 swap 回 GPU 的任务 |
| | `exist_prefill_task_signal` | Engine | Worker | 是否有 prefill 任务 |
| | `engine_forward_signal` | Engine | Worker | 触发一次 forward |
| **生命周期** | `worker_ready_signal` | Worker | Engine | Worker 就绪 |
| | `loaded_model_signal` | Worker | Engine/AsyncLLM | 模型加载完成 |
| | `launched_cache_manager_signal` | CacheManager | Engine | CacheManager 启动 |
| | `cache_ready_signal` | Worker | GPU/XPU Runner | KV Cache 初始化完成 |
| | `worker_healthy_live_signal` | Worker | Engine/Client | 心跳信号 |
| **资源状态** | `model_weights_status` | Engine | Worker/Client | 权重状态 (NORMAL/UPDATING/CLEARING/CLEARED) |
| | `kv_cache_status` | Engine | Worker/CacheTransfer | KV Cache 状态 |
| | `prefix_tree_status` | Engine | CacheManager/Client | 前缀树状态 |
| | `swap_space_ready_signal` | Engine | CacheManager | CPU swap 空间就绪 |
| | `cache_transfer_inited_signal` | CacheTransfer | CacheManager | 传输管理器就绪 |
| **Cache 传输** | `shm_cache_task_flag_broadcast` | CacheManager | CacheTransfer | 任务广播 |
| | `cache_task_is_paused_signal` | CacheManager | CacheTransfer | 传输暂停 |
| | `cache_task_inflight_signal` | CacheTransfer | CacheManager | 传输进行中 |
| **EPLB 专家管理** | `rearrange_experts_signal` | ExpertsManager | Worker | 专家重排状态 |
| | `signal_update_weight_from_tensor` | ExpertsManager | Worker | 权重更新 |

### 2.3 EngineWorkerQueue — BaseManager Server-Client

> 源码：`fastdeploy/inter_communicator/engine_worker_queue.py`

**实现原理**：

- 基于 `multiprocessing.managers.BaseManager`，Server 端管理 `ListProxy`/`ValueProxy`/`AcquirerProxy`
- Engine 做 Server，所有 TP Worker 做 Client 通过网络代理访问共享数据
- 支持多机部署（IP+Port 可配置为实际网络地址）

**核心数据结构**：

- `tasks` (ListProxy)：任务队列
- `client_read_flag` (ListProxy[int])：每个 TP rank 的读取标志
- `lock` (AcquirerProxy)：保护并发访问
- `exist_tasks_intra_signal` (IPCSignal)：单机部署时用共享内存加速任务检测

**同步模式**：Lock + Flag 数组 + 求和检查（确保所有 TP rank 读完后才清空）

### 2.4 EngineCacheQueue — 位掩码同步

> 源码：`fastdeploy/inter_communicator/engine_cache_queue.py`

与 EngineWorkerQueue 类似，但使用**位掩码**（bitmask）代替 Flag 数组来追踪读取状态：

```python
total_num = (1 << num_client) - 1   # 4 个 TP → 0b1111 = 15
position = 1 << client_id           # rank 2 → 0b0100

# 每个 rank 读取后 set_value |= position
# set_value >= total_num 时说明全部读完
```

另外管理了 **13 个 Barrier** 用于 Cache GPU↔CPU/Storage 搬运的多阶段同步。

### 2.5 FastDeploy 方案特点

| 特点 | 说明 |
|------|------|
| ✅ **低延迟状态检测** | IPCSignal 直接读写共享内存，纳秒级 |
| ✅ **多机支持** | BaseManager 天然支持网络通信 |
| ✅ **灵活的状态模型** | 每个信号独立命名，语义清晰 |
| ❌ **信号数量膨胀** | 20+ 个 IPCSignal 实例，名称管理复杂 |
| ❌ **无内存屏障** | 依赖 x86 TSO，非 x86 平台可能有可见性问题 |
| ❌ **BaseManager 性能** | 通过 socket 代理访问，序列化开销大 |
| ❌ **Barrier 数量多** | EngineCacheQueue 有 13 个 barrier，难以维护 |

---

## 三、vLLM 的 ShmRingBuffer + ZMQ + MessageQueue 方案

### 3.1 架构概览

vLLM v1 采用 **ZMQ 消息 + 共享内存环形缓冲区** 的混合架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM v1 通信层                          │
├─────────────────┬───────────────────┬───────────────────────┤
│  ShmRingBuffer  │  MessageQueue     │  ZMQ Sockets          │
│  (SharedMemory) │  (SHM + ZMQ)      │  (DEALER/ROUTER/PUB)  │
├─────────────────┼───────────────────┼───────────────────────┤
│ TP广播数据通道   │ 封装SHM+ZMQ的     │ Engine↔EngineCore     │
│ 高速数据传输     │ 统一消息队列       │ 进程间命令/结果传递    │
└─────────────────┴───────────────────┴───────────────────────┘
```

### 3.2 EngineCoreProc — ZMQ 消息驱动

> 源码：`vllm/v1/engine/core.py`

vLLM v1 的 **EngineCore** 运行在独立进程（`EngineCoreProc`），通过 ZMQ 与前端通信：

- **输入通道**：ZMQ DEALER socket 接收请求（输入线程 → `input_queue` → 主循环）
- **输出通道**：ZMQ DEALER socket 发送结果（主循环 → `output_queue` → 输出线程）
- **序列化**：使用 `msgspec`（MsgpackEncoder/MsgpackDecoder）高性能序列化

```python
# EngineCoreProc 内部
self.input_queue = queue.Queue()   # 输入线程写，主循环读
self.output_queue = queue.Queue()  # 主循环写，输出线程读

# 输入线程: ZMQ socket → deserialize → input_queue
# 主循环:   input_queue → schedule → execute → output_queue
# 输出线程: output_queue → serialize → ZMQ socket
```

**关键设计**：vLLM **不使用共享内存做状态信号**。调度器与执行器在同一进程内通过函数调用交互，不需要 IPC 状态同步。

### 3.3 ShmRingBuffer — 共享内存环形缓冲区

> 源码：`vllm/distributed/device_communicators/shm_broadcast.py`

这是 vLLM 最独特的通信组件，专门用于 **TP 广播**（1 个 writer → N 个 reader）：

**内存布局**：

```
+-------------------------------+----------------------------------------+
| chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
+-------------------------------+----------------------------------------+
| max_chunks × max_chunk_bytes  | max_chunks × (1 + n_reader) bytes      |

metadata 每 chunk:
+--------------+--------------+--------------+-----+--------------+
| written_flag | reader0_flag | reader1_flag | ... | readerN_flag |
+--------------+--------------+--------------+-----+--------------+
```

**状态机**：

- `0???...???`：未写入，可写
- `1000...000`：刚写入，可读
- `1???...???`：部分 reader 已读
- `1111...111`：全部读完，可写

**特点**：
- 数据和元数据都在同一块共享内存中
- 每个 chunk 有独立的 writer/reader flags，无需全局锁
- 使用 `memory_fence()` 保证内存可见性（Python threading.Lock acquire/release 作为内存屏障）
- 默认 24MiB chunk × 10 chunks = 240MiB 共享内存

### 3.4 SpinCondition — 自适应等待

```python
class SpinCondition:
    def wait(self, timeout_ms=None):
        if time.monotonic() <= self.last_read + self.busy_loop_s:
            sched_yield()  # 忙等待（高频读取时）
        else:
            self.poller.poll(timeout=timeout_ms)  # ZMQ poll（空闲时节能）
```

- 高负载时：busy loop（`sched_yield()`），延迟最低
- 空闲时：退化为 ZMQ poll，节省 CPU
- 支持 cancel() 用于优雅关闭

### 3.5 MessageQueue — 统一封装

> 源码：`vllm/distributed/device_communicators/shm_broadcast.py`

`MessageQueue` 将 ShmRingBuffer 和 ZMQ 统一封装为一个消息队列：

- **本地 reader**：通过 ShmRingBuffer 通信（高速）
- **远程 reader**：通过 ZMQ PUB/SUB 通信（支持多机）
- **小消息**：直接写入 ShmRingBuffer chunk
- **大消息**：通过 ZMQ socket 发送（避免占满环形缓冲区）

```python
# Writer 端
mq = MessageQueue(n_reader=4, n_local_reader=4, max_chunk_bytes=24*1024*1024)

# Reader 端 (从 handle 创建)
reader_mq = MessageQueue.create_from_handle(handle, rank=my_rank)
```

### 3.6 MultiprocExecutor — TP 通信

> 源码：`vllm/v1/executor/multiproc_executor.py`

`MultiprocExecutor` 使用 `MessageQueue` 实现 TP 广播：

```python
# Executor 初始化时
self.rpc_broadcast_mq = MessageQueue(world_size, local_world_size, ...)

# collective_rpc: 广播 SchedulerOutput 到所有 Worker
self.rpc_broadcast_mq.enqueue(data)  # Writer (rank 0)
data = worker.input_mq.dequeue()      # Reader (rank 0..N)
```

- 每个 Worker 有自己的 `worker_response_mq`（MessageQueue）回传结果
- 使用 `FutureWrapper` 实现异步非阻塞调用

### 3.7 vLLM 没有单独的"状态信号"机制

**关键区别**：vLLM 不需要 FastDeploy 那样的 IPCSignal，原因：

1. **调度器和执行器同进程**：`EngineCore` 包含 scheduler + model_executor，直接函数调用
2. **Worker 健康检查**：通过 `multiprocessing.connection.wait(sentinels)` 监控进程存活
3. **模型加载状态**：通过 Worker 进程启动完成（`ready_pipe`）通知
4. **KV Cache 状态**：scheduler 直接管理，不需要跨进程同步

```python
# vLLM Worker 就绪通知 — 通过 pipe
class WorkerProc:
    @staticmethod
    def make_worker_process(...):
        ready_pipe, ready_writer = context.Pipe(duplex=False)
        # Worker 进程启动后写入 ready_writer
        # Executor 从 ready_pipe 读取确认就绪
```

### 3.8 vLLM 方案特点

| 特点 | 说明 |
|------|------|
| ✅ **统一的消息队列** | MessageQueue 封装 SHM + ZMQ，接口简洁 |
| ✅ **自适应等待** | SpinCondition 忙等/ZMQ poll 自动切换 |
| ✅ **内存屏障保证** | 显式 `memory_fence()` 避免可见性问题 |
| ✅ **无状态信号膨胀** | 调度器执行器同进程，不需要状态 IPC |
| ✅ **环形缓冲区** | 支持多 chunk pipeline，避免 head-of-line blocking |
| ❌ **内存开销大** | 默认 240MiB 共享内存（24MiB × 10 chunks） |
| ❌ **仅 1-writer-N-reader** | ShmRingBuffer 不支持多 writer |

---

## 四、SGLang 的 ZMQ + NCCL 方案

### 4.1 架构概览

SGLang 采用 **"Scheduler 即 Worker"** 的融合架构，大幅减少了 IPC 需求：

```
┌─────────────────────────────────────────────────────────────┐
│                    SGLang 通信层                              │
├──────────────────┬────────────────────┬─────────────────────┤
│  ZMQ Sockets     │  NCCL broadcast    │  torch.distributed  │
│  (PUSH/PULL/     │  (broadcast_pyobj) │  (barrier/allgather) │
│   DEALER)        │                    │                      │
├──────────────────┼────────────────────┼─────────────────────┤
│ Tokenizer↔Sched  │  TP 调度信息广播    │  模型并行通信         │
│ 请求/响应传递     │  rank0 → all ranks │  AllReduce等         │
└──────────────────┴────────────────────┴─────────────────────┘
```

### 4.2 Scheduler = TP rank 0 Worker

SGLang 的核心设计是 **Scheduler 直接运行在 TP rank 0 的 Worker 进程中**：

```python
class Scheduler:
    def __init__(self, ...):
        # Scheduler 拥有 tp_worker（在同一进程）
        self.tp_worker = TpModelWorker(...)

        # ZMQ 通道：与 Tokenizer/Detokenizer 通信
        self.recv_from_tokenizer = get_zmq_socket(context, zmq.PULL, ...)
        self.send_to_detokenizer = get_zmq_socket(context, zmq.PUSH, ...)
```

这意味着：
- **Scheduler 与 rank 0 Worker 之间不需要 IPC**（同进程直接调用）
- **调度结果广播**：通过 NCCL `broadcast_pyobj` 从 rank 0 发送给其他 TP rank

### 4.3 broadcast_pyobj — NCCL 广播调度数据

SGLang 使用 `broadcast_pyobj` 将调度信息广播给所有 TP Worker：

```python
def broadcast_pyobj(data_list, tp_rank, process_group, src=0):
    """使用 torch.distributed 广播 Python 对象"""
    if tp_rank == src:
        # rank 0: 序列化 → broadcast
        torch.distributed.broadcast_object_list(data_list, src=src, group=process_group)
    else:
        # 其他 rank: 接收 broadcast
        torch.distributed.broadcast_object_list(data_list, src=src, group=process_group)
    return data_list
```

### 4.4 ZMQ 用于 Tokenizer ↔ Scheduler 通信

SGLang 的请求流转通过 ZMQ：

```
Tokenizer → (ZMQ PUSH/PULL) → Scheduler(rank0) → (NCCL broadcast) → Worker(rank1..N)
    ↑                              │
    └── (ZMQ PUSH/PULL) ──────────┘ (results)
```

Scheduler `init_ipc_channels` 初始化 ZMQ 通道：

```python
def init_ipc_channels(self, port_args):
    context = zmq.Context(2)
    if self.tp_rank == 0:  # 只有 rank 0 与 Tokenizer 通信
        self.recv_from_tokenizer = get_zmq_socket(context, zmq.PULL, ...)
        self.send_to_detokenizer = get_zmq_socket(context, zmq.PUSH, ...)
    else:
        self.recv_from_tokenizer = None  # 其他 rank 不需要
```

### 4.5 SGLang 的状态同步

SGLang **几乎不需要显式的 IPC 状态信号**，因为：

1. **调度器和 Worker 融合**：Scheduler 直接管理 KV Cache、模型状态，不需要跨进程通知
2. **TP 同步通过 NCCL**：`broadcast_pyobj` + `torch.distributed.barrier()`
3. **健康检查**：通过进程退出信号和 watchdog 线程
4. **权重更新**：通过 `torch.distributed` group 同步

```python
# SGLang TP 同步 — 使用 torch.distributed.barrier
def init_model_worker(self):
    ...
    barrier()  # 等待所有 TP rank 就绪
```

### 4.6 SGLang 方案特点

| 特点 | 说明 |
|------|------|
| ✅ **架构最简** | Scheduler + Worker 融合，IPC 需求最少 |
| ✅ **NCCL 广播高效** | GPU-aware 通信，延迟低、带宽高 |
| ✅ **无状态信号** | 不需要 SharedMemory 状态信号 |
| ✅ **ZMQ 轻量** | 仅 Tokenizer↔Scheduler 使用 ZMQ |
| ❌ **Scheduler 与 GPU 绑定** | Scheduler 占用 rank 0 的 CPU 资源 |
| ❌ **NCCL 依赖 GPU** | 广播需要 GPU 参与，CPU-only 场景不适用 |
| ❌ **单机限制** | 融合架构不天然支持 Engine 与 Worker 分机部署 |

---

## 五、三者核心对比

### 5.1 架构模型

| 维度 | FastDeploy | vLLM v1 | SGLang |
|------|-----------|---------|--------|
| **进程模型** | Engine / Worker / CacheManager / CacheTransfer 各自独立进程 | EngineCore（含 Scheduler + Executor）独立进程；Worker 子进程 | Scheduler 即 rank 0 Worker，其他 rank 独立进程 |
| **Engine↔Worker** | BaseManager (socket 代理) + IPCSignal (SharedMem) | ZMQ + ShmRingBuffer (MessageQueue) | 同进程直接调用 (rank 0) + NCCL broadcast (其他 rank) |
| **TP 广播** | BaseManager ListProxy + client_read_flag 数组 | ShmRingBuffer (1-writer-N-reader 环形缓冲区) | NCCL broadcast_pyobj |
| **状态同步** | 20+ 个 IPCSignal 实例 | 无（同进程/pipe/process sentinel） | 无（同进程/torch.distributed.barrier） |

### 5.2 通信机制

| 维度 | FastDeploy | vLLM v1 | SGLang |
|------|-----------|---------|--------|
| **主要 IPC 机制** | POSIX SharedMemory + BaseManager | SharedMemory RingBuffer + ZMQ | ZMQ + NCCL |
| **序列化** | pickle（BaseManager 默认） | msgspec (Msgpack) / pickle | pickle / ZMQ msgpack |
| **内存屏障** | 无显式屏障（依赖 x86 TSO） | 显式 `memory_fence()`（threading.Lock acquire/release） | N/A（NCCL 自带同步语义） |
| **等待策略** | 轮询 (`time.sleep(0.001)`) | SpinCondition 自适应（busy loop → ZMQ poll） | ZMQ poll + NCCL 阻塞 |
| **多机支持** | BaseManager (TCP) + IPCSignal (仅单机) | MessageQueue (本地 SHM + 远程 ZMQ PUB/SUB) | NCCL (天然多机) + ZMQ |

### 5.3 信号/状态管理

| 维度 | FastDeploy | vLLM v1 | SGLang |
|------|-----------|---------|--------|
| **任务存在检测** | `IPCSignal(exist_task_signal)` 轮询 | 不需要（ZMQ 消息驱动） | 不需要（Scheduler 自行调度） |
| **Worker 就绪** | `IPCSignal(worker_ready_signal)` | `multiprocessing.Pipe (ready_pipe)` | `torch.distributed.barrier()` |
| **模型加载完成** | `IPCSignal(loaded_model_signal)` | Pipe / ZMQ handshake | barrier + 进程启动完成 |
| **KV Cache 就绪** | `IPCSignal(cache_ready_signal)` | Scheduler 内部状态 | Scheduler 内部状态 |
| **Worker 健康** | `IPCSignal(worker_healthy_live_signal)` 心跳 | `multiprocessing.connection.wait(sentinels)` 进程存活监控 | watchdog 线程 |
| **权重状态** | `IPCSignal(model_weights_status)` + 状态常量 | Executor RPC 调用 | `torch.distributed` group 同步 |
| **Cache 传输协调** | EngineCacheQueue + 13 个 Barrier + 位掩码 | 不适用（无独立 CacheTransferManager） | 不适用（Scheduler 直接管理） |

### 5.4 性能特性

| 维度 | FastDeploy | vLLM v1 | SGLang |
|------|-----------|---------|--------|
| **任务下发延迟** | ~100μs（BaseManager socket） | ~1-10μs（ShmRingBuffer） | ~0（同进程直接调用） |
| **状态检测延迟** | ~10ns（SharedMem 读取） | 不适用 | 不适用 |
| **TP 广播延迟** | ~100μs（BaseManager） | ~1-10μs（ShmRingBuffer） | ~10-50μs（NCCL broadcast） |
| **CPU 开销** | 中（轮询 + BaseManager 线程） | 低-中（SpinCondition 自适应） | 低（NCCL offload 到 GPU） |
| **共享内存开销** | 低（每个信号 4-16 bytes） | 高（240MiB 默认） | 无 |

---

## 六、设计哲学总结

### FastDeploy：显式多进程 + 细粒度信号

- **哲学**：每个功能模块（Engine / Worker / CacheManager / CacheTransfer）独立进程，通过命名共享内存信号 + 管理器代理精确控制状态机
- **优势**：功能解耦彻底，每个组件可以独立重启/升级；支持 PD 分离、EPLB 等复杂场景
- **代价**：通信组件复杂，20+ IPCSignal 实例 + 多个 Barrier，维护成本高

### vLLM v1：消息驱动 + 高性能共享内存

- **哲学**：EngineCore 包含 Scheduler + Executor 在同一进程，消除大部分 IPC 需求；TP Worker 通过高性能 ShmRingBuffer 广播
- **优势**：接口简洁，MessageQueue 统一封装；SpinCondition 自适应等待；msgspec 高性能序列化
- **代价**：ShmRingBuffer 内存开销大；1-writer-N-reader 限制

### SGLang：Scheduler-Worker 融合 + NCCL 原语

- **哲学**：Scheduler 就是 rank 0 Worker，直接函数调用；TP 同步复用 NCCL 通信基础设施
- **优势**：架构最简，几乎不需要自定义 IPC；NCCL 广播高效
- **代价**：Scheduler 与 GPU 绑定，灵活性受限；不天然支持 Engine/Worker 分机

---

## 七、对 FastDeploy 的改进建议

基于以上对比分析，FastDeploy 在进程间通信方面可以考虑以下优化方向：

### 7.1 减少 IPCSignal 数量

当前 20+ 个 IPCSignal 可以考虑**合并**为少量结构化信号：

```python
# 当前：20+ 个独立 IPCSignal
exist_task_signal = IPCSignal("exist_task_signal", ...)
exist_swapped_task_signal = IPCSignal("exist_swapped_task_signal", ...)
model_weights_status = IPCSignal("model_weights_status", ...)
...

# 建议：合并为 1-2 个结构化共享内存块
class EngineState(ctypes.Structure):
    _fields_ = [
        ("exist_task", ctypes.c_int32),
        ("exist_swapped_task", ctypes.c_int32),
        ("exist_prefill_task", ctypes.c_int32),
        ("model_weights_status", ctypes.c_int32),
        ("kv_cache_status", ctypes.c_int32),
        ("worker_healthy", ctypes.c_int32),
        ...
    ]
```

### 7.2 添加内存屏障

参考 vLLM 的 `memory_fence()` 实现，为 IPCSignal 添加显式内存屏障：

```python
_fence_lock = threading.Lock()

def memory_fence():
    with _fence_lock:
        pass

# 在 IPCSignal 写入后调用
signal.value[0] = 1
memory_fence()  # 确保写入对其他进程可见
```

### 7.3 考虑引入 SpinCondition 模式

将当前的 `time.sleep(0.001)` 轮询改为自适应等待：

- 高负载时：busy loop（更低延迟）
- 空闲时：sleep 或 ZMQ poll（节省 CPU）

### 7.4 评估 ShmRingBuffer 替代 BaseManager

对于 TP 广播场景（EngineWorkerQueue.put_tasks → 所有 Worker 读取），可以考虑使用类似 vLLM 的 ShmRingBuffer 替代 BaseManager ListProxy，减少序列化和 socket 开销。

### 7.5 长期：评估 Scheduler-Worker 融合

对于单机场景，可以参考 SGLang 的设计，将 Scheduler 与 rank 0 Worker 融合，消除最大的 IPC 瓶颈。但需要在灵活性（PD 分离、多机部署）和性能之间权衡。
