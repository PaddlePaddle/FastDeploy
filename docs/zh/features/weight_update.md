[English](../../features/weight_update.md)

# 权重清除与更新

FastDeploy 支持面向 RL / RLHF Rollout 服务的动态权重清除、显存卸载和权重更新，主要用于解决以下两类问题：

- Rollout 引擎空闲时释放 GPU 显存；
- Trainer 产出新 checkpoint 后，推理服务在不重启进程的情况下切换到新权重。

本文档介绍 FastDeploy 当前支持的权重控制接口、各接口的语义，以及它们在 RLHF 训练中的典型用法。

## 前置条件

在 RLHF 场景下，FastDeploy 主要通过在线服务模式提供该能力。启动服务时，需要开启动态权重加载：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/model \
    --dynamic-load-weight \
    --load-strategy ipc_snapshot
```

`--dynamic-load-weight` 用于开启动态权重控制能力，`--load-strategy` 用于指定具体的权重更新方式。当前支持的更新模式如下：

| 模式 | `load_strategy` | 典型场景 | 说明 |
| --- | --- | --- | --- |
| CUDA IPC | `ipc` | 训练进程与推理进程同机，直接共享实时张量 | 更新来源是训练侧产出的 IPC 元信息。 |
| IPC 快照 | `ipc_snapshot` | Rollout 从训练侧产出的权重快照文件重载 | 当前仓库里的 RL rollout 示例主要使用该模式。 |
| RDMA / rsync | `rsync` | Trainer 发布新版本，Rollout 远端拉取 | `POST /v1/update_weights` 是这一模式的显式接口。 |

## 接口说明

### 旧版接口

在 FastDeploy <= 2.5 版本中，主要提供以下简化接口，保留给旧版 RL 控制流使用。

| 接口 | 方法 | 含义 | 可用条件 |
| --- | --- | --- | --- |
| `/clear_load_weight` | `GET` | 清除或卸载当前已加载权重 | 需要 `dynamic_load_weight=True` |
| `/update_model_weight` | `GET` | 在清除/卸载后重新加载权重 | 需要 `dynamic_load_weight=True` |

### V1 新版接口

在 FastDeploy >= 2.6 版本中，底层控制信号通信链路经过优化，并引入了 V1 控制接口。相较于旧版接口，V1 接口在通信与执行链路上更稳定，语义更清晰，同时提供了更灵活的控制方式，包括以下接口：

| 接口 | 方法 | 请求参数 | 语义 |
| --- | --- | --- | --- |
| `/v1/pause` | `POST` | 无 | 暂停请求生成，中断 running/inflight 请求，重置调度器，并在开启 cache transfer 时暂停 cache transfer。 |
| `/v1/resume` | `POST` | 无 | 恢复请求生成和 cache transfer。 |
| `/v1/is_paused` | `GET` | 无 | 返回 `{"is_paused": bool}`。 |
| `/v1/sleep` | `POST` | `?tags=weight,kv_cache` | 卸载指定 GPU 内存对象。支持 `weight` 与 `kv_cache`；不传时默认同时处理两者。 |
| `/v1/wakeup` | `POST` | `?tags=weight,kv_cache` | 重新加载之前被卸载的权重和/或 KV Cache。成功后会自动 `resume`。 |
| `/v1/update_weights` | `POST` | JSON `{"version":"...", "verify_checksum": false}` | 通过 worker 控制链路原地刷新模型权重。该接口主要面向 `load_strategy=rsync` 的远端版本更新。 |

### 兼容性说明

底层通信链路的优化同样适用于旧版接口。通过设置环境变量 `FD_ENABLE_V1_UPDATE_WEIGHTS=1`，可以将旧版接口切换到新的控制链路，在保留兼容接口形式的同时，获得更明确的执行路径和更好的可观测性。
- `FD_ENABLE_V1_UPDATE_WEIGHTS=0`：走旧版基于共享内存的控制链路。
- `FD_ENABLE_V1_UPDATE_WEIGHTS=1`：`/clear_load_weight` 底层等价于执行 `/v1/sleep`，`/update_model_weight` 底层等价于执行 `/v1/wakeup`。对应的 pause/resume 动作分别由 `sleep` 和 `wakeup` 内部处理。

**注意**：无论是否设置 V1 环境变量，旧版接口都不是 RLHF 场景下推荐的标准使用方式，后续版本中也可能逐步废弃。建议优先使用 `/v1/*` 控制接口。

## 各接口语义

### `/v1/pause`

`/v1/pause` 是变更模型状态前的安全边界。

它会执行以下动作：

- 停止新请求生成；
- 中断当前 running 和 inflight 请求；
- 重置调度器状态；
- 在启用多级缓存或 KV Cache 存储时暂停 cache 传输。

如果需要在每一轮 rollout 与下一轮训练之间建立清晰的切换边界，建议先调用该接口。

### `/v1/sleep`

`/v1/sleep` 用于从 GPU 显存中卸载指定的运行时状态。

当前支持的 `tags`：

- `weight`：清除设备上的模型权重；如果开启了相关配置，还可能一并释放通信组和 DeepEP buffer。
- `kv_cache`：清除 KV Cache；如果投机解码采用 MTP，还会同步清理 MTP cache。

如果不传 `tags` 参数，FastDeploy 默认等价于：

```bash
/v1/sleep?tags=weight,kv_cache
```

当前实现中，`sleep` 会自动先执行一次 `pause`。但新的接入方不应长期依赖这一隐式行为。

### `/v1/wakeup`

`/v1/wakeup` 用于恢复通过 `/v1/sleep` 卸载的状态。

根据 `tags` 和运行配置，FastDeploy 可能执行：

- 重建通信组；
- 重建 DeepEP buffer；
- 从当前配置的数据源重新加载模型权重；
- 重建 KV Cache；
- 重新捕获 CUDA Graph。

`wakeup` 成功后，FastDeploy 会自动调用一次 `resume`。

### `/v1/update_weights`

`/v1/update_weights` 用于在不卸载模型权重显存占用的情况下，直接刷新模型参数。

当前支持的请求字段：

- `version`：可选字符串，用于指定目标 checkpoint 版本。
- `verify_checksum`：可选布尔值；默认为 `false`。设置为 `true` 时，会在权重同步过程中校验数据完整性。

关键语义：

- 调用前引擎必须已经处于暂停状态，否则请求会失败；
- 实际更新动作只在 worker 侧执行；
- 该接口主要用于显式权重刷新，尤其是 `rsync` 路径；
- 它不会自动执行 `resume`。

推荐调用顺序：

1. `POST /v1/pause`
2. `POST /v1/update_weights`
3. `POST /v1/resume`

如果除更新权重外，还希望在 rollout 轮次之间回收 GPU 显存，则更适合使用 `sleep` / `wakeup` 组合。

## 请求示例

### 基础接口

暂停引擎：

```bash
curl -X POST http://127.0.0.1:8000/v1/pause
```

恢复引擎：

```bash
curl -X POST http://127.0.0.1:8000/v1/resume
```

### Sleep / Wakeup 接口

**卸载权重和 KV Cache**

```bash
# 同时卸载权重和 KV Cache
curl -X POST "http://127.0.0.1:8000/v1/sleep?tags=weight,kv_cache"

# 只卸载权重
curl -X POST "http://127.0.0.1:8000/v1/sleep?tags=weight"

# 不传参数，默认同时卸载两者
curl -X POST "http://127.0.0.1:8000/v1/sleep"
```

**恢复权重和 KV Cache**

```bash
# 恢复权重和 KV Cache
curl -X POST "http://127.0.0.1:8000/v1/wakeup?tags=weight,kv_cache"

# 只恢复权重
curl -X POST "http://127.0.0.1:8000/v1/wakeup?tags=weight"

# 不传参数，默认同时恢复两者
curl -X POST "http://127.0.0.1:8000/v1/wakeup"
```

**注意**：当 `use_cudagraph=True` 时，必须先恢复 KV Cache 再恢复权重。这意味着调用 `/v1/wakeup` 时如果只包含 `weight` tag 而不包含 `kv_cache`，会报错。建议保持 sleep 和 wakeup 的 tags 参数一致。

### Update Weights 接口

切换到远端发布的新版本权重：

```bash
curl -X POST http://127.0.0.1:8000/v1/update_weights \
  -H "Content-Type: application/json" \
  -d '{
    "version": "global_step_1200",
    "verify_checksum": false
  }'
```

## 在 RLHF 中如何使用

### 推荐的 Rollout 服务配置

在 RLHF 场景下，FastDeploy 的 Rollout 服务通常采用以下配置：

- `dynamic_load_weight=True`
- `load_strategy=ipc_snapshot`，适合本地快照式刷新；
- 或 `load_strategy=rsync`，适合远端版本化刷新。

仓库中的 RL rollout 工具已经按该方式接入。典型写法如下：

```python
from fastdeploy.rl.rollout_config import RolloutModelConfig
from fastdeploy.rl.rollout_model import RolloutModel

rollout_config = RolloutModelConfig(
    model_name_or_path=model_path,
    tensor_parallel_size=ranks,
    dynamic_load_weight=True,
    load_strategy="ipc_snapshot",
)
rollout_model = RolloutModel(rollout_config)
```

### FastDeploy 对训练侧的支持

除服务接口外，FastDeploy 还提供以下两类 RLHF 训练侧对接能力：

- `RolloutModel.state_dict()`：暴露 rollout 侧的推理参数；
- `RolloutModel.get_name_mappings_to_training()`：暴露推理参数名到训练参数名的映射关系。

这两个接口可用于训练侧 checkpoint 与 rollout 侧权重布局对齐，尤其适用于推理态和训练态参数命名不完全一致的场景。

### RLHF 常见工作流

以下给出 RLHF 场景下几种常见调用方式。示例中假设服务地址为 `http://127.0.0.1:8000`。

**工作流 1：显存卸载与恢复**

适用于 rollout 服务常驻，但需要在训练阶段前后释放并恢复显存的场景。推荐流程为 `(pause) -> sleep -> wakeup -> (resume)`，其中括号内步骤为可选。

```bash
# 可选：先显式暂停引擎，建立清晰的切换边界
curl -X POST http://127.0.0.1:8000/v1/pause

# 卸载权重和 KV Cache
curl -X POST "http://127.0.0.1:8000/v1/sleep?tags=weight,kv_cache"

# 训练完成后恢复权重和 KV Cache
curl -X POST "http://127.0.0.1:8000/v1/wakeup?tags=weight,kv_cache"

# 可选：如业务侧需要显式恢复，可手动调用
curl -X POST http://127.0.0.1:8000/v1/resume
```

**工作流 2：原地刷新到新 checkpoint**

适用于服务常驻、仅需要切换到新版本权重的场景。推荐流程为 `pause -> update_weights -> resume`。

```bash
# 先暂停引擎
curl -X POST http://127.0.0.1:8000/v1/pause

# 原地刷新到新版本权重
curl -X POST http://127.0.0.1:8000/v1/update_weights \
  -H "Content-Type: application/json" \
  -d '{
    "version": "global_step_1200",
    "verify_checksum": false
  }'

# 更新完成后恢复服务
curl -X POST http://127.0.0.1:8000/v1/resume
```

**工作流 3：兼容旧版接口**

旧版 RL 客户端仍可继续使用兼容接口，流程为 `clear_load_weight -> update_model_weight`。

```bash
# 清除或卸载当前权重
curl -X GET http://127.0.0.1:8000/clear_load_weight

# 训练侧完成 checkpoint 更新后，重新加载权重
curl -X GET http://127.0.0.1:8000/update_model_weight
```

对于新的接入方，建议优先使用 `/v1/*` 接口，因为其控制链路更显式，日志排查和故障定位也更直接。

## 其他相关配置

### 通信组的销毁与重建

FastDeploy 支持通过 `--shutdown-comm-group-if-worker-idle` 和 `--no-shutdown-comm-group-if-worker-idle`，显式控制在卸载权重时是否同时销毁通信组。

保留通信组通常有助于提升权重清除和重新加载过程中的稳定性；相应地，代价是卸载权重后仍会保留更多显存占用，同时 `sleep` / `wakeup` 的执行时间也可能更长。

默认情况下：

- 在 EP 场景下，默认不销毁通信组；
- 在非 EP 场景下，默认销毁通信组。

### CPU 缓存的清除与重建

启用 `--swap-space` 后，可以通过以下环境变量控制在执行 `/v1/sleep` 时，是否同步清理 CPU 侧缓存，以降低训练阶段的内存压力。

默认情况下，FastDeploy 不会主动清理 CPU Cache。如需在 `sleep` 时一并清理，可设置：

```bash
export FD_ENABLE_SWAP_SPACE_CLEARING=1
```
