[简体中文](../zh/features/weight_update.md)

# Weight Clear and Update

FastDeploy supports dynamic weight clear and update for RL and RLHF rollout services. This capability is primarily intended to address the following two requirements:

- release GPU memory when the rollout engine is idle;
- refresh inference weights after the trainer produces a new checkpoint, without restarting the whole service.

This page describes the weight-control interfaces currently supported by FastDeploy, the semantics of each interface, and their typical usage in RLHF training.

## Prerequisites

In RLHF scenarios, FastDeploy mainly provides this capability through the online serving mode. Dynamic weight loading must be enabled when starting the service:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/model \
    --dynamic-load-weight \
    --load_strategy ipc_snapshot
```

`--dynamic-load-weight` enables dynamic weight control, and `--load_strategy` specifies the concrete weight update mechanism. The currently supported update modes are listed below:

| Mode | `load_strategy` | Typical use | Notes |
| --- | --- | --- | --- |
| CUDA IPC | `ipc` | Training and inference processes on the same node share live tensors | Update source comes from IPC metadata produced by the training side. |
| IPC snapshot | `ipc_snapshot` | Rollout reloads a snapshot file produced by training | Used by current RL rollout examples. |
| RDMA / rsync | `rsync` | Trainer publishes a new version and rollout fetches it remotely | `POST /v1/update_weights` is the explicit API for this mode. |

## API Overview

### Compatibility APIs

In FastDeploy <= 2.5, the following simplified APIs are provided for compatibility with the legacy RL control flow.

| API | Method | Meaning | Availability |
| --- | --- | --- | --- |
| `/clear_load_weight` | `GET` | Clear or offload currently loaded weights | Requires `dynamic_load_weight=True` |
| `/update_model_weight` | `GET` | Reload weights after a clear/offload operation | Requires `dynamic_load_weight=True` |

### V1 control APIs

In FastDeploy >= 2.6, the underlying control-signal communication path is optimized and V1 control APIs are introduced. Compared with the legacy APIs, the V1 APIs provide a more stable execution path, clearer semantics, and more flexible control:

| API | Method | Request params | Semantics |
| --- | --- | --- | --- |
| `/v1/pause` | `POST` | none | Pause request generation, abort running and inflight requests, reset scheduler state, and pause cache transfer if enabled. |
| `/v1/resume` | `POST` | none | Resume request generation and cache transfer. |
| `/v1/is_paused` | `GET` | none | Return `{"is_paused": bool}`. |
| `/v1/sleep` | `POST` | `?tags=weight,kv_cache` | Offload selected GPU memory objects. Supported tags are `weight` and `kv_cache`. If omitted, both are used. |
| `/v1/wakeup` | `POST` | `?tags=weight,kv_cache` | Reload previously offloaded weights and/or KV cache. On success, the engine resumes automatically. |
| `/v1/update_weights` | `POST` | JSON `{"version":"...", "rsync_config": {...}}` | Refresh weights in place through the worker control path. This API is intended for remote versioned updates, especially `load_strategy=rsync`. |

### Compatibility Notes

The optimized communication path also applies to the legacy APIs. By setting `FD_ENABLE_V1_UPDATE_WEIGHTS=1`, the legacy APIs can be switched to the new control path while keeping the original API form.

- `FD_ENABLE_V1_UPDATE_WEIGHTS=0`: use the legacy shared-memory-based control path.
- `FD_ENABLE_V1_UPDATE_WEIGHTS=1`: `/clear_load_weight` is effectively handled through `/v1/sleep`, and `/update_model_weight` is effectively handled through `/v1/wakeup`. The corresponding pause/resume actions are handled internally by `sleep` and `wakeup`.

**Note**: regardless of whether V1 is enabled, the legacy APIs are not the recommended standard interface for RLHF scenarios and may be gradually deprecated in future releases. The `/v1/*` control APIs are recommended.

## Interface Semantics

### `/v1/pause`

`/v1/pause` is the safe boundary before changing model state.

It does the following:

- stops new request generation;
- aborts running and inflight requests;
- resets scheduler state;
- pauses cache transfer when multi-level cache or KV cache storage is enabled.

When a clear boundary is required between one rollout round and the next training stage, this API should be called first.

### `/v1/sleep`

`/v1/sleep` offloads selected runtime state from GPU memory.

Supported tags:

- `weight`: clear model weights from device memory; if enabled, communication groups and DeepEP buffers may also be released.
- `kv_cache`: clear KV cache; MTP cache is also cleared when speculative decoding uses MTP.

If the `tags` parameter is omitted, FastDeploy defaults to:

```bash
/v1/sleep?tags=weight,kv_cache
```

In the current implementation, `sleep` automatically performs a `pause` first. New integrations should not rely on this implicit behavior.

### `/v1/wakeup`

`/v1/wakeup` restores the state offloaded by `/v1/sleep`.

Depending on tags and configuration, FastDeploy may:

- restart communication groups;
- recreate DeepEP buffers;
- reload model weights from the configured source;
- rebuild KV cache;
- recapture CUDA Graph.

After `wakeup` succeeds, FastDeploy automatically calls `resume`.

### `/v1/update_weights`

`/v1/update_weights` refreshes model parameters directly, without unloading the GPU memory occupied by model weights.

Current request fields:

- `version`: optional string. Used to choose a target checkpoint version.
- `rsync_config`: optional dictionary. Must contain `etcd_server` when provided.

Important semantics:

- the engine must already be paused, otherwise the request fails;
- the update is executed on workers only;
- this API is meant for explicit weight refresh, especially the `rsync` path;
- it does not implicitly call `resume`.

Recommended sequence:

1. `POST /v1/pause`
2. `POST /v1/update_weights`
3. `POST /v1/resume`

If GPU memory also needs to be reclaimed between rollout rounds, the `sleep` / `wakeup` workflow is more appropriate.

## Example Requests

### Basic APIs

Pause the engine:

```bash
curl -X POST http://127.0.0.1:8000/v1/pause
```

Resume the engine:

```bash
curl -X POST http://127.0.0.1:8000/v1/resume
```

### Sleep / Wakeup APIs

**Offload weights and KV cache**

```bash
# Offload both weights and KV cache
curl -X POST "http://127.0.0.1:8000/v1/sleep?tags=weight,kv_cache"

# Offload only weights
curl -X POST "http://127.0.0.1:8000/v1/sleep?tags=weight"

# Omit parameter, defaults to both
curl -X POST "http://127.0.0.1:8000/v1/sleep"
```

**Restore weights and KV cache**

```bash
# Restore both weights and KV cache
curl -X POST "http://127.0.0.1:8000/v1/wakeup?tags=weight,kv_cache"

# Restore only weights
curl -X POST "http://127.0.0.1:8000/v1/wakeup?tags=weight"

# Omit parameter, defaults to both
curl -X POST "http://127.0.0.1:8000/v1/wakeup"
```

**Note**: When `use_cudagraph=True`, KV cache must be restored before weights. This means `/v1/wakeup` with the `kv_cache` tag must be called before calling `/v1/wakeup` with the `weight` tag. If weights are restored without KV cache, an error will be raised. It is recommended to keep the `tags` parameter consistent between `/v1/sleep` and `/v1/wakeup`.

### Update Weights API

Refresh to a new remotely published version:

```bash
curl -X POST http://127.0.0.1:8000/v1/update_weights \
  -H "Content-Type: application/json" \
  -d '{
    "version": "global_step_1200",
    "rsync_config": {
      "etcd_server": "127.0.0.1:2379"
    }
  }'
```

## RLHF Usage

### Recommended Rollout Service Setup

In RLHF scenarios, FastDeploy rollout services are typically configured as follows:

- `dynamic_load_weight=True`
- `load_strategy=ipc_snapshot` for local snapshot-based refresh;
- or `load_strategy=rsync` for versioned remote refresh.

The rollout utilities in the repository already follow this pattern. A typical example is:

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

### Training-Side Integration Support

In addition to serving endpoints, FastDeploy provides the following training-side integration capabilities for RLHF:

- `RolloutModel.state_dict()`: exposes the rollout-side inference parameters.
- `RolloutModel.get_name_mappings_to_training()`: exposes the mapping from inference parameter names to training parameter names.

These interfaces can be used to align training checkpoints with rollout-side parameter layouts, especially when inference-side and training-side parameter names are not fully identical.

### Common RLHF workflows

The following examples assume the service endpoint is `http://127.0.0.1:8000`.

**Workflow 1: clear and restore**

This workflow is suitable when the rollout service stays resident, but GPU memory should be released before training and restored afterward. The recommended sequence is `(pause) -> sleep -> wakeup -> (resume)`, where the steps in parentheses are optional.

```bash
# Optional: explicitly pause the engine to establish a clear transition boundary
curl -X POST http://127.0.0.1:8000/v1/pause

# Offload both weights and KV cache
curl -X POST "http://127.0.0.1:8000/v1/sleep?tags=weight,kv_cache"

# Restore both weights and KV cache after training completes
curl -X POST "http://127.0.0.1:8000/v1/wakeup?tags=weight,kv_cache"

# Optional: explicitly resume if required by the integration
curl -X POST http://127.0.0.1:8000/v1/resume
```

**Workflow 2: in-place refresh to a new checkpoint**

This workflow is suitable when the service remains resident and only needs to switch to a new checkpoint version. The recommended sequence is `pause -> update_weights -> resume`.

```bash
# Pause the engine first
curl -X POST http://127.0.0.1:8000/v1/pause

# Refresh to a new checkpoint version in place
curl -X POST http://127.0.0.1:8000/v1/update_weights \
  -H "Content-Type: application/json" \
  -d '{
    "version": "global_step_1200",
    "rsync_config": {
      "etcd_server": "127.0.0.1:2379"
    }
  }'

# Resume the service after the update completes
curl -X POST http://127.0.0.1:8000/v1/resume
```

**Workflow 3: legacy compatibility APIs**

Legacy RL clients can continue to use the compatibility flow `clear_load_weight -> update_model_weight`.

```bash
# Clear or offload the current weights
curl -X GET http://127.0.0.1:8000/clear_load_weight

# Reload weights after the trainer updates the checkpoint
curl -X GET http://127.0.0.1:8000/update_model_weight
```

For new integrations, the `/v1/*` APIs are recommended because their control path is more explicit and easier to trace.

## Other Related Configuration

### Communication Group Clear and Rebuild

FastDeploy provides `--shutdown-comm-group-if-worker-idle` and `--no-shutdown-comm-group-if-worker-idle` to explicitly control whether communication groups should also be torn down when weights are offloaded.

Keeping communication groups alive generally improves the stability of weight clearing and reloading. The tradeoff is that more GPU memory remains allocated after weight offload, and the execution time of `sleep` / `wakeup` may also increase.

By default:

- in EP scenarios, communication groups are kept;
- in non-EP scenarios, communication groups are torn down.

### CPU Cache Clear and Rebuild

After `--swap-space` is enabled, the following environment variable can be used to control whether CPU-side cache should also be cleared when `/v1/sleep` is executed, in order to reduce memory pressure during training.

By default, FastDeploy does not actively clear CPU cache. To clear it together with `sleep`, set:

```bash
export FD_ENABLE_SWAP_SPACE_CLEARING=1
```
