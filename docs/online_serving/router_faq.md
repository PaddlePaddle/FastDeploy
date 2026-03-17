[简体中文](../zh/online_serving/router_faq.md)

# Router Troubleshooting Guide

This document is based on the [Golang Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router) implementation and summarizes common log messages, response outputs, and troubleshooting methods encountered during Router usage, helping users quickly locate and resolve issues.

For basic Router usage, please refer to [Load-Balancing Scheduling Router](router.md).

## Common Log Analysis

> **Note**: `{}` represents variables that will be replaced with actual values in logs.

### Error-Level Logs

| Log Message | Meaning | Impact | What to Check |
| :--- | :--- | :--- | :--- |
| `Removed unhealthy prefill instance: {url}` | Prefill instance failed health check and has been removed | This Prefill instance will no longer participate in scheduling | Health status |
| `Removed unhealthy decode instance: {url}` | Decode instance failed health check and has been removed | This Decode instance will no longer participate in scheduling | Health status |
| `Removed unhealthy mixed instance: {url}` | Mixed instance failed health check and has been removed | This Mixed instance will no longer participate in scheduling | Health status |
| `Failed to register instance: {error}` | Instance registration failed | Router cannot register the instance | Health status, registration parameters |
| `Failed to read YAML file {path}: {error}` | Failed to read registration config file at startup | Instances in the config file cannot be registered | File path, file permissions |
| `Failed to unmarshal YAML file {path}: {error}` | Registration config file has invalid format | Instances in the config file cannot be registered | YAML format |
| `Failed to register instance from index {index}: {error}` | Instance at index {index} in config file failed to register | That instance was not registered | Health status, registration parameters |
| `failed to send request to {url} with error: {error}` | Health check request failed to send | The instance may be marked as unhealthy | Network connectivity, proxy settings |
| `scanner error: {error}` | Error occurred while reading backend streaming response | The current request may fail | Backend instance status |

### Warn-Level Logs

| Log Message | Meaning | Impact | What to Check |
| :--- | :--- | :--- | :--- |
| `Server {url} is not healthy` | The instance at this URL failed health check | Router cannot register the instance, or will remove it from the registered list | Health status |
| `Instance {url} role is unknown` | Instance role cannot be recognized | The instance will not be added to the scheduling list | Registration parameters |
| `cache-aware prefill: tokenizer failed, fallback to process_tokens: {error}` | Tokenizer service call failed, automatically falling back to process_tokens strategy | Prefill scheduling temporarily does not use cache_aware strategy; normal request processing is not affected | Tokenizer service status |

### Info-Level Logs

| Log Message | Meaning | Description |
| :--- | :--- | :--- |
| `Starting server on {host:port}` | Router service is starting | Normal startup log |
| `Server {url} is healthy` | Instance passed health check | Normal operation log |
| `Successfully registered instance from index {index}` | Instance from config file registered successfully | Normal startup log |
| `No instances found in config file {path}` | No instances found in the registration config file | Check whether register.yaml is empty |
| `Request completed successfully.` | Request processing completed | Normal operation log |
| `Request failed, retrying...` | Request failed, retrying | Router will retry up to 3 times |

## Common Response Output Analysis

### Inference Request Errors (/v1/chat/completions, /v1/completions)

| Output | HTTP Status | Meaning | What to Check |
| :--- | :---: | :--- | :--- |
| `{"error": "No available prefill/decode workers"}` | 503 | All Prefill or Decode instances are unavailable, no registered healthy instances | Health status |
| `{"error": "Failed to select worker pair"}` | 502 | Failed to select a worker pair in PD disaggregated mode | Health status, scheduling strategy |
| `{"error": "Failed to select worker"}` | 502 | Failed to select a worker in centralized mode | Health status, scheduling strategy |
| `{"error": "Failed to connect to backend service"}` | 502 | Failed to connect to backend inference instance (after 3 retries) | Backend instance status, network connectivity |
| `{"error": "Failed to build disaggregate_info"}` | 500 | Failed to build PD disaggregation communication info | Registration parameters (connector_port, device_ids, etc.) |
| `{"error": "Invalid request body"}` | 400 | Failed to read request body | Request format |
| `{"error": "Invalid JSON format"}` | 400 | Failed to parse request body JSON | Request format |

### Registration Request Errors (/register)

| Output | HTTP Status | Meaning | What to Check |
| :--- | :---: | :--- | :--- |
| `{"code": 503, "msg": "{url} service is not healthy"}` | 503 | Instance failed health check, cannot be registered | Health status |
| `{"code": 400, "msg": "Invalid request body"}` | 400 | Failed to read registration request body | Request format |
| `{"code": 400, "msg": "Invalid InstanceInfo JSON format: {error}"}` | 400 | Registration request has invalid JSON format | Request format |
| `{"code": 400, "msg": "splitwise mode only supports PREFILL/DECODE instances"}` | 400 | MIXED instances are not allowed in PD disaggregated mode | Deployment mode, instance role |
| `{"code": 400, "msg": "only MIXED instances are allowed"}` | 400 | Only MIXED instances are allowed in centralized mode | Deployment mode, instance role |
| `{"code": 400, "msg": "invalid InstanceInfo format: {error}"}` | 400 | Instance registration info validation failed | Registration parameters |
| `{"code": 200, "msg": "Register success"}` | 200 | Registration successful | — |

### Common Registration Parameter Validation Errors

| Error Message | Meaning | Solution |
| :--- | :--- | :--- |
| `role is required` | Missing role field | Add the role field with value: prefill / decode / mixed |
| `invalid role: {role}` | Invalid role value | Use a valid role value: prefill / decode / mixed |
| `host_ip is required` | Missing host_ip field | Add the host_ip field |
| `invalid host_ip: {ip}` | host_ip is not a valid IP address | Provide a valid IP address |
| `port is required` | Missing port field | Add the port field |
| `invalid port: {port}` | port is not a valid port number | Provide a port number in the range 1-65535 |
| `invalid protocol: {protocol}` | Invalid transfer protocol | Use a valid protocol value: ipc / rdma |

## Troubleshooting Guide

### Health Status

Instance health checking is fundamental to Router operation. When instances fail to register or are removed, follow these steps:

**1. Check instance registration status**

View the currently registered instances and their count:
```bash
# View registered instance list
curl -X GET http://{router_url}/registered

# View registered instance count
curl -X GET http://{router_url}/registered_number
```

Verify that all expected instances are registered. If the count does not match, some instances may have failed to register or been removed by health checks.

**2. Check instance health and network connectivity**

Directly access the inference instance's health endpoint from the Router's host:
```bash
curl -X GET http://{server_url}/health
```

- HTTP 200 response indicates the instance is healthy and the network is reachable
- If unreachable or returning a non-200 status code, investigate further:
  - Whether the instance is started and listening on the correct port
  - Whether a proxy is interfering with the connection (try disabling: `unset http_proxy && unset https_proxy`)
  - Whether firewall rules are blocking the connection

**Common solutions:**
- Disable network proxy: `unset http_proxy && unset https_proxy`
- gunicorn version compatibility: If registered instance count is incomplete, it may be due to gunicorn and FastDeploy version incompatibility. Downgrading to `gunicorn==25.0.3` can resolve this issue

### Scheduling Strategy

When encountering `Failed to select worker` or `Failed to select worker pair` errors:

**1. Verify registered instance count**
```bash
curl -X GET http://{router_url}/registered_number
```

If the returned count is 0, there are no available healthy instances. Please refer to [Health Status](#health-status) for troubleshooting.

**2. Check scheduling strategy configuration**

Verify that the scheduling strategy in config.yaml matches your deployment mode. The default scheduling strategies are:

| Deployment Mode | Config Field | Default Strategy |
| :--- | :--- | :--- |
| Centralized mode | `policy` | `request_num` |
| PD disaggregated mode (Prefill) | `prefill-policy` | `process_tokens` |
| PD disaggregated mode (Decode) | `decode-policy` | `request_num` |

If no strategy is specified in the config file, the Router will use the defaults listed above. To use advanced strategies such as `cache_aware` or `fd_metrics_score`, specify them explicitly in the config file. For detailed descriptions of each strategy, see [Scheduling Strategies](router.md#scheduling-strategies).

**3. Check fd_metrics_score strategy dependencies**

When using the `fd_metrics_score` strategy, the Router fetches running/waiting request counts in real time from the `/metrics` endpoint of inference instances. When the `/metrics` endpoint is unavailable (e.g., `metrics_port` is not configured or the metrics service is down), the Router automatically falls back to the internal request counter for scheduling. This does not affect normal request processing, but scheduling accuracy may be reduced.

To ensure optimal scheduling with `fd_metrics_score`, verify that the inference instance's metrics endpoint is responding correctly:
```bash
curl -X GET http://{server_url}/metrics
```

### Registration Parameters

When registration fails with parameter validation errors:

**1. Verify deployment mode and instance role match**
- PD disaggregated mode (`--splitwise`): Only `prefill` and `decode` roles can be registered
- Centralized mode (default): Only `mixed` role can be registered

**2. Check required parameters**

Registration requests must include the following fields:
- `role`: Instance role (prefill / decode / mixed)
- `host_ip`: Instance IP address
- `port`: Instance port number

**3. Check optional parameters for PD disaggregated mode**

In PD disaggregated mode, the following parameters should be fully configured to ensure proper KV Cache transfer:
- `connector_port`: PD communication port
- `transfer_protocol`: Transfer protocol (ipc / rdma)
- `device_ids`: GPU device IDs
- `rdma_ports`: RDMA ports (required when using the rdma protocol)

### Startup Failures

**1. Configuration file loading failure**

If `Failed to load config` appears in startup logs, check:
- Whether the file path specified by `--config_path` is correct
- Whether the configuration file is valid YAML
- Whether configuration parameter values are valid

**2. Port already in use**

If `Failed to start server` appears in startup logs, check:
- Whether the port specified by `--port` is already occupied by another process
- Use `lsof -i:{port}` or `netstat -tlnp | grep {port}` to check port usage

### Tokenizer Service (cache_aware Strategy)

When using the `cache_aware` scheduling strategy, the Router calls a Tokenizer service to tokenize requests for cache hit ratio computation. When the Tokenizer service is unavailable, the Router will log a Warn-level message: `tokenizer failed, fallback to process_tokens`.

**This does not affect normal request processing** — the Router has a built-in degradation mechanism that automatically falls back to the `process_tokens` strategy for continued scheduling. The only impact is the temporary loss of cache-aware optimization.

To restore full cache_aware functionality:

**1. Check if the Tokenizer service is running**
```bash
curl -X POST http://{tokenizer_url}/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

**2. Check related configuration**
- Verify that `tokenizer-url` in config.yaml is set correctly
- If the Tokenizer service responds slowly, consider increasing `tokenizer-timeout-secs` (default: 2 seconds)
