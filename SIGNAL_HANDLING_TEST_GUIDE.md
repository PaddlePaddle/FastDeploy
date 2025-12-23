# 进程退出机制测试指南

本文档描述了如何测试优化后的进程退出机制。

## 问题背景

在之前的版本中，使用 `fastdeploy.entrypoints.openai.api_server` 启动服务后（特别是配置 `workers > 1` 时），通过 `ctrl+c` 退出时容易出现进程残留未退出的情况，包括：
- API server 进程（gunicorn workers）
- Worker process 进程（通过 subprocess.Popen 启动）

## 优化内容

### 1. 信号处理器优化

在以下文件中添加了 SIGINT 和 SIGTERM 信号处理器：
- `fastdeploy/entrypoints/openai/api_server.py`
- `fastdeploy/entrypoints/api_server.py`
- `fastdeploy/worker/worker_process.py`

### 2. 进程清理机制

添加了 `cleanup_processes()` 函数，负责：
- 终止 worker process 进程组 (使用 SIGTERM)
- 如果超时未终止，使用 SIGKILL 强制终止
- 清理 cache manager 进程

### 3. Gunicorn 集成

在 `StandaloneApplication` 类中：
- 添加了 `worker_exit()` 钩子
- 添加了 `on_exit()` 钩子
- 重写了 `run()` 方法以设置信号处理器

### 4. 线程协调

添加了 `shutdown_event` 用于协调多个线程的优雅退出。

## 测试步骤

### 测试 1: 单 Worker 模式

```bash
# 启动服务（单 worker）
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/model \
    --workers 1 \
    --port 8000

# 等待服务完全启动
# 检查进程
ps aux | grep fastdeploy
ps aux | grep worker_process

# 按 Ctrl+C 退出
# 或发送 SIGTERM: kill -TERM <pid>

# 验证所有进程都已退出
ps aux | grep fastdeploy
ps aux | grep worker_process
```

### 测试 2: 多 Worker 模式

```bash
# 启动服务（多 worker）
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/model \
    --workers 4 \
    --port 8000

# 等待服务完全启动
# 检查进程（应该有主进程、4个 worker 进程和 worker_process）
ps aux | grep fastdeploy
ps aux | grep gunicorn
ps aux | grep worker_process

# 按 Ctrl+C 退出
# 或发送 SIGTERM: kill -TERM <master_pid>

# 验证所有进程都已退出
ps aux | grep fastdeploy
ps aux | grep gunicorn
ps aux | grep worker_process
```

### 测试 3: 启动过程中退出

```bash
# 启动服务
python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/model \
    --workers 2 \
    --port 8000 &

# 在服务完全启动前立即发送 SIGTERM
sleep 2
kill -TERM <pid>

# 验证所有进程都已退出
sleep 5
ps aux | grep fastdeploy
ps aux | grep worker_process
```

### 测试 4: 简单 API Server

```bash
# 启动简单的 API server
python -m fastdeploy.entrypoints.api_server \
    --model /path/to/model \
    --workers 2 \
    --port 9904

# 检查进程
ps aux | grep fastdeploy

# 按 Ctrl+C 退出

# 验证所有进程都已退出
ps aux | grep fastdeploy
```

## 验证要点

1. **进程清理完整性**：所有相关进程（主进程、worker 进程、worker_process 子进程）都应该被终止
2. **退出时间**：正常情况下应在 5 秒内完成清理（包括等待 worker_process 终止的超时时间）
3. **日志输出**：应该能看到以下日志：
   - "Received SIGINT/SIGTERM, initiating graceful shutdown..."
   - "Terminating worker process group {pgid}"
   - "Worker {pid} exiting"
   - "Gunicorn master process exiting, cleaning up..."

4. **无僵尸进程**：使用 `ps aux | grep defunct` 确认没有僵尸进程

## 预期结果

- ✅ 按 Ctrl+C 后，所有进程应在 5 秒内退出
- ✅ 不应有残留的 API server 或 worker_process 进程
- ✅ 日志应显示优雅关闭的相关信息
- ✅ 不应出现僵尸进程

## 故障排查

如果测试失败，可以检查：

1. **日志文件**：查看 `log/workerlog.*` 和 `log/launch_worker.log`
2. **进程树**：使用 `pstree -p <pid>` 查看进程树结构
3. **信号传递**：确认信号是否正确传递到子进程
4. **超时设置**：检查 `timeout_graceful_shutdown` 配置

## 已知限制

1. 如果 worker_process 在处理关键任务时收到信号，可能需要完成当前任务才能退出
2. 在某些极端情况下（如系统资源耗尽），可能仍需要使用 SIGKILL 强制终止
