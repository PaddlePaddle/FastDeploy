---
name: nsys-capture
description: 通用 GPU 推理服务 nsys 性能抓取工具。根据用户的启动脚本自动注入 nsys 命令、构建带 nsys 的启动脚本，完成完整的 GPU profiling 抓取流程，输出 .nsys-rep 文件。
---

# Nsys 性能抓取 Skill

## 触发条件

- 用户需要对推理服务进行 nsys GPU profiling / 性能分析
- 用户提到 nsys profile、抓 nsys、GPU profiling、性能抓取

---

## 总体工作流（5 步）

```
Step 1：信息收集         → 从用户脚本和描述中提取关键参数
Step 2：注入 profiling   → 检查并向 FastDeploy 代码注入 nvprof_start/nvprof_stop
Step 3：生成脚本         → 构建 start_nsys.sh（注入 nsys）+ 确认请求脚本
Step 4：用户确认         → 展示生成的脚本，等用户确认后才执行
Step 5：执行抓取         → 启动服务 → 等待就绪 → 发送请求 → 等待文件 → 重命名
```

---

## Step 1：信息收集

读取用户提供的启动脚本，提取以下信息：

| 信息 | 提取方式 | 无法提取时 |
|------|---------|-----------|
| **模型路径** | `--model` 参数 / `MODEL_PATH` 变量 / 用户描述 | 询问用户，必填 |
| **服务端口** | `--port` / `--ports` 参数 / 用户描述 | 询问用户，默认 8080 |
| **Host IP** | 用户描述（"本机"→127.0.0.1，远端则询问） | 默认 127.0.0.1 |
| **nsys 输出目录** | 用户描述 / 脚本中 `OUTPUT_DIR`、`NSYS_PATH` 变量 | 默认 `/tmp/nsys_record` |
| **nsys 版本** | 用户指定（"详细版"/"高版" or "粗略版"/"低版"） | 默认粗略版 |
| **请求脚本** | 用户是否提供了测试请求脚本 | 使用 skill 内置默认脚本 |
| **nsys_start_step** | 用户指定 nvprof_start 触发的 decode 步数 | 默认 40 |
| **nsys_stop_step** | 用户指定 nvprof_stop 触发的 decode 步数 | 默认 60 |

信息收集完毕后，进入 Step 2。

---

## Step 2：检查并注入 FastDeploy profiling 代码

nsys 使用 `-c cudaProfilerApi` 模式，要求 FastDeploy 代码中必须包含 `nvprof_start()` / `nvprof_stop()` 调用。本步骤自动检测并注入。

### 2.1 定位 FastDeploy 代码路径

按优先级查找目标文件 `fastdeploy/worker/gpu_model_runner.py`：

1. **PYTHONPATH 优先**：检查用户启动脚本中是否通过 `PYTHONPATH` 或 `sys.path` 指定了 FastDeploy 路径（如 `export PYTHONPATH=/path/to/FastDeploy:$PYTHONPATH`），若有，使用该路径下的 `fastdeploy/worker/gpu_model_runner.py`
2. **pip 安装路径兜底**：执行 `pip3 show fastdeploy-gpu`，取 `Location` 字段，拼接 `fastdeploy/worker/gpu_model_runner.py`

确认文件存在后继续。若文件不存在，提示用户手动指定路径。

### 2.2 检查是否已有 profiling 代码

在 `gpu_model_runner.py` 中搜索 `nvprof_start` 和 `nvprof_stop`：

```bash
grep -n "nvprof_start\|nvprof_stop" <path_to_gpu_model_runner.py>
```

- **已存在且未被禁用**（无 `if False` 守卫、代码未注释）→ 跳过注入，提示用户已有 profiling 代码，直接进入 Step 3
- **已存在但被禁用**（有 `if False` 守卫或被注释）→ 提示用户发现已有但被禁用的 profiling 代码，询问是否需要启用并修正 step 值
- **不存在** → 进入 2.3 执行自动注入

### 2.3 自动注入 profiling 代码

**注入前先备份原文件：**
```bash
cp <path_to_gpu_model_runner.py> <path_to_gpu_model_runner.py>.bak
```

需注入 3 处代码，参考模板来自 `fastdeploy/worker/gpu_model_runner.py` 的已有实现模式：

#### 位置 1：`__init__` 方法中

在 `self.current_launch_token_num = 0` 之后添加：

```python
self.nsys_step = 0
```

#### 位置 2：`_execute_model` 方法中，执行调度代码之前

找到如下执行调度代码块：
```python
if not self.enable_overlap_schedule:
    self.execute_model_normal(model_forward_batch, num_running_requests)
else:
    self.execute_model_overlap(model_forward_batch, num_running_requests)
```

在其**之前**插入 nvprof_start：

```python
        if self.nsys_step == <nsys_start_step> and self.forward_meta.ids_remove_padding.shape[0] > 0:
            from paddle.framework import core
            core.nvprof_start()
```

> `<nsys_start_step>` 替换为 Step 1 收集到的值，默认 40。

#### 位置 3：`_execute_model` 方法中，执行调度代码之后

在执行调度代码块**之后**插入 nvprof_stop 和步数递增：

```python
        if self.nsys_step == <nsys_stop_step> and self.forward_meta.ids_remove_padding.shape[0] > 0:
            from paddle.framework import core
            core.nvprof_stop()
        if self.forward_meta.ids_remove_padding.shape[0] > 0:
            self.nsys_step += 1
```

> `<nsys_stop_step>` 替换为 Step 1 收集到的值，默认 60。

### 2.4 展示 diff 并确认

注入完成后，向用户展示修改的 diff：

```bash
diff <path_to_gpu_model_runner.py>.bak <path_to_gpu_model_runner.py>
```

说明：
- 注入了 nsys profiling 代码，nvprof_start 在第 `<nsys_start_step>` 步触发，nvprof_stop 在第 `<nsys_stop_step>` 步触发
- 原文件已备份为 `.bak`

**等用户确认后**，进入 Step 3。

---

## Step 3：生成 start_nsys.sh

### 3.1 核心注入原则

**不要** 使用 `nsys profile ... bash start.sh` 的包裹形式。
**要** 在脚本内部，找到实际启动服务的可执行命令行（`python` / `python3` / `torchrun` / `deepspeed` 等），在该行**前面**直接插入 `${NSYS_CMD}`，让 nsys 成为该进程的直接父进程。

```bash
# 原始脚本中的启动命令：
python -m some.serving.module \
    --model /path/to/model \
    --port 8080

# 注入后：
${NSYS_CMD} python -m some.serving.module \
    --model /path/to/model \
    --port 8080
```

### 3.2 生成步骤

1. 完整复制用户原始启动脚本内容
2. 在 shebang 行（`#!/bin/bash`）之后、其他内容之前，插入 **nsys 变量定义块**（见 3.3）
3. 定位脚本中实际启动服务进程的那行 `python` / `python3` 命令，在行首插入 `${NSYS_CMD}`
   - 如果命令跨多行（有 `\` 续行），只在**第一行行首**加 `${NSYS_CMD}`，续行不动
   - 如果脚本里有多个 python 命令，只注入**最后一个**（实际启动服务的那个，通常是前台阻塞的）
4. 文件命名：与原脚本同目录，加 `_nsys` 后缀，如 `start.sh` → `start_nsys.sh`

### 3.3 nsys 变量定义块（插入到脚本顶部）

```bash
# ============================================================
# NSYS INJECTION - 由 nsys-capture skill 自动注入
# ============================================================
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-/tmp/nsys_record}"
mkdir -p "${NSYS_OUTPUT_DIR}"
NSYS_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NSYS_OUTPUT_PATH="${NSYS_OUTPUT_DIR}/${NSYS_TIMESTAMP}_nsys"
echo "${NSYS_OUTPUT_PATH}" > /tmp/nsys_capture_output_path
echo "[nsys-capture] nsys 输出路径: ${NSYS_OUTPUT_PATH}"
```

然后根据用户选择的版本，追加对应的 NSYS_CMD 定义（两版都写，未选中的注释掉）：

**粗略版（默认，日常性能分析，文件约 6-15 MB）**：
```bash
# 粗略版（已启用）
NSYS_CMD="nsys profile -c cudaProfilerApi \
    -t nvtx,osrt,cuda,cublas-verbose,python-gil \
    -f true \
    --cudabacktrace=all \
    --python-backtrace=cuda \
    --cuda-memory-usage=true \
    --output ${NSYS_OUTPUT_PATH}"
# 详细版（深度调试，含完整 cuda graph trace，文件可达数百 MB）
# NSYS_CMD="nsys profile -c cudaProfilerApi \
#     -t nvtx,osrt,cuda,cublas-verbose,python-gil \
#     -f true \
#     --cudabacktrace=all \
#     --python-backtrace=cuda \
#     --cuda-memory-usage=true \
#     --cuda-graph-trace=node \
#     --output ${NSYS_OUTPUT_PATH}"
```

**详细版（用户指定时）**：将上面两段注释状态互换即可。

```bash
# 粗略版（日常性能分析，文件约 6-15 MB）
# NSYS_CMD="nsys profile -c cudaProfilerApi \
#     -t nvtx,osrt,cuda,cublas-verbose,python-gil \
#     -f true \
#     --cudabacktrace=all \
#     --python-backtrace=cuda \
#     --cuda-memory-usage=true \
#     --output ${NSYS_OUTPUT_PATH}"
# 详细版（已启用）
NSYS_CMD="nsys profile -c cudaProfilerApi \
    -t nvtx,osrt,cuda,cublas-verbose,python-gil \
    -f true \
    --cudabacktrace=all \
    --python-backtrace=cuda \
    --cuda-memory-usage=true \
    --cuda-graph-trace=node \
    --output ${NSYS_OUTPUT_PATH}"
```

> **注意**：`NSYS_OUTPUT_PATH` 在定义 `NSYS_CMD` 时已展开，所以两个块的顺序必须是：先定义 `NSYS_OUTPUT_PATH`，再定义 `NSYS_CMD`。
> 如果用户脚本中 `NSYS_OUTPUT_DIR` 已有定义，注入块中的默认值会被覆盖，以用户脚本的为准。

### 3.4 用户自定义 nsys 输出目录

- 如果用户明确提供了输出目录（如 `/data/nsys/`），将注入块中的默认值替换为该路径：
  ```bash
  NSYS_OUTPUT_DIR="/data/nsys/"
  ```
- 若用户脚本中已有 `NSYS_OUTPUT_DIR` 或类似变量，注入块放在该变量**之后**，并删除默认值赋值，直接复用。

---

## Step 4：确认请求脚本

向用户展示生成好的 `start_nsys.sh` 关键内容（注入块 + 被注入的 python 命令行），说明：
- 注入的 nsys 版本（粗略/详细）
- nsys 输出路径
- 使用的请求脚本（默认或用户提供的）

**等用户确认后**，进入 Step 5 执行。

请求脚本优先级：
1. 用户明确提供了测试请求脚本 → 使用用户的
2. 未提供 → 使用 skill 内置默认脚本：
   ```bash
   python3 ~/.claude/skills/nsys-capture/nsys_default_client.py <HOST> <PORT>
   ```

---

## Step 5：执行抓取

### 5.1 启动服务（后台）

```bash
rm -f /tmp/nsys_serve.log
bash <path_to_start_nsys.sh> > /tmp/nsys_serve.log 2>&1 &
echo "服务启动中，PID=$!"
```

### 5.2 等待服务就绪

轮询策略（每 30s 一次）：
- **就绪标志**：`curl -s http://<HOST>:<PORT>/v1/models` 返回 HTTP 200
- **致命错误**：日志出现 `Traceback` / `AssertionError` / `OOM` / `Killed`，且 30s 内日志无新增行 → 停止等待，输出最后 30 行日志供排查
- **超时**：超过 20 分钟未就绪 → 终止

检查日志时，**每次只取最新 50 行**（避免进度条等刷屏内容干扰）：
```bash
tail -50 /tmp/nsys_serve.log | grep -E "Traceback|AssertionError|OOM|Killed|startup complete"
```

### 5.3 发送请求

```bash
python3 <request_script> [HOST] [PORT] 2>/dev/null || true
```

- 流式连接中断（`RemoteProtocolError`）是正常现象，不报错
- 如果 nsys 埋点是在 iter N 触发 stop，请求的 token 数需**足够多**（生成 token > N），保证 stop 被触发

### 5.4 等待 nsys 文件 & 重命名

```bash
NSYS_OUTPUT_BASE=$(cat /tmp/nsys_capture_output_path)
EXPECTED_FILE="${NSYS_OUTPUT_BASE}.nsys-rep"
```

轮询直到文件出现且大小稳定（连续两次 `stat -c%s` 相同，间隔 3s），超时 120s。

稳定后重命名（带类型标记）：
```bash
FINAL_NAME="${NSYS_OUTPUT_DIR}/$(date +%Y%m%d_%H%M%S)_nsys_<type>_<level>.nsys-rep"
# type: text（文生文）/ mm（多模态）/ custom（用户自定义请求）
# level: low（粗略版）/ high（详细版）
mv "${EXPECTED_FILE}" "${FINAL_NAME}"
```

输出最终路径和文件大小，抓取完成。

---

## nsys 两版参数对比

| 版本 | 额外参数 | 文件大小 | 适用场景 |
|------|---------|---------|---------|
| **粗略版（默认）** | 无 `--cuda-graph-trace` | 6-15 MB | 日常性能分析，快速查看算子耗时 |
| **详细版** | `--cuda-graph-trace=node` | 可达数百 MB | 深度调试，完整 cuda graph 节点信息 |

---

## 常见问题

**Q: nsys 文件未生成？**
1. 检查埋点：`nvprof_start` 和 `nvprof_stop` 各一处，顺序正确
2. 检查请求 token 数是否足够（需 > 埋点触发 iter 数）
3. 用 `nsys sessions list` 查看是否进入过 `RangeCollection` 状态

**Q: 服务启动失败？**
- 检查 `/tmp/nsys_serve.log` 最后 50 行
- 常见原因：端口占用、模型路径不存在、GPU 显存不足

**Q: 如何分析 .nsys-rep 文件？**
```bash
nsys-ui <file>.nsys-rep
```

**Q: 需要抓多次怎么办？**
每次抓取后服务会自动退出（`nvprof_stop` 触发），下次需重新启动（需重新加载模型）。
