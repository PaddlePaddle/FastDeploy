# FastDeploy V100 热身打卡教程

## 背景说明

本教程基于 [PR #6306](https://github.com/PaddlePaddle/FastDeploy/pull/6306) 的 V100 (SM70) 支持功能，帮助开发者在 V100 GPU 上完成 FastDeploy 的编译与测试。

**V100 与 A100 的主要区别**：

| 特性 | V100 (SM70) | A100 (SM80) |
|------|-------------|-------------|
| BF16 | fallback 到 FP16 | 原生支持 |
| FP8 | 不支持 | 需 SM89+ |
| APPEND_ATTN | fallback 到 FLASH_ATTN | 支持 |
| MLA_ATTN | fallback 到 FLASH_ATTN | 支持 |

---

## 准备环境

### 1. 硬件要求

- **NVIDIA V100 GPU** (SM70 架构)
- 推荐内存：>=32GB
- CUDA 11.8

### 2. 安装 PaddlePaddle

```bash
# V100 使用 CUDA 11.8 版本
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### 3. 克隆 FastDeploy 源码

```bash
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

# 切换到 V100 支持的 PR 分支
git fetch origin pull/6306/head:pr-6306
git checkout pr-6306
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
pip install pytest pytest-xdist
```

---

## 编译打卡流程

> **重要**：V100 编译时 MAX_JOBS 建议设置为 **8**，过高会导致 OOM 被 Kill。

### Step 1：执行 FastDeploy 编译与打包

```bash
# 参数说明
# 第1个参数: 是否构建 wheel（1=构建，0=仅编译）
# 第2个参数: Python 解释器
# 第3个参数: 是否编译 CPU BF16 算子
# 第4个参数: GPU 架构（V100 = 70）

time MAX_JOBS=8 bash build.sh 1 python false [70]
```

编译完成后，产物位于：`FastDeploy/dist/`

**预期耗时**：约 90 分钟（取决于环境）

### Step 2：二次编译测试

初次编译时间较长，二次编译因为有编译缓存的存在，时间会缩短。

- 修改 kernel_traits 头文件：`custom_ops/gpu_ops/flash_mask_attn/kernel_traits.h`
- 修改 transfer_output 的 cc 文件：`custom_ops/gpu_ops/transfer_output.cc`
- 修改 python 文件：`custom_ops/setup_ops.py`

二次编译方式：对应文件加一个空行/空格保存退出后，执行：

```bash
time MAX_JOBS=8 bash build.sh 0 python false [70]
```

### Step 3：安装 whl 包

```bash
pip install dist/fastdeploy*.whl
```

### Step 4：验证 V100 支持

```bash
python -c "
from fastdeploy.model_executor.layers.utils import get_sm_version
from fastdeploy.platforms import current_platform
print(f'Platform: {current_platform}')
print(f'SM Version: {get_sm_version()}')
print(f'Is V100 (SM70): {get_sm_version() == 70}')
"
```

**预期输出**：

```
Platform: <fastdeploy.platforms.cuda.CUDAPlatform object at 0x...>
SM Version: 70
Is V100 (SM70): True
```

### Step 5：运行单元测试

```bash
# Platform 测试
python -m pytest tests/platforms/test_platforms.py -v

# FFN 测试
python -m pytest tests/layers/test_ffn.py -v

# Quantization 测试
python -m pytest tests/quantization/ -v
```

**V100 测试预期结果**：

| 测试模块 | 通过 | 跳过 | 失败 |
|----------|------|------|------|
| Platform Tests | 28 | 0 | 1* |
| FFN Tests | 1 | 0 | 0 |
| Quantization Tests | 46 | 9 | 0 |

> *注：`test_attention_backend_valid` 失败是预期行为，V100 自动 fallback APPEND_ATTN -> FLASH_ATTN

---

## 邮件格式

**标题**：[Hackathon-FastDeploy V100 热身打卡]

**内容**：

```
飞桨团队你好，

【GitHub ID】：XXX

【打卡内容】：V100 初次编译/二次编译/安装whl包/运行单元测试

【打卡截图】：
```

| 项目 | 内容 |
|------|------|
| 硬件 | V100 (SM70), CUDA 11.8 |
| 编译方式 | 参考 PR #6306 V100 支持 |
| 初次编译命令和时间 | 命令：`time MAX_JOBS=8 bash build.sh 1 python false [70]`<br/>时间：XXX |
| 二次编译时间 | `kernel_traits.h`: XXX<br/>`transfer_output.cc`: XXX<br/>`setup_ops.py`: XXX |
| 安装whl包 | 截图 |
| SM Version 验证 | SM Version: 70, Is V100: True |
| 运行单元测试 | Platform: 28 passed, 1 failed (预期)<br/>FFN: 1 passed<br/>Quantization: 46 passed, 9 skipped |

---

## V100 常见问题

### 1. 编译被 Killed (OOM)

**原因**：nvcc 并发编译消耗大量内存

**解决**：

```bash
# 降低并发数
MAX_JOBS=4 bash build.sh 1 python false [70]

# 或更保守
MAX_JOBS=2 bash build.sh 1 python false [70]
```

### 2. 残留进程清理

```bash
pkill -9 nvcc; pkill -9 cc1plus; pkill -9 cicc; pkill -9 ptxas
rm -rf custom_ops/build custom_ops/tmp build *.egg-info dist
```

### 3. test_attention_backend_valid 失败

**这是预期行为！** V100 不支持 APPEND_ATTN，PR #6306 实现了自动 fallback：

```
WARNING: APPEND_ATTN backend requires SM80+ (cp.async instructions),
but current GPU is SM70. Automatically falling back to FLASH_ATTN backend.
```

### 4. FP8 相关测试跳过

正常现象，FP8 需要 SM89+ (Ada Lovelace) 架构。

### 5. 链接错误：No such file or directory

**错误信息**：

```
x86_64-linux-gnu-g++: error: .../moe_deepgemm_depermute.cu.o: No such file or directory
x86_64-linux-gnu-g++: error: .../min_p_sampling_from_probs.cu.o: No such file or directory
error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1
[FAIL] build wheel failed
```

**原因**：之前编译被中断或部分文件编译失败，导致链接时找不到 .o 文件

**解决**：完全清理构建缓存后重新编译

```bash
cd /home/aistudio/work/FastDeploy
rm -rf custom_ops/build custom_ops/tmp build *.egg-info dist
MAX_JOBS=8 bash build.sh 1 python false [70] 2>&1 | tee "build_v100_$(date +%Y%m%d_%H%M%S).log"
```

---

## 完整一键命令

从零开始的完整流程，可直接复制执行：

```bash
#!/bin/bash
# ============================================================
# FastDeploy V100 完整编译与测试流程（带日志）
# ============================================================

set -e

# 日志配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/aistudio/work/build_v100_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "FastDeploy V100 Build & Test Started"
log "Log file: $LOG_FILE"
log "=========================================="

# 1. 清理残留进程（如有）
log "=== Step 1: 清理残留进程 ==="
pkill -9 nvcc 2>/dev/null || true
pkill -9 cc1plus 2>/dev/null || true
pkill -9 cicc 2>/dev/null || true
pkill -9 ptxas 2>/dev/null || true

# 2. 设置工作目录
log "=== Step 2: 设置工作目录 ==="
cd /home/aistudio/work
rm -rf FastDeploy

# 3. 克隆代码并切换分支
log "=== Step 3: 克隆代码并切换分支 ==="
START_TIME=$(date +%s)
git clone https://github.com/PaddlePaddle/FastDeploy.git 2>&1 | tee -a $LOG_FILE
cd FastDeploy
git fetch origin pull/6306/head:pr-6306 2>&1 | tee -a $LOG_FILE
git checkout pr-6306 2>&1 | tee -a $LOG_FILE
END_TIME=$(date +%s)
log "Step 3 completed in $((END_TIME - START_TIME)) seconds"

# 4. 安装 PaddlePaddle (CUDA 11.8)
log "=== Step 4: 安装 PaddlePaddle ==="
START_TIME=$(date +%s)
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ 2>&1 | tee -a $LOG_FILE
END_TIME=$(date +%s)
log "Step 4 completed in $((END_TIME - START_TIME)) seconds"

# 5. 安装依赖
log "=== Step 5: 安装依赖 ==="
START_TIME=$(date +%s)
pip install -r requirements.txt 2>&1 | tee -a $LOG_FILE
pip install pytest pytest-xdist 2>&1 | tee -a $LOG_FILE
END_TIME=$(date +%s)
log "Step 5 completed in $((END_TIME - START_TIME)) seconds"

# 6. 初次编译 (V100 = SM70)
log "=== Step 6: 初次编译 (MAX_JOBS=8, SM70) ==="
START_TIME=$(date +%s)
MAX_JOBS=8 bash build.sh 1 python false [70] 2>&1 | tee -a $LOG_FILE
END_TIME=$(date +%s)
log "Step 6 completed in $((END_TIME - START_TIME)) seconds"

# 7. 安装 wheel 包
log "=== Step 7: 安装 wheel 包 ==="
START_TIME=$(date +%s)
pip install dist/fastdeploy*.whl 2>&1 | tee -a $LOG_FILE
END_TIME=$(date +%s)
log "Step 7 completed in $((END_TIME - START_TIME)) seconds"

# 8. 验证 SM Version
log "=== Step 8: 验证 SM Version ==="
python -c "
from fastdeploy.model_executor.layers.utils import get_sm_version
from fastdeploy.platforms import current_platform
print(f'Platform: {current_platform}')
print(f'SM Version: {get_sm_version()}')
print(f'Is V100 (SM70): {get_sm_version() == 70}')
" 2>&1 | tee -a $LOG_FILE

# 9. 运行单元测试
log "=== Step 9: 运行单元测试 ==="

log "--- Platform Tests ---"
START_TIME=$(date +%s)
python -m pytest tests/platforms/test_platforms.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "Platform Tests completed in $((END_TIME - START_TIME)) seconds"

log "--- FFN Tests ---"
START_TIME=$(date +%s)
python -m pytest tests/layers/test_ffn.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "FFN Tests completed in $((END_TIME - START_TIME)) seconds"

log "--- Quantization Tests ---"
START_TIME=$(date +%s)
python -m pytest tests/quantization/ -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "Quantization Tests completed in $((END_TIME - START_TIME)) seconds"

log "=========================================="
log "Build & Test Completed"
log "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
log "Full log saved to: $LOG_FILE"
log "=========================================="
```

---

## 参考链接

- [PR #6306: V100 支持](https://github.com/PaddlePaddle/FastDeploy/pull/6306)
- [A100 热身打卡教程](https://github.com/PaddlePaddle/FastDeploy/issues/6225)
- [FastDeploy 源码编译文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/nvidia_gpu.md)
