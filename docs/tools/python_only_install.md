# Python-only 快速安装指南

## 1. 功能概述

`build.sh` 支持 `BUILD_WHEEL=2` 模式，将 Python 源文件直接同步到已安装的 `site-packages` 目录，**跳过 C++ Custom Ops 编译和 Wheel 打包**。适用于仅修改 Python 代码、需要快速验证的开发场景。

## 2. 前提条件

使用 Python-only 模式前，**必须先完成一次完整编译安装**：

```bash
bash build.sh 1 python false "[90]"
```

该模式依赖完整编译产生的 `.so` 文件。若 `site-packages` 中不存在这些产物，`build.sh 2` 会报错退出：

```
[FAIL] fastdeploy is not installed. Please run a full build first (BUILD_WHEEL=1).
```

## 3. 使用方式

```bash
bash build.sh 2 [PYTHON]
```

- **`PYTHON`**：Python 可执行文件的路径或命令名（**不是** `site-packages` 目录路径），默认值为 `python`。
- 脚本内部通过该可执行文件定位 `site-packages` 路径并执行 `pip` 等操作。

**示例：**

```bash
# 使用默认 python
bash build.sh 2

# 显式指定 python
bash build.sh 2 python

# 使用 python3
bash build.sh 2 python3

# 使用完整路径
bash build.sh 2 /root/paddlejob/workspace/env_run/gongweibao/fdenv/bin/python
```

## 4. 工作原理

### 4.1 文件同步

`setup.py` 中通过 `find_packages()` + `package_dir={"fastdeploy": "fastdeploy/"}` 配置，所有 Python 文件安装到 `site-packages/fastdeploy/` 下，目录结构与源码保持一致，无路径重映射。因此可直接使用 `rsync` 将源码目录中的 `.py` 文件同步到安装目录：

```bash
rsync -avc --exclude='__pycache__/' --include='*/' --include='*.py' \
  --filter='P *.so' --filter='P *.txt' --filter='P *.sh' --filter='P *.h' --filter='P *.hpp' \
  --exclude='*' --delete fastdeploy/ ${INSTALL_DIR}/fastdeploy/
```

各参数说明：

| 参数 | 作用 |
|------|------|
| `-c`（`--checksum`） | 基于文件内容校验判断是否需要同步，而非依赖时间戳 |
| `--exclude='__pycache__/'` | 置于最前，完全跳过 `__pycache__` 目录，避免删除非空目录的警告 |
| `--include='*.py'` + `--exclude='*'` | 仅同步 `.py` 文件 |
| `--filter='P ...'` | 保护 `.so`、`.txt` 等已编译产物不被删除 |
| `--delete` | 删除源码中已移除的 `.py` 文件，保持 `site-packages` 与源码一致 |

### 4.2 安装目录定位

脚本通过 `importlib.util.find_spec('fastdeploy')` 定位安装目录。为避免在项目根目录执行时 Python 将当前目录（即源码）误识别为安装位置，脚本会从 `sys.path` 中排除当前工作目录，确保只在 `site-packages` 中查找。

## 5. 同步摘要

同步完成后，脚本输出变更摘要，列出实际更新和删除的文件。

**有文件更新时：**

```
======== Sync Summary ========
[UPDATED] 3 file(s) synced:
  fastdeploy/config.py
  fastdeploy/engine/engine.py
  fastdeploy/entrypoints/api_server.py
[TOTAL] 506 Python files tracked, target: /path/to/site-packages/fastdeploy/
==============================
```

**无文件变化时：**

```
======== Sync Summary ========
[NO CHANGE] All 506 Python files are already up-to-date.
==============================
```

**有文件被删除时：**

```
[DELETED] 1 file(s) removed from site-packages:
  fastdeploy/old_module.py
```

## 6. 防御机制

### 6.1 未安装检测

若 `site-packages` 中找不到已安装的 `fastdeploy`（即从未执行过 `build.sh 1`），脚本直接报错退出，避免在缺少 `.so` 编译产物时产生不完整的安装。

### 6.2 同目录检测

`rsync` 执行前，脚本会比较源目录和目标目录的真实路径（通过 `pwd -P` 解析软链接）。若两者相同（例如 editable install 或从 `site-packages` 目录运行脚本），则跳过同步并提示：

```
[SKIP] Source and target are the same directory: /path/to/fastdeploy
[SKIP] No sync needed (you may be using an editable install or running from site-packages).
```

### 6.3 安装映射校验

同步完成后，脚本自动校验 `setup.py` 的安装映射是否发生变化：通过 `pip show -f` 获取已安装文件列表，检查是否有 `.py` 文件被安装到 `fastdeploy/` 目录之外。若检测到映射变化，脚本会报错并提示执行完整编译。

## 7. 适用场景速查

| 场景 | 推荐方式 |
|------|----------|
| 仅修改 Python 文件 | `bash build.sh 2` |
| 修改了 C++/CUDA 代码 | `bash build.sh 1 python false "[90]"` |
| 首次编译 | `bash build.sh 1 python false "[90]"` |
