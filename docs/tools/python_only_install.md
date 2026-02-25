# Python-only 快速安装

## 功能说明

`build.sh` 支持 `BUILD_WHEEL=2` 模式，只将 Python 文件同步到已安装的 site-packages 目录，跳过 C++ custom ops 编译和 wheel 打包。适用于只修改了 Python 代码、需要快速验证的场景。

## 前提条件

必须先做过一次完整编译安装：
```bash
bash build.sh 1 python false "[90]"
```
Python-only 模式依赖已安装的 `.so` 编译产物，首次使用前需要确保这些产物已存在于 site-packages 中。

## 使用方式

```bash
bash build.sh 2 [PYTHON]
```

第二个参数 `PYTHON` 是 Python **可执行文件**的路径或命令名（不是 site-packages 目录），默认值为 `python`。脚本内部通过该可执行文件来定位 site-packages 路径、执行 pip 等操作。

```bash
# 使用默认 python 命令
bash build.sh 2

# 显式指定 python 命令名
bash build.sh 2 python

# 使用 python3
bash build.sh 2 python3

# 使用完整路径
bash build.sh 2 /root/paddlejob/workspace/env_run/gongweibao/fdenv/bin/python
```

## 原理

`setup.py` 中使用 `find_packages()` + `package_dir={"fastdeploy": "fastdeploy/"}` 配置，所有 Python 文件安装到 `site-packages/fastdeploy/` 下，目录结构与源码一致，无重映射。因此可以直接用 rsync 将源码目录下的 `.py` 文件同步到安装目录：

```bash
rsync -av --exclude='__pycache__/' --include='*/' --include='*.py' \
  --filter='P *.so' --filter='P *.txt' --filter='P *.sh' --filter='P *.h' --filter='P *.hpp' \
  --exclude='*' --delete fastdeploy/ ${INSTALL_DIR}/fastdeploy/
```

- `--exclude='__pycache__/'`：放在最前面，完全跳过 `__pycache__` 目录（避免 "cannot delete non-empty directory" 警告）
- `--include='*.py'` + `--exclude='*'`：只同步 `.py` 文件
- `--filter='P ...'`：保护 `.so`、`.txt` 等已编译产物不被删除
- `--delete`：删除源码中已不存在的 `.py` 文件，保持 site-packages 与源码一致

## 同步摘要

虽然 rsync 会扫描所有 `.py` 文件，但只有**内容有变化**的文件才会实际传输。同步完成后，脚本会输出变更摘要，列出本次实际更新和删除的文件：

```
======== Sync Summary ========
[UPDATED] 3 file(s) synced:
  fastdeploy/config.py
  fastdeploy/engine/engine.py
  fastdeploy/entrypoints/api_server.py
[TOTAL] 506 Python files tracked, target: /path/to/site-packages/fastdeploy/
==============================
```

如果没有任何文件变化，则显示：

```
======== Sync Summary ========
[NO CHANGE] All 506 Python files are already up-to-date.
==============================
```

如果源码中删除了某个 `.py` 文件，site-packages 中对应文件也会被删除，并在摘要中显示：

```
[DELETED] 1 file(s) removed from site-packages:
  fastdeploy/old_module.py
```

## 防御机制

### 同目录检测

rsync 前会比较源目录和目标目录的真实路径（`pwd -P`，解析软链接）。如果两者相同（例如 editable install 或从 site-packages 目录运行脚本），脚本会跳过同步并提示：

```
[SKIP] Source and target are the same directory: /path/to/fastdeploy
[SKIP] No sync needed (you may be using an editable install or running from site-packages).
```

### 安装映射校验

同步完成后，脚本会自动校验 `setup.py` 的安装映射是否发生变化：通过 `pip show -f` 获取已安装文件列表，检查是否有 `.py` 文件被安装到 `fastdeploy/` 目录之外。如果检测到映射变化，脚本会报错并提示执行完整编译。

## 适用场景

| 场景 | 推荐方式 |
|------|----------|
| 只修改了 Python 文件 | `bash build.sh 2` |
| 修改了 C++/CUDA 代码 | `bash build.sh 1 python false "[90]"` |
| 首次编译 | `bash build.sh 1 python false "[90]"` |
