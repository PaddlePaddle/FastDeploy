# XPU CI 测试说明

本文档基于 `/.github/workflows/ci_xpu.yml` 及其调用的子 workflow 整理，重点说明：

- XPU CI 各个 workflow 的职责与执行关系
- 如何在自己的机器上尽可能复现 CI 检查
- 新增测试 case 应该放在哪里
- 不同类型测试 case 的编写要求

## 1. CI 总览

XPU 主流程入口为 `/.github/workflows/ci_xpu.yml`，触发条件为：

- `pull_request` 的 `opened` / `synchronize`
- 目标分支为 `develop` 或 `release/**`

同一 PR 同一 workflow 同时只能运行一个实例（concurrency group），后触发的会取消前一次正在运行的。

整体执行链路如下：

```text
clone ──> xpu_build_test ──┬──> xpu_4cards_case_test ──┐
                           ├──> xpu_8cards_case_test ──┼──> xpu_coverage_report
                           └──> xpu_unit_test ─────────┘
```

1. `clone`：代码克隆 + 归档上传
2. `xpu_build_test`：编译 wheel（依赖 clone）
3. `xpu_4cards_case_test`：4 卡集成测试（依赖 clone + build，**并行**）
4. `xpu_8cards_case_test`：8 卡集成测试（依赖 clone + build，**并行**）
5. `xpu_unit_test`：XPU 单测（依赖 clone + build，**并行**）
6. `xpu_coverage_report`：覆盖率汇总（依赖上述三个测试 job 全部结束）

其中：

- `xpu_build_test` 成功后，4 卡 / 8 卡 / unit test **三个 job 并行**执行
- `xpu_coverage_report` 使用 `if: always()` 条件，即使测试失败也会执行（只要 clone 成功）
- 每个子 workflow 内部都有 `check_bypass` 机制，可通过标签跳过对应阶段
- 整个流程运行在 self-hosted XPU runner 上，不是普通 CPU GitHub Hosted Runner
- Docker 镜像：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:ci`
- 测试 job 超时时间：60 分钟

## 2. 各个 workflow 的作用

### 2.1 `ci_xpu.yml`

主编排入口，只负责组织任务依赖，不直接执行测试逻辑：

- `clone`：代码克隆、PR merge、归档上传到 BOS
- `xpu_build_test`：编译并产出 `fastdeploy-xpu` wheel，上传到 BOS
- `xpu_4cards_case_test`：执行 4 卡集成 case，收集覆盖率
- `xpu_8cards_case_test`：执行 8 卡集成 case，收集覆盖率
- `xpu_unit_test`：执行 XPU 自定义算子单测 + 模型功能单测，收集覆盖率
- `xpu_coverage_report`：汇总三路覆盖率数据，执行增量覆盖率门禁检查，上传 Codecov

### 2.2 `/_clone_linux.yml`

职责：代码克隆、PR 合并、归档上传。

主要工作：

- checkout 目标分支代码（含 submodules，fetch-depth=1000）
- 如果是 PR，fetch PR head 并 merge 到目标分支（模拟合入后的状态）
- 将合并后的代码打包为 `FastDeploy.tar.gz` 上传到 BOS
- 产出 `repo_archive_url` 供下游 workflow 使用

### 2.3 `/_build_xpu.yml`

职责：构建 XPU 版本 FastDeploy。

- runner 标签：`XPU-P800`

主要工作：

- 拉取 CI Docker 镜像
- 解压代码归档
- 安装 Paddle XPU 依赖（支持 nightly / 指定版本 / 指定 wheel URL 三种模式）
- 执行 `bash custom_ops/xpu_ops/download_dependencies.sh develop`
- 设置 `CLANG_PATH` 和 `XVLLM_PATH`
- 执行 `bash build.sh`
- 上传生成的 `fastdeploy*.whl` 到 BOS

本阶段的产物是后续 workflow 使用的 `FASTDEPLOY_WHEEL_URL`。

### 2.4 `/_xpu_4cards_case_test.yml`

职责：在 4 卡机器上运行 `tests/xpu_ci/4cards_cases/` 下的集成测试。

主要特点：

- runner 标签：`XPU-P800-4Cards`
- 超时时间：60 分钟
- 前置操作：下载安装 xre，通过 `xpu-smi -r` 重启 XPU 卡
- 在容器里设置 `XPU_VISIBLE_DEVICES`
- 根据 runner 名末位字符决定 `XPU_ID=0`（使用卡 0-3）或 `XPU_ID=4`（使用卡 4-7）
- 安装上游 build 阶段产出的 wheel（两次安装：一次正常安装，一次 `--no-deps --target` 覆盖到源码目录）
- 执行：

```bash
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short tests/xpu_ci/4cards_cases/
```

- 归档 `case_logs/`（通过 `actions/upload-artifact`）
- 上传 `coverage_4cards.tar.gz` 到 BOS

### 2.5 `/_xpu_8cards_case_test.yml`

职责：在 8 卡机器上运行 `tests/xpu_ci/8cards_cases/` 下的集成测试。

主要特点：

- runner 标签：`XPU-P800-8Cards`
- 超时时间：60 分钟
- 前置操作：下载安装 xre，通过 `xpu-smi -r` 重启 XPU 卡
- 固定设置 `XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- 安装上游 build 阶段产出的 wheel
- 执行：

```bash
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short tests/xpu_ci/8cards_cases/
```

- 归档 `case_logs/`（通过 `actions/upload-artifact`）
- 上传 `coverage_8cards.tar.gz` 到 BOS

### 2.6 `/_xpu_unit_test.yml`

职责：执行 XPU 相关单测。

- runner 标签：`XPU-P800-4Cards`
- 超时时间：60 分钟
- 前置操作：下载安装 xre，通过 `xpu-smi -r` 重启 XPU 卡

当前包含两部分：

1. XPU 自定义算子单测（`custom_ops/xpu_ops/test/` 整个目录）
2. `tests/xpu_ci/unit_test/` 下的模型功能单测

执行命令分别为：

```bash
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short custom_ops/xpu_ops/test/
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short tests/xpu_ci/unit_test/
```

两部分独立运行，任一失败则整体 job 失败。

该 workflow 会上传：

- `ut_logs/`（通过 `actions/upload-artifact`）
- `coverage_unit_test.tar.gz` 到 BOS

### 2.7 `/_xpu_coverage_report.yml`

职责：汇总覆盖率并校验 PR 增量覆盖率。

- runner 标签：`XPU-P800-4Cards`（合并阶段）+ `ubuntu-latest`（Codecov 上传阶段）
- 超时时间：合并阶段 30 分钟，上传阶段 15 分钟

主要工作：

- 从 BOS 下载 4 卡 / 8 卡 / unit test 的覆盖率 tar 包
- 使用 `python -m coverage combine` 合并为统一 coverage 数据
- 生成 `xpu_coverage_all.xml`
- 对 PR 执行 `diff-cover`，生成增量覆盖率报告 `xpu_diff_coverage.json`
- 增量覆盖率阈值：**80%**（低于则 CI 失败）
- 上传覆盖率报告到 BOS
- 在独立的 `ubuntu-latest` runner 上将 XML 上传到 Codecov（flags: XPU）

这一步的意义是：

- 不是再跑一次测试
- 而是对前面几个 workflow 的覆盖率结果做统一汇总和门禁检查

## 3. 目录结构与测试分类

当前 XPU CI 测试主要分为三类：

```text
tests/xpu_ci/
├── 4cards_cases/   # 4 卡集成测试
├── 8cards_cases/   # 8 卡集成测试
├── unit_test/      # XPU 模型功能单测
├── conftest.py     # 公共 fixture / 启停服务 / 日志归档 / 环境管理
└── README.md
```

建议按下面规则理解：

- `4cards_cases/`
  - 适合单机 4 卡可完成的服务级联调 case
  - 常见场景：TP4、量化、V1 模式、VL、logprobs 等
- `8cards_cases/`
  - 适合必须依赖 8 卡资源的 case
  - 常见场景：EP4TP4、PD/EP 组合、多角色部署
- `unit_test/`
  - 适合不需要完整在线服务编排的模型功能单测
- `custom_ops/xpu_ops/test/`
  - 适合 XPU 自定义算子层面的单测

## 4. 如何在自己的机器上执行 CI 检查

先说明限制：当前 XPU CI 并不是“普通开发机开箱即跑”的测试体系。要尽量复现 CI，机器至少需要满足以下条件：

- 有可用 XPU 机器
- 可使用 Docker
- 能访问 CI 用到的依赖源和模型目录
- 有 `xpu-smi` 能力
- 能准备 `MODEL_PATH` 对应模型
- 最好具备与 CI 接近的目录映射，例如 `/workspace`、`/ssd3/model`

如果你的机器不满足这些条件，建议优先做“最小本地验证”，而不是强求 1:1 复现整条 CI。

### 4.1 最推荐：先做最小本地验证

在仓库根目录执行：

```bash
cd /paddle/sjx_cuda12.6_py310/fd_test/FastDeploy
export MODEL_PATH=/your/model/path
export XPU_ID=0
export PYTHONPATH=$(pwd):$(pwd)/tests/xpu_ci:$PYTHONPATH
```

#### 运行单个 4 卡 case

```bash
python -m pytest -v -s --tb=short tests/xpu_ci/4cards_cases/test_w4a8.py
```

#### 运行整个 4 卡目录

```bash
python -m pytest -v -s --tb=short tests/xpu_ci/4cards_cases/
```

#### 运行整个 8 卡目录

```bash
python -m pytest -v -s --tb=short tests/xpu_ci/8cards_cases/
```

#### 运行模型功能单测

```bash
python -m pytest -v -s --tb=short tests/xpu_ci/unit_test/
```

#### 运行自定义算子单测

```bash
python -m pytest -v -s --tb=short custom_ops/xpu_ops/test/
```

适用场景：

- 改了某个测试 case，先验证 case 自己是否能过
- 改了 `conftest.py` 或启动逻辑，先做快速回归
- 不希望先做完整 build / 覆盖率上传 / Docker 封装

### 4.2 复现 4 卡 CI 检查

如果你要尽量模拟 `/_xpu_4cards_case_test.yml`，建议按下面步骤执行。

#### 步骤 1：准备环境

```bash
cd /paddle/sjx_cuda12.6_py310/fd_test/FastDeploy
export MODEL_PATH=/your/model/path
export XPU_ID=0
```

如果你要模拟另一组卡，可以改成：

```bash
export XPU_ID=4
```

注意：`conftest.py` 会根据 `XPU_ID` 映射：

- `XPU_ID=0` -> `XPU_VISIBLE_DEVICES=0,1,2,3`
- 其他值 -> `XPU_VISIBLE_DEVICES=4,5,6,7`

#### 步骤 2：准备 Python 依赖和编译产物

你有两种方式：

- 方式 A：直接本地编译并安装当前仓库
- 方式 B：安装 CI build 阶段产出的 `fastdeploy-xpu` wheel

如果只是自测新增 case，通常方式 A 就够用：

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt
python -m pip uninstall paddlepaddle-xpu fastdeploy-xpu -y
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
bash custom_ops/xpu_ops/download_dependencies.sh develop
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
bash build.sh
```

然后补充测试依赖：

```bash
python -m pip install -U openai pytest pytest-timeout coverage
export PYTHONPATH=$(pwd):$(pwd)/tests/xpu_ci:$PYTHONPATH
export COVERAGE_RCFILE=$(pwd)/scripts/.coveragerc_xpu
mkdir -p case_logs coveragedata
```

#### 步骤 3：运行 4 卡 case

```bash
COVERAGE_FILE=$(pwd)/coveragedata/.coverage.4cards \
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short tests/xpu_ci/4cards_cases/
```

#### 步骤 4：合并并查看覆盖率

```bash
COVERAGE_FILE=$(pwd)/coveragedata/.coverage.4cards \
python -m coverage combine --rcfile=${COVERAGE_RCFILE} coveragedata/ || true
python -m coverage report --rcfile=${COVERAGE_RCFILE} || true
```

### 4.3 复现 8 卡 CI 检查

和 4 卡类似，但需要保证机器具备完整 8 卡资源：

```bash
cd /paddle/sjx_cuda12.6_py310/fd_test/FastDeploy
export MODEL_PATH=/your/model/path
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m pip install -U openai pytest pytest-timeout coverage
export PYTHONPATH=$(pwd):$(pwd)/tests/xpu_ci:$PYTHONPATH
export COVERAGE_RCFILE=$(pwd)/scripts/.coveragerc_xpu
mkdir -p case_logs coveragedata

COVERAGE_FILE=$(pwd)/coveragedata/.coverage.8cards \
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short tests/xpu_ci/8cards_cases/
```

### 4.4 复现 unit test CI 检查

```bash
cd /paddle/sjx_cuda12.6_py310/fd_test/FastDeploy
export XPU_ID=0
export PYTHONPATH=$(pwd):$(pwd)/tests/xpu_ci:$PYTHONPATH
export COVERAGE_RCFILE=$(pwd)/scripts/.coveragerc_xpu
mkdir -p ut_logs coveragedata

# 自定义算子单测（整个目录）
COVERAGE_FILE=$(pwd)/coveragedata/.coverage.unit_ops \
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short custom_ops/xpu_ops/test/

# 模型功能单测
COVERAGE_FILE=$(pwd)/coveragedata/.coverage.unit_model \
python -m coverage run --rcfile=${COVERAGE_RCFILE} -m pytest -v -s --tb=short tests/xpu_ci/unit_test/
```

### 4.5 直接执行现成脚本

仓库中还提供了入口脚本：`scripts/run_xpu_ci_pytest.sh`

```bash
cd /paddle/sjx_cuda12.6_py310/fd_test/FastDeploy
export MODEL_PATH=/your/model/path
export XPU_ID=0
bash scripts/run_xpu_ci_pytest.sh
```

这个脚本适合：

- 希望一次跑完整的 `tests/xpu_ci/`
- 希望复用脚本里的依赖安装与 build 逻辑

但要注意：

- 脚本默认更像“4 卡本地全量入口”，不是 GitHub workflow 的完整等价替身
- 它会执行依赖安装、重启 XPU 卡、重新 build，动作较重
- 它调用的是：

```bash
python -m pytest -v -s --tb=short tests/xpu_ci/
```

因此会依赖 pytest 自动发现测试文件。

## 5. 日志与失败排查

### 5.1 自动归档日志

`tests/xpu_ci/conftest.py` 已实现 pytest hook，会在每个 case 执行后自动把日志归档到：

```text
case_logs/<test_file_name>/
```

常见日志包括：

- `server.log`
- `log/workerlog.0`
- `log_router/`
- `log_prefill/`
- `log_decode/`

### 5.2 常用排查点

如果 case 失败，优先检查：

1. `MODEL_PATH` 是否正确，模型目录是否真实存在
2. `XPU_VISIBLE_DEVICES` 是否与 case 需求匹配
3. 端口是否冲突
4. `/health` 和 `/v1/models` 是否都能返回成功
5. RDMA 网卡是否可用（PD/EP 类 case 特别重要）

## 6. 新增测试 case 放在哪里

### 6.1 放置规则

按测试目标选择目录：

- 新增 4 卡集成 case：放到 `tests/xpu_ci/4cards_cases/`
- 新增 8 卡集成 case：放到 `tests/xpu_ci/8cards_cases/`
- 新增模型功能单测：放到 `tests/xpu_ci/unit_test/`
- 新增自定义算子单测：放到 `custom_ops/xpu_ops/test/`

### 6.2 文件命名规则

测试文件名必须使用：

```text
test_*.py
```

例如：

- `test_v1_mode_xxx.py`
- `test_pd_xxx.py`
- `test_quant_xxx.py`

原因不是文档约定，而是当前 CI workflow 直接对目录执行 `pytest`，依赖 pytest 自动发现规则。如果文件名不是 `test_*.py`，通常不会被自动收集。

### 6.3 什么时候放 4 卡，什么时候放 8 卡

按“资源最小充分”原则：

- 4 卡能覆盖，就放 `4cards_cases/`
- 必须使用 8 卡才能成立，才放 `8cards_cases/`

不要把 4 卡就能覆盖的 case 放到 8 卡目录，否则会无谓提高 CI 成本。

## 7. 新增测试 case 的编写要求

### 7.1 通用要求

新增 case 建议复用 `tests/xpu_ci/conftest.py` 中的公共能力，不要在每个文件里重复实现：

- `get_port_num()`：按 `XPU_ID` 计算基准端口
- `get_model_path()`：统一读取模型目录
- `start_server()`：统一启动服务并做健康检查
- `print_logs_on_failure()`：失败时打印关键日志
- `xpu_env`：统一设置 `XPU_VISIBLE_DEVICES` 并在 case 结束后清理进程

普通在线服务 case 的推荐结构：

```python
import openai
import pytest
from conftest import get_model_path, get_port_num, print_logs_on_failure, start_server


def test_xxx(xpu_env):
    port_num = get_port_num()
    model_path = get_model_path()
    server_args = [
        "--model", f"{model_path}/YOUR_MODEL",
        "--port", str(port_num),
    ]

    if not start_server(server_args):
        pytest.fail("服务启动失败")

    try:
        client = openai.Client(base_url=f"http://0.0.0.0:{port_num}/v1", api_key="EMPTY_API_KEY")
        response = client.chat.completions.create(...)
        assert response is not None
    except Exception as e:
        print_logs_on_failure()
        pytest.fail(str(e))
```

### 7.2 断言要求

断言要满足以下原则：

- 能稳定复现，不依赖随机结果
- 尽量验证核心能力，而不是过度依赖长文本完全一致
- 优先校验关键字段、关键词、接口成功与基本语义正确
- 失败信息要可读，便于 CI 直接定位

推荐做法：

- 先校验服务可用
- 再校验 response 结构合法
- 最后校验关键结果

不推荐：

- 对整段生成文本做完全字符串匹配
- 使用极易波动的 prompt 或阈值
- 把多个大场景硬塞进一个 case 导致定位困难

### 7.3 启动参数要求

新增 case 的 `server_args` 要尽量最小化，只保留该场景真正需要的参数。

同时建议保持以下字段清晰：

- `--model`
- `--port`
- `--engine-worker-queue-port`
- `--metrics-port`
- 并行配置（如 `--tensor-parallel-size`）
- 场景配置（如 `--quantization`、`--enable-expert-parallel`）

如果你的场景依赖特殊环境变量，不要在 README 里口头说明，要直接在 case 中显式设置或通过 `conftest.py` 公共函数设置。

### 7.4 EP / PD / 特殊场景要求

如果是特殊并行或部署模式，请优先复用已有 helper：

- EP 场景：`setup_ep_env()` / `restore_env()`
- PD 场景：`setup_pd_env()` / `restore_pd_env()`
- PD + EP 场景：`setup_pd_ep_env()` / `restore_pd_ep_env()`
- logprobs 特殊场景：`setup_logprobs_zmq_env()`
测试失败时会自动打印 `server.log` 和 `log/paddle/workerlog.0` 的内容。
你也可以在测试运行时手动查看:

```bash
tail -f server.log
tail -f log/paddle/workerlog.0
```

这类 case 额外要注意：

- 是否需要 RDMA 网卡
- 是否需要多角色进程（router / prefill / decode）
- 是否需要手动切换不同 `XPU_VISIBLE_DEVICES`
- 超时时间是否需要比普通 case 更长

### 7.5 日志与清理要求

新增 case 必须保证失败时能留下足够信息：

- 普通场景失败时调用 `print_logs_on_failure()`
- 特殊场景需要打印对应角色日志
- 尽量把清理逻辑放到 `finally` 中
- 不要在 case 里自行发散式清理一堆无关进程

公共清理逻辑已经放在 `conftest.py`，新增 case 应优先复用，而不是自己重复写一套。

### 7.6 单测要求

如果新增的是 `tests/xpu_ci/unit_test/` 下的测试，要求和集成 case 不完全一样：

- 优先聚焦单一功能点
- 尽量避免引入完整在线服务依赖
- 保持执行时间短、定位明确
- 如果存在临时跳过项，可在 `tests/xpu_ci/unit_test/pytest.ini` 中统一管理

如果新增的是 `custom_ops/xpu_ops/test/` 下的测试，则应遵循对应目录现有单测风格，不要混入在线服务 case 的写法。

## 8. 新增 case 后需要做什么

新增 case 后，建议至少完成下面几步：

1. 先单独跑新 case
2. 再跑对应目录全量测试
3. 如果改了公共逻辑，再补跑 unit test
4. 如果改动涉及覆盖率敏感代码，关注 PR 的 `xpu_coverage_report`

推荐命令：

```bash
cd /paddle/sjx_cuda12.6_py310/fd_test/FastDeploy
export MODEL_PATH=/your/model/path
export XPU_ID=0
export PYTHONPATH=$(pwd):$(pwd)/tests/xpu_ci:$PYTHONPATH

python -m pytest -v -s --tb=short tests/xpu_ci/4cards_cases/test_your_case.py
python -m pytest -v -s --tb=short tests/xpu_ci/4cards_cases/
```

如果是 8 卡 case，把目录替换为 `tests/xpu_ci/8cards_cases/` 即可。

## 9. 常见问题

### 9.1 为什么我新增了文件，但 CI 没跑到？

先检查文件名是否为 `test_*.py`。当前 workflow 是对目录直接执行 `pytest`，不是手工枚举文件列表。

### 9.2 为什么本地能跑，CI 跑不过？

优先检查以下差异：

- 本地是否真的安装了 XPU 版本 Paddle / FastDeploy
- 本地模型目录是否和 CI 一致
- 本地是否少了 `XPU_VISIBLE_DEVICES`、RDMA、代理或覆盖率配置
- 本地是否直接用了源码，而 CI 实际验证的是 build 阶段产出的 wheel

### 9.3 为什么 health check 过了，但测试还是失败？

`conftest.py` 不仅检查 `/health`，还会继续检查 `/v1/models` 是否返回有效模型列表。很多“服务已起但模型未就绪”的问题会在第二阶段暴露。

## 10. 结论

新增 XPU CI 用例时，请牢记四点：

1. 放对目录：4 卡 / 8 卡 / unit test / custom ops
2. 命名正确：必须使用 `test_*.py`
3. 复用公共能力：优先使用 `conftest.py` 的 fixture 和 helper
4. 先做最小本地验证，再尝试复现完整 CI
