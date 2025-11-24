# XPU CI Test Framework

基于 pytest 的 XPU CI 测试框架，使用面向对象的设计，便于维护和扩展。

## 目录结构

```
tests/xpu_ci/
├── __init__.py
├── conftest.py          # pytest fixtures 和配置
├── pytest.ini           # pytest 配置文件
├── core/                # 核心模块
│   ├── __init__.py
│   ├── config.py        # 测试配置类
│   ├── server_manager.py # 服务管理器
│   └── base_test.py     # 测试基类
└── cases/               # 测试用例
    ├── __init__.py
    ├── test_v1_mode.py      # V1 模式测试
    ├── test_w4a8.py         # W4A8 量化测试
    ├── test_vl_model.py     # VL 多模态测试
    └── test_expert_parallel.py  # Expert Parallel 测试
```

## 快速开始

### 运行所有测试

```bash
# 设置环境变量
export MODEL_PATH=/path/to/models
export XPU_ID=0

# 运行所有测试
bash scripts/run_ci_xpu_pytest.sh --all
```

### 运行特定测试

```bash
# 只运行 V1 模式测试
bash scripts/run_ci_xpu_pytest.sh --test v1

# 只运行 W4A8 测试
bash scripts/run_ci_xpu_pytest.sh --test w4a8

# 只运行 VL 模型测试
bash scripts/run_ci_xpu_pytest.sh --test vl

# 只运行 EP4TP4 测试
bash scripts/run_ci_xpu_pytest.sh --test ep4tp4

# 只运行所有 Expert Parallel 测试
bash scripts/run_ci_xpu_pytest.sh --test ep
```

### 直接使用 pytest

```bash
# 运行所有测试
python -m pytest tests/xpu_ci/cases/ --model-path $MODEL_PATH --xpu-id 0

# 使用 marker 过滤测试
python -m pytest tests/xpu_ci/cases/ -m v1_mode --model-path $MODEL_PATH
python -m pytest tests/xpu_ci/cases/ -m w4a8 --model-path $MODEL_PATH
python -m pytest tests/xpu_ci/cases/ -m vl_model --model-path $MODEL_PATH
python -m pytest tests/xpu_ci/cases/ -m expert_parallel --model-path $MODEL_PATH

# 运行特定测试文件
python -m pytest tests/xpu_ci/cases/test_v1_mode.py --model-path $MODEL_PATH
```

## 添加新的测试用例

### 1. 创建新的测试配置

在 `core/config.py` 中添加新的配置工厂方法：

```python
@classmethod
def create_new_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
    """Create new test configuration."""
    port = 8188 + xpu_id * 100
    xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

    return cls(
        name="new_test",
        description="新测试描述",
        server_config=ServerConfig(
            model_path=os.path.join(model_path, "MODEL_NAME"),
            port=port,
            tensor_parallel_size=4,
            # ... 其他配置
        ),
        env_config=EnvironmentConfig(
            xpu_id=xpu_id,
            xpu_visible_devices=xpu_devices,
        ),
    )
```

### 2. 创建新的测试文件

在 `cases/` 目录下创建新的测试文件：

```python
# cases/test_new_feature.py

import pytest
from ..core import TestConfig
from ..core.base_test import TextModelTest

@pytest.mark.new_feature
class TestNewFeature(TextModelTest):
    """新功能测试类"""

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        return TestConfig.create_new_test(model_path, xpu_id)

    def test_new_functionality(self, openai_client, test_config):
        """测试新功能"""
        response = self._chat_completion(
            openai_client,
            messages=[{"role": "user", "content": "测试问题"}],
        )
        # 添加断言...
```

### 3. 注册新的 marker

在 `conftest.py` 中添加新的 marker：

```python
config.addinivalue_line(
    "markers",
    "new_feature: marks tests as new feature tests",
)
```

## 测试标记 (Markers)

| Marker | 描述 |
|--------|------|
| `v1_mode` | V1 模式测试 |
| `w4a8` | W4A8 量化测试 |
| `vl_model` | VL 多模态模型测试 |
| `expert_parallel` | 所有 Expert Parallel 测试 |
| `ep4tp4` | EP4TP4 测试 |
| `ep4tp1` | EP4TP1 测试 |
| `all2all` | all2all 通信测试 |

## 核心类说明

### ServerConfig

服务器配置类，包含启动 FastDeploy API 服务器所需的所有参数。

### EnvironmentConfig

环境配置类，管理 XPU 设备和 BKCL 相关的环境变量。

### TestConfig

测试配置类，组合服务器配置和环境配置，并提供工厂方法创建预定义的测试配置。

### ServerManager

服务管理器类，负责服务的启动、健康检查和停止。支持上下文管理器使用方式。

### BaseXPUTest

测试基类，提供：
- 服务管理 fixture
- OpenAI 客户端 fixture
- 通用的聊天完成方法
- 响应验证方法

### TextModelTest / VLModelTest

针对不同模型类型的测试基类，提供特定的测试方法。

## 与旧版脚本的对比

| 特性 | 旧版 (run_ci_xpu.sh) | 新版 (pytest) |
|------|---------------------|---------------|
| 测试组织 | 单一脚本 | 模块化文件 |
| 代码复用 | 复制粘贴 | 继承基类 |
| 配置管理 | 硬编码 | 配置类 |
| 扩展性 | 修改脚本 | 添加测试类 |
| 测试选择 | 全部运行 | marker 过滤 |
| 报告 | 手动打印 | pytest 报告 |
| 超时控制 | 手动实现 | pytest-timeout |
