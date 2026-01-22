# XPU CI 测试框架

基于pytest的XPU硬件CI测试框架,用于自动化测试FastDeploy在XPU硬件上的各种配置和模型。

## 目录结构

```
tests/xpu_ci/
├── 4cards_cases     # 使用4张卡的case
│   ├── test_ep4tp1_online.py
│   ├── test_ep4tp4_all2all.py
│   ├── test_ep4tp4_online.py
│   ├── test_logprobs_21b_tp4.py
│   ├── test_mtp.py
│   ├── test_pd_03b_tp1.py
│   ├── test_pd_21b_tp2.py
│   ├── test_v1_mode.py
│   ├── test_vl_model.py
│   └── test_w4a8.py
├── 8cards_cases    # 使用8张卡的case
│   ├── test_pd_21b_ep4tp1.py
│   ├── test_pd_21b_ep4tp4.py
│   └── test_pd_p_tp4ep4_d_tp1ep4.py
├── conftest.py
└── README.md

## 使用方法

### 运行所有测试

```bash
# 设置环境变量
export XPU_ID=0  # 或 1
export MODEL_PATH=/path/to/models
# 注意 需要设置PYTHONPATH环境变量,否则CI脚本导入模块会失败
export PYTHONPATH=$(pwd)/tests/xpu_ci:$PYTHONPATH
# 运行CI测试
bash scripts/run_xpu_ci_pytest.sh
```

### 运行单个测试

```bash
# 进入项目根目录
cd /path/to/FastDeploy

# 设置环境变量
export XPU_ID=0
export MODEL_PATH=/path/to/models
# 注意 需要设置PYTHONPATH环境变量,否则CI脚本导入模块会失败
export PYTHONPATH=$(pwd)/tests/xpu_ci:$PYTHONPATH
# 运行单个测试
python -m pytest -v -s tests/xpu_ci/4cards_cases/test_ep4tp1_online.py

```

### 运行指定的测试

```bash
# 运行多个测试
# 设置环境变量
export XPU_ID=0
export MODEL_PATH=/path/to/models
# 注意 需要设置PYTHONPATH环境变量,否则CI脚本导入模块会失败
export PYTHONPATH=$(pwd)/tests/xpu_ci:$PYTHONPATH

python -m pytest -v -s \
    tests/xpu_ci/4cards_cases/test_v1_mode.py \
    tests/xpu_ci/4cards_cases/test_w4a8.py

# 使用pytest的过滤功能
python -m pytest -v -s -k "v1_mode or w4a8" tests/xpu_ci/
```

## 添加新的测试Case

### 步骤1: 创建新的测试文件

在 `tests/xpu_ci/` 对应卡数（目前有8卡或者4卡）目录下创建新的测试文件,文件名必须以 `test_` 开头,例如 `test_new_feature.py`

### 步骤2: 编写测试代码

参考现有的测试case,复制一个最相似的测试文件作为模板。基本结构如下:

```python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ... (许可证声明)

"""
测试说明 - 简短描述这个测试的目的

测试配置:
- 模型: 模型名称
- 量化: 量化方式
- 其他重要配置
"""

import os
import pytest
import openai
from conftest import (
    get_port_num,
    get_model_path,
    start_server,
    print_logs_on_failure,
    xpu_env,
)


def test_new_feature(xpu_env):
    """新功能测试"""

    print("\n============================开始新功能测试!============================")

    # 获取配置
    port_num = get_port_num()
    model_path = get_model_path()

    # 构建服务器启动参数
    server_args = [
        "--model", f"{model_path}/YOUR_MODEL_NAME",
        "--port", str(port_num),
        # ... 其他参数
    ]

    # 启动服务器
    if not start_server(server_args):
        pytest.fail("服务启动失败")

    # 执行测试
    try:
        ip = "0.0.0.0"
        client = openai.Client(
            base_url=f"http://{ip}:{port_num}/v1",
            api_key="EMPTY_API_KEY"
        )

        # 调用API进行测试
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": "你好,你是谁?"},
            ],
            temperature=1,
            top_p=0,
            max_tokens=64,
            stream=False,
        )

        print(f"\n模型回复: {response.choices[0].message.content}")

        # 验证响应
        assert "预期的关键词" in response.choices[0].message.content

        print("\n新功能测试通过!")

    except Exception as e:
        print(f"\n新功能测试失败: {str(e)}")
        print_logs_on_failure()
        pytest.fail(f"新功能测试失败: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

### 步骤3: 添加到CI流程

 `scripts/run_xpu_ci_pytest.sh`会自动扫描 tests/xpu_ci/ 对应卡数目录下 test_ 开头的测试文件进行测试

### 步骤4: 测试验证

```bash
# 先单独运行新的测试case,确保能够正常工作
export PYTHONPATH=$(pwd)/tests/xpu_ci:$PYTHONPATH
python -m pytest -v -s tests/xpu_ci/test_new_feature.py

# 然后运行完整的CI流程
export PYTHONPATH=$(pwd)/tests/xpu_ci:$PYTHONPATH
bash scripts/run_xpu_ci_pytest.sh
```

## 通用函数说明

在 `conftest.py` 中提供了以下通用函数,可以在测试中直接使用:

### 基础配置函数

- `get_xpu_id()` - 获取XPU_ID环境变量
- `get_port_num()` - 根据XPU_ID计算端口号
- `get_model_path()` - 获取MODEL_PATH环境变量

### 进程管理函数

- `stop_processes()` - 停止所有相关进程
- `cleanup_resources()` - 清理资源(log目录、core文件、消息队列)

### 服务器管理函数

- `start_server(server_args, wait_before_check=60)` - 启动API服务器
  - `server_args`: 服务器启动参数列表
  - `wait_before_check`: 启动后等待多少秒再进行健康检查
  - 返回: bool,服务是否启动成功

- `wait_for_health_check(timeout=900, interval=10)` - 等待服务健康检查通过
  - `timeout`: 超时时间(秒)
  - `interval`: 检查间隔(秒)
  - 返回: bool,服务是否启动成功

### 日志函数

- `print_logs_on_failure()` - 失败时打印日志(server.log和workerlog.0)

### EP并行测试函数

- `setup_ep_env()` - 设置EP(Expert Parallel)相关环境变量
  - 返回: dict,原始环境变量值,用于后续恢复

- `restore_env(original_values)` - 恢复环境变量
  - `original_values`: setup_ep_env()返回的原始环境变量值

- `download_and_build_xdeepep()` - 下载并编译xDeepEP(用于EP并行测试)
  - 返回: bool,是否成功

### Pytest Fixture

- `xpu_env` - 设置XPU环境变量的fixture
  - 自动设置XPU_VISIBLE_DEVICES
  - 测试结束后自动停止服务
  - 使用方法: 在测试函数参数中声明即可

## 测试Case模板

### 普通测试模板

用于不需要EP并行的测试:

```python
def test_example(xpu_env):
    """示例测试"""
    print("\n============================开始示例测试!============================")

    port_num = get_port_num()
    model_path = get_model_path()

    server_args = [
        "--model", f"{model_path}/YOUR_MODEL",
        "--port", str(port_num),
        # 添加其他参数...
    ]

    if not start_server(server_args):
        pytest.fail("服务启动失败")

    try:
        # 执行测试逻辑
        client = openai.Client(base_url=f"http://0.0.0.0:{port_num}/v1", api_key="EMPTY_API_KEY")
        response = client.chat.completions.create(...)
        assert "预期结果" in response.choices[0].message.content
        print("\n示例测试通过!")
    except Exception as e:
        print_logs_on_failure()
        pytest.fail(f"测试失败: {str(e)}")
```

### EP并行测试模板

用于需要EP并行的测试:

```python
def test_ep_example(xpu_env):
    """EP并行示例测试"""
    print("\n============================开始EP并行示例测试!============================")

    original_env = setup_ep_env()

    try:
        port_num = get_port_num()
        model_path = get_model_path()

        server_args = [
            "--model", f"{model_path}/YOUR_MODEL",
            "--enable-expert-parallel",
            # 添加其他参数...
        ]

        if not start_server(server_args):
            pytest.fail("服务启动失败")

        # 执行测试逻辑
        client = openai.Client(base_url=f"http://0.0.0.0:{port_num}/v1", api_key="EMPTY_API_KEY")
        response = client.chat.completions.create(...)
        assert "预期结果" in response.choices[0].message.content
        print("\nEP并行示例测试通过!")
    except Exception as e:
        print_logs_on_failure()
        pytest.fail(f"测试失败: {str(e)}")
    finally:
        restore_env(original_env)
```

## 常见问题

### 1. 如何调试单个测试?

```bash
# 使用pytest的调试选项
python -m pytest -v -s --pdb tests/xpu_ci/test_xxx.py

# 或者直接在代码中添加断点
import pdb; pdb.set_trace()
```

### 2. 如何查看服务器日志?

测试失败时会自动打印 `server.log` 和 `log/workerlog.0` 的内容。
你也可以在测试运行时手动查看:

```bash
tail -f server.log
tail -f log/workerlog.0
```

### 3. 如何跳过某个测试?

```python
@pytest.mark.skip(reason="暂时跳过此测试")
def test_example(xpu_env):
    pass
```

### 4. 如何添加超时控制?

```python
@pytest.mark.timeout(300)  # 5分钟超时
def test_example(xpu_env):
    pass
```

## 与旧版本的对比

### 旧版本 (run_ci_xpu.sh)

- 所有测试逻辑都在一个大的shell脚本中
- 代码重复率高(每个测试都重复启动服务、健康检查等逻辑)
- 难以维护和扩展
- 添加新测试需要修改主脚本

### 新版本 (基于pytest)

- 每个测试case独立成文件
- 通用逻辑抽象到conftest.py中
- 易于维护和扩展
- 添加新测试只需新建文件,无需修改主脚本(只需在run_xpu_ci_pytest.sh中添加文件名)
- 支持pytest的所有功能(参数化、fixture、插件等)

## 注意事项

1. **环境变量**: 确保设置了 `XPU_ID` 和 `MODEL_PATH` 环境变量
2. **端口冲突**: 每个测试会自动根据XPU_ID分配不同的端口,避免冲突
3. **资源清理**: 使用 `xpu_env` fixture会自动清理资源,无需手动清理
4. **测试顺序**: pytest会按文件名顺序执行测试,可以通过pytest参数调整
5. **日志输出**: 使用 `-s` 参数可以看到print输出,方便调试

## 参考资料

- [pytest官方文档](https://docs.pytest.org/)
- [pytest fixture文档](https://docs.pytest.org/en/stable/fixture.html)
- [FastDeploy文档](https://github.com/PaddlePaddle/FastDeploy)
