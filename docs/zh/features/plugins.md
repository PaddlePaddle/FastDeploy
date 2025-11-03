[English](../../features/plugins.md)

# FastDeploy 插件机制说明文档

FastDeploy 支持插件机制，允许用户在不修改核心代码的前提下扩展功能。插件通过 Python 的 `entry_points` 机制实现自动发现与加载。

## 插件工作原理

插件本质上是在 FastDeploy 启动时被自动调用的注册函数。系统使用 `load_plugins_by_group` 函数确保所有进程（包括分布式训练场景下的子进程）在正式运行前都已加载所需的插件。

## 插件发现机制

FastDeploy 利用 Python 的 `entry_points` 机制来发现并加载插件。开发者需在自己的项目中将插件注册到指定的 entry point 组中。

### 示例：创建一个插件

#### 1. 编写插件逻辑

假设你有一个自定义模型类 `MyModelForCasualLM` 和预训练类 `MyPretrainedModel`，你可以编写如下注册函数：

```python
# 文件：fd_add_dummy_model/__init__.py
from fastdeploy.model_executor.models.model_base import ModelRegistry
from my_custom_model import MyModelForCasualLM, MyPretrainedModel

def register():
    if "MyModelForCasualLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model_class(MyModelForCasualLM)
        ModelRegistry.register_pretrained_model(MyPretrainedModel)
```

#### 2. 注册插件到 `setup.py`

```python
# setup.py
from setuptools import setup

setup(
    name="fastdeploy-plugins",
    version="0.1",
    packages=["fd_add_dummy_model"],
    entry_points={
        "fastdeploy.model_register_plugins": [
            "fd_add_dummy_model = fd_add_dummy_model:register",
        ],
    },
)
```

## 插件结构说明

插件由三部分组成：

| 组件 | 说明 |
|------|------|
| **插件组（Group）** | 插件所属的功能分组，例如：<br> - `fastdeploy.model_register_plugins`: 用于注册模型<br> - `fastdeploy.model_runner_plugins`: 用于注册模型运行器<br> 用户可根据需要自定义分组。 |
| **插件名（Name）** | 每个插件的唯一标识名（如 `fd_add_dummy_model`），可通过环境变量 `FD_PLUGINS` 控制是否加载该插件。 |
| **插件值（Value）** | 格式为 `模块名:函数名`，指向实际执行注册逻辑的入口函数。 |

## 控制插件加载行为

默认情况下，FastDeploy 会加载所有已注册的插件。若只想加载特定插件，可以设置环境变量：

```bash
export FD_PLUGINS=fastdeploy-plugins
```

多个插件名之间可以用逗号分隔：

```bash
export FD_PLUGINS=plugin_a,plugin_b
```

## 参考示例

请参见项目目录下的示例插件实现：
```
./test/plugins/
```

其中包含完整的插件结构和 `setup.py` 配置示例。

## 总结

通过插件机制，用户可以轻松地为 FastDeploy 添加自定义模型或功能模块，而无需修改核心源码。这不仅提升了系统的可扩展性，也方便了第三方开发者进行功能拓展。

如需进一步开发插件，请参考 FastDeploy 源码中的 `model_registry` 和 `plugin_loader` 模块。
