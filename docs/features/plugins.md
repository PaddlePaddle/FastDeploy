[简体中文](../zh/features/plugins.md)

# FastDeploy Plugin Mechanism Documentation

FastDeploy supports a plugin mechanism that allows users to extend functionality without modifying the core code. Plugins are automatically discovered and loaded through Python's `entry_points` mechanism.

## How Plugins Work

Plugins are essentially registration functions that are automatically called when FastDeploy starts. The system uses the `load_plugins_by_group` function to ensure that all processes (including child processes in distributed training scenarios) have loaded the required plugins before official operations begin.

## Plugin Discovery Mechanism

FastDeploy uses Python's `entry_points` mechanism to discover and load plugins. Developers need to register their plugins in the specified entry point group in their project.

### Example: Creating a Plugin

#### 1. How Plugin Work

Assuming you have a custom model class `MyModelForCasualLM` and a pretrained class `MyPretrainedModel`, you can write the following registration function:

```python
# File: fd_add_dummy_model/__init__.py or fd_add_dummy_model/register.py
from fastdeploy.model_executor.models.model_base import ModelRegistry
from my_custom_model import MyModelForCasualLM, MyPretrainedModel
from fastdeploy.config import ErnieArchitectures

def register():
    if "MyModelForCasualLM" not in ModelRegistry.get_supported_archs():
        if MyModelForCasualLM.name().startswith("Ernie"):
            ErnieArchitectures.register_ernie_model_arch(MyModelForCasualLM)
        ModelRegistry.register_model_class(MyModelForCasualLM)
        ModelRegistry.register_pretrained_model(MyPretrainedModel)
```
Assuming you have a custom model_runner class `MyModelRunner`, you can write the following registration function:
```python
# File: fd_add_dummy_model_runner/__init__.py
from .my_model_runner import MyModelRunner

def get_runner():
    return MyModelRunner
```

#### 2. Register Plugin in `setup.py`

```python
# setup.py
from setuptools import setup

setup(
    name="fastdeploy-plugins",
    version="0.1",
    packages=["fd_add_dummy_model", "fd_add_dummy_model_runner"],
    entry_points={
        "fastdeploy.model_register_plugins": [
            "fd_add_dummy_model = fd_add_dummy_model:register",
        ],
        "fastdeploy.model_runner_plugins": [
            "model_runner = fd_add_dummy_model:get_runner"
        ],
    },
)
```

## Plugin Structure

Plugins consist of three components:

| Component | Description |
|-----------|-------------|
| **Plugin Group** | The functional group to which the plugin belongs, for example:<br> - `fastdeploy.model_register_plugins`: for model registration<br> - `fastdeploy.model_runner_plugins`: for model runner registration<br> Users can customize groups as needed. |
| **Plugin Name** | The unique identifier for each plugin (e.g., `fd_add_dummy_model`), which can be controlled via the `FD_PLUGINS` environment variable to determine whether to load the plugin. |
| **Plugin Value** | Format is `module_name:function_name`, pointing to the entry function that executes the registration logic. |

## Controlling Plugin Loading Behavior

By default, FastDeploy loads all registered plugins. To load only specific plugins, you can set the environment variable:

```bash
export FD_PLUGINS=fastdeploy-plugins
```

Multiple plugin names can be separated by commas:

```bash
export FD_PLUGINS=plugin_a,plugin_b
```

## Reference Example

Please refer to the example plugin implementation in the project directory:
```
./test/plugins/
```

It contains a complete plugin structure and `setup.py` configuration example.

## Summary

Through the plugin mechanism, users can easily add custom models or functional modules to FastDeploy without modifying the core source code. This not only enhances system extensibility but also facilitates third-party developers in extending functionality.

For further plugin development, please refer to the `model_registry` and `plugin_loader` modules in the FastDeploy source code.
