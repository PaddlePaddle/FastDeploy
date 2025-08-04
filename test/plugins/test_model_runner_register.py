from fastdeploy.plugins import load_model_runner_plugins

runner_class = load_model_runner_plugins()

# 创建 runner 实例
if not callable(runner_class):
    print("The returned runner constructor is not callable.")

device_id = 7
runner_instance = runner_class(device_id)
runner_instance.test()
print(f"Model runner initialized successfully on device {device_id}.")
