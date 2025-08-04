from setuptools import setup

setup(
    name="fastdeploy-plugins",
    version="0.1",
    packages=["fd_add_dummy_model"],
    entry_points={
        "fastdeploy.model_register_plugins": [
            "fd_add_dummy_model = fd_add_dummy_model:register",
        ],
        # 'fastdeploy.model_runner_plugins': [
        #     "model_runner = model_runner:get_runner"
        # ]
    },
)
