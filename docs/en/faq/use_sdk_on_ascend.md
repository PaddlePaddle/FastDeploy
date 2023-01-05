[简体中文](../../cn/faq/use_sdk_on_ascend.md) | English


# # Linux deployment with C++ on Huawei Ascend

After the deployment example is compiled, we need to import some environment variables to initialize the deployment environment before running the program, because we need to use the Huawei Ascend toolkit.
Users can use the following script (located in the directory of the compiled FastDeploy library) to initialize the Huawei Ascend deployment environment.


```
# The path to our default Ascend Toolkit is as follows,
# HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
# HUAWEI_ASCEND_DRIVER_PATH="/usr/local/Ascend/driver"
# If the user's installation directory is different from this, you need to export it manually first.
# export HUAWEI_ASCEND_TOOLKIT_HOME="Your_ascend_toolkit_path"
# export HUAWEI_ASCEND_DRIVER_PATH="Your_ascend_driver_path"

source fastdeploy-ascend/fastdeploy_init.sh
```

Note that this command only takes effect in the current command environment after execution (switching to a new terminal window, or closing the window and reopening it will not work), if you need to keep it in effect on the system, add these environment variables to the `~/.bashrc` file.
