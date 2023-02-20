# FastDeploy C++ Benchmarks

在跑benchmark前，需确认以下两个步骤

* 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../docs/cn/build_and_install/download_prebuilt_libraries.md)
* 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../docs/cn/build_and_install/download_prebuilt_libraries.md)

FastDeploy 目前支持多种推理后端，下面以 PaddleYOLOv8 为例，跑出多后端在 CPU/GPU 对应 benchmark 数据
