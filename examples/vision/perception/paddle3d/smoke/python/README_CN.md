[English](README.md) | 简体中文

# Smoke Python 部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl 包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供 `infer.py` 快速完成 Smoke 在 CPU/GPU上部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/vision/paddle3d/smoke/python

wget https://bj.bcebos.com/fastdeploy/models/smoke.tar.gz
tar -xf smoke.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

# CPU推理
python infer.py --model smoke --image smoke_test.png --device cpu
# GPU推理
python infer.py --model smoke --image smoke_test.png --device gpu
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/30516196/230387825-53ac0a09-4137-4e49-9564-197cbc30ff08.png">

## Smoke Python接口

```python
fastdeploy.vision.detection.Smoke(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

Smoke模型加载和初始化。

**参数**
> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 配置文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict 函数

> ```python
> Smoke.predict(image_data)
> ```
>
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.PerceptionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)


## 其它文档

- [Smoke 模型介绍](..)
- [Smoke C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
