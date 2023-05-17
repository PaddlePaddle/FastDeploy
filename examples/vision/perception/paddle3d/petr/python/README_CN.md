[English](README.md) | 简体中文

# Petr Python 部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl 包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供 `infer.py` 快速完成 Petr 在 CPU/GPU上部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/vision/paddle3d/petr/python

wget https://bj.bcebos.com/fastdeploy/models/petr.tar.gz
tar -xf petr.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/petr_test.png

# CPU推理
python infer.py --model petr --image petr_test.png --device cpu
# GPU推理
python infer.py --model petr --image petr_test.png --device gpu
```


## Petr Python接口

```python
fastdeploy.vision.perception.Petr(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

Petr模型加载和初始化。

**参数**
> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 配置文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict 函数

> ```python
> Petr.predict(image_data)
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

- [Petr 模型介绍](..)
- [Petr C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
