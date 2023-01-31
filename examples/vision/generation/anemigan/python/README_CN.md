[English](README.md) | 简体中文
# AnimeGAN Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成AnimeGAN在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/generation/anemigan/python
# 下载准备好的测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/style_transfer_testimg.jpg

# CPU推理
python infer.py --model animegan_v1_hayao_60  --image style_transfer_testimg.jpg  --device cpu
# GPU推理
python infer.py --model animegan_v1_hayao_60 --image style_transfer_testimg.jpg  --device gpu
```

## AnimeGAN Python接口

```python
fd.vision.generation.AnimeGAN(model_file, params_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

AnimeGAN模型加载和初始化，其中model_file和params_file为用于Paddle inference的模型结构文件和参数文件。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式


### predict函数

> ```python
> AnimeGAN.predict(input_image)
> ```
>
> 模型预测入口，输入图像输出风格迁移后的结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回** np.ndarray, 风格转换后的图像，BGR格式

### batch_predict函数
> ```python
> AnimeGAN.batch_predict函数(input_images)
> ```
>
> 模型预测入口，输入一组图像并输出风格迁移后的结果。
>
> **参数**
>
> > * **input_images**(list(np.ndarray)): 输入数据，一组图像数据，注意需为HWC，BGR格式

> **返回** list(np.ndarray), 风格转换后的一组图像，BGR格式

## 其它文档

- [风格迁移 模型介绍](..)
- [C++部署](../cpp)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
