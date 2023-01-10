[English](README.md) | 简体中文
# PP-MSVSR Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成PP-MSVSR在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/sr/ppmsvsr/python

# 下载VSR模型文件和测试视频
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-MSVSR_reds_x4.tar
tar -xvf PP-MSVSR_reds_x4.tar
wget https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4
# CPU推理
python infer.py --model PP-MSVSR_reds_x4 --video vsr_src.mp4 --frame_num 2 --device cpu
# GPU推理
python infer.py --model PP-MSVSR_reds_x4 --video vsr_src.mp4 --frame_num 2 --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model PP-MSVSR_reds_x4 --video vsr_src.mp4 --frame_num 2 --device gpu --use_trt True
```

## VSR Python接口

```python
fd.vision.sr.PPMSVSR(model_file, params_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PP-MSVSR模型加载和初始化，其中model_file和params_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict函数

> ```python
> PPMSVSR.predict(frames)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **frames**(list[np.ndarray]): 输入数据，注意需为HWC，BGR格式, frames为视频帧序列

> **返回** list[np.ndarray] 为超分后的视频帧序列


## 其它文档

- [PP-MSVSR 模型介绍](..)
- [PP-MSVSR C++部署](../cpp)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
