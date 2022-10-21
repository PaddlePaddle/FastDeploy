# PP-Tracking Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成PP-Matting在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/tracking/pptracking/python

# 下载PP-Tracking模型文件和测试视频
wget https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
tar -xvf fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4
# CPU推理
python infer.py --model fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video person.mp4 --device cpu
# GPU推理
python infer.py --model fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video person.mp4  --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video person.mp4  --device gpu --use_trt True
```

## PP-Tracking Python接口

```python
fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PP-Tracking模型加载和初始化，其中model_file, params_file以及config_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict函数

> ```python 
> PPTracking.predict(frame)
> ```
>
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **frame**(np.ndarray): 输入数据，注意需为HWC，BGR格式,frame为视频帧如：_,frame=cap.read()得到

> **返回**
>
> > 返回`fastdeploy.vision.MOTResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果



## 其它文档

- [PP-Matting 模型介绍](..)
- [PP-Matting C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
