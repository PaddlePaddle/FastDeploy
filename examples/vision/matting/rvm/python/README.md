# RobustVideoMatting Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成RobustVideoMatting在CPU/GPU。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/matting/rvm/python

# 下载RobustVideoMatting模型文件和测试图片以及视频
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_fp32.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/video.mp4

# CPU推理
## 图片
python infer.py --model rvm_mobilenetv3_fp32.onnx --image matting_input.jpg --bg matting_bgr.jpg --device cpu
## 视频
python infer.py --model rvm_mobilenetv3_fp32.onnx --video video.mp4 --bg matting_bgr.jpg --device cpu
# GPU推理
## 图片
python infer.py --model rvm_mobilenetv3_fp32.onnx --image matting_input.jpg --bg matting_bgr.jpg --device gpu
## 视频
python infer.py --model rvm_mobilenetv3_fp32.onnx --video video.mp4 --bg matting_bgr.jpg --device gpu
```

运行完成可视化结果如下图所示
<div width="1240">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/19977378/196653716-f7043bd5-dfc2-4e7d-be0f-e12a6af4c55b.gif">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/19977378/196654529-866bff5d-47a2-4584-9627-39b587799228.gif">
</div>

## RobustVideoMatting Python接口

```python
fd.vision.matting.RobustVideoMatting(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

RobustVideoMatting模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX

### predict函数

> ```python
> RobustVideoMatting.predict(input_image)
> ```
>
> 模型预测结口，输入图像直接输出抠图结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.MattingResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果



## 其它文档

- [RobustVideoMatting 模型介绍](..)
- [RobustVideoMatting C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
