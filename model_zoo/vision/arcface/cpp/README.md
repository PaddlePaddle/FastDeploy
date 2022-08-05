# 编译ArcFace示例

## 0. 简介
当前支持模型版本为：[ArcFace CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5)

## 1. 下载和解压预测库
```bash
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.3.0.tgz
tar xvf fastdeploy-linux-x64-0.3.0.tgz
```

## 1. 编译示例代码
```bash
mkdir build & cd build
cmake ..
make -j
```

## 3. 特别说明  
fastdeploy支持 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition) 的人脸识别模块recognition中大部分模型的部署，包括ArcFace、CosFace、Partial FC、VPL等，由于用法类似，这里仅用ArcFace来演示部署流程。所有支持的模型结构，可参考 [ArcFace API文档](../api.md).

## 4. 获取ONNX文件

访问[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)官方github库，按照指引下载安装，下载pt模型文件，利用 `torch2onnx.py` 得到`onnx`格式文件。

* 下载ArcFace模型文件
  ```
  Link: https://pan.baidu.com/share/init?surl=CL-l4zWqsI1oDuEEYVhj-g code: e8pw  
  ```

* 导出onnx格式文件
  ```bash
  PYTHONPATH=. python ./torch2onnx.py partial_fc/pytorch/ms1mv3_arcface_r100_fp16/backbone.pth --output ms1mv3_arcface_r100.onnx --network r100 --simplify 1
  ```
* 移动onnx文件到model_zoo/arcface的目录
  ```bash
  cp PATH/TO/ms1mv3_arcface_r100.onnx PATH/TO/model_zoo/vision/arcface/
  ```


## 5. 准备测试图片
准备3张仅包含人脸的测试图片，命名为face_recognition_*.jpg，并拷贝到可执行文件所在的目录，比如
```bash
face_recognition_0.png  # 0,1 同一个人
face_recognition_1.png
face_recognition_2.png  # 0,2 不同的人
```

## 6. 执行
```bash
./arcface_demo
```

执行完成后会输出检测结果如下
```
FaceRecognitionResult: [Dim(512), Min(-0.141219), Max(0.121645), Mean(-0.003172)]
FaceRecognitionResult: [Dim(512), Min(-0.117939), Max(0.141897), Mean(0.000407)]
FaceRecognitionResult: [Dim(512), Min(-0.124471), Max(0.112567), Mean(-0.001320)]
Cosine 01:  0.7211584683376316
Cosine 02:  -0.06262668682788906
```
