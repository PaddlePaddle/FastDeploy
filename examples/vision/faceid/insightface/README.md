English | [简体中文](README_CN.md)
# InsightFace Ready-to-deploy Model

- [InsightFace](https://github.com/deepinsight/insightface/commit/babb9a5)
  - （1）The *.pt provided by the [Official Library](https://github.com/deepinsight/insightface/) can be deployed after the [Export ONNX Model](#export-onnx-model)；
  - （2）As for InsightFace model trained on customized data, please follow the operations guidelines in [Export ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B) to complete the deployment.


## List of Supported Models
Now FastDeploy supports the deployment of the following models
- ArcFace
- CosFace
- PartialFC
- VPL


##  Export ONNX Model
Taking ArcFace as an example:
Visit [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) official github repository, follow the guidelines to download pt model files, and employ `torch2onnx.py` to get the file in `onnx` format.

* Download ArcFace model files
  ```bash
  Link: https://pan.baidu.com/share/init?surl=CL-l4zWqsI1oDuEEYVhj-g code: e8pw  
  ```

* Export files in onnx format
  ```bash
  PYTHONPATH=. python ./torch2onnx.py ms1mv3_arcface_r100_fp16/backbone.pth --output ms1mv3_arcface_r100.onnx --network r100 --simplify 1
  ```

## Download Pre-trained ONNX Model

For developers' testing, models exported by InsightFace are provided below. Developers can download and use them directly. (The accuracy of the models in the table is sourced from the official library) The accuracy metric is sourced from the model description in InsightFace. Refer to the introduction in InsightFace for more details.

| Model                                                                                         | Size    | Accuracy (AgeDB_30) |
|:-------------------------------------------------------------------------------------------|:------|:--------------|
| [CosFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r18.onnx)       | 92MB  | 97.7          |
| [CosFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r34.onnx)       | 131MB | 98.3          |
| [CosFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r50.onnx)       | 167MB | 98.3          |
| [CosFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/glint360k_cosface_r100.onnx)     | 249MB | 98.4          |
| [ArcFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r18.onnx)          | 92MB  | 97.7          |
| [ArcFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r34.onnx)          | 131MB | 98.1          |
| [ArcFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r50.onnx)          | 167MB | -             |
| [ArcFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx)        | 249MB | 98.4          |
| [ArcFace-r100_lr0.1](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_r100_lr01.onnx)     | 249MB | 98.4          |
| [PartialFC-r34](https://bj.bcebos.com/paddlehub/fastdeploy/partial_fc_glint360k_r50.onnx)  | 167MB | -             |
| [PartialFC-r50](https://bj.bcebos.com/paddlehub/fastdeploy/partial_fc_glint360k_r100.onnx) | 249MB | -             |




## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [InsightFace CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5) 
