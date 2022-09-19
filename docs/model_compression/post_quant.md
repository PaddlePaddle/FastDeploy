# FastDeploy 支持量化模型部署
量化是一种流行的模型压缩方法，量化后的模型拥有更小的体积和更快的推理速度. FastDeploy支持部署量化后的模型，帮助用户实现推理加速.

## 1. FastDeploy 多个引擎支持量化模型部署

当前，FastDeploy中多个推理后端可以在不同硬件上支持量化模型的部署. 支持情况如下:

| 硬件/推理后端 | ONNXRuntime | Paddle Inference | TensorRT |
| :-----------| :--------   | :--------------- | :------- |
|   CPU       |  支持        |  支持            |          |  
|   GPU       |             |                  | 支持      |


## 2. 用户如何量化模型

### 量化方式
用户可以通过PaddleSlim来量化模型, 量化主要有量化训练和离线量化两种方式, 量化训练通过模型训练来获得量化模型, 离线量化不需要模型训练即可完成模型的量化. FastDeploy 对两种方式产出的量化模型均能部署.
两种方法的主要对比如下表所示:
| 量化方法 | 量化过程耗时 | 量化模型精度 | 模型体积 | 推理速度 |
| :-----------| :--------| :-------| :------- | :------- |
|   离线量化      |  无需训练，耗时短 |  比量化训练稍低       | 两者一致   | 两者一致   |  
|   量化训练      |  需要训练，耗时高 |  较未量化模型有少量损失 | 两者一致   |两者一致   |  

### 用户使用fdquant命令量化模型
Fastdeploy 为用户提供了一键离线量化的功能，请参考如下文档进行模型量化.
- [FastDeploy 一键模型量化](../../quant_tools/)
当用户获得产出的量化模型之后，即可以使用FastDeploy来部署量化模型.

## 3. FastDeploy 部署量化模型
用户只需要简单地传入量化后的模型路径及相应参数，即可以使用FastDeploy进行部署.
具体请用户参考示例文档:
- [YOLOv5s 量化模型Python部署](../../examples/slim/yolov5s/python/)
- [YOLOv5s 量化模型C++部署](../../examples/slim/yolov5s/cpp/)
- [YOLOv6s 量化模型Python部署](../../examples/slim/yolov6s/python/)
- [YOLOv6s 量化模型C++部署](../../examples/slim/yolov6s/cpp/)
- [YOLOv7 量化模型Python部署](../../examples/slim/yolov7/python/)
- [YOLOv7 量化模型C++部署](../../examples/slim/yolov7/cpp/)
