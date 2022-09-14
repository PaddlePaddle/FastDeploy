### Demo 运行步骤：

1. 使用模型为百度飞桨 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 的 Picodet模型。

2. 先下在本目录下下载[模型](https://paddlelite-demo.bj.bcebos.com/onnx_model/picodet_fp32.tar.gz)，然后当前目录直接解压，会生成 model 文件夹。

3. 将导出的onnx模型复制到该demo目录下，执行命令:

   ```
   python picodet_demo.py
   ```

   

### 注意事项：

1. 使用模型为百度飞桨 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 的 Picodet模型

2. 使用 [paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX) 将 paddle 模型转换为 onnx 模型

3. 切换成自己训练的模型时，请注意对齐,NMS_THRESH等后处理参数，否则会导致后处理解析出错。

4. 使用的飞桨 picodet 模型为无后处理的版本，与 yolov5 一样，后处理都在 python 代码中实现。

5. 目前 rknn-toolkit2 暂不支持对 picodet 的量化，会在后续版本中更新。
