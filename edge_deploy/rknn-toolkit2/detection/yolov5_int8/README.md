### Demo 运行步骤：

1. 使用yolov5官方仓库导出模型，链接：https://github.com/ultralytics/yolov5。该demo创建时yolov5的最新节点的commit id为:c5360f6e7009eb4d05f14d1cc9dae0963e949213

2. 先下在本目录下下载[模型](https://paddlelite-demo.bj.bcebos.com/onnx_model/yolov5.tar.gz)，然后当前目录直接解压，会生成 model 文件夹。

3. 将导出的onnx模型复制到该demo目录下，执行命令:

   ```
   python picodet_demo.py
   ```

   

### 注意事项：

1. 切换成自己训练的模型时，请注意对齐anchor,BOX_THESH,NMS_THRESH等后处理参数，否则会导致后处理解析出错。

2. 本 demo 提供瑞芯微官方优化的 yolov5s，以及百度飞桨 Paddle 导出的yolov5 s/m/l。

3. 如果使用瑞芯微官方版本或者 yolov5官方版本导出的模型，为兼容 rknpu/rknn/rknn_api/examples/rknn_yolov5_demo 示例, 测试代码导出RKNN模型时指定了输出节点['378', '439', '500']。这三个输出节点分别是原模型第2、3、4输出节点的前一个卷积(去掉卷积后面的reshape和transpose), 对应输出的shape是[1,255,80,80],[1,255,40,40],[1,255,20,20]。实际部署代码请灵活处理。该demo只是提供了其中一种方式，并不是唯一的。对于其他模型，如yolov5m, yolov5l 等，请使用工具自行查找对应输出节点的名称，这里推荐使用可视化工具 netron。

3. 开启量化时，rknn.config中的quantize_input_node 被设置为True。若不启用，yolov5模型由于模型头部是slice算子而非常规算子，这种情况有可能导致yolov5模型的输入层没有被转为量化算子，导致RKNN模型在板端使用 RKNN C api 部署时，input_set耗时异常。该参数具体作用请参考RKNN Toolkit用户手册7.2节。
