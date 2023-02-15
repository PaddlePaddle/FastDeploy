[English](faq.md)| 简体中文
# 视觉模型预测结果常见问题

## 将视觉模型预测结果转换为numpy格式

这里以[SegmentationResult](./segmentation_result_CN.md)为例，展示如何抽取SegmentationResult中的label_map或者score_map来转为numpy格式，同时也可以利用已有数据new SegmentationResult结构体
``` python
import fastdeploy as fd
import cv2
import numpy as np

model = fd.vision.segmentation.PaddleSegModel(
    model_file, params_file, config_file)
im = cv2.imread(image)
result = model.predict(im)
# convert label_map and score_map to numpy format
numpy_label_map = np.array(result.label_map)
numpy_score_map = np.array(result.score_map)

# create SegmentationResult object
result = fd.C.vision.SegmentationResult()
result.label_map = numpy_label_map.tolist()
result.score_map = numpy_score_map.tolist()
```
>> **注意**: 以上为示例代码，具体请参考[PaddleSeg example](../../../examples/vision/segmentation/paddleseg/)
