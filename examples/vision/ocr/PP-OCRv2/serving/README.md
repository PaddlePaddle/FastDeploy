# PP-OCRv2服务化部署示例

除了`下载的模型`和`rec前处理的1个参数`以外PP-OCRv2的服务化部署与PP-OCRv3服务化部署全部一样，请参考[PP-OCRv3服务化部署](../../PP-OCRv3/serving)。

## 下载模型
将下载链接中的`v3`改为`v2`即可。

## 修改rec前处理参数
在[model.py](../../PP-OCRv3/serving/models/det_postprocess/1/model.py)文件**109行添加以下代码**：
```
self.rec_preprocessor.cls_image_shape[1] = 32
```
