# 编译YOLOv5Lite示例

当前支持模型版本为：[YOLOv5-Lite-v1.4](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)

## 获取ONNX文件
- 自动获取
  访问[YOLOv5Lite](https://github.com/ppogg/YOLOv5-Lite)
官方github库，按照指引下载安装，下载`yolov5-lite-xx.onnx` 模型(Tips：官方提供的ONNX文件目前是没有decode模块的)
  ```
  #下载yolov5-lite模型文件(.onnx)
  Download from https://drive.google.com/file/d/1bJByk9eoS6pv8Z3N4bcLRCV3i7uk24aU/view
  官方Repo也支持百度云下载
  ```

- 手动获取

  访问[YOLOv5Lite](https://github.com/ppogg/YOLOv5-Lite)
官方github库，按照指引下载安装，下载`yolov5-lite-xx.pt` 模型，利用 `export.py` 得到`onnx`格式文件。

  - 导出含有decode模块的ONNX文件

  首先需要参考[YOLOv5-Lite#189](https://github.com/ppogg/YOLOv5-Lite/pull/189)的解决办法，修改代码。

  ```
  #下载yolov5-lite模型文件(.pt)
  Download from https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view
  官方Repo也支持百度云下载

  # 导出onnx格式文件
  python export.py --grid --dynamic --concat --weights PATH/TO/yolov5-lite-xx.pt

  # 移动onnx文件到demo目录
  cp PATH/TO/yolov5lite.onnx PATH/TO/model_zoo/vision/yolov5lite/
  ```
  - 导出无decode模块的ONNX文件(不需要修改代码)

  ```
  #下载yolov5-lite模型文件
  Download from https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view
  官方Repo也支持百度云下载

  # 导出onnx格式文件
  python export.py --grid --dynamic --weights PATH/TO/yolov5-lite-xx.pt

  # 移动onnx文件到demo目录
  cp PATH/TO/yolov5lite.onnx PATH/TO/model_zoo/vision/yolov5lite/
  ```

## 设置ONNX文件处理方式

如果ONNX文件是含有decode模块的，设置`model.is_decode_exported = true`(解除yolov5lite.cc第27行注释)

如果ONNX文件是无decode模块的，不用做任何处理，默认是`model.is_decode_exported = false`

## 运行demo

```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 移动onnx文件到demo目录
cp PATH/TO/yolov5lite.onnx PATH/TO/model_zoo/vision/yolov5lite/cpp/build/

# 下载图片
wget https://raw.githubusercontent.com/ppogg/YOLOv5-Lite/master/cpp_demo/mnn/test.jpg

# 执行
./yolov5lite_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
513.642029,772.265076, 653.019897, 1124.715576, 0.921402, 0
304.034210,1025.680542, 452.700409, 1298.065552, 0.919130, 0
1291.161011,703.538208, 1407.240967, 1023.016602, 0.907900, 0
308.813660,88.560013, 409.269318, 380.412476, 0.896667, 0
5.142653,1.688710, 101.452621, 260.087219, 0.895105, 0
1346.397461,192.343491, 1451.643066, 448.248474, 0.888778, 0
155.137177,1142.878174, 258.905548, 1297.865601, 0.888149, 0
779.170532,817.150024, 892.359314, 1120.427246, 0.882556, 0
1171.451172,725.544006, 1288.958496, 1032.780640, 0.880074, 0
627.751099,825.236572, 718.951965, 1132.701294, 0.875676, 0
906.220215,510.590088, 996.494202, 823.648071, 0.868565, 0
783.026672,491.062164, 872.556519, 775.829956, 0.853596, 0
1331.119263,374.009216, 1443.430420, 638.995850, 0.838086, 0
133.456970,37.730553, 211.912766, 285.147705, 0.836074, 0
1256.783936,30.545942, 1349.071655, 315.149597, 0.814443, 0
414.291351,986.187073, 541.678040, 1295.738647, 0.811473, 0
262.399963,303.828979, 373.234070, 569.670776, 0.810324, 0
660.811646,467.742737, 762.443787, 785.119812, 0.800616, 0
960.292908,76.071686, 1068.249023, 338.966492, 0.738156, 0
914.498779,312.663727, 1030.258911, 623.590393, 0.720789, 0
489.880371,1034.852051, 564.550781, 1179.285645, 0.681449, 24
499.645538,350.272461, 593.194397, 626.313293, 0.658708, 0
1321.224121,2.669750, 1421.656860, 100.821770, 0.636028, 0
955.810547,120.606491, 1013.012695, 244.129440, 0.595005, 26
842.028809,531.238831, 876.928711, 617.188538, 0.587429, 26
450.080841,318.434631, 537.059143, 581.682983, 0.531207, 0
```
