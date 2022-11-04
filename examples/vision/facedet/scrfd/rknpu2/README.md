# SCRFD RKNPU2部署模型


- [SCRFD](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950)
    - （1）[官方库](https://github.com/deepinsight/insightface/)中提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
    - （2）开发者基于自己数据训练的SCRFD模型，可按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)后，完成部署。

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了SCRFD导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [SCRFD-500M-kps-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape160x160.onnx) | 2.5MB | - |
| [SCRFD-500M-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_shape160x160.onnx) | 2.2MB | - |
| [SCRFD-500M-kps-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape320x320.onnx) | 2.5MB | - |
| [SCRFD-500M-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_shape320x320.onnx) | 2.2MB | - |
| [SCRFD-500M-kps-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape640x640.onnx) | 2.5MB | 90.97% |
| [SCRFD-500M-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_shape640x640.onnx) | 2.2MB | 90.57% |
| [SCRFD-1G-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_1g_shape160x160.onnx ) | 2.5MB | - |
| [SCRFD-1G-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_1g_shape320x320.onnx) | 2.5MB | - |
| [SCRFD-1G-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_1g_shape640x640.onnx) | 2.5MB | 92.38% |
| [SCRFD-2.5G-kps-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape160x160.onnx) | 3.2MB | - |
| [SCRFD-2.5G-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_shape160x160.onnx) | 2.6MB | - |
| [SCRFD-2.5G-kps-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape320x320.onnx) | 3.2MB | - |
| [SCRFD-2.5G-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_shape320x320.onnx) | 2.6MB | - |
| [SCRFD-2.5G-kps-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape640x640.onnx) | 3.2MB | 93.8% |
| [SCRFD-2.5G-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_shape640x640.onnx) | 2.6MB | 93.78% |
| [SCRFD-10G-kps-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_bnkps_shape320x320.onnx) | 17MB | - |
| [SCRFD-10G-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_shape320x320.onnx) | 15MB | - |
| [SCRFD-10G-kps-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_bnkps_shape640x640.onnx) | 17MB | 95.4% |
| [SCRFD-10G-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_shape640x640.onnx) | 15MB | 95.16% |
| [SCRFD-10G-kps-1280](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_bnkps_shape1280x1280.onnx) | 17MB | - |
| [SCRFD-10G-1280](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_shape1280x1280.onnx) | 15MB | - |


## ONNX模型转换RKNN模型
```bash
wget  https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape640x640.onnx
python tools/rknpu2/export.py --config_path tools/rknpu2/config/RK3588/scrfd.yaml
```

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)


## 版本说明

- 本版本文档和代码基于[SCRFD CommitID:17cdeab](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950) 编写
