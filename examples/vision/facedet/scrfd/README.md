English | [简体中文](README_CN.md)
# SCRFD Ready-to-deploy Model


- [SCRFD](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950)
  - （1）The *.pt provided by the [Official Library](https://github.com/deepinsight/insightface/) can be deployed after the [Export ONNX Model](#export-onnx-model) to complete the deployment；
  - （2）As for SCRFD model trained on customized data, please follow [Export ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B) to complete the deployment.


## Export ONNX Model

  ```bash
  # Download scrfd model files
  e.g. download from  https://onedrive.live.com/?authkey=%21ABbFJx2JMhNjhNA&id=4A83B6B633B029CC%215542&cid=4A83B6B633B029CC

  # Install the official library to configure the environment. This version should be exported in the following environment:
  - Configure the environment manually
    torch==1.8.0
    mmcv==1.3.5
    mmdet==2.7.0

  - Configure via docker
    docker pull qyjdefdocker/onnx-scrfd-converter:v0.3

  # Export files in onnx format
  - Manual generation
    python tools/scrfd2onnx.py configs/scrfd/scrfd_500m.py weights/scrfd_500m.pth --shape 640 --input-img face-xxx.jpg

  - docker
    onnx files are in docker's onnx directory

  ```

## Download Pre-trained ONNX Models

For developers' testing, models exported by SCRFD are provided below. Developers can download and use them directly. (The accuracy of the models in the table is sourced from the official library)
| Model                                                               | Size    | Accuracy    |
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

## Detailed Deployment Tutorials

- [Python Deployement](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [SCRFD CommitID:17cdeab](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950) 
