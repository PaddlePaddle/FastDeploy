[English](README.md) | 简体中文
# 使用 FastDeploy 服务化部署 PaddleSeg 模型
## FastDeploy 服务化部署介绍
在线推理作为企业或个人线上部署模型的最后一环，是工业界必不可少的环节，其中最重要的就是服务化推理框架。FastDeploy 目前提供两种服务化部署方式：simple_serving和fastdeploy_serving。simple_serving 基于Flask框架具有简单高效的特点，可以快速验证线上部署模型的可行性。fastdeploy_serving基于Triton Inference Server框架，是一套完备且性能卓越的服务化部署框架，可用于实际生产。

## 详细部署文档

- [fastdeploy serving](fastdeploy_serving)
- [simple serving](simple_serving)
