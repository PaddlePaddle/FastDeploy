[English](README.md) | 简体中文
# 图像生成模型

FastDeploy目前支持PaddleHub预训练模型库中如下风格迁移模型的部署

| 模型 | 说明 | 模型格式 |
| :--- | :--- | :------- |
|[animegan_v1_hayao_60](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v1_hayao_60&en_category=GANs)|可将输入的图像转换成宫崎骏动漫风格，模型权重转换自AnimeGAN V1官方开源项目|paddle|
|[animegan_v2_paprika_97](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_97&en_category=GANs)|可将输入的图像转换成今敏红辣椒动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_hayao_64](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_hayao_64&en_category=GANs)|可将输入的图像转换成宫崎骏动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_shinkai_53](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_shinkai_53&en_category=GANs)|可将输入的图像转换成新海诚动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_shinkai_33](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_shinkai_33&en_category=GANs)|可将输入的图像转换成新海诚动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_paprika_54](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_54&en_category=GANs)|可将输入的图像转换成今敏红辣椒动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_hayao_99](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_hayao_99&en_category=GANs)|可将输入的图像转换成宫崎骏动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_paprika_74](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_74&en_category=GANs)|可将输入的图像转换成今敏红辣椒动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|
|[animegan_v2_paprika_98](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_98&en_category=GANs)|可将输入的图像转换成今敏红辣椒动漫风格，模型权重转换自AnimeGAN V2官方开源项目|paddle|

## FastDeploy paddle backend部署和hub速度对比(ips, 越高越好)
| Device | FastDeploy | Hub |
| :--- | :--- | :------- |
|  CPU   |  0.075     | 0.069|
|  GPU   |  8.33      | 8.26 |



## 下载预训练模型
使用fastdeploy.download_model即可以下载模型, 例如下载animegan_v1_hayao_60
```python
import fastdeploy as fd
fd.download_model(name='animegan_v1_hayao_60', path='./', format='paddle')
```
将会在当前目录获得animegan_v1_hayao_60的预训练模型。

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
