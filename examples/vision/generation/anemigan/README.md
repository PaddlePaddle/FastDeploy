English | [简体中文](README_CN.md)
# Image Generation Model

Now FastDeploy supports the deployment of the following style transfer models in the PaddleHub pre-trained model repository

| Model | Description | Model Format |
| :--- | :--- | :------- |
|[animegan_v1_hayao_60](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v1_hayao_60&en_category=GANs)| Convert the input image into one in Miyazaki anime style with model weights converting from AnimeGAN V1 official open source project |paddle|
|[animegan_v2_paprika_97](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_97&en_category=GANs)| Convert the input image into one in Satoshi Paprika anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_hayao_64](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_hayao_64&en_category=GANs)| Convert the input image into one in Miyazaki anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_shinkai_53](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_shinkai_53&en_category=GANs)| Convert the input image into one in Shinkai anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_shinkai_33](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_shinkai_33&en_category=GANs)| Convert the input image into one in Shinkai anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_paprika_54](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_54&en_category=GANs)| Convert the input image into one in Satoshi Paprika anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_hayao_99](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_hayao_99&en_category=GANs)| Convert the input image into one in Miyazaki anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_paprika_74](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_74&en_category=GANs)| Convert the input image into one in Satoshi Paprika anime style with model weights converting from AnimeGAN V2 official open source project |paddle|
|[animegan_v2_paprika_98](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_paprika_98&en_category=GANs)| Convert the input image into one in Satoshi Paprika anime style with model weights converting from AnimeGAN V2 official open source project |paddle|

## Speed comparison between hub and FastDeploy paddle backend Deployment (ips, higher is better)
| Device | FastDeploy | Hub |
| :--- | :--- | :------- |
|  CPU   |  0.075     | 0.069|
|  GPU   |  8.33      | 8.26 |



## Downloading pre-trained models
Use fastdeploy.download_model to download models. For example, download animegan_v1_hayao_60
```python
import fastdeploy as fd
fd.download_model(name='animegan_v1_hayao_60', path='./', format='paddle')
```
The pre-trained model of animegan_v1_hayao_60 will be available in the current directory.

## Detailed deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)
