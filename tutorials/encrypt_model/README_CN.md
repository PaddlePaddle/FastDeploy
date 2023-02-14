[English](README.md) | 中文

# 使用FastDeploy生成加密模型

本目录下提供`encrypt.py`快速完成ResNet50_vd的模型和参数文件加密

FastDeploy支持对称加密的方案，通过调用OpenSSL中的对称加密算法（AES）对模型进行加密并产生密钥

## 加密
```bash
#下载加密示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/tutorials/encrypt_model

# 下载ResNet50_vd模型文件
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz

python encrypt.py --model_file ResNet50_vd_infer/inference.pdmodel  --params_file ResNet50_vd_infer/inference.pdiparams --encrypted_model_dir ResNet50_vd_infer_encrypt
```
>> **注意** 加密完成后会生成ResNet50_vd_infer_encrypt文件夹，包含`__model__.encrypted`,`__params__.encrypted`,`encryption_key.txt`三个文件，其中`encryption_key.txt`包含加密后的秘钥，同时需要将原文件夹中的、`inference_cls.yaml`配置文件 拷贝至ResNet50_vd_infer_encrypt文件夹，以便后续部署使用

### Python加密接口

通过如下接口的设定，使用加密接口（解密）
```python
import fastdeploy as fd
import os
# when key is not given, key will be automatically generated.
# otherwise, the file will be encrypted by specific key
encrypted_model, key = fd.encryption.encrypt(model_file.read())
encrypted_params, key= fd.encryption.encrypt(params_file.read(), key)
```

### FastDeploy 部署加密模型

通过如下接口的设定，完成加密模型的推理
```python
import fastdeploy as fd
option = fd.RuntimeOption()
option.set_encryption_key(key)
```

```C++
fastdeploy::RuntimeOption option;
option.SetEncryptionKey(key)
```
>> **注意** RuntimeOption的更多详细信息，请参考[RuntimeOption Python文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/runtime_option.html)，[RuntimeOption C++文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/structfastdeploy_1_1RuntimeOption.html)
