English | [中文](README_CN.md)

# FastDeploy generates an encrypted model

This directory provides `encrypt.py` to quickly complete the encryption of the model and parameter files of ResNet50_vd

## encryption
```bash
# Download deployment example code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/tutorials/encrypt_model

# Download the ResNet50_vd model file
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz

python encrypt.py --model_file ResNet50_vd_infer/inference.pdmodel  --params_file ResNet50_vd_infer/inference.pdiparams --encrypted_model_dir ResNet50_vd_infer_encrypt
```
>> **Note** After the encryption is completed, the ResNet50_vd_infer_encrypt folder will be generated, including `__model__.encrypted`, `__params__.encrypted`, `encryption_key.txt` three files, where `encryption_key.txt` contains the encrypted key. At the same time, you need to copy the `inference_cls.yaml` configuration file in the original folder to the ResNet50_vd_infer_encrypt folder for subsequent deployment

### Python encryption interface

Use the encrypted interface through the following interface settings
```python
import fastdeploy as fd
import os
# when key is not given, key will be automatically generated.
# otherwise, the file will be encrypted by specific key
encrypted_model, key = fd.encryption.encrypt(model_file.read())
encrypted_params, key= fd.encryption.encrypt(params_file.read(), key)
```

### FastDeploy deployment encryption model (decryption)

Through the setting of the following interface, FastDeploy can deploy the encryption model
```python
import fastdeploy as fd
option = fd.RuntimeOption()
option.set_encryption_key(key)
```

```C++
fastdeploy::RuntimeOption option;
option.SetEncryptionKey(key)
```
>> **Note** For more details about RuntimeOption, please refer to [RuntimeOption Python Documentation](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/runtime_option.html), [ RuntimeOption C++ Documentation](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/structfastdeploy_1_1RuntimeOption.html)
