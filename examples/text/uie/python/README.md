# 通用信息抽取 UIE Python部署示例


## UIEModel Python接口

```python
fd.text.uie.UIEModel(model_file, params_file, vocab_file, position_prob=0.5, max_length=128, schema=[], runtime_option=None, model_format=Frontend.PADDLE)
```

UIEModel模型加载和初始化，其中`model_file`, `params_file`为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)，`vocab_file`为词表文件，UIE模型的词表可在[UIE配置文件](https://github.com/PaddlePaddle/PaddleNLP/blob/5401f01af85f1c73d8017c6b3476242fce1e6d52/model_zoo/uie/utils.py)中下载相应的UIE模型的vocab_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **vocab_file**(str): 词表文件
> * **position_prob**(str): 位置概率，模型将输出位置概率大于`position_prob`的位置，默认为0.5
> * **max_length**(int): 输入文本的最大长度。输入文本下标超过`max_length`的部分将被截断。默认为128
> * **schema**(list|dict): 抽取任务的目标信息。
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为Paddle格式

### set_schema函数

> ```python
> set_schema(schema)
> ```
> 设置UIE模型的schema接口。
>
> **参数**
> > * **schema**(list|dict): 输入数据，待抽取文本列表。
>
> **返回**
> 空。

### predict函数

> ```python
> UIEModel.predict(texts, return_dict=False)
> ```
>
> 模型预测接口，输入文本列表直接输出抽取结果。
>
> **参数**
>
> > * **texts**(list(str)): 输入数据，待抽取文本列表。
> > * **return_dict**(bool): 是否以字典形式输出UIE结果，默认为False。
> **返回**
>
> > 返回`dict(schema_key, list(fastdeploy.text.UIEResult))`。

## 相关文档

[UIE模型详细介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md)
[UIE导出方法](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)
[UIE C++部署方法](../cpp/README.md)
