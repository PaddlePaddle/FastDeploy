
# tokenizer
## 说明
  Tokenizer 子命令提供文本与 token 序列之间的编码与解码功能，并可查看或导出模型的词表信息。支持文本模型与多模态模型。

## 用法
```
fastdeploy tokenizer --model MODEL (--encode TEXT | --decode TOKENS | --vocab-size | --info)
```

## 参数
|参数|说明|默认值|
|-|-|-|
|--model, -m|模型路径或名称|None|
|--encode, -e|将文本编码为 token 列表|None|
|--decode, -d|将 token 列表解码为文本|None|
|--vocab-size, -vs|查看词表大小|None|
|--info, -i|查看 tokenizer 详细信息（特殊符号、ID、最大长度等）|None|
|--vocab-export FILE, -ve FILE|导出词表到文件|None|

## 示例
```
# 1. 编码文本为 tokens
# 将输入文本转换为模型可识别的 token 序列
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --encode "Hello, world!"

# 2. 解码 tokens 为文本
# 将 token 序列转换回可读文本
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --decode "[1, 2, 3]"

# 3. 查看词表大小
# 输出模型 tokenizer 的总词表数量
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --vocab-size

# 4. 查看 tokenizer 详细信息
# 包括特殊符号、ID 映射、最大长度等信息
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --info

# 5. 导出词表到文件
# 将 tokenizer 的词表保存到本地文件
fastdeploy tokenizer --model baidu/ERNIE-4.5-0.3B-Paddle --vocab-export ./vocab.txt

# 6. 支持多模模型
# 对多模态模型进行解码
fastdeploy tokenizer --model baidu/EB-VL-Lite-d --decode "[5300, 96382]"

# 7. 多功能组合使用
# 可以同时进行编码、解码、查看词表、导出词表等操作
fastdeploy tokenizer \
    -m baidu/ERNIE-4.5-0.3B-PT \
    -e "你好哇" \
    -d "[5300, 96382]" \
    -i \
    -vs \
    -ve vocab.json

```
