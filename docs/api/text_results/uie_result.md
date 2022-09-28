# UIEResult 图像分类结果

UIEResult代码定义在`fastdeploy/text/uie/model.h`中，用于表明UIE模型抽取结果和置信度。

## C++ 定义

`fastdeploy::text::UIEResult`

```c++
struct UIEResult {
  size_t start_;
  size_t end_;
  double probability_;
  std::string text_;
  std::unordered_map<std::string, std::vector<UIEResult>> relation_;
  std::string Str() const;
};
```

- **start_**: 成员变量，表示抽取结果text_在原文本（Unicode编码）中的起始位置。
- **end**: 成员变量，表示抽取结果text_在原文本（Unicode编码）中的结束位置。
- **text_**: 成员函数，表示抽取的结果，以UTF-8编码方式保存。
- **relation_**: 成员函数，表示当前结果关联的结果。常用于关系抽取。
- **Str()**: 成员函数，将结构体中的信息以字符串形式输出（用于Debug）

## Python 定义

`fastdeploy.text.C.UIEResult`

- **start_**(int): 成员变量，表示抽取结果text_在原文本（Unicode编码）中的起始位置。
- **end**(int): 成员变量，表示抽取结果text_在原文本（Unicode编码）中的结束位置。
- **text_**(str): 成员函数，表示抽取的结果，以UTF-8编码方式保存。
- **relation_**(dict(str, list(fastdeploy.text.C.UIEResult))): 成员函数，表示当前结果关联的结果。常用于关系抽取。
- **get_dict()**: 以dict形式返回fastdeploy.text.C.UIEResult。
