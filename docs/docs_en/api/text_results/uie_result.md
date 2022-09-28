# UIEResult Image - Claasification Results

The UIEResult function is defined in `fastdeploy/text/uie/model.h`, indicating the UIE model abstraction results and confidence levels.

## C++ Definition

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

- **start_**: Member variable that indicates the starting position of the abstraction result text_ in the original text (Unicode encoding).
- **end**: Member variable that indicates the ending position of the abstraction result text_ in the original text (Unicode encoding).
- **text_**: Member function that indicates the result of the abstraction, saved in UTF-8 format.
- **relation_**: Member function that indicates the current result association. It is commonly used for relationship abstraction.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

## Python Definition

`fastdeploy.text.C.UIEResult`

- **start_**(int): Member variable that indicates the starting position of the abstraction result text_ in the original text (Unicode encoding).
- **end**(int): Member variable that indicates the ending position of the abstraction result text_ in the original text (Unicode encoding).
- **text_**(str): Member function that indicates the result of the abstraction, saved in UTF-8 format.
- **relation_**(dict(str, list(fastdeploy.text.C.UIEResult))): Member function that indicates the current result association. It is commonly used for relationship abstraction.
- **get_dict()**: give fastdeploy.text.C.UIEResult in dict format.
