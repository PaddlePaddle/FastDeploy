English | [中文](matting_result_CN.md)

# Matting Result 

The MattingResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the predicted value of alpha transparency predicted and the predicted foreground, etc.

## C++ Definition

``fastdeploy::vision::MattingResult`''

```c++
struct MattingResult {
  std::vector<float> alpha;
  std::vector<float> foreground;
  std::vector<int64_t> shape;
  bool contain_foreground = false;
  void Clear();
  std::string Str();
};
```

- **alpha**: It is a one-dimensional vector, indicating the predicted value of alpha transparency. The value range is [0.,1.], and the length is hxw, in which h,w represent the height and the width of the input image seperately.
- **foreground**: It is a one-dimensional vector, indicating the predicted foreground. The value range is [0.,255.], and the length is hxwxc, in which h,w represent the height and the width of the input image, and c is generally 3. This vector is valid only when the model itself predicts the foreground.
- **contain_foreground**: Used to indicate whether the result contains foreground.
- **shape**: Used to indicate the shape of the output. When contain_foreground is false, the shape only contains (h,w), while when contain_foreground is true, the shape contains (h,w,c), in which c is generally 3.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).


## Python Definition

`fastdeploy.vision.MattingResult`

- **alpha**(list of float): It is a one-dimensional vector, indicating the predicted value of alpha transparency. The value range is [0.,1.], and the length is hxw, in which h,w represent the height and the width of the input image seperately.
- **foreground**(list of float): It is a one-dimensional vector, indicating the predicted foreground. The value range is [0.,255.], and the length is hxwxc, in which h,w represent the height and the width of the input image, and c is generally 3. This vector is valid only when the model itself predicts the foreground.
- **contain_foreground**(bool): Used to indicate whether the result contains foreground.
- **shape**(list of int): Used to indicate the shape of the output. When contain_foreground is false, the shape only contains (h,w), while when contain_foreground is true, the shape contains (h,w,c), in which c is generally 3.
