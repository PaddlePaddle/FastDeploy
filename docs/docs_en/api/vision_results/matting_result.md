# Matting Results

The MattingResult function is defined in `csrcs/fastdeploy/vision/common/result.h` , indicating the value of alpha transparency predicted by the model, the predicted foreground.

## C++ Definition

`fastdeploy::vision::MattingResult`

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

- **alpha**: a one-dimensional vector of predicted alpha transparency value in the range [0.,1.] and length hxw, with h,w being the height and width of the input image
- **foreground**: a one-dimensional vector for the predicted foreground, with a value range of [0.,255.] and a length of hxwxc. 'h,w' is the height and width of the input image, and c=3 in general. The foreground feature is not always available. It is only valid if the model predicts the foreground
- **contain_foreground**: indicates whether the predicted result contains a foreground
- **shape**: indicates the results shape. When contain_foreground is false, the shape only contains (h,w); when contain_foreground is true, the shape contains (h,w,c), and c is generally 3
- **Clear()**: Member function that clears the results stored in a struct.
- **Str()**: Member function that outputs the information in the struct as a string (for Debug)

## Python Definition

`fastdeploy.vision.MattingResult`

- **alpha**(list of float): a one-dimensional vector of predicted alpha transparency value in the range [0.,1.] and length hxw, with h,w being the height and width of the input image.
- **foreground**(list of float): a one-dimensional vector for the predicted foreground, with a value range of [0.,255.] and a length of hxwxc. 'h,w' is the height and width of the input image, and c=3 in general. The foreground feature is not always available. It is only valid if the model predicts the foreground.
- **contain_foreground**(bool): indicates whether the predicted result contains a foreground
- **shape**(list of int): indicates the results shape. When contain_foreground is false, the shape only contains (h,w); when contain_foreground is true, the shape contains (h,w,c), and c is generally 3
