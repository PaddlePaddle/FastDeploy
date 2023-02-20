English | [简体中文](detection_result_CN.md)

# Target Detection Result

The DetectionResult code is defined in `fastdeploy/vision/common/result.h`, and is used to indicate the target frame, target class and target confidence level detected in the image.

## C++ Definition

```c++
fastdeploy::vision::DetectionResult
```  

```c++
struct DetectionResult {
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
  std::vector<Mask> masks;
  bool contain_masks = false;
  void Clear();
  std::string Str();
};
```

- **boxes**: Member variable which indicates the coordinates of all detected target boxes in a single image. `boxes.size()` indicates the number of boxes, each box is represented by 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**: Member variable which indicates the confidence level of all targets detected in a single image, where the number of elements is the same as `boxes.size()`.
- **label_ids**: Member variable which indicates all target categories detected in a single image, where the number of elements is the same as `boxes.size()`.
- **masks**: Member variable which indicates all detected instance masks of a single image, where the number of elements and the shape size are the same as `boxes`.
- **contain_masks**: Member variable which indicates whether the detected result contains instance masks, which is generally true for the instance segmentation model.
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

```c++
fastdeploy::vision::Mask
```  
```c++
struct Mask {
  std::vector<int32_t> data;
  std::vector<int64_t> shape; // (H,W) ...

  void Clear();
  std::string Str();
};
```  
- **data**: Member variable which indicates a detected mask.
- **shape**: Member variable which indicates the shape of the mask, e.g. (h,w).
- **Clear()**: Member function used to clear the results stored in the structure.
- **Str()**: Member function used to output the information in the structure as string (for Debug).

## Python Definition

```python
fastdeploy.vision.DetectionResult  
```

- **boxes**(list of list(float)): Member variable which indicates the coordinates of all detected target boxes in a single frame. It is a list, and each element in it is also a list of length 4, representing a box with 4 float values representing xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**(list of float): Member variable which indicates the confidence level of all targets detected in a single image.
- **label_ids**(list of int): Member variable which indicates all target categories detected in a single image.
- **masks**: Member variable which indicates all detected instance masks of a single image, where the number of elements and the shape size are the same as `boxes`.
- **contain_masks**: Member variable which indicates whether the detected result contains instance masks, which is generally true for the instance segmentation model.

```python
fastdeploy.vision.Mask  
```
- **data**: Member variable which indicates a detected mask.
- **shape**: Member variable which indicates the shape of the mask, e.g. (h,w).

## C# Definition

```c#
fastdeploy.vision.DetectionResult
```  

```C#
public struct DetectionResult {
  public List<float[]> boxes;
  public List<float> scores;
  public List<int> label_ids;
  public List<Mask> masks;
  public bool contain_masks;
}
```

- **boxes**(list of array(float)): Member variable which indicates the coordinates of all detected target boxes in a single frame. It is a list, and each element in it is also a list of length 4, representing a box with 4 float values representing xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner.
- **scores**(list of float): Member variable which indicates the confidence level of all targets detected in a single image.
- **label_ids**(list of int): Member variable which indicates all target categories detected in a single image.
- **masks**: Member variable which indicates all detected instance masks of a single image, where the number of elements and the shape size are the same as `boxes`.
- **contain_masks**: Member variable which indicates whether the detected result contains instance masks, which is generally true for the instance segmentation model.

```C#
public struct Mask {
  public List<byte> data;
  public List<long> shape;
}
```

- **data**: Member variable which indicates a detected mask.
- **shape**: Member variable which indicates the shape of the mask, e.g. (h,w).

## C Definition

```c
typedef struct FD_C_DetectionResult {
  FD_C_TwoDimArrayFloat boxes;
  FD_C_OneDimArrayFloat scores;
  FD_C_OneDimArrayInt32 label_ids;
  FD_C_OneDimMask masks;
  FD_C_Bool contain_masks;
} FD_C_DetectionResult;
```

- **boxes**(FD_C_TwoDimArrayFloat): Member variable which indicates the coordinates of all detected target boxes in a single frame. It is a list, and each element in it is also a list of length 4, representing a box with 4 float values representing xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corner. FD_C_TwoDimArrayFloat includes two fields，i.e. size and data，in which size represents the number of elements，and data is the array to store elements of type FD_C_OneDimArrayFloat.

```c
typedef struct FD_C_TwoDimArrayFloat {
  size_t size;
  FD_C_OneDimArrayFloat* data;
}
```


- **scores**(FD_C_OneDimArrayFloat): Member variable which indicates the confidence level of all targets detected in a single image. FD_C_OneDimArrayFloat includes two fields，i.e. size and data，in which size represents the number of elements，and data is the array to store elements.

```c
typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;
```

- **label_ids**(FD_C_OneDimArrayInt32): Member variable which indicates all target categories detected in a single image. FD_C_OneDimArrayInt32 includes two fields，i.e. size and data，in which size represents the number of elements，and data is the array to store elements.

```c
typedef struct FD_C_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} FD_C_OneDimArrayInt32;
```

- **masks**(FD_C_OneDimMask): Member variable which indicates all detected instance masks of a single image, where the number of elements and the shape size are the same as `boxes`.

```c
typedef struct FD_C_OneDimMask {
  size_t size;
  FD_C_Mask* data;
} FD_C_OneDimMask;
```

```c
typedef struct FD_C_Mask {
  FD_C_OneDimArrayUint8 data;
  FD_C_OneDimArrayInt64 shape;
} FD_C_Mask;
```
- **contain_masks**: Member variable which indicates whether the detected result contains instance masks, which is generally true for the instance segmentation model.
