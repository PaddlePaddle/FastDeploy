# 视觉模型预测结果说明

## ClassifyResult
ClassifyResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像的分类结果和置信度.

API:`fastdeploy.vision.ClassifyResult`, 该结果返回:
- **label_ids**(list of int): 成员变量，表示单张图片的分类结果，其个数根据在使用分类模型时传入的`topk`决定，例如可以返回`top5`的分类结果.
- **scores**(list of float): 成员变量，表示单张图片在相应分类结果上的置信度，其个数根据在使用分类模型时传入的`topk`决定，例如可以返回`top5`的分类置信度.


## SegmentationResult
SegmentationResult代码定义在`fastdeploy/vision/ttommon/result.h`中，用于表明图像中每个像素预测出来的分割类别和分割类别的概率值.

API:`fastdeploy.vision.SegmentationResult`, 该结果返回:
- **label_map**(list of int): 成员变量，表示单张图片每个像素点的分割类别.
- **score_map**(list of float): 成员变量，与label_map一一对应的所预测的分割类别概率值(当导出模型时指定`--output_op argmax`)或者经过softmax归一化化后的概率值(当导出模型时指定`--output_op softmax`或者导出模型时指定`--output_op none`同时模型初始化的时候设置模型类成员属性`apply_softmax=true`).
- **shape**(list of int): 成员变量，表示输出图片的尺寸，为`H*W`.

## DetectionResult
DetectionResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像检测出来的目标框、目标类别和目标置信度.

API:`fastdeploy.vision.DetectionResult` , 该结果返回:
- **boxes**(list of list(float)): 成员变量，表示单张图片检测出来的所有目标框坐标. boxes是一个list，其每个元素为一个长度为4的list， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标.
- **scores**(list of float): 成员变量，表示单张图片检测出来的所有目标置信度.
- **label_ids**(list of int): 成员变量，表示单张图片检测出来的所有目标类别.
- **masks**: 成员变量，表示单张图片检测出来的所有实例mask，其元素个数及shape大小与boxes一致.
- **contain_masks**: 成员变量，表示检测结果中是否包含实例mask，实例分割模型的结果此项一般为`True`.

`fastdeploy.vision.Mask` , 该结果返回:
- **data**: 成员变量，表示检测到的一个mask.
- **shape**: 成员变量，表示mask的尺寸，如 `H*W`.


## FaceDetectionResult
FaceDetectionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明人脸检测出来的目标框、人脸landmarks，目标置信度和每张人脸的landmark数量.

API:`fastdeploy.vision.FaceDetectionResult` , 该结果返回:
- **boxes**(list of list(float)): 成员变量，表示单张图片检测出来的所有目标框坐标。boxes是一个list，其每个元素为一个长度为4的list， 表示为一个框，每个框以4个float数值依次表示xmin, ymin, xmax, ymax， 即左上角和右下角坐标.
- **scores**(list of float): 成员变量，表示单张图片检测出来的所有目标置信度.
- **landmarks**(list of list(float)): 成员变量，表示单张图片检测出来的所有人脸的关键点.
- **landmarks_per_face**(int): 成员变量，表示每个人脸框中的关键点的数量.

## KeyPointDetectionResult
KeyPointDetectionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像中目标行为的各个关键点坐标和置信度。

API:`fastdeploy.vision.KeyPointDetectionResult` , 该结果返回:
- **keypoints**(list of list(float)): 成员变量，表示识别到的目标行为的关键点坐标。`keypoints.size()= N * J * 2`，
    - `N`：图片中的目标数量
    - `J`：num_joints（一个目标的关键点数量）
    - `3`:坐标信息[x, y]
- **scores**(list of float): 成员变量，表示识别到的目标行为的关键点坐标的置信度。`scores.size()= N * J`
    - `N`：图片中的目标数量
    - `J`:num_joints（一个目标的关键点数量）
- **num_joints**(int): 成员变量，表示一个目标的关键点数量


## FaceRecognitionResult
FaceRecognitionResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明人脸识别模型对图像特征的embedding.

API:`fastdeploy.vision.FaceRecognitionResult`, 该结果返回:
- **embedding**(list of float): 成员变量，表示人脸识别模型最终提取的特征embedding，可以用来计算人脸之间的特征相似度.


## MattingResult
MattingResult 代码定义在`fastdeploy/vision/common/result.h`中，用于表明模型预测的alpha透明度的值，预测的前景等.

API:`fastdeploy.vision.MattingResult`, 该结果返回:
- **alpha**(list of float): 是一维向量，为预测的alpha透明度的值，值域为`[0.,1.]`，长度为`H*W`，H,W为输入图像的高和宽.
- **foreground**(list of float): 是一维向量，为预测的前景，值域为`[0.,255.]`，长度为`H*W*C`，H,W为输入图像的高和宽，C一般为3，`foreground`不是一定有的，只有模型本身预测了前景，这个属性才会有效.
- **contain_foreground**(bool): 表示预测的结果是否包含前景.
- **shape**(list of int): 表示输出结果的shape，当`contain_foreground`为`false`，shape只包含`(H,W)`，当`contain_foreground`为`true`，shape包含`(H,W,C)`, C一般为3.

## OCRResult
OCRResult代码定义在`fastdeploy/vision/common/result.h`中，用于表明图像检测和识别出来的文本框，文本框方向分类，以及文本框内的文本内容.

API:`fastdeploy.vision.OCRResult`, 该结果返回:
- **boxes**(list of list(int)): 成员变量，表示单张图片检测出来的所有目标框坐标，boxes.size()表示单张图内检测出的框的个数，每个框以8个int数值依次表示框的4个坐标点，顺序为左下，右下，右上，左上.
- **text**(list of string): 成员变量，表示多个文本框内被识别出来的文本内容，其元素个数与`boxes.size()`一致.
- **rec_scores**(list of float): 成员变量，表示文本框内识别出来的文本的置信度，其元素个数与`boxes.size()`一致.
- **cls_scores**(list of float): 成员变量，表示文本框的分类结果的置信度，其元素个数与`boxes.size()`一致.
- **cls_labels**(list of int): 成员变量，表示文本框的方向分类类别，其元素个数与`boxes.size()`一致.
