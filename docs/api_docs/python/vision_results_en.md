# Description of Vision Results

## ClassifyResult
The code of ClassifyResult is defined in `fastdeploy/vision/common/result.h` and is used to indicate the classification label result and confidence the image.

API: `fastdeploy.vision.ClassifyResult`, The ClassifyResult will return:
- **label_ids**(list of int):Member variables that represent the classification label results of a single image, the number of which is determined by the  `topk` passed in when using the classification model. For example, you can return the label results of the Top 5 categories.

- **scores**(list of float):Member variables that indicate the confidence level of a single image on the corresponding classification result, the number of which is determined by the  `topk` passed in when using the classification model, e.g. the confidence level of a Top 5 classification can be returned.

## SegmentationResult
The code of SegmentationResult is defined in `fastdeploy/vision/common/result.h` and is used to indicate the segmentation category predicted for each pixel in the image and the probability of the segmentation category.

API: `fastdeploy.vision.SegmentationResult`, The SegmentationResult will return:
- **label_ids**(list of int):Member variable indicating the segmentation category for each pixel of a single image.
- **score_map**(list of float):Member variable, the predicted probability value of the segmentation category corresponding to  `label_map` (specified when exporting the model `--output_op argmax`) or the probability value normalized by softmax (specified when exporting the model `--output_op softmax` or when exporting the model `--output_op none` and set the model class member attribute `apply_softmax=true` when initializing the model).
- **shape**(list of int):Member variable indicating the shape of the output image, as  `H*W`.


## DetectionResult
The code of DetectionResult is defined in `fastdeploy/vision/common/result.h` and is used to indicate the target location (detection box), target class and target confidence level detected by the image.

API: `fastdeploy.vision.DetectionResult`, The DetectionResult will return:
- **boxes**(list of list(float)):Member variable, represents the coordinates of all target boxes detected by a single image. boxes is a list, each element of which is a list of length 4, representing a box with 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corners.
- **socres**(list of float):Member variable indicating the confidence of all targets detected by a single image.
- **label_ids**(list of int):Member variable indicating all target categories detected for a single image.
- **masks**:Member variable that represents all instances of mask detected from a single image, with the same number of elements and shape size as boxes.
- **contain_masks**:Member variable indicating whether the detection result contains the instance mask, the result of the instance segmentation model is generally set to True.

API: `fastdeploy.vision.Mask`, The Mask will return:
- **data**:Member variable indicating a detected mask.
- **shape**:Member variable representing the shape of the mask, e.g.  `(H,W)`.

## FaceDetectionResult
The FaceDetectionResult code is defined in `fastdeploy/vision/common/result.h` and is used to indicate the target frames detected by face detection, face landmarks, target confidence and the number of landmarks per face.

API: `fastdeploy.vision.FaceDetectionResult`, The FaceDetectionResult will return:
- **data**(list of list(float)):Member variables that represent the coordinates of all target boxes detected by a single image. boxes is a list, each element of which is a list of length 4, representing a box with 4 float values in order of xmin, ymin, xmax, ymax, i.e. the coordinates of the top left and bottom right corners.
- **scores**(list of float):Member variable indicating the confidence of all targets detected by a single image.
- **landmarks**(list of list(float)): Member variables that represent the key points of all faces detected by a single image.
- **landmarks_per_face**(int):Member variable indicating the number of key points in each face frame.

## FaceRecognitionResult
The FaceRecognitionResult code is defined in `fastdeploy/vision/common/result.h` and is used to indicate the embedding of the image features by the face recognition model.

API: `fastdeploy.vision.FaceRecognitionResult`, The FaceRecognitionResult will return:
- **landmarks_per_face**(list of float):Member variables, which indicate the final extracted features embedding of the face recognition model, can be used to calculate the feature similarity between faces.

## MattingResult
The MattingResult code is defined in `fastdeploy/vision/common/result.h` and is used to indicate the value of alpha transparency predicted by the model, the predicted outlook, etc.

API:`fastdeploy.vision.MattingResult`, The MattingResult will return:
- **alpha**(list of float):This is a one-dimensional vector of predicted alpha transparency values in the range `[0.,1.]`, with length `H*W`, H,W being the height and width of the input image.
- **foreground**(list of float):This is a one-dimensional vector for the predicted foreground, the value domain is `[0.,255.]`, the length is `H*W*C`, H,W is the height and width of the input image, C is generally 3, foreground is not necessarily there, only if the model itself predicts the foreground, this property will be valid.
- **contain_foreground**(bool):Indicates whether the predicted outcome includes the foreground.
- **shape**(list of int): When `contain_foreground` is false, the shape only contains `(H,W)`, when `contain_foreground` is `true,` the shape contains `(H,W,C)`, C is generally 3.

## OCRResult
The OCRResult code is defined in `fastdeploy/vision/common/result.h` and is used to indicate the text box detected in the image, the text box orientation classification, and the text content recognized inside the text box.

API:`fastdeploy.vision.OCRResult`, The OCRResult will return:
- **boxes**(list of list(int)): Member variable, indicates the coordinates of all target boxes detected in a single image, `boxes.size()` indicates the number of boxes detected in a single image, each box is represented by 8 int values in order of the 4 coordinate points of the box, the order is lower left, lower right, upper right, upper left.
- **text**(list of string):Member variable indicating the content of the recognized text in multiple text boxes, with the same number of elements as `boxes.size()`.
- **rec_scores**(list of float):Member variable indicating the confidence level of the text identified in the box, the number of elements is the same as `boxes.size()`.
- **cls_scores**(list of float):Member variable indicating the confidence level of the classification result of the text box, with the same number of elements as `boxes.size()`.
- **cls_labels**(list of int):Member variable indicating the orientation category of the text box, the number of elements is the same as `boxes.size()`.
