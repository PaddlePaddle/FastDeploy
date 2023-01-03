English | [简体中文](README.md)
# PP-OCRv2 Serving Deployment 

The serving deployment of PP-OCRv2 is identical to that of PP-OCRv3 except for `downloaded models` and `1 parameter for rec pre-processing`. Refer to [PP-OCRv3 serving deployment](../../PP-OCRv3/serving)

## Download models 
Change `v3` into `v2` in the download link.

## Modify the rec pre-processing parameter
**Add the following code to line 109** in the file [model.py](../../PP-OCRv3/serving/models/det_postprocess/1/model.py#L109):
```
self.rec_preprocessor.cls_image_shape[1] = 32
```
