English | [简体中文](README_CN.md)
# RobustVideoMatting supports TRT’s dynamic ONNX export

## Environment Dependencies

- python >= 3.5
- pytorch 1.12.0
- onnx 1.10.0
- onnxsim 0.4.8

## Step 1: Pull the RobustVideoMatting onnx branch code

```shell
git clone -b onnx https://github.com/PeterL1n/RobustVideoMatting.git
cd RobustVideoMatting
```

## Step 2: Remove downsample_ratio dynamic input

Remove the ```downsample_ratio``` input in ```model/model.py```.

```python
def forward(self, src, r1, r2, r3, r4,
                # downsample_ratio: float = 0.25,
                segmentation_pass: bool = False):

    if torch.onnx.is_in_onnx_export():
        # src_sm = CustomOnnxResizeByFactorOp.apply(src, 0.25)
        src_sm = self._interpolate(src, scale_factor=0.25)
    elif downsample_ratio != 1:
        src_sm = self._interpolate(src, scale_factor=0.25)
    else:
        src_sm = src

    f1, f2, f3, f4 = self.backbone(src_sm)
    f4 = self.aspp(f4)
    hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)

    if not segmentation_pass:
        fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
        # if torch.onnx.is_in_onnx_export() or downsample_ratio != 1:
        if torch.onnx.is_in_onnx_export():
            fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
        fgr = fgr_residual + src
        fgr = fgr.clamp(0., 1.)
        pha = pha.clamp(0., 1.)
        return [fgr, pha, *rec]
    else:
        seg = self.project_seg(hid)
        return [seg, *rec]
```

## Step 3: Modify the export ONNX script

Modify ```export_onnx.py``` script to remove the ```downsample_ratio``` input

```python
def export(self):
    rec = (torch.zeros([1, 1, 1, 1]).to(self.args.device, self.precision),) * 4
    # src = torch.randn(1, 3, 1080, 1920).to(self.args.device, self.precision)
    src = torch.randn(1, 3, 1920, 1080).to(self.args.device, self.precision)
    # downsample_ratio = torch.tensor([0.25]).to(self.args.device)

    dynamic_spatial = {0: 'batch_size', 2: 'height', 3: 'width'}
    dynamic_everything = {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}

    torch.onnx.export(
        self.model,
        # (src, *rec, downsample_ratio),
        (src, *rec),
        self.args.output,
        export_params=True,
        opset_version=self.args.opset,
        do_constant_folding=True,
        # input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio'],
        input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i'],
        output_names=['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'],
        dynamic_axes={
            'src': {0: 'batch_size0', 2: 'height0', 3: 'width0'},
            'fgr': {0: 'batch_size1', 2: 'height1', 3: 'width1'},
            'pha': {0: 'batch_size2', 2: 'height2', 3: 'width2'},
            'r1i': {0: 'batch_size3', 1: 'channels3', 2: 'height3', 3: 'width3'},
            'r2i': {0: 'batch_size4', 1: 'channels4', 2: 'height4', 3: 'width4'},
            'r3i': {0: 'batch_size5', 1: 'channels5', 2: 'height5', 3: 'width5'},
            'r4i': {0: 'batch_size6', 1: 'channels6', 2: 'height6', 3: 'width6'},
            'r1o': {0: 'batch_size7', 2: 'height7', 3: 'width7'},
            'r2o': {0: 'batch_size8', 2: 'height8', 3: 'width8'},
            'r3o': {0: 'batch_size9', 2: 'height9', 3: 'width9'},
            'r4o': {0: 'batch_size10', 2: 'height10', 3: 'width10'},
        })
```

Run the following commands

```shell
python export_onnx.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output rvm_mobilenetv3.onnx
```

**Note**：
- For the dynamic shape of the multi-input ONNX model in trt, if the shapes of x0 and x1 are different, we cannot use height and width but height0 and height1 to differentiate them, otherwise, there will be some mistakes when building engine

## Step 4: Simplify with onnxsim

Install onnxsim and simplify the ONNX model exported in step 3

```shell
pip install onnxsim
onnxsim rvm_mobilenetv3.onnx rvm_mobilenetv3_trt.onnx
```

```rvm_mobilenetv3_trt.onnx```: The ONNX model in dynamic shape that can run the TRT backend
