English | [中文](../../../cn/faq/horizon/export.md)

# Export Model

## Introduction

The Horizon model conversion and quantization tools are both encapsulated in the provided Docker image. Before performing model conversion, please ensure that the environment has been installed successfully according to [How to Build Horizon Deployment Environment](../../build_and_install/horizon.md).

## Model Conversion
Due to the lack of direct support for converting Paddle models to Horizon models, the first step is to convert the Paddle model to an ONNX model. The main opset versions supported by Horizon currently are opset10 and opset11, and ir_version <= 7. The conversion process requires special attention, and [the official documentation provided by Horizon](https://developer.horizon.ai/api/v1/fileData/doc/ddk_doc/navigation/ai_toolchain/docs_cn/horizon_ai_toolchain_user_guide/model_conversion.html#fp-model-preparation) can be referred to for more details.。

To convert a Paddle model to an ONNX model, you can run the following command:
```bash
paddle2onnx --model_dir model/ \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file model.onnx \
            --enable_dev_version True \
            --opset_version 11
```
To change the ir_version, you can refer to the following code snippet:
```python
import onnx
model = onnx.load("model.onnx")
model.ir_version = 7
onnx.save(model, "model.onnx")
```
Once you have converted the model to ONNX format, you can begin the process of converting it to a Horizon model. The official documentation provides detailed instructions on how to do this. Here is an example of converting a MobileNetv2 model.

Assuming you have already entered the Docker instance, you can navigate to the following directory by running the command:
```bash
cd ddk/samples/ai_toolchain/horizon_model_convert_sample/03_classification/04_mobilenet_onnx/mapper/
```
The directory contains three scripts that are mainly used for model conversion: `01_check.sh`, `02_preprocess.sh`, and `03_build.sh`. Below are some important points to keep in mind when using these scripts:

`01_check.sh` script is used to check the model and the environment before starting the conversion process. You only need to modify the caffe_model variable to point to the path of your ONNX model.

```bash
set -ex
cd $(dirname $0) || exit

model_type="onnx"
caffe_model="../../../01_common/model_zoo/mapper/classification/mobilenet_onnx/mobilenetv2.onnx"
march="bernoulli2"

hb_mapper checker --model-type ${model_type} \
                  --model ${caffe_model} \
                  --march ${march}
```

`02_preprocess.sh`, Preparing data for quantization requires selecting a configuration. For FastDeploy, the following configuration is selected.


```bash
python3 ../../../data_preprocess.py \
  --src_dir ../../../01_common/calibration_data/imagenet \
  --dst_dir ./calibration_data_rgb \
  --pic_ext .rgb \
  --read_mode opencv \
  --saved_data_type uint8
```

To convert the ONNX model to a Horizon runnable model, you can use the `03_build.sh` script. This script requires several input parameters to configure the conversion process, including the input data format, batch size, and input and output node names
For configuring the model path in FastDeploy, you need to specify the following parameters:


```yaml
model_parameters:
  # the model file of floating-point ONNX neural network data
  onnx_model: '../../../01_common/model_zoo/mapper/classification/mobilenet_onnx/mobilenetv2.onnx'
 
  # the applicable BPU architecture
  march: "bernoulli2"

  # specifies whether or not to dump the intermediate results of all layers in conversion
  # if set to True, then the intermediate results of all layers shall be dumped
  layer_out_dump: False

  # the directory in which model conversion results are stored
  working_dir: 'model_output_rgb'

  # model conversion generated name prefix of those model files used for dev board execution
  output_model_file_prefix: 'mobilenetv2_224x224_rgb'

```
The configuration for the input format of the model is as follows:

```yaml

input_parameters:

  # (Optional) node name of model input,
  # it shall be the same as the name of model file, otherwise an error will be reported,
  # the node name of model file will be used when left blank
  input_name: ""

  # the data formats to be passed into neural network when actually performing neural network
  # available options: nv12/rgb/bgr/yuv444/gray/featuremap,
  input_type_rt: 'rgb'

  # the data layout formats to be passed into neural network when actually performing neural network, available options: NHWC/NCHW
  # If input_type_rt is configured as nv12, then this parameter does not need to be configured
  input_layout_rt: 'NHWC'

  # the data formats in network training
  # available options: rgb/bgr/gray/featuremap/yuv444
  input_type_train: 'rgb'

  # the data layout in network training, available options: NHWC/NCHW
  input_layout_train: 'NCHW'

  # (Optional)the input size of model network, seperated by 'x'
  # note that the network input size of model file will be used if left blank
  # otherwise it will overwrite the input size of model file
  input_shape: ''

  # the data batch_size to be passed into neural network when actually performing neural network, default value: 1
  #input_batch: 1
  
  # preprocessing methods of network input, available options:
  # 'no_preprocess' indicates that no preprocess will be made 
  # 'data_mean' indicates that to minus the channel mean, i.e. mean_value
  # 'data_scale' indicates that image pixels to multiply data_scale ratio
  # 'data_mean_and_scale' indicates that to multiply scale ratio after channel mean is minused
  norm_type: 'data_mean_and_scale'

  # the mean value minused by image
  # note that values must be seperated by space if channel mean value is used
  mean_value: 123.675 116.28 103.53

  # scale value of image preprocess
  # note that values must be seperated by space if channel scale value is used
  scale_value: 0.01712 0.0175 0.01743

```
The configuration for the quantization parameters of the model is as follows:

```yaml
calibration_parameters:

  # the directory where reference images of model quantization are stored
  # image formats include JPEG, BMP etc.
  # should be classic application scenarios, usually 20~100 images are picked out from test datasets
  # in addition, note that input images should cover typical scenarios
  # and try to avoid those overexposed, oversaturated, vague, 
  # pure blank or pure white images
  # use ';' to seperate when there are multiple input nodes
  cal_data_dir: './calibration_data_rgb'

  # calibration data binary file save type, available options: float32, uint8
#   cal_data_type: 'float32'

  # In case the size of input image file is different from that of in model training
  # and that preprocess_on is set to True,
  # shall the default preprocess method(skimage resize) be used
  # i.e., to resize or crop input image into specified size
  # otherwise user must keep image size as that of in training in advance
  # preprocess_on: False

  # The algorithm type of model quantization, support default, mix, kl, max, load, usually use default can meet the requirements.
  # If it does not meet the expectation, you can try to change it to mix first. If there is still no expectation, try kl or max again.
  # When using QAT to export the model, this parameter should be set to load.
  # For more details of the parameters, please refer to the parameter details in PTQ Principle And Steps section of the user manual.
  calibration_type: 'max'

  # this is the parameter of the 'max' calibration method and it is used for adjusting the intercept point of the 'max' calibration.
  # this parameter will only become valid when the calibration_type is specified as 'max'.
  # RANGE: 0.0 - 1.0. Typical options includes: 0.99999/0.99995/0.99990/0.99950/0.99900.
  max_percentile: 0.9999
```
The remaining parameters are set to their default values, and run `03_build.sh`.
```bash
config_file="./mobilenetv2_config.yaml"
model_type="onnx"
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
```
By now, the converted model file (with the suffix .bin) will be generated in `model_output_rgb` in the same directory.




