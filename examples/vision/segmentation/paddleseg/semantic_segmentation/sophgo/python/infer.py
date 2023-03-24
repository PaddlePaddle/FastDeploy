import fastdeploy as fd
import cv2
import os
from subprocess import run

def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto", required=True, help="Auto download, convert, compile and infer if True")
    parser.add_argument("--model", required=True, help="Path of model.")
    parser.add_argument(
        "--config_file", required=True, help="Path of config file.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")

    return parser.parse_args()


def download():
    download_model_str = 'wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz'
    if not os.path.exists('PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz'):
        print(download_model_str)
        run(download_model_str, shell=True)
    tar_str = 'tar xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz'
    if not os.path.exists('./PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer'):
        print(tar_str)
        run(tar_str, shell=True)
    download_script_str = 'wget https://raw.githubusercontent.com/PaddlePaddle/Paddle2ONNX/develop/tools/paddle/paddle_infer_shape.py'
    if not os.path.exists('paddle_infer_shape.py'):
        print(download_script_str)
        run(download_script_str, shell=True)
    download_img_str = 'wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png'
    if not os.path.exists('cityscapes_demo.png'):
        print(download_img_str)
        run(download_img_str, shell=True)

def paddle2onnx():
    paddle_infer_shape_str = 'python3 paddle_infer_shape.py --model_dir PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir pp_liteseg_fix \
                             --input_shape_dict="{\'x\':[1,3,512,512]}"'
    print(paddle_infer_shape_str)
    run(paddle_infer_shape_str, shell=True)
    pp2onnx_str = 'paddle2onnx --model_dir pp_liteseg_fix \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --save_file pp_liteseg.onnx \
                --enable_dev_version True'
    print(pp2onnx_str)
    run(pp2onnx_str, shell=True)

def mlir_prepare():
    mlir_path = os.getenv("MODEL_ZOO_PATH")
    mlir_path = mlir_path[:-13]
    regression_path = os.path.join(mlir_path, 'regression')
    mv_str_list = ['mkdir pp_liteseg', 
                'cp -rf ' + os.path.join(regression_path, 'dataset/COCO2017/') + ' ./pp_liteseg', 
                'cp -rf ' + os.path.join(regression_path, 'image/') + ' ./pp_liteseg',
                'mv pp_liteseg.onnx ./pp_liteseg',
                'mkdir ./pp_liteseg/workspace']
    for str in mv_str_list:
        print(str)
        run(str, shell=True)

def onnx2mlir():
    transform_str = 'model_transform.py \
        --model_name pp_liteseg \
        --model_def ../pp_liteseg.onnx \
        --input_shapes [[1,3,512,512]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names bilinear_interp_v2_6.tmp_0 \
        --test_input ../image/dog.jpg \
        --test_result pp_liteseg_top_outputs.npz \
        --mlir pp_liteseg.mlir'
    print(transform_str)
    os.chdir('./pp_liteseg/workspace')
    run(transform_str, shell=True)
    os.chdir('../../')

def mlir2bmodel():
    deploy_str = 'model_deploy.py \
                --mlir pp_liteseg.mlir \
                --quantize F32 \
                --chip bm1684x \
                --test_input pp_liteseg_in_f32.npz \
                --test_reference pp_liteseg_top_outputs.npz \
                --model pp_liteseg_1684x_f32.bmodel'
    print(deploy_str)
    os.chdir('./pp_liteseg/workspace')
    run(deploy_str, shell=True)
    os.chdir('../../')

args = parse_arguments()


if args.auto:
    download()
    paddle2onnx()
    mlir_prepare()
    onnx2mlir()
    mlir2bmodel()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

model_file = './pp_liteseg/workspace/pp_liteseg_1684x_f32.bmodel' if args.auto else args.model
params_file = ""
config_file = './PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer/deploy.yaml' if args.auto else args.config_file
img_file = './cityscapes_demo.png' if args.auto else args.image

model = fd.vision.segmentation.PaddleSegModel(
    model_file,
    params_file,
    config_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.SOPHGO)

# 预测图片分类结果
im_org = cv2.imread(img_file)
#bmodel 是静态模型，模型输入固定，这里设置为[512, 512]
im = cv2.resize(im_org, [512, 512], interpolation=cv2.INTER_LINEAR)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_segmentation(im, result, weight=0.5)
vis_im = cv2.resize(vis_im, [im_org.shape[1], im_org.shape[0]], interpolation=cv2.INTER_LINEAR)
cv2.imwrite("sophgo_img.png", vis_im)
