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
    parser.add_argument("--model", help="Path of model.")
    parser.add_argument(
        "--image", type=str, help="Path of test image file.")

    return parser.parse_args()


def download():
    download_model_str = 'wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx'
    if not os.path.exists('yolov5s.onnx'):
        print(download_model_str)
        run(download_model_str, shell=True)
    download_img_str = 'wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg'
    if not os.path.exists('000000014439.jpg'):
        print(download_img_str)
        run(download_img_str, shell=True)
    
    
def mlir_prepare():
    mlir_path = os.getenv("MODEL_ZOO_PATH")
    mlir_path = mlir_path[:-13]
    regression_path = os.path.join(mlir_path, 'regression')
    mv_str_list = ['mkdir YOLOv5s', 
                'cp -rf ' + os.path.join(regression_path, 'dataset/COCO2017/') + ' ./YOLOv5s', 
                'cp -rf ' + os.path.join(regression_path, 'image/') + ' ./YOLOv5s',
                'cp yolov5s.onnx ./YOLOv5s',
                'mkdir ./YOLOv5s/workspace']
    for str in mv_str_list:
        print(str)
        run(str, shell=True)


def onnx2mlir():
    transform_str = 'model_transform.py \
            --model_name yolov5s \
            --model_def ../yolov5s.onnx \
            --input_shapes [[1,3,640,640]] \
            --mean 0.0,0.0,0.0 \
            --scale 0.0039216,0.0039216,0.0039216 \
            --keep_aspect_ratio \
            --pixel_format rgb \
            --output_names output0 \
            --test_input ../image/dog.jpg \
            --test_result yolov5s_top_outputs.npz \
            --mlir yolov5s.mlir'
    os.chdir('./YOLOv5s/workspace')
    print(transform_str)
    run(transform_str, shell=True)
    os.chdir('../../')

def mlir2bmodel():
    deploy_str = 'model_deploy.py \
                --mlir yolov5s.mlir \
                --quantize F32 \
                --chip bm1684x \
                --test_input yolov5s_in_f32.npz \
                --test_reference yolov5s_top_outputs.npz \
                --model yolov5s_1684x_f32.bmodel'

    os.chdir('./YOLOv5s/workspace')
    print(deploy_str)
    run(deploy_str, shell=True)
    os.chdir('../../')


args = parse_arguments()

if args.auto:
    download()
    mlir_prepare()
    onnx2mlir()
    mlir2bmodel()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

model_file = './YOLOv5s/workspace/yolov5s_1684x_f32.bmodel' if args.auto else args.model
params_file = ""
img_file = './000000014439.jpg' if args.auto else args.image

model = fd.vision.detection.YOLOv5(
    model_file,
    params_file,
    runtime_option=runtime_option,
    model_format=fd.ModelFormat.SOPHGO)


# 预测图片分类结果
im = cv2.imread(img_file)
result = model.predict(im)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_detection(im, result)
cv2.imwrite("sophgo_result.jpg", vis_im)
print("Visualized result save in ./sophgo_result.jpg")
