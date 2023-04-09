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
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization label of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")

    return parser.parse_args()


def getPPOCRv3():
    cmd_str_det = 'wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar'
    tar_str_det = 'tar xvf ch_PP-OCRv3_det_infer.tar'
    cmd_str_cls = 'wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    tar_str_cls = 'tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar'
    cmd_str_rec = 'wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar'
    tar_str_rec = 'tar xvf ch_PP-OCRv3_rec_infer.tar'
    cmd_str_img = 'wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg'
    cmd_str_label = 'wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt'
    script_str = 'wget https://raw.githubusercontent.com/PaddlePaddle/Paddle2ONNX/develop/tools/paddle/paddle_infer_shape.py'
    if not os.path.exists('ch_PP-OCRv3_det_infer.tar'):
        print(cmd_str_det, tar_str_det)
        run(cmd_str_det, shell=True)
        run(tar_str_det, shell=True)
    if not os.path.exists('ch_ppocr_mobile_v2.0_cls_infer.tar'):
        print(cmd_str_cls, tar_str_cls)
        run(cmd_str_cls, shell=True)
        run(tar_str_cls, shell=True)
    if not os.path.exists('ch_PP-OCRv3_rec_infer.tar'):
        print(cmd_str_rec, tar_str_rec)
        run(cmd_str_rec, shell=True)
        run(tar_str_rec, shell=True)
    if not os.path.exists('12.jpg'):
        print(cmd_str_img)
        run(cmd_str_img, shell=True)
    if not os.path.exists('ppocr_keys_v1.txt'):
        print(cmd_str_label)
        run(cmd_str_label, shell=True)
    if not os.path.exists('paddle_infer_shape.py'):
        print(script_str)
        run(script_str, shell=True)

def fix_input_shape():
    fix_det_str = 'python paddle_infer_shape.py --model_dir ch_PP-OCRv3_det_infer \
                    --model_filename inference.pdmodel \
                    --params_filename inference.pdiparams \
                    --save_dir ch_PP-OCRv3_det_infer_fix \
                    --input_shape_dict="{\'x\':[1,3,960,608]}"'
    fix_rec_str = 'python paddle_infer_shape.py --model_dir ch_PP-OCRv3_rec_infer \
                    --model_filename inference.pdmodel \
                    --params_filename inference.pdiparams \
                    --save_dir ch_PP-OCRv3_rec_infer_fix \
                    --input_shape_dict="{\'x\':[1,3,48,320]}"'
    fix_cls_str = 'python paddle_infer_shape.py --model_dir ch_ppocr_mobile_v2.0_cls_infer \
                    --model_filename inference.pdmodel \
                    --params_filename inference.pdiparams \
                    --save_dir ch_PP-OCRv3_cls_infer_fix \
                    --input_shape_dict="{\'x\':[1,3,48,192]}"'
    print(fix_det_str)
    run(fix_det_str, shell=True)
    print(fix_rec_str)
    run(fix_rec_str, shell=True)
    print(fix_cls_str)
    run(fix_cls_str, shell=True)


def paddle2onnx():
    cmd_str_det = 'paddle2onnx --model_dir ch_PP-OCRv3_det_infer_fix \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_det_infer.onnx \
            --enable_dev_version True'
    cmd_str_cls = 'paddle2onnx --model_dir ch_PP-OCRv3_cls_infer_fix \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_cls_infer.onnx \
            --enable_dev_version True'
    cmd_str_rec = 'paddle2onnx --model_dir ch_PP-OCRv3_rec_infer_fix \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_rec_infer.onnx \
            --enable_dev_version True'
    print(cmd_str_det)
    run(cmd_str_det, shell=True)
    print(cmd_str_cls)
    run(cmd_str_cls, shell=True)
    print(cmd_str_rec)
    run(cmd_str_rec, shell=True)

def mlir_prepare():
    mlir_path = os.getenv("MODEL_ZOO_PATH")
    mlir_path = mlir_path[:-13]
    regression_path = os.path.join(mlir_path, 'regression')
    mv_str_list = ['mkdir ch_PP-OCRv3', 
                'cp -rf ' + os.path.join(regression_path, 'dataset/COCO2017/') + ' ./ch_PP-OCRv3', 
                'cp -rf ' + os.path.join(regression_path, 'image/') + ' ./ch_PP-OCRv3',
                'mv ch_PP-OCRv3_det_infer.onnx ./ch_PP-OCRv3',
                'mv ch_PP-OCRv3_rec_infer.onnx ./ch_PP-OCRv3',
                'mv ch_PP-OCRv3_cls_infer.onnx ./ch_PP-OCRv3',
                'mkdir ./ch_PP-OCRv3/workspace']
    for str in mv_str_list:
        print(str)
        run(str, shell=True)


def onnx2mlir():
    transform_str_det = 'model_transform.py \
        --model_name ch_PP-OCRv3_det \
        --model_def ../ch_PP-OCRv3_det_infer.onnx \
        --input_shapes [[1,3,960,608]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names sigmoid_0.tmp_0 \
        --test_input ../image/dog.jpg \
        --test_result ch_PP-OCRv3_det_top_outputs.npz \
        --mlir ./ch_PP-OCRv3_det.mlir'
    transform_str_rec = 'model_transform.py \
        --model_name ch_PP-OCRv3_rec \
        --model_def ../ch_PP-OCRv3_rec_infer.onnx \
        --input_shapes [[1,3,48,320]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names softmax_5.tmp_0 \
        --test_input ../image/dog.jpg \
        --test_result ch_PP-OCRv3_rec_top_outputs.npz \
        --mlir ./ch_PP-OCRv3_rec.mlir'
    transform_str_cls = 'model_transform.py \
        --model_name ch_PP-OCRv3_cls \
        --model_def ../ch_PP-OCRv3_cls_infer.onnx \
        --input_shapes [[1,3,48,192]] \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb \
        --output_names softmax_0.tmp_0 \
        --test_input ../image/dog.jpg \
        --test_result ch_PP-OCRv3_cls_top_outputs.npz \
        --mlir ./ch_PP-OCRv3_cls.mlir'

    os.chdir('./ch_PP-OCRv3/workspace/')

    print(transform_str_det)
    run(transform_str_det, shell=True)

    print(transform_str_rec)
    run(transform_str_rec, shell=True)

    print(transform_str_cls)
    run(transform_str_cls, shell=True)

    os.chdir('../../')

def mlir2bmodel():
    det_str = 'model_deploy.py \
        --mlir ./ch_PP-OCRv3_det.mlir \
        --quantize F32 \
        --chip bm1684x \
        --test_input ./ch_PP-OCRv3_det_in_f32.npz \
        --test_reference ./ch_PP-OCRv3_det_top_outputs.npz \
        --model ./ch_PP-OCRv3_det_1684x_f32.bmodel'
    rec_str = 'model_deploy.py \
        --mlir ./ch_PP-OCRv3_rec.mlir \
        --quantize F32 \
        --chip bm1684x \
        --test_input ./ch_PP-OCRv3_rec_in_f32.npz \
        --test_reference ./ch_PP-OCRv3_rec_top_outputs.npz \
        --model ./ch_PP-OCRv3_rec_1684x_f32.bmodel'
    cls_str = 'model_deploy.py \
        --mlir ./ch_PP-OCRv3_cls.mlir \
        --quantize F32 \
        --chip bm1684x \
        --test_input ./ch_PP-OCRv3_cls_in_f32.npz \
        --test_reference ./ch_PP-OCRv3_cls_top_outputs.npz \
        --model ./ch_PP-OCRv3_cls_1684x_f32.bmodel'
    os.chdir('./ch_PP-OCRv3/workspace/')
    print(det_str)
    run(det_str, shell=True)

    print(rec_str)
    run(rec_str, shell=True)

    print(cls_str)
    run(cls_str, shell=True)
    os.chdir('../../')

args = parse_arguments()

if (args.auto):
    getPPOCRv3()
    fix_input_shape()
    paddle2onnx()
    mlir_prepare()
    onnx2mlir()
    mlir2bmodel()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

# Detection模型, 检测文字框
det_model_file = './ch_PP-OCRv3/workspace/ch_PP-OCRv3_det_1684x_f32.bmodel' if args.auto else args.det_model
det_params_file = ""
# Classification模型，方向分类，可选
cls_model_file = './ch_PP-OCRv3/workspace/ch_PP-OCRv3_cls_1684x_f32.bmodel' if args.auto else args.cls_model
cls_params_file = ""
# Recognition模型，文字识别模型
rec_model_file = './ch_PP-OCRv3/workspace/ch_PP-OCRv3_rec_1684x_f32.bmodel' if args.auto else args.rec_model
rec_params_file = ""
rec_label_file = './ppocr_keys_v1.txt' if args.auto else args.rec_label_file
image_file = './12.jpg' if args.auto else args.image

# PPOCR的cls和rec模型现在已经支持推理一个Batch的数据
# 定义下面两个变量后, 可用于设置trt输入shape, 并在PPOCR模型初始化后, 完成Batch推理设置
cls_batch_size = 1
rec_batch_size = 1

# 当使用TRT时，分别给三个模型的runtime设置动态shape,并完成模型的创建.
# 注意: 需要在检测模型创建完成后，再设置分类模型的动态输入并创建分类模型, 识别模型同理.
# 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
det_option = runtime_option
det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])
# 用户可以把TRT引擎文件保存至本地
# det_option.set_trt_cache_file(args.det_model  + "/det_trt_cache.trt")
det_model = fd.vision.ocr.DBDetector(
    det_model_file,
    det_params_file,
    runtime_option=det_option,
    model_format=fd.ModelFormat.SOPHGO)

cls_option = runtime_option
cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [cls_batch_size, 3, 48, 320],
                               [cls_batch_size, 3, 48, 1024])
# 用户可以把TRT引擎文件保存至本地
# cls_option.set_trt_cache_file(args.cls_model  + "/cls_trt_cache.trt")
cls_model = fd.vision.ocr.Classifier(
    cls_model_file,
    cls_params_file,
    runtime_option=cls_option,
    model_format=fd.ModelFormat.SOPHGO)

rec_option = runtime_option
rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [rec_batch_size, 3, 48, 320],
                               [rec_batch_size, 3, 48, 2304])
# 用户可以把TRT引擎文件保存至本地
# rec_option.set_trt_cache_file(args.rec_model  + "/rec_trt_cache.trt")
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file,
    rec_params_file,
    rec_label_file,
    runtime_option=rec_option,
    model_format=fd.ModelFormat.SOPHGO)

# 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 需要使用下行代码, 来启用rec模型的静态shape推理，这里rec模型的静态输入为[3, 48, 320]
rec_model.preprocessor.static_shape_infer = True
rec_model.preprocessor.rec_image_shape = [3, 48, 320]

# 给cls和rec模型设置推理时的batch size
# 此值能为-1, 和1到正无穷
# 当此值为-1时, cls和rec模型的batch size将默认和det模型检测出的框的数量相同
ppocr_v3.cls_batch_size = cls_batch_size
ppocr_v3.rec_batch_size = rec_batch_size

# 预测图片准备
im = cv2.imread(image_file)

#预测并打印结果
result = ppocr_v3.predict(im)

print(result)

# 可视化结果
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("sophgo_result.jpg", vis_im)
print("Visualized result save in ./sophgo_result.jpg")
