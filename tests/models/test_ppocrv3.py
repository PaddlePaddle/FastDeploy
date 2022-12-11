import fastdeploy as fd
import cv2
import os
import runtime_config as rc
import numpy as np
import math
import pickle

det_model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"
cls_model_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
rec_model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"
img_url = "https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg"
label_url = "https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt"
result_url = "https://bj.bcebos.com/fastdeploy/tests/data/ocr_result.pickle"
fd.download_and_decompress(det_model_url, "resources")
fd.download_and_decompress(cls_model_url, "resources")
fd.download_and_decompress(rec_model_url, "resources")
fd.download(img_url, "resources")
fd.download(result_url, "resources")
fd.download(label_url, "resources")


def get_rotate_crop_image(img, box):
    points = []
    for i in range(4):
        points.append([box[2 * i], box[2 * i + 1]])
    points = np.array(points, dtype=np.float32)
    img = img.astype(np.float32)
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


option = fd.RuntimeOption()

# det_model
det_model_path = "resources/ch_PP-OCRv3_det_infer/"
det_model_file = det_model_path + "inference.pdmodel"
det_params_file = det_model_path + "inference.pdiparams"

det_preprocessor = fd.vision.ocr.DBDetectorPreprocessor()

rc.test_option.set_model_path(det_model_file, det_params_file)
det_runtime = fd.Runtime(rc.test_option)

det_postprocessor = fd.vision.ocr.DBDetectorPostprocessor()

det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=option)

# cls_model
cls_model_path = "resources/ch_ppocr_mobile_v2.0_cls_infer/"
cls_model_file = cls_model_path + "inference.pdmodel"
cls_params_file = cls_model_path + "inference.pdiparams"

cls_preprocessor = fd.vision.ocr.ClassifierPreprocessor()

rc.test_option.set_model_path(cls_model_file, cls_params_file)
cls_runtime = fd.Runtime(rc.test_option)

cls_postprocessor = fd.vision.ocr.ClassifierPostprocessor()

cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=option)

#rec_model
rec_model_path = "resources/ch_PP-OCRv3_rec_infer/"
rec_model_file = rec_model_path + "inference.pdmodel"
rec_params_file = rec_model_path + "inference.pdiparams"
rec_label_file = "resources/ppocr_keys_v1.txt"

rec_preprocessor = fd.vision.ocr.RecognizerPreprocessor()

rc.test_option.set_model_path(rec_model_file, rec_params_file)
rec_runtime = fd.Runtime(rc.test_option)

rec_postprocessor = fd.vision.ocr.RecognizerPostprocessor(rec_label_file)

rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=option)

#pp_ocrv3
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

#pp_ocrv3_no_cls
ppocr_v3_no_cls = fd.vision.ocr.PPOCRv3(
    det_model=det_model, rec_model=rec_model)

#input image
img_file = "resources/12.jpg"
im = []
im.append(cv2.imread(img_file))
im.append(cv2.imread(img_file))

result_file = "resources/ocr_result.pickle"
with open(result_file, 'rb') as f:
    boxes, cls_labels, cls_scores, text, rec_scores = pickle.load(f)
    base_boxes = np.array(boxes)
    base_cls_labels = np.array(cls_labels)
    base_cls_scores = np.array(cls_scores)
    base_text = text
    base_rec_scores = np.array(rec_scores)


def compare_result(pred_boxes, pred_cls_labels, pred_cls_scores, pred_text,
                   pred_rec_scores):
    pred_boxes = np.array(pred_boxes)
    pred_cls_labels = np.array(pred_cls_labels)
    pred_cls_scores = np.array(pred_cls_scores)
    pred_text = pred_text
    pred_rec_scores = np.array(pred_rec_scores)

    diff_boxes = np.fabs(base_boxes - pred_boxes).max()
    diff_cls_labels = np.fabs(base_cls_labels - pred_cls_labels).max()
    diff_cls_scores = np.fabs(base_cls_scores - pred_cls_scores).max()
    diff_text = (base_text != pred_text)
    diff_rec_scores = np.fabs(base_rec_scores - pred_rec_scores).max()

    print('diff:', diff_boxes, diff_cls_labels, diff_cls_scores, diff_text,
          diff_rec_scores)
    diff_threshold = 1e-01
    assert diff_boxes < diff_threshold, "There is diff in boxes"
    assert diff_cls_labels < diff_threshold, "There is diff in cls_label"
    assert diff_cls_scores < diff_threshold, "There is diff in cls_scores"
    assert diff_text < diff_threshold, "There is diff in text"
    assert diff_rec_scores < diff_threshold, "There is diff in rec_scores"


def compare_result_no_cls(pred_boxes, pred_text, pred_rec_scores):
    pred_boxes = np.array(pred_boxes)
    pred_text = pred_text
    pred_rec_scores = np.array(pred_rec_scores)

    diff_boxes = np.fabs(base_boxes - pred_boxes).max()
    diff_text = (base_text != pred_text)
    diff_rec_scores = np.fabs(base_rec_scores - pred_rec_scores).max()

    print('diff:', diff_boxes, diff_text, diff_rec_scores)
    diff_threshold = 1e-01
    assert diff_boxes < diff_threshold, "There is diff in boxes"
    assert diff_text < diff_threshold, "There is diff in text"
    assert diff_rec_scores < diff_threshold, "There is diff in rec_scores"


def test_ppocr_v3():
    ppocr_v3.cls_batch_size = -1
    ppocr_v3.rec_batch_size = -1
    ocr_result = ppocr_v3.predict(im[0])
    compare_result(ocr_result.boxes, ocr_result.cls_labels,
                   ocr_result.cls_scores, ocr_result.text,
                   ocr_result.rec_scores)

    ppocr_v3.cls_batch_size = 2
    ppocr_v3.rec_batch_size = 2
    ocr_result = ppocr_v3.predict(im[0])
    compare_result(ocr_result.boxes, ocr_result.cls_labels,
                   ocr_result.cls_scores, ocr_result.text,
                   ocr_result.rec_scores)


def test_ppocr_v3_1():
    ppocr_v3_no_cls.cls_batch_size = -1
    ppocr_v3_no_cls.rec_batch_size = -1
    ocr_result = ppocr_v3_no_cls.predict(im[0])
    compare_result_no_cls(ocr_result.boxes, ocr_result.text,
                          ocr_result.rec_scores)

    ppocr_v3_no_cls.cls_batch_size = 2
    ppocr_v3_no_cls.rec_batch_size = 2
    ocr_result = ppocr_v3_no_cls.predict(im[0])
    compare_result_no_cls(ocr_result.boxes, ocr_result.text,
                          ocr_result.rec_scores)


def test_ppocr_v3_2():
    det_input_tensors, batch_det_img_info = det_preprocessor.run(im)
    det_output_tensors = det_runtime.infer({"x": det_input_tensors[0]})
    det_results = det_postprocessor.run(det_output_tensors, batch_det_img_info)

    batch_boxes = []

    batch_cls_labels = []
    batch_cls_scores = []

    batch_rec_texts = []
    batch_rec_scores = []

    for i_batch in range(len(det_results)):
        cls_labels = []
        cls_scores = []
        rec_texts = []
        rec_scores = []
        box_list = fd.vision.ocr.sort_boxes(det_results[i_batch])
        batch_boxes.append(box_list)
        image_list = []
        if len(box_list) == 0:
            image_list.append(im[i_batch])
        else:
            for box in box_list:
                crop_img = get_rotate_crop_image(im[i_batch], box)
                image_list.append(crop_img)

        cls_input_tensors = cls_preprocessor.run(image_list)
        cls_output_tensors = cls_runtime.infer({"x": cls_input_tensors[0]})
        cls_labels, cls_scores = cls_postprocessor.run(cls_output_tensors)

        batch_cls_labels.append(cls_labels)
        batch_cls_scores.append(cls_scores)

        for index in range(len(image_list)):
            if cls_labels[index] == 1 and cls_scores[
                    index] > cls_postprocessor.cls_thresh:
                image_list[index] = cv2.rotate(
                    image_list[index].astype(np.float32), 1)
                image_list[index] = np.astype(np.uint8)

        rec_input_tensors = rec_preprocessor.run(image_list)
        rec_output_tensors = rec_runtime.infer({"x": rec_input_tensors[0]})
        rec_texts, rec_scores = rec_postprocessor.run(rec_output_tensors)

        batch_rec_texts.append(rec_texts)
        batch_rec_scores.append(rec_scores)

        compare_result(box_list, cls_labels, cls_scores, rec_texts, rec_scores)


if __name__ == "__main__":
    print("test test_ppocr_v3")
    test_ppocr_v3()
    test_ppocr_v3()
    print("test test_ppocr_v3_1")
    test_ppocr_v3_1()
    test_ppocr_v3_1()
    print("test test_ppocr_v3_2")
    test_ppocr_v3_2()
    test_ppocr_v3_2()
