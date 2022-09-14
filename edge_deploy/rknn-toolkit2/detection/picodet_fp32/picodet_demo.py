import argparse
from picodet_tool import PicodetPreProcess, hard_nms, softmax, warp_boxes, draw_box, label_list
import numpy as np
import cv2
import os
from pathlib import Path
import json
import time

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="./model/picodet.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        '--export_rknn_path',
        dest='export_rknn_path',
        default="./model/picodet.rknn",
        type=str,
        help="path of rknn_model to export")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        default="./images/bus.jpg",
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--target',
        dest='target',
        help='The target for build rknn.',
        type=str,
        default='RK3568')
    parser.add_argument(
        '--device',
        dest='device',
        help='The device id for build rknn.',
        type=str,
        default='500f14f3ce049dc0')
    return parser.parse_args()

class RKNNConfig:
    def __init__(self, onnx_path='./model/picodet.onnx',
                 export_rknn_path='./model/picodet.rknn',
                 target='RK3568',
                 device='500f14f3ce049dc0'):
        self.model_path = onnx_path
        self.export_rknn_path = export_rknn_path
        self.target = target
        self.device = device

    def create_rknn(self):
        from rknn.api import RKNN
        # create rknn
        self.rknn = RKNN()
        # pre-process config
        self.rknn.config(mean_values=[[0, 0, 0]],
                         std_values=[[1, 1, 1]],
                         target_platform=self.target)

        # Load ONNX model
        ret = self.rknn.load_onnx(model=self.model_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)

        # Build model
        ret = self.rknn.build(do_quantization=False)
        if ret != 0:
            print('[RKNNConfig]error :Build model failed!')
            exit(ret)

        self.rknn.export_rknn(self.export_rknn_path)

        if self.device == 'pc':
            ret = self.rknn.init_runtime()
        else:
            ret = self.rknn.init_runtime(self.target, device_id=self.device)
        if ret != 0:
            print('[RKNNConfig] error :Init runtime environment failed!')
            exit(ret)
        return self.rknn

class Picodet:
    def __init__(self,
                 onnx_model_path,
                 export_rknn_path,
                 target_size=None,
                 strides=None,
                 score_threshold=0.01,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100,
                 re_shape=320,
                 target="RK3566",
                 device='500f14f3ce049dc0'):
        if strides is None:
            strides = [8, 16, 32, 64]
        if target_size is None:
            target_size = [320, 320]
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.re_shape = re_shape
        self.strides = strides
        self.target_size = target_size
        rknn_config = RKNNConfig(onnx_model_path, export_rknn_path, target, device)
        self.model = rknn_config.create_rknn()

    def infer_by_rknn(self, img):
        pic_pre_process = PicodetPreProcess(target_size=self.target_size)
        inputs, src_image = pic_pre_process.get_inputs(img)
        new_inputs = inputs.transpose((1, 2, 0))  # chw hwc
        inputs = np.array((inputs,))
        start_time = time.time()
        result = self.model.inference([new_inputs])
        end_time = time.time()
        #perf_results = self.model.eval_perf(inputs=[new_inputs])
        return result, inputs, src_image, end_time - start_time

    def detect(self, scores, raw_boxes, input_shape):
        # detect
        test_im_shape = np.array([[self.re_shape, self.re_shape]]).astype('float32')
        test_scale_factor = np.array([[1, 1]]).astype('float32')
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes, scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = input_shape[0] / stride
                fm_w = input_shape[1] / stride
                # print("fm_h = {},fm_w = {}".format(fm_h, fm_w))
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                # print("ct_row.shape = {},ct_col.shape = {}".format(ct_row.shape, ct_col.shape))
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                # print("softmax shape =", box_distance.shape)
                box_distance = softmax(box_distance)
                # print("softmax shape =", box_distance.shape)
                box_distance = box_distance * np.expand_dims(reg_range, axis=0)
                box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
                box_distance = box_distance * stride

                # top K candidate
                topk_idx = np.argsort(score.max(axis=1))[::-1]
                topk_idx = topk_idx[:self.nms_top_k]
                center = center[topk_idx]
                score = score[topk_idx]
                box_distance = box_distance[topk_idx]

                # decode box
                decode_box = center + [-1, -1, 1, 1] * box_distance

                select_scores.append(score)
                decode_boxes.append(decode_box)

            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k, )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))
                out_boxes_num.append(0)

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, :4] = warp_boxes(
                    picked_box_probs[:, :4], test_im_shape[batch_id])
                im_scale = np.concatenate([
                    test_scale_factor[batch_id][::-1],
                    test_scale_factor[batch_id][::-1]
                ])
                picked_box_probs[:, :4] /= im_scale
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                            picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))
                out_boxes_num.append(len(picked_labels))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        out_boxes_num = np.asarray(out_boxes_num).astype(np.int32)
        return out_boxes_list, out_boxes_num

    def predict(self, img):
        result, image, src_image, predict_time= self.infer_by_rknn(img)
        np_score_list = []
        np_boxes_list = []
        num_outs = int(len(result) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(result[out_idx])
            np_boxes_list.append(result[out_idx + num_outs])

        out_boxes_list, out_boxes_num = self.detect(np_score_list, np_boxes_list, self.target_size)

        scale_x = src_image.shape[1] / image.shape[3]
        scale_y = src_image.shape[0] / image.shape[2]

        res_image = draw_box(src_image, out_boxes_list, label_list, scale_x, scale_y)
        cv2.imwrite('result.jpg', res_image)

if __name__ == "__main__":
    FLAGS = parse_args()
    picodet = Picodet(onnx_model_path=FLAGS.model_path,
                      export_rknn_path=FLAGS.export_rknn_path,
                      target = FLAGS.target,
                      device = FLAGS.device,
                      score_threshold=FLAGS.threshold)
    picodet.predict(FLAGS.image_path)