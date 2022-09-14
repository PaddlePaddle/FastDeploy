import cv2
import numpy as np

label_list = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv monitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

class PicodetPreProcess:
    def __init__(self,
                 target_size=None,
                 interp=cv2.INTER_CUBIC,
                 mean=None,
                 std=None,
                 is_scale=True,
                 stride=32):
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if target_size is None:
            self.target_size = [416, 416]
        else:
            self.target_size = target_size
        self.interp = interp
        self.is_scale = is_scale
        self.stride = stride

    def resize(self, im, im_info):
        target_size = self.target_size
        interp = self.interp
        assert len(target_size) == 2
        assert target_size[0] > 0 and target_size[1] > 0
        origin_shape = im.shape[:2]
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        im = cv2.resize(
            im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def normalizeImage(self, im):
        mean = self.mean
        std = self.std
        is_scale = self.is_scale

        im = im.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]

        if is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im

    def padStride(self, im):
        coarsest_stride = self.stride
        coarsest_stride = coarsest_stride
        if coarsest_stride <= 0:
            return im
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im

    def get_inputs(self, im):
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }
        if isinstance(im, str):
            src_image = cv2.imread(im)
            im = cv2.imread(im)
        else:
            src_image = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im, _ = self.resize(im, im_info)
        im = self.normalizeImage(im)
        im = im.transpose((2, 0, 1))  # chw
        # if FLAGS.model_type == "picodet":
        im = self.padStride(im)
        # im = im.transpose((1, 2, 0))  # hwc

        return im, src_image


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


# 计算IoU，矩形框的坐标形式为xyxy
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def softmax(m):
    shape = m.shape[0]
    m_row_max = m.max(axis=1)
    m = m - m_row_max.reshape(shape, 1)
    m_exp = np.exp(m)
    m_exp_row_sum = m_exp.sum(axis=1).reshape(shape, 1)
    result = m_exp / m_exp_row_sum
    return result


def warp_boxes(boxes, ori_shape):
    """Apply transform to boxes
    """
    width, height = ori_shape[1], ori_shape[0]
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def draw_box(img, results, class_label, scale_x, scale_y):
    label_list = list(
        # 改动点
        map(lambda x: x.strip(), class_label))
    for i in range(len(results)):
        bbox = results[i, 2:]
        label_id = int(results[i, 0])
        score = results[i, 1]
        if (score > 0.20):
            xmin, ymin, xmax, ymax = [
                int(bbox[0] * scale_x), int(bbox[1] * scale_y),
                int(bbox[2] * scale_x), int(bbox[3] * scale_y)
            ]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            # cv2.imshow("name", img)
            # cv2.waitKey(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_text = label_list[label_id]
            print("label: {} ,pred:{}, loc:(min:{},max:{})".format(label_text, str(round(score, 3)), (xmin, ymin),
                                                                   (xmax, ymax)))
            img = cv2.putText(img, (label_text), (xmin, ymin - 40), font, 1,
                              (0, 0, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, str(round(score, 3)), (xmin, ymin), font, 1,
                              (0, 0, 0), 1, cv2.LINE_AA)
    return img
