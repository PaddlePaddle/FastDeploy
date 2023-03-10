#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from openvino.inference_engine import IECore
import numpy as np


class Predictor:
    def __init__(self, xml_file, bin_file, device="CPU"):
        ie = IECore()
        net = ie.read_network(model=xml_file, weights=bin_file)
        net.batch_size = 1
        network_config = dict()
        if device == "MYRIAD":
            network_config = {'VPU_HW_STAGES_OPTIMIZATION': 'NO'}
        self.exec_net = ie.load_network(
            network=net, device_name=device, config=network_config)

    def run(self, input_dict):
        return self.exec_net.infer(inputs=input_dict)


class YOLOv3:
    def __init__(self,
                 xml_file,
                 bin_file,
                 model_input_shape=[608, 608],
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        Parameters
        ------
        xml_file: path of converted openvino ir file, e.g yolov3.xml
        bin_file: path of converted openvino ir file, e.g yolov3.bin
        model_input_shape: 模型的输入大小，输入的图像将被resize到此大小，如(608, 608); 
            注意不是输入图像大小，而是模型的输入大小，需与模型一致
        mean: 图像均值，输入的图像将以此均值归一化
        std: 图像方差，输入的图像将以此方差归一化
        """

        self.model_input_shape = model_input_shape
        self.mean = mean
        self.std = std
        self.predictor = Predictor(xml_file, bin_file)

    def preprocess(self, image_file):
        """
        输入的图像进行预处理，包括resize, 归一化
        并返回模型所需的3个输入
        image: 预处理后的图像数据
        im_shape: 预处理后的图像大小
        scale_factor: 原图在处理后，分别在高和宽上的缩放系数
        """
        import cv2

        def resize(im, height, width, interp=cv2.INTER_CUBIC):
            scale_h = height / float(im.shape[0])
            scale_w = width / float(im.shape[1])
            im = cv2.resize(
                im, None, None, fx=scale_w, fy=scale_h, interpolation=interp)
            return im, scale_h, scale_w

        def normalize(im, mean, std, is_scale=True):
            if is_scale:
                im = im / 255.0
            im -= mean
            im /= std
            return im

        im = cv2.imread(image_file)
        if im is None:
            raise Exception("Can not read image file: {} by cv2.imread".format(
                image_file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im, scale_h, scale_w = resize(im, self.model_input_shape[0],
                                      self.model_input_shape[1])
        im = normalize(im, self.mean, self.std)

        # 数据格式由HWC转为NCHW
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        inputs = dict()
        inputs["image"] = im
        inputs["im_shape"] = np.array(im.shape[2:]).astype("float32")
        inputs["scale_factor"] = np.array([scale_h, scale_w]).astype("float32")
        return inputs

    def postprocess(self, results, threshold):
        """
        返回结果为numpy.ndarray, shape为[N, 6]
        其中N表示检测框的个数
        每个检测框共6个数值，分别为[类别ID，置信度Score, 左上角xmin, 左上角ymin，右下角xmax, 右下角ymax]

        Parameters
        ----------
        results: 模型的直接输出结果，其中类别ID小于0的皆为无效框
        threshold: 根据threshold过滤低置信度框
        """
        boxes = None
        for k, v in results.items():
            if len(v.shape) == 2 and v.shape[1] == 6:
                boxes = v
                break
        # 过滤类别ID小于0的结果
        filtered_boxes = boxes[boxes[:, 0] > -1e-06]
        # 过滤低置信度结果
        filtered_boxes = filtered_boxes[filtered_boxes[:, 1] >= threshold]
        return filtered_boxes

    def visualize(self, image_file, boxes, save_path):
        import cv2
        try:
            import paddlex as pdx
        except Exception as e:
            raise Exception(
                "Error happend: {}, if you haven't installed paddlex, please try to `pip install paddlex`"
            )
        packed_result = list()
        for i in range(len(boxes)):
            label_id, score, xmin, ymin, xmax, ymax = boxes[i].flatten().tolist(
            )
            label_id = int(label_id)
            w = xmax - xmin
            h = ymax - ymin
            result = dict()
            result['category'] = str(label_id)
            result['score'] = score
            result['bbox'] = [xmin, ymin, w, h]
            packed_result.append(result)
        vis_result = pdx.det.visualize(image_file, packed_result, 0.0, None)
        cv2.imwrite(save_path, vis_result)

    def predict(self,
                image_file,
                visualize_out='./visualized_result.jpg',
                threshold=0.0):
        input_dict = self.preprocess(image_file)
        results = self.predictor.run(input_dict)
        boxes = self.postprocess(results, threshold)
        if visualize_out is not None:
            self.visualize(image_file, boxes, visualize_out)
        return boxes
