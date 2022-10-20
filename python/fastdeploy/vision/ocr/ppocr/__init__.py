# # Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from __future__ import absolute_import
import logging
from .... import FastDeployModel, ModelFormat
from .... import c_lib_wrap as C


class DBDetector(FastDeployModel):
    def __init__(self,
                 model_file="",
                 params_file="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load OCR detection model provided by PaddleOCR.

        :param model_file: (str)Path of model file, e.g ./ch_PP-OCRv3_det_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_PP-OCRv3_det_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model.
        """
        super(DBDetector, self).__init__(runtime_option)

        if (len(model_file) == 0):
            self._model = C.vision.ocr.DBDetector()
        else:
            self._model = C.vision.ocr.DBDetector(
                model_file, params_file, self._runtime_option, model_format)
            assert self.initialized, "DBDetector initialize failed."

    # 一些跟DBDetector模型有关的属性封装
    @property
    def max_side_len(self):
        return self._model.max_side_len

    @property
    def det_db_thresh(self):
        return self._model.det_db_thresh

    @property
    def det_db_box_thresh(self):
        return self._model.det_db_box_thresh

    @property
    def det_db_unclip_ratio(self):
        return self._model.det_db_unclip_ratio

    @property
    def det_db_score_mode(self):
        return self._model.det_db_score_mode

    @property
    def use_dilation(self):
        return self._model.use_dilation

    @property
    def is_scale(self):
        return self._model.max_wh

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int), "The value to set `max_side_len` must be type of int."
        self._model.max_side_len = value

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `det_db_thresh` must be type of float."
        self._model.det_db_thresh = value

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._model.det_db_box_thresh = value

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._model.det_db_unclip_ratio = value

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        assert isinstance(
            value,
            str), "The value to set `det_db_score_mode` must be type of str."
        self._model.det_db_score_mode = value

    @use_dilation.setter
    def use_dilation(self, value):
        assert isinstance(
            value,
            bool), "The value to set `use_dilation` must be type of bool."
        self._model.use_dilation = value

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._model.is_scale = value


class Classifier(FastDeployModel):
    def __init__(self,
                 model_file="",
                 params_file="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load OCR classification model provided by PaddleOCR.

        :param model_file: (str)Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model.
        """
        super(Classifier, self).__init__(runtime_option)

        if (len(model_file) == 0):
            self._model = C.vision.ocr.Classifier()
        else:
            self._model = C.vision.ocr.Classifier(
                model_file, params_file, self._runtime_option, model_format)
            assert self.initialized, "Classifier initialize failed."

    @property
    def cls_thresh(self):
        return self._model.cls_thresh

    @property
    def cls_image_shape(self):
        return self._model.cls_image_shape

    @property
    def cls_batch_num(self):
        return self._model.cls_batch_num

    @cls_thresh.setter
    def cls_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `cls_thresh` must be type of float."
        self._model.cls_thresh = value

    @cls_image_shape.setter
    def cls_image_shape(self, value):
        assert isinstance(
            value, list), "The value to set `cls_thresh` must be type of list."
        self._model.cls_image_shape = value

    @cls_batch_num.setter
    def cls_batch_num(self, value):
        assert isinstance(
            value,
            int), "The value to set `cls_batch_num` must be type of int."
        self._model.cls_batch_num = value


class Recognizer(FastDeployModel):
    def __init__(self,
                 model_file="",
                 params_file="",
                 label_path="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load OCR recognition model provided by PaddleOCR

        :param model_file: (str)Path of model file, e.g ./ch_PP-OCRv3_rec_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_PP-OCRv3_rec_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param label_path: (str)Path of label file used by OCR recognition model. e.g ./ppocr_keys_v1.txt
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model.
        """
        super(Recognizer, self).__init__(runtime_option)

        if (len(model_file) == 0):
            self._model = C.vision.ocr.Recognizer()
        else:
            self._model = C.vision.ocr.Recognizer(
                model_file, params_file, label_path, self._runtime_option,
                model_format)
            assert self.initialized, "Recognizer initialize failed."

    @property
    def rec_img_h(self):
        return self._model.rec_img_h

    @property
    def rec_img_w(self):
        return self._model.rec_img_w

    @property
    def rec_batch_num(self):
        return self._model.rec_batch_num

    @rec_img_h.setter
    def rec_img_h(self, value):
        assert isinstance(
            value, int), "The value to set `rec_img_h` must be type of int."
        self._model.rec_img_h = value

    @rec_img_w.setter
    def rec_img_w(self, value):
        assert isinstance(
            value, int), "The value to set `rec_img_w` must be type of int."
        self._model.rec_img_w = value

    @rec_batch_num.setter
    def rec_batch_num(self, value):
        assert isinstance(
            value,
            int), "The value to set `rec_batch_num` must be type of int."
        self._model.rec_batch_num = value


class PPOCRv3(FastDeployModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (FastDeployModel) The detection model object created by fastdeploy.vision.ocr.DBDetector.
        :param cls_model: (FastDeployModel) The classification model object created by fastdeploy.vision.ocr.Classifier.
        :param rec_model: (FastDeployModel) The recognition model object created by fastdeploy.vision.ocr.Recognizer.
        """
        assert det_model is not None and rec_model is not None, "The det_model and rec_model cannot be None."
        if cls_model is None:
            self.system = C.vision.ocr.PPOCRv3(det_model._model,
                                               rec_model._model)
        else:
            self.system = C.vision.ocr.PPOCRv3(
                det_model._model, cls_model._model, rec_model._model)

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system.predict(input_image)


class PPOCRSystemv3(PPOCRv3):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv3 is deprecated, "
            "please use fd.vision.ocr.PPOCRv3 instead.")
        super(PPOCRSystemv3, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv3, self).predict(input_image)


class PPOCRv2(FastDeployModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (FastDeployModel) The detection model object created by fastdeploy.vision.ocr.DBDetector.
        :param cls_model: (FastDeployModel) The classification model object created by fastdeploy.vision.ocr.Classifier.
        :param rec_model: (FastDeployModel) The recognition model object created by fastdeploy.vision.ocr.Recognizer.
        """
        assert det_model is not None and rec_model is not None, "The det_model and rec_model cannot be None."
        if cls_model is None:
            self.system = C.vision.ocr.PPOCRv2(det_model._model,
                                               rec_model._model)
        else:
            self.system = C.vision.ocr.PPOCRv2(
                det_model._model, cls_model._model, rec_model._model)

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system.predict(input_image)


class PPOCRSystemv2(PPOCRv2):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv2 is deprecated, "
            "please use fd.vision.ocr.PPOCRv2 instead.")
        super(PPOCRSystemv2, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv2, self).predict(input_image)
