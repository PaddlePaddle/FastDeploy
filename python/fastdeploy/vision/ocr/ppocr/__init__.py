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


def sort_boxes(boxes):
    return C.vision.ocr.sort_boxes(boxes)


class DBDetectorPreprocessor:
    def __init__(self):
        """Create a preprocessor for DBDetectorModel
        """
        self._preprocessor = C.vision.ocr.DBDetectorPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for DBDetectorModel
        :param: input_ims: (list of numpy.ndarray) The input image
        :return: pair(list of FDTensor, list of std::array<int, 4>)
        """
        return self._preprocessor.run(input_ims)

    @property
    def max_side_len(self):
        return self._preprocessor.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int), "The value to set `max_side_len` must be type of int."
        self._preprocessor.max_side_len = value

    @property
    def is_scale(self):
        return self._preprocessor.is_scale

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._preprocessor.is_scale = value

    @property
    def scale(self):
        return self._preprocessor.scale

    @scale.setter
    def scale(self, value):
        assert isinstance(
            value, list), "The value to set `scale` must be type of list."
        self._preprocessor.scale = value

    @property
    def mean(self):
        return self._preprocessor.mean

    @mean.setter
    def mean(self, value):
        assert isinstance(
            value, list), "The value to set `mean` must be type of list."
        self._preprocessor.mean = value


class DBDetectorPostprocessor:
    def __init__(self):
        """Create a postprocessor for DBDetectorModel
        """
        self._postprocessor = C.vision.ocr.DBDetectorPostprocessor()

    def run(self, runtime_results, batch_det_img_info):
        """Postprocess the runtime results for DBDetectorModel
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :param: batch_det_img_info: (list of std::array<int, 4>)The output of det_preprocessor
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results, batch_det_img_info)

    @property
    def det_db_thresh(self):
        return self._postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `det_db_thresh` must be type of float."
        self._postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        return self._postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        return self._postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_score_mode(self):
        return self._postprocessor.det_db_score_mode

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        assert isinstance(
            value,
            str), "The value to set `det_db_score_mode` must be type of str."
        self._postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        return self._postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        assert isinstance(
            value,
            bool), "The value to set `use_dilation` must be type of bool."
        self._postprocessor.use_dilation = value


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
            self._runnable = False
        else:
            self._model = C.vision.ocr.DBDetector(
                model_file, params_file, self._runtime_option, model_format)
            assert self.initialized, "DBDetector initialize failed."
            self._runnable = True

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: boxes
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: batch_boxes
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value

    # Det Preprocessor Property
    @property
    def max_side_len(self):
        return self._model.preprocessor.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int), "The value to set `max_side_len` must be type of int."
        self._model.preprocessor.max_side_len = value

    @property
    def is_scale(self):
        return self._model.preprocessor.is_scale

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._model.preprocessor.is_scale = value

    @property
    def scale(self):
        return self._model.preprocessor.scale

    @scale.setter
    def scale(self, value):
        assert isinstance(
            value, list), "The value to set `scale` must be type of list."
        self._model.preprocessor.scale = value

    @property
    def mean(self):
        return self._model.preprocessor.mean

    @mean.setter
    def mean(self, value):
        assert isinstance(
            value, list), "The value to set `mean` must be type of list."
        self._model.preprocessor.mean = value

    # Det Ppstprocessor Property
    @property
    def det_db_thresh(self):
        return self._model.postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `det_db_thresh` must be type of float."
        self._model.postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        return self._model.postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._model.postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        return self._model.postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._model.postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_score_mode(self):
        return self._model.postprocessor.det_db_score_mode

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        assert isinstance(
            value,
            str), "The value to set `det_db_score_mode` must be type of str."
        self._model.postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        return self._model.postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        assert isinstance(
            value,
            bool), "The value to set `use_dilation` must be type of bool."
        self._model.postprocessor.use_dilation = value


class ClassifierPreprocessor:
    def __init__(self):
        """Create a preprocessor for ClassifierModel
        """
        self._preprocessor = C.vision.ocr.ClassifierPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for ClassifierModel
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)

    @property
    def is_scale(self):
        return self._preprocessor.is_scale

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._preprocessor.is_scale = value

    @property
    def scale(self):
        return self._preprocessor.scale

    @scale.setter
    def scale(self, value):
        assert isinstance(
            value, list), "The value to set `scale` must be type of list."
        self._preprocessor.scale = value

    @property
    def mean(self):
        return self._preprocessor.mean

    @mean.setter
    def mean(self, value):
        assert isinstance(
            value, list), "The value to set `mean` must be type of list."
        self._preprocessor.mean = value

    @property
    def cls_image_shape(self):
        return self._preprocessor.cls_image_shape

    @cls_image_shape.setter
    def cls_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `cls_image_shape` must be type of list."
        self._preprocessor.cls_image_shape = value


class ClassifierPostprocessor:
    def __init__(self):
        """Create a postprocessor for ClassifierModel
        """
        self._postprocessor = C.vision.ocr.ClassifierPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for ClassifierModel
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)

    @property
    def cls_thresh(self):
        return self._postprocessor.cls_thresh

    @cls_thresh.setter
    def cls_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `cls_thresh` must be type of float."
        self._postprocessor.cls_thresh = value


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
            self._runnable = False
        else:
            self._model = C.vision.ocr.Classifier(
                model_file, params_file, self._runtime_option, model_format)
            assert self.initialized, "Classifier initialize failed."
            self._runnable = True

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: cls_label, cls_score
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of cls_label, list of cls_score
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value

    # Cls Preprocessor Property
    @property
    def is_scale(self):
        return self._model.preprocessor.is_scale

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._model.preprocessor.is_scale = value

    @property
    def scale(self):
        return self._model.preprocessor.scale

    @scale.setter
    def scale(self, value):
        assert isinstance(
            value, list), "The value to set `scale` must be type of list."
        self._model.preprocessor.scale = value

    @property
    def mean(self):
        return self._model.preprocessor.mean

    @mean.setter
    def mean(self, value):
        assert isinstance(
            value, list), "The value to set `mean` must be type of list."
        self._model.preprocessor.mean = value

    @property
    def cls_image_shape(self):
        return self._model.preprocessor.cls_image_shape

    @cls_image_shape.setter
    def cls_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `cls_image_shape` must be type of list."
        self._model.preprocessor.cls_image_shape = value

    # Cls Postprocessor Property
    @property
    def cls_thresh(self):
        return self._model.postprocessor.cls_thresh

    @cls_thresh.setter
    def cls_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `cls_thresh` must be type of float."
        self._model.postprocessor.cls_thresh = value


class RecognizerPreprocessor:
    def __init__(self):
        """Create a preprocessor for RecognizerModel
        """
        self._preprocessor = C.vision.ocr.RecognizerPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for RecognizerModel
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)

    @property
    def static_shape(self):
        return self._preprocessor.static_shape

    @static_shape.setter
    def static_shape(self, value):
        assert isinstance(
            value,
            bool), "The value to set `static_shape` must be type of bool."
        self._preprocessor.static_shape = value

    @property
    def is_scale(self):
        return self._preprocessor.is_scale

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._preprocessor.is_scale = value

    @property
    def scale(self):
        return self._preprocessor.scale

    @scale.setter
    def scale(self, value):
        assert isinstance(
            value, list), "The value to set `scale` must be type of list."
        self._preprocessor.scale = value

    @property
    def mean(self):
        return self._preprocessor.mean

    @mean.setter
    def mean(self, value):
        assert isinstance(
            value, list), "The value to set `mean` must be type of list."
        self._preprocessor.mean = value

    @property
    def rec_image_shape(self):
        return self._preprocessor.rec_image_shape

    @rec_image_shape.setter
    def rec_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `rec_image_shape` must be type of list."
        self._preprocessor.rec_image_shape = value


class RecognizerPostprocessor:
    def __init__(self, label_path):
        """Create a postprocessor for RecognizerModel
        :param label_path: (str)Path of label file
        """
        self._postprocessor = C.vision.ocr.RecognizerPostprocessor(label_path)

    def run(self, runtime_results):
        """Postprocess the runtime results for RecognizerModel
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


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
            self._runnable = False
        else:
            self._model = C.vision.ocr.Recognizer(
                model_file, params_file, label_path, self._runtime_option,
                model_format)
            assert self.initialized, "Recognizer initialize failed."
            self._runnable = True

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: rec_text, rec_score
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of rec_text, list of rec_score
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value

    @property
    def static_shape(self):
        return self._model.preprocessor.static_shape

    @static_shape.setter
    def static_shape(self, value):
        assert isinstance(
            value,
            bool), "The value to set `static_shape` must be type of bool."
        self._model.preprocessor.static_shape = value

    @property
    def is_scale(self):
        return self._model.preprocessor.is_scale

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._model.preprocessor.is_scale = value

    @property
    def scale(self):
        return self._model.preprocessor.scale

    @scale.setter
    def scale(self, value):
        assert isinstance(
            value, list), "The value to set `scale` must be type of list."
        self._model.preprocessor.scale = value

    @property
    def mean(self):
        return self._model.preprocessor.mean

    @mean.setter
    def mean(self, value):
        assert isinstance(
            value, list), "The value to set `mean` must be type of list."
        self._model.preprocessor.mean = value

    @property
    def rec_image_shape(self):
        return self._model.preprocessor.rec_image_shape

    @rec_image_shape.setter
    def rec_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `rec_image_shape` must be type of list."
        self._model.preprocessor.rec_image_shape = value


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

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """
        return self.system.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `cls_batch_size` must be type of int."
        self.system.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `rec_batch_size` must be type of int."
        self.system.rec_batch_size = value


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

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """

        return self.system.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `cls_batch_size` must be type of int."
        self.system.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `rec_batch_size` must be type of int."
        self.system.rec_batch_size = value


class PPOCRSystemv2(PPOCRv2):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv2 is deprecated, "
            "please use fd.vision.ocr.PPOCRv2 instead.")
        super(PPOCRSystemv2, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv2, self).predict(input_image)
