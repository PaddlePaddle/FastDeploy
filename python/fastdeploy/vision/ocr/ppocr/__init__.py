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
from ...common import ProcessorManager


def sort_boxes(boxes):
    return C.vision.ocr.sort_boxes(boxes)


class DBDetectorPreprocessor(ProcessorManager):
    def __init__(self):
        """
        Create a preprocessor for DBDetectorModel
        """
        super(DBDetectorPreprocessor, self).__init__()
        self._manager = C.vision.ocr.DBDetectorPreprocessor()

    @property
    def max_side_len(self):
        """Get max_side_len value.
        """
        return self._manager.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        """Set max_side_len value.
        :param: value: (int) max_side_len value
        """
        assert isinstance(
            value, int), "The value to set `max_side_len` must be type of int."
        self._manager.max_side_len = value

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

    @property
    def static_shape_infer(self):
        return self._manager.static_shape_infer

    @static_shape_infer.setter
    def static_shape_infer(self, value):
        assert isinstance(
            value,
            bool), "The value to set `static_shape_infer` must be type of bool."
        self._manager.static_shape_infer = value

    def disable_normalize(self):
        """
        This function will disable normalize in preprocessing step.
        """
        self._manager.disable_normalize()

    def disable_permute(self):
        """
        This function will disable hwc2chw in preprocessing step.
        """
        self._manager.disable_permute()


class DBDetectorPostprocessor:
    def __init__(self):
        """
        Create a postprocessor for DBDetectorModel
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
        """
        Return the det_db_thresh of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        """Set the det_db_thresh for DBDetectorPostprocessor

        :param: value : the det_db_thresh value
        """
        assert isinstance(
            value,
            float), "The value to set `det_db_thresh` must be type of float."
        self._postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        """
        Return the det_db_box_thresh of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        """Set the det_db_box_thresh for DBDetectorPostprocessor

        :param: value : the det_db_box_thresh value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        """
        Return the det_db_unclip_ratio of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        """Set the det_db_unclip_ratio for DBDetectorPostprocessor

        :param: value : the det_db_unclip_ratio value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_score_mode(self):
        """
        Return the det_db_score_mode of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_score_mode

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        """Set the det_db_score_mode for DBDetectorPostprocessor

        :param: value : the det_db_score_mode value
        """
        assert isinstance(
            value,
            str), "The value to set `det_db_score_mode` must be type of str."
        self._postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        """
        Return the use_dilation of DBDetectorPostprocessor
        """
        return self._postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        """Set the use_dilation for DBDetectorPostprocessor

        :param: value : the use_dilation value
        """
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

    def clone(self):
        """Clone OCR detection model object

        :return: a new OCR detection model object
        """

        class DBDetectorClone(DBDetector):
            def __init__(self, model):
                self._model = model

        clone_model = DBDetectorClone(self._model.clone())
        return clone_model

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

    @property
    def postprocessor(self):
        return self._model.postprocessor

    # Det Preprocessor Property
    @property
    def max_side_len(self):
        return self._model.preprocessor.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int), "The value to set `max_side_len` must be type of int."
        self._model.preprocessor.max_side_len = value

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


class ClassifierPreprocessor(ProcessorManager):
    def __init__(self):
        """Create a preprocessor for ClassifierModel
        """
        super(ClassifierPreprocessor, self).__init__()
        self._manager = C.vision.ocr.ClassifierPreprocessor()

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

    @property
    def cls_image_shape(self):
        return self._manager.cls_image_shape

    @cls_image_shape.setter
    def cls_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `cls_image_shape` must be type of list."
        self._manager.cls_image_shape = value

    def disable_normalize(self):
        """
        This function will disable normalize in preprocessing step.
        """
        self._manager.disable_normalize()

    def disable_permute(self):
        """
        This function will disable hwc2chw in preprocessing step.
        """
        self._manager.disable_permute()


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
        """
        Return the cls_thresh of ClassifierPostprocessor
        """
        return self._postprocessor.cls_thresh

    @cls_thresh.setter
    def cls_thresh(self, value):
        """Set the cls_thresh for ClassifierPostprocessor

        :param: value: the value of cls_thresh
        """
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

    def clone(self):
        """Clone OCR classification model object
        :return: a new OCR classification model object
        """

        class ClassifierClone(Classifier):
            def __init__(self, model):
                self._model = model

        clone_model = ClassifierClone(self._model.clone())
        return clone_model

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


class RecognizerPreprocessor(ProcessorManager):
    def __init__(self):
        """Create a preprocessor for RecognizerModel
        """
        super(RecognizerPreprocessor, self).__init__()
        self._manager = C.vision.ocr.RecognizerPreprocessor()

    @property
    def static_shape_infer(self):
        return self._manager.static_shape_infer

    @static_shape_infer.setter
    def static_shape_infer(self, value):
        assert isinstance(
            value,
            bool), "The value to set `static_shape_infer` must be type of bool."
        self._manager.static_shape_infer = value

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

    @property
    def rec_image_shape(self):
        return self._manager.rec_image_shape

    @rec_image_shape.setter
    def rec_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `rec_image_shape` must be type of list."
        self._manager.rec_image_shape = value

    def disable_normalize(self):
        """
        This function will disable normalize in preprocessing step.
        """
        self._manager.disable_normalize()

    def disable_permute(self):
        """
        This function will disable hwc2chw in preprocessing step.
        """
        self._manager.disable_permute()


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

    def clone(self):
        """Clone OCR recognition model object
        :return: a new OCR recognition model object
        """

        class RecognizerClone(Recognizer):
            def __init__(self, model):
                self._model = model

        clone_model = RecognizerClone(self._model.clone())
        return clone_model

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
    def static_shape_infer(self):
        return self._model.preprocessor.static_shape_infer

    @static_shape_infer.setter
    def static_shape_infer(self, value):
        assert isinstance(
            value,
            bool), "The value to set `static_shape_infer` must be type of bool."
        self._model.preprocessor.static_shape_infer = value

    @property
    def rec_image_shape(self):
        return self._model.preprocessor.rec_image_shape

    @rec_image_shape.setter
    def rec_image_shape(self, value):
        assert isinstance(
            value,
            list), "The value to set `rec_image_shape` must be type of list."
        self._model.preprocessor.rec_image_shape = value


class StructureV2TablePreprocessor:
    def __init__(self):
        """Create a preprocessor for StructureV2Table Model
        """
        self._preprocessor = C.vision.ocr.StructureV2TablePreprocessor()

    def run(self, input_ims):
        """Preprocess input images for StructureV2TableModel
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)


class StructureV2TablePostprocessor:
    def __init__(self):
        """Create a postprocessor for StructureV2Table Model
        """
        self._postprocessor = C.vision.ocr.StructureV2TablePostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for StructureV2Table Model
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class StructureV2Table(FastDeployModel):
    def __init__(self,
                 model_file="",
                 params_file="",
                 table_char_dict_path="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load StructureV2Table model provided by PP-StructureV2.

        :param model_file: (str)Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param table_char_dict_path: (str)Path of table_char_dict file, e.g ../ppocr/utils/dict/table_structure_dict_ch.txt
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model.
        """
        super(StructureV2Table, self).__init__(runtime_option)

        if (len(model_file) == 0):
            self._model = C.vision.ocr.StructureV2Table()
            self._runnable = False
        else:
            self._model = C.vision.ocr.StructureV2Table(
                model_file, params_file, table_char_dict_path,
                self._runtime_option, model_format)
            assert self.initialized, "Classifier initialize failed."
            self._runnable = True

    def clone(self):
        """Clone StructureV2Table model object
        :return: a new StructureV2Table model object
        """

        class StructureV2TableClone(StructureV2Table):
            def __init__(self, model):
                self._model = model

        clone_model = StructureV2TableClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: bbox, structure
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of bbox list, list of structure
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


class StructureV2LayoutPreprocessor:
    def __init__(self):
        """Create a preprocessor for StructureV2Layout Model
        """
        self._preprocessor = C.vision.ocr.StructureV2LayoutPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for StructureV2Layout Model
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)


class StructureV2LayoutPostprocessor:
    def __init__(self):
        """Create a postprocessor for StructureV2Layout Model
        """
        self._postprocessor = C.vision.ocr.StructureV2LayoutPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for StructureV2Layout Model
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class StructureV2Layout(FastDeployModel):
    def __init__(self,
                 model_file="",
                 params_file="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load StructureV2Layout model provided by PP-StructureV2.

        :param model_file: (str)Path of model file, e.g ./picodet_lcnet_x1_0_fgd_layout_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./picodet_lcnet_x1_0_fgd_layout_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model.
        """
        super(StructureV2Layout, self).__init__(runtime_option)

        if (len(model_file) == 0):
            self._model = C.vision.ocr.StructureV2Layout()
            self._runnable = False
        else:
            self._model = C.vision.ocr.StructureV2Layout(
                model_file, params_file, self._runtime_option, model_format)
            assert self.initialized, "StructureV2Layout model initialize failed."
            self._runnable = True

    def clone(self):
        """Clone StructureV2Layout model object
        :return: a new StructureV2Table model object
        """

        class StructureV2LayoutClone(StructureV2Layout):
            def __init__(self, model):
                self._model = model

        clone_model = StructureV2LayoutClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: bboxes
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of bboxes list
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

class PPOCRv4(FastDeployModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (FastDeployModel) The detection model object created by fastdeploy.vision.ocr.DBDetector.
        :param cls_model: (FastDeployModel) The classification model object created by fastdeploy.vision.ocr.Classifier.
        :param rec_model: (FastDeployModel) The recognition model object created by fastdeploy.vision.ocr.Recognizer.
        """
        assert det_model is not None and rec_model is not None, "The det_model and rec_model cannot be None."
        if cls_model is None:
            self.system_ = C.vision.ocr.PPOCRv4(det_model._model,
                                                rec_model._model)
        else:
            self.system_ = C.vision.ocr.PPOCRv4(
                det_model._model, cls_model._model, rec_model._model)

    def clone(self):
        """Clone PPOCRv4 pipeline object
        :return: a new PPOCRv4 pipeline object
        """

        class PPOCRv4Clone(PPOCRv4):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPOCRv4Clone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """
        return self.system_.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system_.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `cls_batch_size` must be type of int."
        self.system_.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system_.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `rec_batch_size` must be type of int."
        self.system_.rec_batch_size = value

class PPOCRSystemv4(PPOCRv4):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv4 is deprecated, "
            "please use fd.vision.ocr.PPOCRv4 instead.")
        super(PPOCRSystemv4, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv4, self).predict(input_image)

class PPOCRv3(FastDeployModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (FastDeployModel) The detection model object created by fastdeploy.vision.ocr.DBDetector.
        :param cls_model: (FastDeployModel) The classification model object created by fastdeploy.vision.ocr.Classifier.
        :param rec_model: (FastDeployModel) The recognition model object created by fastdeploy.vision.ocr.Recognizer.
        """
        assert det_model is not None and rec_model is not None, "The det_model and rec_model cannot be None."
        if cls_model is None:
            self.system_ = C.vision.ocr.PPOCRv3(det_model._model,
                                                rec_model._model)
        else:
            self.system_ = C.vision.ocr.PPOCRv3(
                det_model._model, cls_model._model, rec_model._model)

    def clone(self):
        """Clone PPOCRv3 pipeline object
        :return: a new PPOCRv3 pipeline object
        """

        class PPOCRv3Clone(PPOCRv3):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPOCRv3Clone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """
        return self.system_.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system_.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `cls_batch_size` must be type of int."
        self.system_.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system_.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `rec_batch_size` must be type of int."
        self.system_.rec_batch_size = value


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
            self.system_ = C.vision.ocr.PPOCRv2(det_model._model,
                                                rec_model._model)
        else:
            self.system_ = C.vision.ocr.PPOCRv2(
                det_model._model, cls_model._model, rec_model._model)

    def clone(self):
        """Clone PPOCRv3 pipeline object
        :return: a new PPOCRv3 pipeline object
        """

        class PPOCRv2Clone(PPOCRv2):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPOCRv2Clone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """

        return self.system_.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system_.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `cls_batch_size` must be type of int."
        self.system_.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system_.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value,
            int), "The value to set `rec_batch_size` must be type of int."
        self.system_.rec_batch_size = value


class PPOCRSystemv2(PPOCRv2):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv2 is deprecated, "
            "please use fd.vision.ocr.PPOCRv2 instead.")
        super(PPOCRSystemv2, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv2, self).predict(input_image)


class PPStructureV2Table(FastDeployModel):
    def __init__(self, det_model=None, rec_model=None, table_model=None):
        """Consruct a pipeline with text detector, text recognizer and table recognizer models

        :param det_model: (FastDeployModel) The detection model object created by fastdeploy.vision.ocr.DBDetector.
        :param rec_model: (FastDeployModel) The recognition model object created by fastdeploy.vision.ocr.Recognizer.
        :param table_model: (FastDeployModel) The table recognition model object created by fastdeploy.vision.ocr.Table.
        """
        assert det_model is not None and rec_model is not None and table_model is not None, "The det_model, rec_model and table_model cannot be None."
        self.system_ = C.vision.ocr.PPStructureV2Table(
            det_model._model,
            rec_model._model,
            table_model._model, )

    def clone(self):
        """Clone PPStructureV2Table pipeline object
        :return: a new PPStructureV2Table pipeline object
        """

        class PPStructureV2TableClone(PPStructureV2Table):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPStructureV2TableClone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """

        return self.system_.batch_predict(images)


class PPStructureV2TableSystem(PPStructureV2Table):
    def __init__(self, det_model=None, rec_model=None, table_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPStructureV2TableSystem is deprecated, "
            "please use fd.vision.ocr.PPStructureV2Table instead.")
        super(PPStructureV2TableSystem, self).__init__(det_model, rec_model,
                                                       table_model)

    def predict(self, input_image):
        return super(PPStructureV2TableSystem, self).predict(input_image)
