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
import os
from unittest import result

from .... import FastDeployModel, ModelFormat
from .... import c_lib_wrap as C

from paddleocr import PaddleOCR

from .vqa_utils import *
from .transforms import *
from .operators import *
from pathlib import Path


class SER_Preprocessor():
    def __init__(self, ser_dict_path):
        self._manager = None
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            det_model_dir=None,
            rec_model_dir=None,
            show_log=False,
            use_gpu=True)

        pre_process_list = [{
            'VQATokenLabelEncode': {
                'class_path': ser_dict_path,
                'contains_re': False,
                'ocr_engine': self.ocr_engine,
                'order_method': "tb-yx"
            }
        }, {
            'VQATokenPad': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'VQASerTokenChunk': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'Resize': {
                'size': [224, 224]
            }
        }, {
            'NormalizeImage': {
                'std': [58.395, 57.12, 57.375],
                'mean': [123.675, 116.28, 103.53],
                'scale': '1',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]
            }
        }]

        self.preprocess_op = create_operators(pre_process_list,
                                              {'infer_mode': True})

    def _transform(self, data, ops=None):
        """ transform """
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def run(self, input_ims):
        ori_im = input_ims.copy()
        data = {'image': input_ims}
        data = transform(data, self.preprocess_op)

        for idx in range(len(data)):
            if isinstance(data[idx], np.ndarray):
                data[idx] = np.expand_dims(data[idx], axis=0)
            else:
                data[idx] = [data[idx]]

        return data


class SER_Postprocessor():
    def __init__(self, class_path):
        self.postprocessor_op = VQASerTokenLayoutLMPostProcess(class_path)

    def run(self, preds, batch=None, *args, **kwargs):
        return self.postprocessor_op(preds, batch, *args, **kwargs)


class SERViLayoutxlmModel(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 ser_dict_path,
                 class_path,
                 config_file="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        super(SERViLayoutxlmModel, self).__init__(runtime_option)

        assert self._runtime_option.backend != 0, \
            "Runtime Option required backend setting."
        self._model = C.vision.structure_test.SERViLayoutxlmModel(
            model_file, params_file, config_file, self._runtime_option,
            model_format)

        assert self.initialized, "SERViLayoutxlm model initialize failed."

        self.preprocessor = SER_Preprocessor(ser_dict_path)
        self.postprocesser = SER_Postprocessor(class_path)

        self.input_name_0 = self._model.get_input_info(0).name
        self.input_name_1 = self._model.get_input_info(1).name
        self.input_name_2 = self._model.get_input_info(2).name
        self.input_name_3 = self._model.get_input_info(3).name

    def predict(self, image):
        assert isinstance(image,
                          np.ndarray), "predict recives numpy.ndarray(BGR)"

        data = self.preprocessor.run(image)
        infer_input = {
            self.input_name_0: data[0],
            self.input_name_1: data[1],
            self.input_name_2: data[2],
            self.input_name_3: data[3],
        }

        infer_result = self._model.infer(infer_input)
        infer_result = infer_result[0]

        post_result = self.postprocesser.run(infer_result,
                                             segment_offset_ids=data[6],
                                             ocr_infos=data[7])

        return post_result

    def batch_predict(self, image_list):
        assert isinstance(image_list, list) and \
             isinstance(image_list[0], np.ndarray), \
              "batch_predict recives list of numpy.ndarray(BGR)"

        # reading and preprocessing images
        datas = None
        for image in image_list:
            data = self.preprocessor.run(image)

            # concatenate data to batch
            if datas == None:
                datas = data
            else:
                for idx in range(len(data)):
                    if isinstance(data[idx], np.ndarray):
                        datas[idx] = np.concatenate(
                            (datas[idx], data[idx]), axis=0)
                    else:
                        datas[idx].extend(data[idx])

        # infer
        infer_inputs = {
            self.input_name_0: datas[0],
            self.input_name_1: datas[1],
            self.input_name_2: datas[2],
            self.input_name_3: datas[3],
        }

        infer_results = self._model.infer(infer_inputs)
        infer_results = infer_results[0]

        # postprocessing
        post_results = self.postprocesser.run(infer_results,
                                              segment_offset_ids=datas[6],
                                              ocr_infos=datas[7])

        return post_results
