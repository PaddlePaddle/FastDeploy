# -*- coding: utf-8 -*-
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import numpy as np
import os

import fastdeploy
from fastdeploy.text import UIEModel, SchemaLanguage

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("input:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("output:", self.output_names)

        #Init fastdeploy.RuntimeOption
        runtime_option = fastdeploy.RuntimeOption()
        options = None
        if (args['model_instance_kind'] == 'GPU'):
            runtime_option.use_gpu(int(args['model_instance_device_id']))
            options = self.model_config['optimization'][
                'execution_accelerators']['gpu_execution_accelerator']
        else:
            runtime_option.use_cpu()
            options = self.model_config['optimization'][
                'execution_accelerators']['cpu_execution_accelerator']

        for option in options:
            if option['name'] == 'paddle':
                runtime_option.use_paddle_infer_backend()
            elif option['name'] == 'onnxruntime':
                runtime_option.use_ort_backend()
            elif option['name'] == 'openvino':
                runtime_option.use_openvino_backend()

            if option['parameters']:
                if 'cpu_threads' in option['parameters']:
                    runtime_option.set_cpu_thread_num(
                        int(option['parameters']['cpu_threads']))

        model_path = os.path.abspath(os.path.dirname(
            __file__)) + "/model.pdmodel"
        param_path = os.path.abspath(os.path.dirname(
            __file__)) + "/model.pdiparams"
        vocab_path = os.path.abspath(os.path.dirname(__file__)) + "/vocab.txt"
        schema = []
        # init UIE model
        self.uie_model_ = UIEModel(
            model_path,
            param_path,
            vocab_path,
            position_prob=0.5,
            max_length=128,
            schema=schema,
            runtime_option=runtime_option,
            schema_language=SchemaLanguage.ZH)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request,
                                                      self.input_names[0])
            schema = pb_utils.get_input_tensor_by_name(request,
                                                       self.input_names[1])
            texts = texts.as_numpy()
            schema = schema.as_numpy()
            # not support batch predict
            texts = json.loads(texts[0][0])
            schema = json.loads(schema[0][0])

            if schema:
                self.uie_model_.set_schema(schema)
            results = self.uie_model_.predict(texts, return_dict=True)

            results = np.array(results, dtype=np.object_)
            out_tensor = pb_utils.Tensor(self.output_names[0], results)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor, ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
